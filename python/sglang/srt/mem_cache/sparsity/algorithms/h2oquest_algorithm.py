"""
H2OQuest sparse attention algorithm.

Novel hybrid combining Quest's bounding-box page scoring (prefill-time,
query-aware) with H2O-style temporal accumulation of scores across decode
steps (captures persistent importance). Addresses Quest's statelessness
and H2O's phase restriction.

Key design choices:
- Scoring and accumulation are separated: _retrieve_page_scores is a pure
  read; _finalize_step_accumulation runs once per decode step at the last layer.
- Score buffers are shared across layer groups (score_layer_group_size=4)
  using float32. Within each layer group, the last layer's fresh score
  overwrites earlier layers'. This is intentional — H2O paper Fig 10 shows
  heavy hitter patterns are highly consistent across adjacent layers
  (validated on OPT; Qwen3-VL cross-layer consistency is untested — W6).
- Attention sinks (first N pages) and vision token pages are force-retained.
- GQA: max-across-groups aggregation (retain if ANY query head thinks page
  is important), then sum across KV heads for page-level score. Base class
  retrieve_topk calls _retrieve_page_scores per-request (B=1), so the 5D
  broadcast intermediate is ~33MB, acceptable on any modern GPU.
- Full-buffer decay each step prevents stale scores on pages that temporarily
  drop out of topk. Global decay over-decays inactive requests' pages (I3);
  acceptable for browser agent workloads where batch size is typically 1.
- Stale scores cleared on page reallocation (physical page recycling).
- Alpha resets on new prefill (construct_representations).
- Both fresh and accumulated scores are normalized to [0,1] before alpha
  blending to prevent scale mismatch (W2 fix).
- _pending_fresh accumulates across requests in a batch via torch.cat (F2 fix).
  scatter_add_ used for deterministic accumulation with duplicate indices (W4 fix).
"""

import logging
import math
from typing import TYPE_CHECKING

import torch

from sglang.srt.mem_cache.sparsity.algorithms.base_algorithm import (
    BaseSparseAlgorithmImpl,
)

if TYPE_CHECKING:
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch

logger = logging.getLogger(__name__)


class H2OQuestAlgorithm(BaseSparseAlgorithmImpl):
    """H2OQuest: Quest bounding-box scoring with H2O-style temporal accumulation.

    Uses Quest's bounding-box criticality (upper bound on attention) as the base
    scoring signal, accumulated across decode steps with exponential decay.
    Novel hybrid addressing Quest's statelessness and H2O's phase restriction.
    """

    def __init__(self, config, device: torch.device, **kwargs):
        super().__init__(config, device, **kwargs)
        self.score_decay = config.sparse_extra_config.get("score_decay", 0.95)
        self.initial_alpha = config.sparse_extra_config.get("initial_alpha", 1.0)
        self.alpha_decay = config.sparse_extra_config.get("alpha_decay", 0.9)
        self.num_sink_pages = config.sparse_extra_config.get("num_sink_pages", 2)
        self.score_layer_group_size = config.sparse_extra_config.get(
            "score_layer_group_size", 4
        )
        self.protect_vision_tokens = config.sparse_extra_config.get(
            "protect_vision_tokens", True
        )
        self.image_token_id = config.sparse_extra_config.get("image_token_id", 151655)
        self.video_token_id = config.sparse_extra_config.get("video_token_id", 151656)

        # Quest bounding-box representations (per layer)
        self.page_k_min = {}
        self.page_k_max = {}
        self.page_valid = {}

        # H2O accumulated scores (per layer group, float32 — F1 fix: float16 overflows
        # after ~10 decode steps since Quest scores sum to ~1000-9000 per step and
        # accumulated sum reaches ~18x that, exceeding float16 max of 65504)
        self.accumulated_scores = {}
        self.num_groups = 0

        # Per-request decode step counter
        self.decode_steps = None

        # Vision page mask (precomputed via mark_vision_pages after prefill)
        self.vision_page_mask = None

        # Pending fresh scores for deferred accumulation.
        # F2 fix: accumulates across requests in a batch via torch.cat,
        # not overwritten per-request.
        self._pending_fresh = {}

    def _initialize_representation_pools(
        self, start_layer: int, end_layer: int, total_num_pages: int
    ):
        key_buf = self.token_to_kv_pool.get_key_buffer(start_layer)
        head_num, head_dim = key_buf.shape[1], key_buf.shape[2]

        # F3 fix: Warn if CUDA graphs are enabled (incompatible with Python
        # dict/loop/branch control flow in retrieve_topk and _finalize_step_accumulation)
        try:
            from sglang.srt.server_args import get_global_server_args

            server_args = get_global_server_args()
            if server_args is not None and not getattr(
                server_args, "disable_cuda_graph", True
            ):
                logger.warning(
                    "H2OQuest sparse attention is incompatible with CUDA graphs. "
                    "Use --disable-cuda-graph to avoid incorrect results."
                )
        except (ImportError, ValueError):
            pass

        # Warn if page_size=1 would require excessive memory
        if self.page_size == 1 and total_num_pages > 50000:
            logger.warning(
                "page_size=1 with %d pages requires ~%.1f GB for bounding-box "
                "representations. Consider page_size>=4.",
                total_num_pages,
                total_num_pages
                * head_num
                * head_dim
                * 4
                * 2
                * (end_layer - start_layer)
                / 1e9,
            )

        # Quest bounding-box representations (per layer, full precision)
        for layer_id in range(start_layer, end_layer):
            self.page_k_min[layer_id] = torch.zeros(
                (total_num_pages, head_num, head_dim),
                dtype=torch.float32,
                device=self.device,
            )
            self.page_k_max[layer_id] = torch.zeros_like(self.page_k_min[layer_id])
            self.page_valid[layer_id] = torch.zeros(
                total_num_pages, dtype=torch.bool, device=self.device
            )

        # H2O accumulated scores (per layer GROUP, float32 — F1 fix)
        self.num_groups = math.ceil(
            (end_layer - start_layer) / self.score_layer_group_size
        )
        for g in range(self.num_groups):
            self.accumulated_scores[g] = torch.zeros(
                (total_num_pages,), dtype=torch.float32, device=self.device
            )

        # Per-request decode step counter
        max_pool_size = self.req_to_token_pool.req_to_token.shape[0]
        self.decode_steps = torch.zeros(
            max_pool_size, dtype=torch.int32, device=self.device
        )

        # Vision page mask
        self.vision_page_mask = torch.zeros(
            total_num_pages, dtype=torch.bool, device=self.device
        )

        logger.info(
            "Initialized H2OQuest: %d pages, %d layers (%d groups), "
            "head_num=%d, head_dim=%d, score_decay=%.3f, "
            "alpha_decay=%.3f, num_sink_pages=%d, protect_vision=%s",
            total_num_pages,
            end_layer - start_layer,
            self.num_groups,
            head_num,
            head_dim,
            self.score_decay,
            self.alpha_decay,
            self.num_sink_pages,
            self.protect_vision_tokens,
        )

    def construct_representations(
        self, layer_id, req_pool_indices, seq_lens, k_buffer, forward_batch
    ):
        """Override to reset decode steps, set prompt_lens, and mark vision pages."""
        if layer_id == self.start_layer:
            self.decode_steps[req_pool_indices] = 0
            # Set prompt_lens for new requests so _compute_sparse_mask works.
            # Without this, prompt_lens stays 0 (from RequestTrackers init) and
            # sparse attention is never applied during decode.
            new_mask = ~self.states.repr_constructed[req_pool_indices]
            if new_mask.any():
                self.states.prompt_lens[req_pool_indices[new_mask]] = seq_lens[
                    new_mask
                ]
            # Mark vision pages from batch token IDs (first prefill only)
            self._mark_vision_pages_from_batch(req_pool_indices, forward_batch)
        super().construct_representations(
            layer_id, req_pool_indices, seq_lens, k_buffer, forward_batch
        )

    def _compute_page_representations(
        self,
        layer_id: int,
        reqs: torch.Tensor,
        seq_lens: torch.Tensor,
        start_page,
        end_page: torch.Tensor,
        k_buffer: torch.Tensor,
    ):
        """Compute Quest bounding-box reps. Reuses Quest's exact logic."""
        if isinstance(start_page, int):
            start_page = torch.full_like(end_page, start_page)

        device = k_buffer.device
        req_to_token = self.req_to_token_pool.req_to_token
        n = reqs.shape[0]
        max_pages = int((end_page - start_page).max().item())
        if max_pages <= 0:
            return

        # Gather keys for each page (identical to Quest)
        pg_off = torch.arange(max_pages, device=device).unsqueeze(0)
        pg_id = start_page.unsqueeze(1) + pg_off
        pg_mask = pg_id < end_page.unsqueeze(1)

        tok_start = pg_id * self.page_size
        tok_off = torch.arange(self.page_size, device=device).view(1, 1, -1)
        tok_pos = tok_start.unsqueeze(2) + tok_off
        tok_mask = (
            tok_pos
            < (tok_start + self.page_size)
            .clamp(max=seq_lens.unsqueeze(1))
            .unsqueeze(2)
        ) & pg_mask.unsqueeze(2)

        phys_tok = req_to_token[
            reqs.view(n, 1, 1).expand(n, max_pages, self.page_size),
            tok_pos.clamp(0, req_to_token.shape[1] - 1),
        ].clamp(0, k_buffer.shape[0] - 1)

        keys = k_buffer[phys_tok].to(torch.float32)
        mask = tok_mask.unsqueeze(-1).unsqueeze(-1)

        # Quest bounding-box: per-page min/max
        page_min = torch.where(
            mask, keys, torch.full_like(keys, float("inf"))
        ).amin(dim=2)
        page_max = torch.where(
            mask, keys, torch.full_like(keys, float("-inf"))
        ).amax(dim=2)

        phys_pg = (
            req_to_token[
                reqs.unsqueeze(1).expand(n, max_pages),
                tok_start.clamp(0, req_to_token.shape[1] - 1),
            ]
            // self.page_size
        )

        idx = pg_mask.nonzero(as_tuple=False)
        if idx.numel() == 0:
            return

        target_pages = phys_pg[idx[:, 0], idx[:, 1]].clamp(
            0, self.page_k_min[layer_id].shape[0] - 1
        )
        self.page_k_min[layer_id][target_pages] = page_min[idx[:, 0], idx[:, 1]]
        self.page_k_max[layer_id][target_pages] = page_max[idx[:, 0], idx[:, 1]]
        self.page_valid[layer_id][target_pages] = True

        # H2OQuest-specific: clear stale scores on page reallocation
        if layer_id == self.start_layer:
            for g in range(self.num_groups):
                self.accumulated_scores[g][target_pages] = 0

            # Vision token protection: mark_vision_pages is called from
            # construct_representations via _mark_vision_pages_from_batch,
            # which has access to token IDs from the ForwardBatch.
            if self.protect_vision_tokens:
                logger.debug(
                    "Vision token protection active. Pages marked via "
                    "_mark_vision_pages_from_batch in construct_representations."
                )

    def _mark_vision_pages_from_batch(
        self, req_pool_indices: torch.Tensor, forward_batch: "ForwardBatch"
    ):
        """Mark vision pages using token IDs from the current extend batch.

        Called at the first layer during construct_representations (prefill).
        Uses extend_start_loc and extend_seq_lens to split the flattened
        input_ids back into per-request token ID segments.
        """
        if not self.protect_vision_tokens:
            return
        if not hasattr(forward_batch, "extend_start_loc"):
            return
        if forward_batch.extend_start_loc is None:
            return

        input_ids = forward_batch.input_ids
        extend_start_loc = forward_batch.extend_start_loc
        extend_seq_lens = forward_batch.extend_seq_lens

        for i in range(req_pool_indices.shape[0]):
            req_idx = req_pool_indices[i].item()
            # Only mark on first prefill (repr not yet constructed)
            if self.states.repr_constructed[req_idx]:
                continue
            start = extend_start_loc[i].item()
            length = extend_seq_lens[i].item()
            token_ids = input_ids[start : start + length]
            # Prefix cache offset: extend tokens start at absolute position
            # prefix_len in the request. mark_vision_pages looks up
            # req_to_token[req_pool_idx, pos], so positions must be absolute.
            total_seq = forward_batch.seq_lens[i].item()
            prefix_len = total_seq - length
            self.mark_vision_pages(req_idx, token_ids, length, position_offset=prefix_len)

    def mark_vision_pages(
        self,
        req_pool_idx: int,
        token_ids: torch.Tensor,
        seq_len: int,
        position_offset: int = 0,
    ):
        """Mark pages containing vision tokens.

        Args:
            req_pool_idx: Request pool index.
            token_ids: Token IDs for the request (1D tensor).
            seq_len: Actual sequence length (may be less than token_ids length).
            position_offset: Offset to add to token positions before looking up
                req_to_token. Needed when token_ids is a slice from a prefix-cached
                extend (positions are relative to extend start, but req_to_token
                uses absolute request positions).
        """
        tokens = token_ids[:seq_len]
        is_vision = (tokens == self.image_token_id) | (
            tokens == self.video_token_id
        )
        vision_positions = is_vision.nonzero(as_tuple=True)[0]
        if vision_positions.numel() == 0:
            return

        req_to_token = self.req_to_token_pool.req_to_token
        for pos in vision_positions:
            abs_pos = pos.item() + position_offset
            if abs_pos >= req_to_token.shape[1]:
                continue
            phys_tok = req_to_token[req_pool_idx, abs_pos]
            phys_page = phys_tok // self.page_size
            if phys_page < self.vision_page_mask.shape[0]:
                self.vision_page_mask[phys_page] = True

    def _retrieve_page_scores(
        self,
        layer_id: int,
        phys_pages: torch.Tensor,
        req_pool_indices: torch.Tensor,
        queries: torch.Tensor,
    ) -> torch.Tensor:
        """Score pages using Quest bounding-box + H2O accumulated scores.

        PURE READ — no mutation of accumulated_scores buffer. Fresh scores are
        stashed in _pending_fresh for deferred accumulation at end of step.
        """
        # 1. Fresh Quest bounding-box score (query-aware)
        fresh_score = self._quest_bounding_box_score(
            layer_id, phys_pages, req_pool_indices, queries
        )

        # 2. Read accumulated buffer (NO MUTATION — F1 fix: now float32)
        group_id = (layer_id - self.start_layer) // self.score_layer_group_size
        phys_clamped = phys_pages.clamp(
            0, self.accumulated_scores[group_id].shape[0] - 1
        )
        acc = self.accumulated_scores[group_id][phys_clamped]

        # 3. Stash fresh score for deferred accumulation.
        # DESIGN: Within each layer group, the last layer's fresh score overwrites
        # earlier layers'. Cross-layer Quest scores are highly correlated (H2O
        # paper Fig 10). Cross-layer consistency for Qwen3-VL is untested (W6).
        # F2 fix: accumulate across requests in a batch, not overwrite.
        # Flatten to 1D before storing — requests may have different page counts
        # (P_i varies per request), so 2D torch.cat along dim=0 would fail on
        # shape mismatch. The consumer (scatter_add_) uses 1D indices anyway.
        if group_id not in self._pending_fresh:
            self._pending_fresh[group_id] = (
                phys_clamped.view(-1),
                fresh_score.detach().view(-1),
            )
        else:
            prev_pages, prev_fresh = self._pending_fresh[group_id]
            self._pending_fresh[group_id] = (
                torch.cat([prev_pages, phys_clamped.view(-1)], dim=0),
                torch.cat([prev_fresh, fresh_score.detach().view(-1)], dim=0),
            )

        # 4. Inject attention sinks (match by physical page ID, not array position)
        # Robust against potential reordering of phys_pages in future base class changes.
        scores_for_blend = fresh_score.clone()
        if self.num_sink_pages > 0:
            num_pages = phys_pages.shape[1]
            if num_pages > 0:
                sink_count = min(self.num_sink_pages, num_pages)
                req_to_token = self.req_to_token_pool.req_to_token
                # Compute physical page IDs for the first sink_count logical pages
                sink_logical_starts = (
                    torch.arange(sink_count, device=self.device) * self.page_size
                )
                sink_logical_starts = sink_logical_starts.clamp(
                    max=req_to_token.shape[1] - 1
                )
                sink_phys_tok = req_to_token[
                    req_pool_indices.unsqueeze(1),
                    sink_logical_starts.unsqueeze(0).expand(
                        req_pool_indices.shape[0], -1
                    ),
                ]
                sink_phys_pages = sink_phys_tok // self.page_size
                # Match: True where phys_clamped equals any sink physical page ID
                # sink_phys_pages: [B, sink_count] -> [B, 1, sink_count]
                # phys_clamped:    [B, P]          -> [B, P, 1]
                sink_mask = (
                    phys_clamped.unsqueeze(2) == sink_phys_pages.unsqueeze(1)
                ).any(dim=2)
                sink_mask = sink_mask & self.page_valid[layer_id][phys_clamped]
                scores_for_blend = torch.where(
                    sink_mask,
                    torch.full_like(
                        scores_for_blend,
                        torch.finfo(scores_for_blend.dtype).max,
                    ),
                    scores_for_blend,
                )

        # 5. Inject vision token protection (intersected with page validity)
        if self.protect_vision_tokens:
            vision_mask = (
                self.vision_page_mask[phys_clamped]
                & self.page_valid[layer_id][phys_clamped]
            )
            if vision_mask.any():
                scores_for_blend = torch.where(
                    vision_mask,
                    torch.full_like(
                        scores_for_blend,
                        torch.finfo(scores_for_blend.dtype).max,
                    ),
                    scores_for_blend,
                )

        # 6. Alpha blending with normalization (W2 fix)
        # Without normalization, accumulated scores (~12.8x fresh after 20 steps)
        # overwhelm fresh scores, making alpha meaningless. Normalizing both to
        # [0,1] ensures alpha has its intuitive meaning: at alpha=0.5, fresh and
        # accumulated contribute equally to the ranking.
        steps = self.decode_steps[req_pool_indices].float()
        alpha = self.initial_alpha * (self.alpha_decay**steps)

        # Normalize fresh scores to [0,1] (exclude -inf from invalid/sink pages)
        valid_fresh = torch.where(
            scores_for_blend < torch.finfo(scores_for_blend.dtype).max * 0.5,
            scores_for_blend,
            torch.zeros_like(scores_for_blend),
        )
        fresh_max = valid_fresh.max(dim=-1, keepdim=True).values.clamp(min=1e-8)
        fresh_min = valid_fresh.min(dim=-1, keepdim=True).values
        fresh_range = (fresh_max - fresh_min).clamp(min=1e-8)
        fresh_normalized = (scores_for_blend - fresh_min) / fresh_range

        # Normalize accumulated scores to [0,1]
        valid_acc = torch.where(acc > float("-inf"), acc, torch.zeros_like(acc))
        acc_max = valid_acc.max(dim=-1, keepdim=True).values.clamp(min=1e-8)
        acc_min = valid_acc.min(dim=-1, keepdim=True).values
        acc_range = (acc_max - acc_min).clamp(min=1e-8)
        acc_normalized = (acc - acc_min) / acc_range

        # Re-inject float_max for sinks/vision (normalization may have changed them)
        if self.num_sink_pages > 0 or self.protect_vision_tokens:
            force_retain = scores_for_blend >= torch.finfo(scores_for_blend.dtype).max * 0.5
            fresh_normalized = torch.where(force_retain, torch.ones_like(fresh_normalized), fresh_normalized)
            acc_normalized = torch.where(force_retain, torch.ones_like(acc_normalized), acc_normalized)

        scores = alpha.unsqueeze(1) * fresh_normalized + (
            1 - alpha.unsqueeze(1)
        ) * acc_normalized

        return scores

    def _quest_bounding_box_score(
        self,
        layer_id: int,
        phys_pages: torch.Tensor,
        req_pool_indices: torch.Tensor,
        queries: torch.Tensor,
    ) -> torch.Tensor:
        """Compute Quest-style bounding-box criticality with GQA max-aggregation.

        Base class retrieve_topk calls this per-request (B=1 always), so the 5D
        broadcast intermediate is ~33MB for P=2048, kv=8, group=4, dim=128.
        Acceptable on any modern GPU. Loop kept for clarity and future-proofing
        if the base class is ever batched (W1).
        """
        phys_pages_clamped = phys_pages.clamp(
            0, self.page_k_min[layer_id].shape[0] - 1
        )

        k_min = self.page_k_min[layer_id][phys_pages_clamped]
        k_max = self.page_k_max[layer_id][phys_pages_clamped]
        valid_mask = self.page_valid[layer_id][phys_pages_clamped]

        # Align query shape to KV heads
        head_dim = k_min.shape[-1]
        if queries.dim() == 2:
            bs, hidden = queries.shape
            if hidden % head_dim != 0:
                raise ValueError(
                    f"H2OQuest query hidden size {hidden} not divisible "
                    f"by head_dim {head_dim}"
                )
            q_heads = hidden // head_dim
            q = queries.view(bs, q_heads, head_dim)
        elif queries.dim() == 3:
            q = queries
        else:
            raise ValueError(
                f"Unsupported query shape for H2OQuest: {queries.shape}"
            )

        kv_heads = k_min.shape[-2]
        q_heads = q.shape[1]

        if q_heads != kv_heads:
            if q_heads % kv_heads != 0:
                raise ValueError(
                    f"Query heads {q_heads} not divisible by KV heads {kv_heads}"
                )
            group = q_heads // kv_heads

            # GQA max-across-groups: loop over group members. B=1 at call site
            # (base class retrieve_topk iterates per-request), so peak intermediate
            # per iteration is 1 × P × kv_heads × dim × 4 bytes ≈ 8MB for P=2048.
            q_grouped = q.to(k_min.dtype).view(
                q.shape[0], kv_heads, group, head_dim
            )
            q_pos_all = q_grouped.clamp(min=0)
            q_neg_all = q_grouped.clamp(max=0)

            max_crit = torch.full(
                (q.shape[0], phys_pages_clamped.shape[1], kv_heads),
                float("-inf"),
                device=q.device,
                dtype=k_min.dtype,
            )
            for g_idx in range(group):
                # [B, 1, kv_heads, dim]
                q_g_pos = q_pos_all[:, :, g_idx, :].unsqueeze(1)
                q_g_neg = q_neg_all[:, :, g_idx, :].unsqueeze(1)
                # [B, P, kv_heads]
                crit_g = (q_g_pos * k_max + q_g_neg * k_min).sum(dim=-1)
                max_crit = torch.max(max_crit, crit_g)

            # Sum across KV heads for page-level score
            criticality = max_crit.sum(dim=-1)  # [B, P]
        else:
            # MHA: standard Quest scoring
            q = q.to(k_min.dtype).unsqueeze(1)  # [bs, 1, kv_heads, dim]
            criticality = torch.where(q >= 0, q * k_max, q * k_min).sum(
                dim=(2, 3)
            )

        criticality = torch.where(
            valid_mask, criticality, torch.full_like(criticality, float("-inf"))
        )

        return criticality

    def retrieve_topk(
        self,
        queries: torch.Tensor,
        layer_id: int,
        req_pool_indices: torch.Tensor,
        sparse_mask: torch.Tensor,
        **kwargs,
    ) -> tuple:
        """Override to clear pending at start and finalize accumulation at end."""
        # Clear pending at first layer of each step (prevents stale entries
        # from previous batch if a request finished mid-step)
        if layer_id == self.start_layer:
            self._pending_fresh.clear()

        # Call parent's retrieve_topk (calls _retrieve_page_scores per-request
        # in a Python loop with B=1, handles topk + recent page selection)
        selected, lengths = super().retrieve_topk(
            queries, layer_id, req_pool_indices, sparse_mask, **kwargs
        )

        # At last layer: finalize step accumulation (ONCE per step)
        if layer_id == self.end_layer - 1:
            self._finalize_step_accumulation(req_pool_indices)

        return selected, lengths

    def _finalize_step_accumulation(self, req_pool_indices: torch.Tensor):
        """Apply decay to ALL pages + add fresh scores. Called once per decode step.

        Global decay affects all pages including those belonging to other requests
        not in the current batch. For single-request or low-batch browser agent
        workloads this is negligible. For high-throughput multi-batch serving,
        pages of intermittently-batched requests may be over-decayed. The
        stale-score clearing on page reallocation prevents incorrect scores;
        the worst case is slightly faster effective decay (I3).
        """
        for g in range(self.num_groups):
            # Decay ALL pages globally (not just selected — prevents stale high
            # scores on pages that temporarily drop out of topk)
            self.accumulated_scores[g] *= self.score_decay
            # Add fresh scores only for pages scored this step.
            # F2 fix: _pending_fresh now accumulates across requests in batch.
            # W4 fix: scatter_add_ is deterministic for duplicate indices,
            # unlike __iadd__ with advanced indexing on CUDA.
            if g in self._pending_fresh:
                phys_clamped, fresh = self._pending_fresh[g]
                self.accumulated_scores[g].scatter_add_(
                    0, phys_clamped.view(-1).long(), fresh.view(-1)
                )
        self._pending_fresh.clear()
        self.decode_steps[req_pool_indices] += 1

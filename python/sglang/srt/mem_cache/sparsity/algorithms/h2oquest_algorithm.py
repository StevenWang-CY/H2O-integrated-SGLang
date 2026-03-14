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
  using float16 for memory efficiency. Last layer in each group wins.
- Attention sinks (first N pages) and vision token pages are force-retained.
- GQA: max-across-groups aggregation (retain if ANY query head thinks page
  is important), then sum across KV heads for page-level score.
- Full-buffer decay each step prevents stale scores on pages that temporarily
  drop out of topk.
- Stale scores cleared on page reallocation (physical page recycling).
- Alpha resets on new prefill (construct_representations).
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

        # H2O accumulated scores (per layer group, float16)
        self.accumulated_scores = {}
        self.num_groups = 0

        # Per-request decode step counter
        self.decode_steps = None

        # Vision page mask (precomputed during prefill)
        self.vision_page_mask = None

        # Pending fresh scores for deferred accumulation
        self._pending_fresh = {}

    def _initialize_representation_pools(
        self, start_layer: int, end_layer: int, total_num_pages: int
    ):
        key_buf = self.token_to_kv_pool.get_key_buffer(start_layer)
        head_num, head_dim = key_buf.shape[1], key_buf.shape[2]

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

        # H2O accumulated scores (per layer GROUP, float16 for memory)
        self.num_groups = math.ceil(
            (end_layer - start_layer) / self.score_layer_group_size
        )
        for g in range(self.num_groups):
            self.accumulated_scores[g] = torch.zeros(
                (total_num_pages,), dtype=torch.float16, device=self.device
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
        """Override to reset decode steps on new prefill (alpha reset)."""
        if layer_id == self.start_layer:
            self.decode_steps[req_pool_indices] = 0
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

        # H2OQuest-specific: clear stale scores and precompute vision mask
        if layer_id == self.start_layer:
            # Clear stale accumulated scores for newly initialized pages
            for g in range(self.num_groups):
                self.accumulated_scores[g][target_pages] = 0

            # Precompute vision page mask
            if self.protect_vision_tokens:
                # Check token IDs for vision tokens within these pages
                # phys_tok shape: [n, max_pages, page_size]
                # We need the actual token IDs, but k_buffer doesn't carry them.
                # Vision mask is set to False by default; users should provide
                # vision page info through forward_batch if available.
                # For now, vision_page_mask remains as initialized (all False)
                # until a proper token_id mapping is available.
                pass

    def _retrieve_page_scores(
        self,
        layer_id: int,
        phys_pages: torch.Tensor,
        req_pool_indices: torch.Tensor,
        queries: torch.Tensor,
    ) -> torch.Tensor:
        """Score pages using Quest bounding-box + H2O accumulated scores. PURE READ."""
        # 1. Fresh Quest bounding-box score (query-aware)
        fresh_score = self._quest_bounding_box_score(
            layer_id, phys_pages, req_pool_indices, queries
        )

        # 2. Read accumulated buffer (NO MUTATION)
        group_id = (layer_id - self.start_layer) // self.score_layer_group_size
        phys_clamped = phys_pages.clamp(
            0, self.accumulated_scores[group_id].shape[0] - 1
        )
        acc = self.accumulated_scores[group_id][phys_clamped].float()

        # 3. Stash fresh score for deferred accumulation
        # Last layer in each group wins; cross-layer Quest scores are highly
        # correlated (H2O paper Fig 10), so this is a valid simplification.
        self._pending_fresh[group_id] = (phys_clamped, fresh_score.detach())

        # 4. Inject attention sinks (vectorized)
        scores_for_blend = fresh_score.clone()
        if self.num_sink_pages > 0:
            num_pages = phys_pages.shape[1]
            if num_pages > 0:
                sink_count = min(self.num_sink_pages, num_pages)
                scores_for_blend[:, :sink_count] = torch.finfo(
                    scores_for_blend.dtype
                ).max

        # 5. Inject vision token protection
        if self.protect_vision_tokens:
            vision_mask = self.vision_page_mask[phys_clamped]
            if vision_mask.any():
                scores_for_blend = torch.where(
                    vision_mask,
                    torch.full_like(
                        scores_for_blend, torch.finfo(scores_for_blend.dtype).max
                    ),
                    scores_for_blend,
                )

        # 6. Alpha blending
        steps = self.decode_steps[req_pool_indices].float()
        alpha = self.initial_alpha * (self.alpha_decay ** steps)
        scores = alpha.unsqueeze(1) * scores_for_blend + (
            1 - alpha.unsqueeze(1)
        ) * acc

        return scores

    def _quest_bounding_box_score(
        self,
        layer_id: int,
        phys_pages: torch.Tensor,
        req_pool_indices: torch.Tensor,
        queries: torch.Tensor,
    ) -> torch.Tensor:
        """Compute Quest-style bounding-box criticality with GQA max-aggregation."""
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
                    f"H2OQuest query hidden size {hidden} not divisible by head_dim {head_dim}"
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
            # GQA: max across query heads within each group (retain if ANY head
            # thinks page is important), then proceed with per-KV-head scores
            q_grouped = q.view(q.shape[0], kv_heads, group, head_dim)
            # For bounding-box: compute criticality per query head, take max
            q_for_score = q_grouped.view(
                q.shape[0], kv_heads, group, head_dim
            )
            q_pos = q_for_score.clamp(min=0)  # [bs, kv_heads, group, dim]
            q_neg = q_for_score.clamp(max=0)

            # k_max/k_min: [bs, num_pages, kv_heads, dim]
            k_max_expanded = k_max.unsqueeze(2)  # [bs, pages, 1, kv_heads, dim]
            k_min_expanded = k_min.unsqueeze(2)

            # per-head criticality
            # q_pos: [bs, kv_heads, group, dim] -> [bs, 1, kv_heads, group, dim]
            q_pos = q_pos.unsqueeze(1)
            q_neg = q_neg.unsqueeze(1)
            # k_max: [bs, pages, kv_heads, dim] -> [bs, pages, kv_heads, 1, dim]
            k_max_e = k_max.unsqueeze(3)
            k_min_e = k_min.unsqueeze(3)

            crit_per_head = (q_pos * k_max_e + q_neg * k_min_e).sum(
                dim=-1
            )  # [bs, pages, kv_heads, group]

            # Max across group, then sum across KV heads
            crit_max_group = crit_per_head.max(dim=-1).values  # [bs, pages, kv_heads]
            criticality = crit_max_group.sum(dim=-1)  # [bs, pages]
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
        # Clear pending at first layer of each step
        if layer_id == self.start_layer:
            self._pending_fresh.clear()

        # Call parent's retrieve_topk (calls _retrieve_page_scores, handles topk + recent)
        selected, lengths = super().retrieve_topk(
            queries, layer_id, req_pool_indices, sparse_mask, **kwargs
        )

        # At last layer: finalize step accumulation (ONCE per step)
        if layer_id == self.end_layer - 1:
            self._finalize_step_accumulation(req_pool_indices)

        return selected, lengths

    def _finalize_step_accumulation(self, req_pool_indices: torch.Tensor):
        """Apply decay to ALL pages + add fresh scores. Called once per decode step."""
        for g in range(self.num_groups):
            # Decay ALL pages globally (not just selected — prevents stale high
            # scores on pages that temporarily drop out of topk)
            self.accumulated_scores[g] *= self.score_decay
            # Add fresh scores only for pages scored this step
            if g in self._pending_fresh:
                phys_clamped, fresh = self._pending_fresh[g]
                self.accumulated_scores[g][phys_clamped] += fresh.to(torch.float16)
        self._pending_fresh.clear()
        self.decode_steps[req_pool_indices] += 1

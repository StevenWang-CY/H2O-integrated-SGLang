"""Standalone Quest and H2OQuest page scoring functions.

Pure PyTorch implementation with NO SGLang dependency. Extracts the exact
mathematical operations from the actual algorithm implementations:
- quest_algorithm.py: mean-aggregation GQA for Quest baseline
- h2oquest_algorithm.py: max-aggregation GQA for H2OQuest fresh scoring,
  alpha-blended with accumulated temporal scores

IMPORTANT: Quest (mean-agg) and H2OQuest (max-agg) scores are NOT directly
comparable in magnitude. Benchmarks must compare selected page SETS (recall,
Jaccard), never raw score values.

Divergence from SGLang implementation:
- Accumulation uses simple addition (not scatter_add_) because offline
  benchmarks are single-request with no duplicate page indices. This is
  identical to scatter_add_ when there are no duplicates.
- Keys are assumed to be provided directly (pre-rotation or post-rotation
  depending on caller). The adapter is agnostic to RoPE.
- All accumulated buffers are float32 (F1 fix: float16 overflows after ~5
  decode steps since Quest scores sum to ~10,000 per step).
"""

import torch
from typing import Optional


def quest_score_mean_agg(
    query: torch.Tensor,
    k_min: torch.Tensor,
    k_max: torch.Tensor,
    num_kv_heads: int = 8,
    gqa_group_size: int = 4,
) -> torch.Tensor:
    """Quest bounding-box scoring with mean-aggregation GQA.

    This matches quest_algorithm.py _retrieve_page_scores (line 157):
        q = q.view(bs, kv_heads, group, head_dim).mean(dim=2)

    For each page, computes upper bound on max(q·k) for any k in the page:
        score_d = q_d * k_max_d  if q_d >= 0
                  q_d * k_min_d  if q_d < 0
        page_score = sum over heads and dims of score_d

    GQA handling: mean across query heads within each KV group, then standard
    bounding-box scoring on the averaged query.

    Args:
        query: [batch, q_heads, head_dim] query vectors.
        k_min: [num_pages, kv_heads, head_dim] per-page key minimums.
        k_max: [num_pages, kv_heads, head_dim] per-page key maximums.
        num_kv_heads: Number of KV heads.
        gqa_group_size: Number of query heads per KV head group.

    Returns:
        [batch, num_pages] page scores.
    """
    batch_size = query.shape[0]
    head_dim = query.shape[-1]

    # Mean-aggregate query heads within each KV group
    # [batch, q_heads, dim] -> [batch, kv_heads, group, dim] -> mean -> [batch, kv_heads, dim]
    q = query.to(k_min.dtype)
    if gqa_group_size > 1:
        q = q.view(batch_size, num_kv_heads, gqa_group_size, head_dim).mean(dim=2)
    # q: [batch, kv_heads, head_dim]

    # Expand for broadcasting: [batch, 1, kv_heads, head_dim]
    q = q.unsqueeze(1)

    # k_min/k_max: [num_pages, kv_heads, head_dim] -> [1, num_pages, kv_heads, head_dim]
    k_min_exp = k_min.unsqueeze(0)
    k_max_exp = k_max.unsqueeze(0)

    # Bounding-box score: where(q >= 0, q * k_max, q * k_min)
    # [batch, num_pages, kv_heads, head_dim]
    criticality = torch.where(q >= 0, q * k_max_exp, q * k_min_exp)

    # Sum over heads and head_dim -> [batch, num_pages]
    return criticality.sum(dim=(2, 3))


def quest_score_max_agg(
    query: torch.Tensor,
    k_min: torch.Tensor,
    k_max: torch.Tensor,
    num_kv_heads: int = 8,
    gqa_group_size: int = 4,
) -> torch.Tensor:
    """Quest bounding-box scoring with max-aggregation GQA.

    This matches h2oquest_algorithm.py _quest_bounding_box_score (lines 489-575).
    Uses loop over group members to avoid 270MB+ 5D broadcast intermediate.

    GQA handling: max across query heads within each KV group (retain page if
    ANY query head deems it important), then sum across KV heads.

    Args:
        query: [batch, q_heads, head_dim] query vectors.
        k_min: [num_pages, kv_heads, head_dim] per-page key minimums.
        k_max: [num_pages, kv_heads, head_dim] per-page key maximums.
        num_kv_heads: Number of KV heads.
        gqa_group_size: Number of query heads per KV head group.

    Returns:
        [batch, num_pages] page scores.
    """
    batch_size = query.shape[0]
    num_pages = k_min.shape[0]
    head_dim = query.shape[-1]

    q = query.to(k_min.dtype)

    if gqa_group_size <= 1:
        # MHA: standard scoring (no grouping needed)
        q_exp = q.unsqueeze(1)  # [batch, 1, kv_heads, head_dim]
        criticality = torch.where(
            q_exp >= 0,
            q_exp * k_max.unsqueeze(0),
            q_exp * k_min.unsqueeze(0),
        ).sum(dim=(2, 3))
        return criticality

    # GQA: reshape query -> [batch, kv_heads, group, head_dim]
    q_grouped = q.view(batch_size, num_kv_heads, gqa_group_size, head_dim)
    q_pos_all = q_grouped.clamp(min=0)
    q_neg_all = q_grouped.clamp(max=0)

    # Initialize max criticality across group members
    max_crit = torch.full(
        (batch_size, num_pages, num_kv_heads),
        float("-inf"),
        device=query.device,
        dtype=k_min.dtype,
    )

    # Loop over group members (matching actual implementation)
    for g_idx in range(gqa_group_size):
        # [batch, 1, kv_heads, head_dim]
        q_g_pos = q_pos_all[:, :, g_idx, :].unsqueeze(1)
        q_g_neg = q_neg_all[:, :, g_idx, :].unsqueeze(1)
        # [batch, num_pages, kv_heads]
        crit_g = (q_g_pos * k_max.unsqueeze(0) + q_g_neg * k_min.unsqueeze(0)).sum(
            dim=-1
        )
        max_crit = torch.max(max_crit, crit_g)

    # Sum across KV heads -> [batch, num_pages]
    return max_crit.sum(dim=-1)


def h2oquest_score(
    query: torch.Tensor,
    k_min: torch.Tensor,
    k_max: torch.Tensor,
    accumulated: torch.Tensor,
    decode_step: int,
    cfg: dict,
    num_kv_heads: int = 8,
    gqa_group_size: int = 4,
    num_sink_pages: int = 2,
) -> tuple[torch.Tensor, torch.Tensor]:
    """H2OQuest scoring: alpha-blended fresh Quest + accumulated history.

    Matches h2oquest_algorithm.py _retrieve_page_scores lines 353-487:
      1. Compute fresh Quest bounding-box score (max-agg GQA)
      2. Inject float_max for sink pages (lines 397-433)
      3. Normalize both fresh and accumulated to [0,1], excluding +inf from
         min/max computation (W2 fix, lines 459-468)
      4. Re-inject 1.0 for sink pages after normalization (lines 478-481)
      5. Alpha-blend: alpha * fresh_norm + (1-alpha) * acc_norm

    Also matches _finalize_step_accumulation lines 603-627:
      1. Decay ALL pages: accumulated *= score_decay
      2. Add RAW fresh scores (pre-normalization): accumulated += fresh

    Accumulation uses simple addition (not scatter_add_) because offline
    benchmarks are single-request with no duplicate page indices.

    All buffers are float32 (F1 fix: float16 overflows after ~5 steps).

    Args:
        query: [batch, q_heads, head_dim] query vectors.
        k_min: [num_pages, kv_heads, head_dim] per-page key minimums.
        k_max: [num_pages, kv_heads, head_dim] per-page key maximums.
        accumulated: [num_pages] float32 accumulated score buffer.
        decode_step: Current decode step (0-indexed). Controls alpha decay.
        cfg: Dict with keys: score_decay (float), initial_alpha (float),
            alpha_decay (float).
        num_kv_heads: Number of KV heads.
        gqa_group_size: Number of query heads per KV head group.
        num_sink_pages: Number of initial pages to force-retain via score
            injection (matching actual H2OQuest sink injection).

    Returns:
        (scores [batch, num_pages], updated_accumulated [num_pages])
        Both float32.
    """
    score_decay = cfg.get("score_decay", 0.95)
    initial_alpha = cfg.get("initial_alpha", 1.0)
    alpha_decay = cfg.get("alpha_decay", 0.9)

    # 1. Fresh Quest bounding-box score (max-aggregation GQA)
    fresh = quest_score_max_agg(query, k_min, k_max, num_kv_heads, gqa_group_size)
    # fresh: [batch, num_pages]

    # 2. Inject float_max for sink pages (matching lines 397-433)
    #    In the actual code, sink pages are identified by physical page ID.
    #    In offline mode, logical page index == physical page index.
    scores_for_blend = fresh.clone()
    num_pages = k_min.shape[0]
    float_max = torch.finfo(scores_for_blend.dtype).max
    if num_sink_pages > 0:
        sink_count = min(num_sink_pages, num_pages)
        scores_for_blend[:, :sink_count] = float_max

    # 3. Read accumulated scores
    acc = accumulated.unsqueeze(0).to(scores_for_blend.dtype)

    # 4. Alpha blending with normalization (W2 fix, lines 459-487)
    alpha = initial_alpha * (alpha_decay ** decode_step)

    # Normalize fresh scores to [0,1], EXCLUDING +inf pages from min/max
    # (matching lines 460-468: valid_fresh filters out float_max pages)
    valid_fresh = torch.where(
        scores_for_blend < float_max * 0.5,
        scores_for_blend,
        torch.zeros_like(scores_for_blend),
    )
    fresh_max = valid_fresh.max(dim=-1, keepdim=True).values.clamp(min=1e-8)
    fresh_min = valid_fresh.min(dim=-1, keepdim=True).values
    fresh_range = (fresh_max - fresh_min).clamp(min=1e-8)
    fresh_normalized = (scores_for_blend - fresh_min) / fresh_range

    # Normalize accumulated scores to [0,1] (lines 470-475)
    valid_acc = torch.where(acc > float("-inf"), acc, torch.zeros_like(acc))
    acc_max = valid_acc.max(dim=-1, keepdim=True).values.clamp(min=1e-8)
    acc_min = valid_acc.min(dim=-1, keepdim=True).values
    acc_range = (acc_max - acc_min).clamp(min=1e-8)
    acc_normalized = (acc - acc_min) / acc_range

    # Re-inject 1.0 for sink pages AFTER normalization (lines 478-481)
    if num_sink_pages > 0:
        force_retain = scores_for_blend >= float_max * 0.5
        fresh_normalized = torch.where(
            force_retain, torch.ones_like(fresh_normalized), fresh_normalized
        )
        acc_normalized = torch.where(
            force_retain, torch.ones_like(acc_normalized), acc_normalized
        )

    # Blend
    scores = alpha * fresh_normalized + (1 - alpha) * acc_normalized

    # 5. Update accumulated buffer (matching _finalize_step_accumulation)
    # Decay ALL pages globally FIRST (line 616), then add RAW fresh scores
    # (lines 623-625). Note: accumulated uses RAW fresh (pre-normalization,
    # pre-sink-injection), matching actual code where _pending_fresh stores
    # fresh_score from _quest_bounding_box_score (line 387-388).
    new_accumulated = accumulated.clone().float() * score_decay
    new_accumulated = new_accumulated + fresh[0].detach().float()

    return scores, new_accumulated


def select_topk(
    scores: torch.Tensor,
    sparsity_ratio: float = 0.5,
    num_recent_pages: int = 32,
    total_pages: Optional[int] = None,
) -> set[int]:
    """Select top-k pages matching base_algorithm.py retrieve_topk (lines 262-355).

    Logic:
    1. Compute recent_start = max(num_pages - num_recent_pages, 0)
    2. Mask recent pages to -inf in scores
    3. history_pages = max(recent_start, 1)
    4. k = max(int(history_pages * sparsity_ratio), 1), clamped to history_pages
    5. Select top-k from masked scores (history region)
    6. Combine topk + recent pages

    Sink pages are NOT force-included here. In H2OQuest, sinks are retained
    via score injection (float_max → normalized to 1.0) which naturally makes
    them appear in topk. In Quest (no sink injection), sinks compete on merit
    only — matching the actual Quest implementation which has no sink handling.

    Args:
        scores: [batch, num_pages] page scores (batch=1 expected).
        sparsity_ratio: Fraction of history pages to select (default 0.5,
            actual default in base_algorithm.py is 0.7).
        num_recent_pages: Number of trailing pages to force-retain.
        total_pages: Override for total page count (default: scores.shape[1]).

    Returns:
        Set of selected page indices.
    """
    total_pages = total_pages or scores.shape[1]

    if total_pages <= num_recent_pages:
        # All pages are "recent" — no sparse selection needed
        return set(range(total_pages))

    # Recent pages: always included
    recent_start = max(0, total_pages - num_recent_pages)
    recent_pages = set(range(recent_start, total_pages))

    # History pages: [0, recent_start), scored and top-k selected
    history_pages = max(recent_start, 1)
    k = max(int(history_pages * sparsity_ratio), 1)
    k = min(k, history_pages)

    # Mask recent pages to -inf so topk picks from history only
    masked_scores = scores[0, :total_pages].clone()
    masked_scores[recent_start:] = float("-inf")

    topk_indices = masked_scores.topk(k).indices
    selected = recent_pages | set(topk_indices.tolist())

    return selected

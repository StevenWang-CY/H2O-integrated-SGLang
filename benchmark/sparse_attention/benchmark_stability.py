"""B3: Selection Stability Benchmark.

Measures how stable page selections are across multi-turn browser episodes,
particularly at "return steps" where the user revisits a previously-seen page.

H2OQuest's temporal accumulation should retain historically-important pages
that Quest (stateless) forgets. This benchmark uses SYNTHETIC random key
vectors — no GPU or model required.

Expected results:
- H2OQuest Jaccard > 0.80 vs Quest < 0.60 at return steps
- H2OQuest shows smoother selection transitions across steps

Usage:
    python benchmark/sparse_attention/benchmark_stability.py \
        --output-dir results/stability/
"""

import argparse
import csv
import os
import time
from collections import defaultdict
from datetime import datetime
from itertools import product

import torch

from scoring_adapter import (
    h2oquest_score,
    quest_score_mean_agg,
    select_topk,
)

# Qwen3-VL-4B config
NUM_KV_HEADS = 8
GQA_GROUP_SIZE = 4
NUM_Q_HEADS = NUM_KV_HEADS * GQA_GROUP_SIZE
HEAD_DIM = 128
PAGE_SIZE = 16

# Benchmark config
NUM_EPISODES = 50
NUM_TURNS = 5
PAGES_PER_TURN = 128  # ~2048 tokens per turn
SPARSITY_RATIO = 0.5
NUM_RECENT_PAGES = 32


def generate_synthetic_episode(
    num_turns: int = NUM_TURNS,
    pages_per_turn: int = PAGES_PER_TURN,
    seed: int = 0,
) -> list[dict]:
    """Generate a synthetic multi-turn browser episode.

    Turn structure:
        Turn 0-1: Topic A (pages share a distribution)
        Turn 2-3: Topic B (different distribution — topic shift)
        Turn 4:   Topic A again (return step — revisit turn 0-1's content)

    Each turn produces:
        - k_min, k_max: [pages_per_turn, kv_heads, head_dim] page bounds
        - query: [1, q_heads, head_dim] query vector

    Returns:
        List of turn dicts with keys: k_min, k_max, query, is_return_step
    """
    rng = torch.Generator().manual_seed(seed)

    # Generate two topic distributions
    topic_a_center = torch.randn(NUM_KV_HEADS, HEAD_DIM, generator=rng) * 2.0
    topic_b_center = torch.randn(NUM_KV_HEADS, HEAD_DIM, generator=rng) * 2.0

    turns = []
    for t in range(num_turns):
        is_return = t == num_turns - 1  # Last turn is the return step

        # Select topic distribution
        if t <= 1 or is_return:
            center = topic_a_center
        else:
            center = topic_b_center

        # Generate page key bounds around the topic center
        # k_min and k_max are constructed so that k_min < k_max per dimension
        noise = torch.randn(pages_per_turn, NUM_KV_HEADS, HEAD_DIM, generator=rng)
        k_center = center.unsqueeze(0) + noise * 0.5
        spread = torch.rand(pages_per_turn, NUM_KV_HEADS, HEAD_DIM, generator=rng) * 0.3 + 0.1
        k_min = k_center - spread
        k_max = k_center + spread

        # Generate query vector (from same topic)
        q_noise = torch.randn(NUM_Q_HEADS, HEAD_DIM, generator=rng)
        query = center.repeat(GQA_GROUP_SIZE, 1) + q_noise * 0.3
        query = query.unsqueeze(0)  # [1, q_heads, head_dim]

        turns.append({
            "k_min": k_min,
            "k_max": k_max,
            "query": query,
            "is_return_step": is_return,
        })

    return turns


def compute_jaccard(set_a: set, set_b: set) -> float:
    """Compute Jaccard similarity between two sets."""
    if not set_a and not set_b:
        return 1.0
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    return intersection / union if union > 0 else 0.0


NUM_DECODE_STEPS_PER_TURN = 20  # Simulate 20 decode tokens per turn


def run_episode(
    turns: list[dict],
    score_decay: float = 0.95,
    alpha_decay: float = 0.9,
    decode_steps_per_turn: int = NUM_DECODE_STEPS_PER_TURN,
) -> dict:
    """Run Quest and H2OQuest on a synthetic episode, measuring stability.

    C4 fix: Simulates multiple decode steps per turn. In the actual system,
    each turn generates N tokens, and the accumulated buffer is updated at
    each decode step. The final selection (and Jaccard measurement) uses the
    LAST decode step of each turn. Alpha resets to initial_alpha at each new
    prefill (turn), then decays across decode steps within the turn.

    Returns list of per-turn metric dicts.
    """
    cfg = {
        "score_decay": score_decay,
        "initial_alpha": 1.0,
        "alpha_decay": alpha_decay,
    }

    num_pages = turns[0]["k_min"].shape[0]

    # H2OQuest accumulated buffer (persistent across turns)
    accumulated = torch.zeros(num_pages, dtype=torch.float32)

    prev_quest_sel = None
    prev_h2oquest_sel = None
    turn_0_quest_sel = None
    turn_0_h2oquest_sel = None

    results = []

    for t_idx, turn in enumerate(turns):
        k_min = turn["k_min"]
        k_max = turn["k_max"]
        query = turn["query"]

        # Quest scoring (stateless — uses mean-agg)
        quest_scores = quest_score_mean_agg(
            query, k_min, k_max, NUM_KV_HEADS, GQA_GROUP_SIZE
        )
        quest_sel = select_topk(
            quest_scores, SPARSITY_RATIO, NUM_RECENT_PAGES, num_pages
        )

        # H2OQuest: simulate N decode steps per turn (C4 fix).
        # Alpha resets at each new prefill (decode_step=0 for first token).
        # Accumulated buffer is updated at every decode step.
        for decode_step in range(decode_steps_per_turn):
            h2oquest_scores, accumulated = h2oquest_score(
                query, k_min, k_max, accumulated, decode_step, cfg,
                NUM_KV_HEADS, GQA_GROUP_SIZE,
            )

        # Use the LAST decode step's scores for selection
        h2oquest_sel = select_topk(
            h2oquest_scores, SPARSITY_RATIO, NUM_RECENT_PAGES, num_pages
        )

        # Compute Jaccard with previous turn
        quest_jaccard_prev = (
            compute_jaccard(quest_sel, prev_quest_sel)
            if prev_quest_sel is not None else float("nan")
        )
        h2oquest_jaccard_prev = (
            compute_jaccard(h2oquest_sel, prev_h2oquest_sel)
            if prev_h2oquest_sel is not None else float("nan")
        )

        # Compute Jaccard with turn 0 (for return-step analysis)
        if t_idx == 0:
            turn_0_quest_sel = quest_sel
            turn_0_h2oquest_sel = h2oquest_sel
        quest_jaccard_t0 = compute_jaccard(quest_sel, turn_0_quest_sel)
        h2oquest_jaccard_t0 = compute_jaccard(h2oquest_sel, turn_0_h2oquest_sel)

        # Overlap between Quest and H2OQuest selections
        method_overlap = compute_jaccard(quest_sel, h2oquest_sel)

        results.append({
            "turn": t_idx,
            "is_return_step": turn["is_return_step"],
            "quest_jaccard_prev": quest_jaccard_prev,
            "h2oquest_jaccard_prev": h2oquest_jaccard_prev,
            "quest_jaccard_t0": quest_jaccard_t0,
            "h2oquest_jaccard_t0": h2oquest_jaccard_t0,
            "method_overlap": method_overlap,
            "quest_num_selected": len(quest_sel),
            "h2oquest_num_selected": len(h2oquest_sel),
        })

        prev_quest_sel = quest_sel
        prev_h2oquest_sel = h2oquest_sel

    return results


def parse_args():
    parser = argparse.ArgumentParser(description="B3: Selection Stability")
    parser.add_argument("--output-dir", type=str, default="results/stability/")
    parser.add_argument("--num-episodes", type=int, default=NUM_EPISODES)
    parser.add_argument(
        "--score-decays", nargs="+", type=float, default=[0.8, 0.9, 0.95, 0.99]
    )
    parser.add_argument(
        "--alpha-decays", nargs="+", type=float, default=[0.5, 0.7, 0.9, 0.95]
    )
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(args.output_dir, f"stability_{timestamp}.csv")

    fieldnames = [
        "score_decay", "alpha_decay", "episode", "turn", "is_return_step",
        "quest_jaccard_prev", "h2oquest_jaccard_prev",
        "quest_jaccard_t0", "h2oquest_jaccard_t0",
        "method_overlap",
        "quest_num_selected", "h2oquest_num_selected",
    ]

    print(f"B3: Selection Stability Benchmark")
    print(f"Episodes: {args.num_episodes}")
    print(f"Score decays: {args.score_decays}")
    print(f"Alpha decays: {args.alpha_decays}")
    print(f"Output: {csv_path}")

    total_configs = len(args.score_decays) * len(args.alpha_decays)
    start_time = time.time()

    with open(csv_path, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        config_idx = 0
        for score_decay, alpha_decay in product(args.score_decays, args.alpha_decays):
            config_idx += 1
            print(
                f"\n[{config_idx}/{total_configs}] "
                f"score_decay={score_decay}, alpha_decay={alpha_decay}"
            )

            # Aggregate metrics for summary
            return_step_quest_jaccards = []
            return_step_h2oquest_jaccards = []

            for ep in range(args.num_episodes):
                turns = generate_synthetic_episode(seed=ep * 1000 + config_idx)
                turn_results = run_episode(turns, score_decay, alpha_decay)

                for tr in turn_results:
                    row = {
                        "score_decay": score_decay,
                        "alpha_decay": alpha_decay,
                        "episode": ep,
                        **tr,
                    }
                    writer.writerow(row)

                    if tr["is_return_step"]:
                        return_step_quest_jaccards.append(tr["quest_jaccard_t0"])
                        return_step_h2oquest_jaccards.append(tr["h2oquest_jaccard_t0"])

                csvfile.flush()

            # Print summary for this config
            if return_step_quest_jaccards:
                q_mean = sum(return_step_quest_jaccards) / len(return_step_quest_jaccards)
                h_mean = sum(return_step_h2oquest_jaccards) / len(return_step_h2oquest_jaccards)
                delta = h_mean - q_mean
                print(
                    f"  Return-step Jaccard(t0): "
                    f"Quest={q_mean:.3f}, H2OQuest={h_mean:.3f}, "
                    f"delta={delta:+.3f}"
                )

    elapsed = time.time() - start_time
    print(f"\nCompleted in {elapsed:.1f}s")
    print(f"Results saved to: {csv_path}")


if __name__ == "__main__":
    main()

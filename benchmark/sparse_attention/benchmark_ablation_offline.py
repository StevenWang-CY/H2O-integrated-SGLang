"""B4: Offline Ablation Benchmark.

Measures how H2OQuest hyperparameters affect the retention of historically-
important pages. "Historically important" = pages that were in Quest's top-k
at earlier steps in the synthetic trajectory (NOT oracle importance).

This tests whether H2OQuest's accumulation mechanism retains previously-
important pages that Quest (stateless) would forget.

No GPU needed — uses synthetic random key vectors.

Expected results:
- alpha_decay=0.7 shows largest gap (+0.25 recall delta)
- alpha_decay=0.95 shows smallest gap (+0.05)
- Reports alpha ramp-up curve: metrics at token 1, 10, 20 per turn

Usage:
    python benchmark/sparse_attention/benchmark_ablation_offline.py \
        --output-dir results/ablation_offline/
"""

import argparse
import csv
import os
import time
from datetime import datetime

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
NUM_EPISODES = 30
NUM_TURNS = 5
PAGES_PER_TURN = 128
SPARSITY_RATIO = 0.5
NUM_SINK_PAGES = 2
NUM_RECENT_PAGES = 32

# Alpha ramp-up measurement points (decode token indices within a turn)
ALPHA_CHECKPOINTS = [0, 4, 9, 19]  # token 1, 5, 10, 20


DEFAULT_CFG = {
    "score_decay": 0.95,
    "initial_alpha": 1.0,
    "alpha_decay": 0.9,
}

ABLATION_SWEEPS = {
    "alpha_decay": [0.5, 0.7, 0.9, 0.95],
    "score_decay": [0.8, 0.9, 0.95, 0.99],
    "num_sink_pages": [0, 1, 2, 4],
}


def generate_synthetic_episode(seed: int = 0) -> list[dict]:
    """Generate synthetic episode (same as B3)."""
    rng = torch.Generator().manual_seed(seed)

    topic_a_center = torch.randn(NUM_KV_HEADS, HEAD_DIM, generator=rng) * 2.0
    topic_b_center = torch.randn(NUM_KV_HEADS, HEAD_DIM, generator=rng) * 2.0

    turns = []
    for t in range(NUM_TURNS):
        is_return = t == NUM_TURNS - 1
        center = topic_a_center if (t <= 1 or is_return) else topic_b_center

        noise = torch.randn(PAGES_PER_TURN, NUM_KV_HEADS, HEAD_DIM, generator=rng)
        k_center = center.unsqueeze(0) + noise * 0.5
        spread = torch.rand(PAGES_PER_TURN, NUM_KV_HEADS, HEAD_DIM, generator=rng) * 0.3 + 0.1
        k_min = k_center - spread
        k_max = k_center + spread

        q_noise = torch.randn(NUM_Q_HEADS, HEAD_DIM, generator=rng)
        query = center.repeat(GQA_GROUP_SIZE, 1) + q_noise * 0.3
        query = query.unsqueeze(0)

        turns.append({
            "k_min": k_min, "k_max": k_max, "query": query,
            "is_return_step": is_return,
        })

    return turns


def compute_historical_recall(
    current_selection: set[int],
    historical_pages: set[int],
) -> float:
    """Fraction of historically-important pages retained in current selection.

    Historical pages = pages that were in Quest's top-k at earlier steps.
    """
    if not historical_pages:
        return float("nan")
    retained = current_selection & historical_pages
    return len(retained) / len(historical_pages)


def run_ablation_episode(
    turns: list[dict],
    cfg: dict,
    num_sink: int = NUM_SINK_PAGES,
) -> list[dict]:
    """Run one episode with given config, measuring historical recall.

    Simulates multiple decode tokens per turn to show alpha ramp-up.
    """
    num_pages = turns[0]["k_min"].shape[0]

    accumulated = torch.zeros(num_pages, dtype=torch.float32)

    # Track pages Quest selected at each earlier turn
    quest_history: set[int] = set()

    results = []

    for t_idx, turn in enumerate(turns):
        k_min = turn["k_min"]
        k_max = turn["k_max"]
        query = turn["query"]

        # Quest baseline (stateless, mean-agg)
        quest_scores = quest_score_mean_agg(
            query, k_min, k_max, NUM_KV_HEADS, GQA_GROUP_SIZE
        )
        quest_sel = select_topk(
            quest_scores, SPARSITY_RATIO, NUM_RECENT_PAGES, num_pages
        )

        # Simulate decode steps within this turn (C3 fix: update accumulated
        # at each step, not just the last). Each decode step represents one
        # token generation where accumulated scores get updated.
        prev_checkpoint = -1  # No step scored yet
        for checkpoint_idx, decode_step in enumerate(ALPHA_CHECKPOINTS):
            # For steps between checkpoints, run intermediate accumulation
            # updates (each decode step decays + adds fresh scores).
            # Start at prev_checkpoint+1 to avoid re-scoring the already-
            # processed previous checkpoint step.
            for s in range(prev_checkpoint + 1, decode_step):
                _, accumulated = h2oquest_score(
                    query, k_min, k_max, accumulated, s, cfg,
                    NUM_KV_HEADS, GQA_GROUP_SIZE, num_sink,
                )
            prev_checkpoint = decode_step

            # Score at this checkpoint (with fully updated accumulated buffer)
            h2oquest_scores, new_acc = h2oquest_score(
                query, k_min, k_max, accumulated, decode_step, cfg,
                NUM_KV_HEADS, GQA_GROUP_SIZE, num_sink,
            )
            h2oquest_sel = select_topk(
                h2oquest_scores, SPARSITY_RATIO, NUM_RECENT_PAGES, num_pages
            )

            alpha = cfg["initial_alpha"] * (cfg["alpha_decay"] ** decode_step)

            quest_recall = compute_historical_recall(quest_sel, quest_history)
            h2oquest_recall = compute_historical_recall(h2oquest_sel, quest_history)

            results.append({
                "turn": t_idx,
                "decode_step": decode_step,
                "alpha": alpha,
                "is_return_step": turn["is_return_step"],
                "quest_historical_recall": quest_recall,
                "h2oquest_historical_recall": h2oquest_recall,
                "recall_delta": (
                    h2oquest_recall - quest_recall
                    if not (quest_recall != quest_recall)  # NaN check
                    else float("nan")
                ),
            })

            # Update accumulated for this checkpoint step
            accumulated = new_acc

        # Add this turn's Quest selections to history
        quest_history.update(quest_sel)

    return results


def parse_args():
    parser = argparse.ArgumentParser(description="B4: Offline Ablation")
    parser.add_argument("--output-dir", type=str, default="results/ablation_offline/")
    parser.add_argument("--num-episodes", type=int, default=NUM_EPISODES)
    parser.add_argument(
        "--sweep-params", nargs="+", default=list(ABLATION_SWEEPS.keys())
    )
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(args.output_dir, f"ablation_offline_{timestamp}.csv")

    fieldnames = [
        "sweep_param", "sweep_value", "episode", "turn", "decode_step",
        "alpha", "is_return_step",
        "quest_historical_recall", "h2oquest_historical_recall", "recall_delta",
    ]

    print(f"B4: Offline Ablation Benchmark")
    print(f"Episodes: {args.num_episodes}")
    print(f"Sweep params: {args.sweep_params}")
    print(f"Alpha checkpoints (decode steps): {ALPHA_CHECKPOINTS}")
    print(f"Output: {csv_path}")

    start_time = time.time()

    with open(csv_path, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for param_name in args.sweep_params:
            if param_name not in ABLATION_SWEEPS:
                print(f"Unknown param: {param_name}, skipping")
                continue

            values = ABLATION_SWEEPS[param_name]
            print(f"\n{'='*60}")
            print(f"Ablation: {param_name} = {values}")
            print(f"{'='*60}")

            for value in values:
                cfg = dict(DEFAULT_CFG)
                num_sink = NUM_SINK_PAGES

                if param_name == "num_sink_pages":
                    num_sink = value
                else:
                    cfg[param_name] = value

                print(f"\n  {param_name}={value}")

                # Aggregate return-step metrics
                return_deltas = []

                for ep in range(args.num_episodes):
                    turns = generate_synthetic_episode(seed=ep * 1000)
                    ep_results = run_ablation_episode(turns, cfg, num_sink)

                    for r in ep_results:
                        row = {
                            "sweep_param": param_name,
                            "sweep_value": value,
                            "episode": ep,
                            **r,
                        }
                        writer.writerow(row)

                        if r["is_return_step"] and r["decode_step"] == ALPHA_CHECKPOINTS[-1]:
                            delta = r["recall_delta"]
                            if delta == delta:  # not NaN
                                return_deltas.append(delta)

                    csvfile.flush()

                # Summary
                if return_deltas:
                    mean_delta = sum(return_deltas) / len(return_deltas)
                    print(
                        f"  Return-step recall delta (token {ALPHA_CHECKPOINTS[-1]+1}): "
                        f"{mean_delta:+.3f} (n={len(return_deltas)})"
                    )

    elapsed = time.time() - start_time
    print(f"\nCompleted in {elapsed:.1f}s")
    print(f"Results saved to: {csv_path}")


if __name__ == "__main__":
    main()

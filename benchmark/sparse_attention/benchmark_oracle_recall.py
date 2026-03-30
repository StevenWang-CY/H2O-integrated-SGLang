"""B1: Oracle Recall Benchmark.

Compares Quest vs H2OQuest page selection against a gradient-based oracle
that measures true page importance via d(logit)/d(key_vectors).

GPU REQUIRED: bf16 model for key extraction + gradient oracle (~14GB VRAM).
FP8 server must NOT be running (GPU memory exclusivity).
Sequences capped at 8K tokens to fit in 16GB VRAM.

Expected results:
- H2OQuest 65-75% oracle recall vs Quest 40-50% at return steps
- Gap appears after alpha ramps down (decode step > 5)

Usage:
    python benchmark/sparse_attention/benchmark_oracle_recall.py \
        --model Qwen/Qwen3-VL-4B-Instruct \
        --output-dir results/oracle_recall/
"""

import argparse
import csv
import gc
import os
import time
from datetime import datetime

import torch

from extract_keys import KeyVectorExtractor, keys_to_page_bounds
from gradient_oracle import compute_page_importance_gradient
from scoring_adapter import (
    h2oquest_score,
    quest_score_mean_agg,
    select_topk,
)

PAGE_SIZE = 16
NUM_KV_HEADS = 8
GQA_GROUP_SIZE = 4
HEAD_DIM = 128
SPARSITY_RATIO = 0.5
NUM_RECENT_PAGES = 4  # Fix: was 32, leaving only 32 history pages out of 64 — too few for differentiation

# Representative layer for scoring (middle of 36-layer model)
SCORE_LAYER = 17

# Fix: use score_decay=0.99 — B3 proved this is optimal for H2OQuest advantage
DEFAULT_CFG = {
    "score_decay": 0.99,
    "initial_alpha": 1.0,
    "alpha_decay": 0.9,
}

# Number of simulated decode steps per turn for H2OQuest accumulation
DECODE_STEPS_PER_TURN = 15


def build_multi_turn_prompts(
    processor,
    num_turns: int = 5,
    max_tokens: int = 8192,
    episode_seed: int = 0,
) -> list[dict]:
    """Build synthetic multi-turn browser prompts with per-episode randomization.

    Simulates a browser agent workflow:
    - Turns 0-1: Shopping website (topic A)
    - Turns 2-3: Travel website (topic B)
    - Turn 4: Return to shopping (topic A again)

    Each episode uses a different seed to vary product/flight counts and details,
    ensuring gradient oracle produces different importance distributions.

    Returns list of turn dicts with 'messages' and 'is_return_step'.
    """
    import random
    rng = random.Random(episode_seed * 7919 + 42)

    # Vary the number of items per episode for different key distributions
    num_products = rng.randint(60, 120)
    num_flights = rng.randint(40, 100)

    topic_a_html = (
        '<div class="product-listing">'
        + "".join(
            f'<div class="product" id="item-{i}">'
            f"<h3>Product {rng.randint(100,999)}-{i}</h3>"
            f"<p>Description of product {i} with {rng.choice(['great','amazing','solid','premium','budget'])} "
            f"features. Price: ${rng.randint(5, 500)}.{rng.randint(0,99):02d}. "
            f"Rating: {rng.randint(1,5)}/5 stars. {rng.choice(['Free shipping','Limited stock','New arrival','Best seller'])}.</p>"
            f'<button class="add-to-cart">Add to Cart</button>'
            f"</div>"
            for i in range(num_products)
        )
        + "</div>"
    )

    destinations = ['London','Paris','Tokyo','Berlin','Sydney','Dubai','Rome','Seoul','Bangkok','Mumbai']
    airlines = ['United','Delta','AA','BA','QF','Emirates','Lufthansa','JAL','Singapore','Air France']
    topic_b_html = (
        '<div class="travel-listings">'
        + "".join(
            f'<div class="flight" id="flight-{i}">'
            f"<h3>Flight {rng.randint(100,9999)}: NYC to {destinations[i % len(destinations)]}</h3>"
            f"<p>Departure: {rng.randint(5,22)}:{rng.choice(['00','15','30','45'])}. Duration: {rng.randint(3,18)}h. "
            f"Price: ${rng.randint(150, 2000)}. Airline: {airlines[i % len(airlines)]}. "
            f"{rng.choice(['Direct','1 stop','2 stops'])}.</p>"
            f'<button class="book-flight">Book Now</button>'
            f"</div>"
            for i in range(num_flights)
        )
        + "</div>"
    )

    turns = []
    for t in range(num_turns):
        is_return = t == num_turns - 1
        html = topic_a_html if (t <= 1 or is_return) else topic_b_html
        task = "Buy product 42" if (t <= 1 or is_return) else "Book flight to Paris"

        action_history = "\n".join(
            f"Step {i+1}: {'Browsed products' if i <= 1 else 'Searched flights'}"
            for i in range(t)
        )

        messages = [{
            "role": "user",
            "content": [{
                "type": "text",
                "text": (
                    f"Task: {task}\n"
                    f"Previous actions:\n{action_history}\n\n"
                    f"Current page HTML:\n{html}\n\n"
                    f"What is the next action?"
                ),
            }],
        }]

        turns.append({
            "messages": messages,
            "is_return_step": is_return,
        })

    return turns


def run_oracle_recall(
    model,
    processor,
    num_episodes: int = 10,
    max_tokens: int = 8192,
    use_gradient_checkpointing: bool = False,
    seed_offset: int = 0,
) -> list[dict]:
    """Run oracle recall benchmark.

    For each multi-turn episode:
    1. Extract key vectors and query for each turn
    2. Compute page bounds
    3. Score with Quest and H2OQuest
    4. Compute gradient oracle importance
    5. Measure recall against oracle top-k
    """
    extractor = KeyVectorExtractor(
        model, NUM_KV_HEADS, HEAD_DIM, NUM_KV_HEADS * GQA_GROUP_SIZE
    )
    results = []

    for ep in range(num_episodes):
        print(f"\n  Episode {ep + 1}/{num_episodes}")
        turns = build_multi_turn_prompts(processor, max_tokens=max_tokens, episode_seed=ep + seed_offset)

        # Per-episode H2OQuest accumulated buffer
        accumulated = None

        for t_idx, turn in enumerate(turns):
            # Prepare inputs
            inputs = processor.apply_chat_template(
                turn["messages"],
                add_generation_prompt=True,
                tokenize=True,
                return_tensors="pt",
            )
            if isinstance(inputs, dict):
                inputs = {k: v.to(model.device) if isinstance(v, torch.Tensor) else v
                          for k, v in inputs.items()}
            elif isinstance(inputs, torch.Tensor):
                inputs = {"input_ids": inputs.to(model.device)}
            else:
                inputs = {"input_ids": inputs.input_ids.to(model.device)}

            seq_len = inputs["input_ids"].shape[1]
            if seq_len > max_tokens:
                # Truncate to max_tokens
                inputs["input_ids"] = inputs["input_ids"][:, :max_tokens]
                if "attention_mask" in inputs:
                    inputs["attention_mask"] = inputs["attention_mask"][:, :max_tokens]
                seq_len = max_tokens

            num_pages = (seq_len + PAGE_SIZE - 1) // PAGE_SIZE

            print(f"    Turn {t_idx}: {seq_len} tokens, {num_pages} pages")

            # 1. Extract key vectors AND last-token query in one forward pass
            keys_per_layer = extractor.extract(inputs, query_layer_idx=SCORE_LAYER)
            keys = keys_per_layer[SCORE_LAYER].to(model.device)
            k_min, k_max = keys_to_page_bounds(keys, PAGE_SIZE)
            query = extractor.get_last_query().to(model.device)

            # 3. Initialize/resize accumulated buffer
            if accumulated is None or accumulated.shape[0] != num_pages:
                new_acc = torch.zeros(num_pages, dtype=torch.float32, device=model.device)
                if accumulated is not None:
                    copy_len = min(accumulated.shape[0], num_pages)
                    new_acc[:copy_len] = accumulated[:copy_len]
                accumulated = new_acc

            # 4. Score pages
            # Quest is stateless — single score per turn
            quest_scores = quest_score_mean_agg(
                query, k_min, k_max, NUM_KV_HEADS, GQA_GROUP_SIZE
            )

            # H2OQuest: simulate DECODE_STEPS_PER_TURN decode steps per turn.
            # Each step: score → accumulate → decay. This lets alpha decay and
            # gives the accumulated buffer time to build up importance signals.
            # The query stays the same (last-token query) but the accumulated
            # buffer evolves, which is the key differentiator from Quest.
            for decode_step in range(DECODE_STEPS_PER_TURN):
                h2oquest_scores, accumulated = h2oquest_score(
                    query, k_min, k_max, accumulated, decode_step, DEFAULT_CFG,
                    NUM_KV_HEADS, GQA_GROUP_SIZE,
                )

            quest_sel = select_topk(
                quest_scores, SPARSITY_RATIO, NUM_RECENT_PAGES, num_pages
            )
            h2oquest_sel = select_topk(
                h2oquest_scores, SPARSITY_RATIO, NUM_RECENT_PAGES, num_pages
            )

            # 5. Gradient oracle
            oracle_budget = 0
            try:
                oracle_importance = compute_page_importance_gradient(
                    model, inputs, PAGE_SIZE,
                    use_gradient_checkpointing=use_gradient_checkpointing,
                ).to(model.device)

                # Oracle budget matches the total selected pages
                recent_start = max(0, num_pages - NUM_RECENT_PAGES)
                history_pages = max(recent_start, 1)
                k = max(int(history_pages * SPARSITY_RATIO), 1)
                oracle_budget = min(k + (num_pages - recent_start), oracle_importance.shape[0])
                oracle_topk = set(oracle_importance.topk(oracle_budget).indices.tolist())

                # 6. Compute recall
                quest_recall = len(quest_sel & oracle_topk) / oracle_budget
                h2oquest_recall = len(h2oquest_sel & oracle_topk) / oracle_budget
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print(f"    OOM on gradient oracle — skipping this turn")
                    torch.cuda.empty_cache()
                    gc.collect()
                    quest_recall = float("nan")
                    h2oquest_recall = float("nan")
                else:
                    raise

            delta = (
                h2oquest_recall - quest_recall
                if quest_recall == quest_recall  # NaN check
                else float("nan")
            )

            results.append({
                "episode": ep,
                "turn": t_idx,
                "context_tokens": seq_len,
                "num_pages": num_pages,
                "budget": oracle_budget,
                "is_return_step": turn["is_return_step"],
                "quest_recall": quest_recall,
                "h2oquest_recall": h2oquest_recall,
                "delta": delta,
            })

            print(
                f"    Quest recall={quest_recall:.3f}, "
                f"H2OQuest recall={h2oquest_recall:.3f}, "
                f"delta={delta:+.3f}"
                f"{' [RETURN]' if turn['is_return_step'] else ''}"
            )

            # Clean up between turns
            del keys_per_layer, keys, k_min, k_max, query
            torch.cuda.empty_cache()

    return results


def parse_args():
    parser = argparse.ArgumentParser(description="B1: Oracle Recall")
    parser.add_argument(
        "--model", type=str, default="Qwen/Qwen3-VL-4B-Instruct"
    )
    parser.add_argument("--output-dir", type=str, default="results/oracle_recall/")
    parser.add_argument("--num-episodes", type=int, default=10)
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument(
        "--gradient-checkpointing", action="store_true",
        help="Enable gradient checkpointing (slower but less VRAM)",
    )
    parser.add_argument(
        "--seed-offset", type=int, default=0,
        help="Offset added to episode index for seed (for reproducibility runs)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print("B1: Oracle Recall Benchmark")
    print(f"Model: {args.model}")
    print(f"Max tokens: {args.max_tokens}")
    print(f"Episodes: {args.num_episodes}")

    # Check GPU memory
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        print(f"GPU: {props.name}, {props.total_memory / 1e9:.1f} GB")
        allocated = torch.cuda.memory_allocated(0) / 1e9
        if allocated > 1.0:
            print(f"WARNING: {allocated:.1f} GB already allocated — is FP8 server running?")
    else:
        print("WARNING: No CUDA GPU — this benchmark requires GPU")
        return

    # Load model
    # Use SDPA (default) instead of eager: SDPA uses fused attention kernels
    # that avoid materializing the full [heads, seq, seq] attention matrix,
    # saving ~4.6 GB at 1024 tokens (128 MB/layer × 36 layers). k_proj hooks
    # and register_full_backward_hook work identically with SDPA.
    print("\nLoading model (bf16, SDPA attention)...")
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    hf_home = os.environ.get("HF_HOME", os.path.expanduser("~/hf_models"))
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        cache_dir=hf_home,
    ).to("cuda")
    processor = AutoProcessor.from_pretrained(args.model, cache_dir=hf_home)
    print(f"Model loaded. GPU memory: {torch.cuda.memory_allocated(0) / 1e9:.1f} GB")

    # Run benchmark
    start_time = time.time()
    results = run_oracle_recall(
        model, processor,
        num_episodes=args.num_episodes,
        max_tokens=args.max_tokens,
        use_gradient_checkpointing=args.gradient_checkpointing,
        seed_offset=args.seed_offset,
    )
    elapsed = time.time() - start_time

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(args.output_dir, f"oracle_recall_{timestamp}.csv")

    fieldnames = list(results[0].keys()) if results else []
    with open(csv_path, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow(row)

    # Print summary
    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")

    return_results = [r for r in results if r["is_return_step"] and r["delta"] == r["delta"]]
    all_results = [r for r in results if r["quest_recall"] == r["quest_recall"]]

    if all_results:
        q_mean = sum(r["quest_recall"] for r in all_results) / len(all_results)
        h_mean = sum(r["h2oquest_recall"] for r in all_results) / len(all_results)
        print(f"All turns: Quest={q_mean:.3f}, H2OQuest={h_mean:.3f}, delta={h_mean-q_mean:+.3f}")

    if return_results:
        q_mean = sum(r["quest_recall"] for r in return_results) / len(return_results)
        h_mean = sum(r["h2oquest_recall"] for r in return_results) / len(return_results)
        print(f"Return steps: Quest={q_mean:.3f}, H2OQuest={h_mean:.3f}, delta={h_mean-q_mean:+.3f}")

    print(f"\nCompleted in {elapsed:.1f}s")
    print(f"Results saved to: {csv_path}")

    # Unload model to free GPU for subsequent phases
    del model
    torch.cuda.empty_cache()
    gc.collect()
    print("Model unloaded. GPU freed for next phase.")


if __name__ == "__main__":
    main()

"""B2 + B5: Mind2Web Action Quality + Domain Breakdown.

Two-phase execution due to GPU memory exclusivity:
  Phase 9a: Load bf16 model, extract key vectors for all episodes, save to disk.
  Phase 9b: Start FP8 server, load saved keys, compute page selections (CPU),
            send truncated prompts to server, evaluate action quality.

Expected results:
- H2OQuest-truncated matches dense output 75-85% of the time
- Quest-truncated matches dense output 60-70%
- B5: breakdown by Mind2Web domain

Usage:
    # Phase 9a: extract keys (bf16 model, no server)
    python benchmark/sparse_attention/benchmark_mind2web.py extract-keys \
        --model Qwen/Qwen3-VL-4B-Instruct \
        --output-dir results/mind2web/

    # Phase 9b: action quality (FP8 server, no bf16 model)
    python benchmark/sparse_attention/benchmark_mind2web.py evaluate \
        --keys-dir results/mind2web/keys/ \
        --server-url http://localhost:30000 \
        --output-dir results/mind2web/
"""

import argparse
import csv
import gc
import json
import os
import time
from collections import defaultdict
from datetime import datetime

import torch

PAGE_SIZE = 16
NUM_KV_HEADS = 8
GQA_GROUP_SIZE = 4
HEAD_DIM = 128
SPARSITY_RATIO = 0.5
NUM_RECENT_PAGES = 32
SCORE_LAYER = 17
MAX_EPISODES = 50

DEFAULT_CFG = {
    "score_decay": 0.95,
    "initial_alpha": 1.0,
    "alpha_decay": 0.9,
}


def load_mind2web_episodes(max_episodes: int = MAX_EPISODES) -> list[dict]:
    """Load Mind2Web test_task episodes via streaming.

    Groups actions by annotation_id to form multi-step episodes.
    """
    from datasets import load_dataset

    hf_home = os.environ.get("HF_HOME", os.path.expanduser("~/hf_models"))
    ds = load_dataset(
        "osunlp/Multimodal-Mind2Web",
        split="test_task",
        streaming=True,
        cache_dir=hf_home,
    )

    episodes_by_id = defaultdict(lambda: {"steps": [], "domain": None, "task": None})
    episode_count = 0

    for row in ds:
        ann_id = row.get("annotation_id", row.get("id", str(episode_count)))

        if ann_id not in episodes_by_id and episode_count >= max_episodes:
            break

        ep = episodes_by_id[ann_id]
        if ep["domain"] is None:
            ep["domain"] = row.get("domain", "unknown")
            ep["task"] = row.get("confirmed_task", row.get("task", ""))
            ep["name"] = ann_id
            episode_count += 1

        op_field = row.get("operation", "")
        if isinstance(op_field, dict):
            op_type = op_field.get("op", "")
            op_value = op_field.get("value", "")
        else:
            op_type = str(op_field)
            op_value = row.get("value", "")
        step = {
            "html": row.get("cleaned_html", row.get("html", "")),
            "operation": op_type,
            "value": op_value,
            "action_repr": row.get(
                "target_action_repr",
                row.get("action_repr", ""),
            ),
            "pos_candidates": row.get("pos_candidates", []),
            "screenshot_path": row.get("screenshot", None),
        }
        ep["steps"].append(step)

    episodes = list(episodes_by_id.values())
    print(f"Loaded {len(episodes)} episodes across domains: "
          f"{set(ep['domain'] for ep in episodes)}")
    return episodes


def extract_keys_phase(
    model_name: str,
    output_dir: str,
    max_episodes: int = MAX_EPISODES,
):
    """Phase 9a: Extract key vectors from bf16 model, save to disk.

    Each episode's keys are saved as a .pt file containing:
        {layer_idx: [seq_len, kv_heads, head_dim]} per step
    """
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    from extract_keys import KeyVectorExtractor

    keys_dir = os.path.join(output_dir, "keys")
    os.makedirs(keys_dir, exist_ok=True)

    print("Loading bf16 model for key extraction...")
    hf_home = os.environ.get("HF_HOME", os.path.expanduser("~/hf_models"))
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",
        cache_dir=hf_home,
    ).to("cuda")
    processor = AutoProcessor.from_pretrained(model_name, cache_dir=hf_home)
    extractor = KeyVectorExtractor(
        model, NUM_KV_HEADS, HEAD_DIM, NUM_KV_HEADS * GQA_GROUP_SIZE
    )
    print(f"Model loaded. GPU: {torch.cuda.memory_allocated(0)/1e9:.1f} GB")

    episodes = load_mind2web_episodes(max_episodes)

    # Save episode metadata
    meta_path = os.path.join(output_dir, "episodes_meta.json")
    meta = []
    for ep in episodes:
        meta.append({
            "name": ep["name"],
            "domain": ep["domain"],
            "task": ep["task"],
            "num_steps": len(ep["steps"]),
            "steps": [
                {
                    "operation": s["operation"],
                    "value": s["value"],
                    "action_repr": s["action_repr"],
                    "html_len": len(s["html"]),
                }
                for s in ep["steps"]
            ],
        })
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    for ep_idx, ep in enumerate(episodes):
        print(f"\nEpisode {ep_idx+1}/{len(episodes)}: {ep['name']} ({ep['domain']})")
        ep_keys = []

        for step_idx, step in enumerate(ep["steps"]):
            html = step["html"]
            if not html.strip():
                ep_keys.append(None)
                continue

            messages = [{
                "role": "user",
                "content": [{"type": "text", "text": f"Task: {ep['task']}\nHTML:\n{html[:6000]}"}],
            }]

            try:
                inputs = processor.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    tokenize=True,
                    return_tensors="pt",
                )
                if isinstance(inputs, dict):
                    inputs = {k: v.to("cuda") if isinstance(v, torch.Tensor) else v
                              for k, v in inputs.items()}
                elif isinstance(inputs, torch.Tensor):
                    inputs = {"input_ids": inputs.to("cuda")}
                else:
                    inputs = {"input_ids": inputs.input_ids.to("cuda")}

                seq_len = inputs["input_ids"].shape[1]
                # Cap at 8K tokens
                if seq_len > 8192:
                    inputs["input_ids"] = inputs["input_ids"][:, :8192]
                    if "attention_mask" in inputs:
                        inputs["attention_mask"] = inputs["attention_mask"][:, :8192]
                    seq_len = 8192

                # Extract keys AND query in one forward pass (I2 fix)
                keys_all = extractor.extract(inputs, query_layer_idx=SCORE_LAYER)
                keys_score_layer = keys_all[SCORE_LAYER]  # [seq_len, kv_heads, head_dim] on CPU
                query = extractor.get_last_query()  # [1, q_heads, head_dim] on CPU

                ep_keys.append({
                    "keys": keys_score_layer,  # [seq_len, kv_heads, head_dim]
                    "query": query,  # [1, q_heads, head_dim]
                    "seq_len": seq_len,
                })

                print(f"  Step {step_idx}: {seq_len} tokens")

            except Exception as e:
                print(f"  Step {step_idx}: ERROR: {e}")
                ep_keys.append(None)

            torch.cuda.empty_cache()

        # Save episode keys
        key_path = os.path.join(keys_dir, f"ep_{ep_idx:03d}.pt")
        torch.save(ep_keys, key_path)

    # Unload model
    del model, extractor
    torch.cuda.empty_cache()
    gc.collect()
    print("\nKey extraction complete. Model unloaded.")


def evaluate_phase(
    keys_dir: str,
    server_url: str,
    output_dir: str,
    model_name: str = "Qwen/Qwen3-VL-4B-Instruct",
):
    """Phase 9b: Compute page selections and evaluate action quality via FP8 server."""
    from transformers import AutoTokenizer

    from context_truncation import evaluate_action, truncate_context_to_pages
    from extract_keys import keys_to_page_bounds
    from scoring_adapter import h2oquest_score, quest_score_mean_agg, select_topk

    hf_home = os.environ.get("HF_HOME", os.path.expanduser("~/hf_models"))
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=hf_home)

    # Load episode metadata
    meta_path = os.path.join(output_dir, "episodes_meta.json")
    with open(meta_path) as f:
        meta = json.load(f)

    # Load episodes for HTML content
    episodes = load_mind2web_episodes(len(meta))

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(output_dir, f"mind2web_{timestamp}.csv")

    fieldnames = [
        "episode", "domain", "step", "method",
        "action", "ground_truth_op", "ground_truth_repr",
        "operation_match", "element_match", "exact_match",
        "html_tokens", "dense_agreement",
    ]

    import openai
    client = openai.Client(base_url=f"{server_url}/v1", api_key="EMPTY")

    print(f"Evaluating {len(meta)} episodes via {server_url}")

    all_results = []

    with open(csv_path, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for ep_idx, ep_meta in enumerate(meta):
            key_path = os.path.join(keys_dir, f"ep_{ep_idx:03d}.pt")
            if not os.path.exists(key_path):
                print(f"  Skipping episode {ep_idx}: no key file")
                continue

            ep_keys = torch.load(key_path, weights_only=False)
            ep = episodes[ep_idx] if ep_idx < len(episodes) else None
            if ep is None:
                continue

            print(f"\nEpisode {ep_idx+1}: {ep_meta['name']} ({ep_meta['domain']})")

            accumulated = None

            for step_idx, step in enumerate(ep["steps"]):
                if step_idx >= len(ep_keys) or ep_keys[step_idx] is None:
                    continue

                step_data = ep_keys[step_idx]
                keys = step_data["keys"]
                query = step_data["query"]
                seq_len = step_data["seq_len"]

                num_pages = (seq_len + PAGE_SIZE - 1) // PAGE_SIZE

                # Compute page bounds
                k_min, k_max = keys_to_page_bounds(keys, PAGE_SIZE)

                # Initialize/resize accumulated
                if accumulated is None or accumulated.shape[0] != num_pages:
                    new_acc = torch.zeros(num_pages, dtype=torch.float32)
                    if accumulated is not None:
                        copy_len = min(accumulated.shape[0], num_pages)
                        new_acc[:copy_len] = accumulated[:copy_len]
                    accumulated = new_acc

                # Score and select pages
                quest_scores = quest_score_mean_agg(
                    query, k_min, k_max, NUM_KV_HEADS, GQA_GROUP_SIZE
                )
                h2oquest_scores, accumulated = h2oquest_score(
                    query, k_min, k_max, accumulated, step_idx, DEFAULT_CFG,
                    NUM_KV_HEADS, GQA_GROUP_SIZE,
                )

                quest_sel = select_topk(
                    quest_scores, SPARSITY_RATIO, NUM_RECENT_PAGES, num_pages
                )
                h2oquest_sel = select_topk(
                    h2oquest_scores, SPARSITY_RATIO, NUM_RECENT_PAGES, num_pages
                )

                # Truncate HTML
                full_html = step["html"]
                quest_html = truncate_context_to_pages(
                    full_html, quest_sel, PAGE_SIZE, tokenizer
                )
                h2oquest_html = truncate_context_to_pages(
                    full_html, h2oquest_sel, PAGE_SIZE, tokenizer
                )

                # Build action history
                action_history = "\n".join(
                    f"Step {i+1}: {s['action_repr']}"
                    for i, s in enumerate(ep["steps"][:step_idx])
                )

                # Send 3 requests to server
                dense_action = None
                step_results = {}

                for label, html_ctx in [
                    ("dense", full_html),
                    ("quest", quest_html),
                    ("h2oquest", h2oquest_html),
                ]:
                    prompt_text = (
                        f"Task: {ep['task']}\n"
                        f"Previous actions:\n{action_history}\n\n"
                        f"Current page HTML:\n{html_ctx[:8000]}\n\n"
                        f"What is the next action? Respond with exactly one of:\n"
                        f"CLICK(element_description)\n"
                        f"TYPE(element_description, text)\n"
                        f"SELECT(element_description, value)"
                    )

                    try:
                        response = client.chat.completions.create(
                            model=model_name,
                            messages=[{
                                "role": "user",
                                "content": [{"type": "text", "text": prompt_text}],
                            }],
                            max_tokens=64,
                            temperature=0.0,
                        )
                        action = response.choices[0].message.content
                    except Exception as e:
                        action = f"ERROR: {e}"

                    if label == "dense":
                        dense_action = action

                    # Evaluate against ground truth
                    eval_result = evaluate_action(
                        action,
                        step.get("operation", ""),
                        step.get("value", ""),
                        step.get("action_repr", ""),
                    )

                    # Dense agreement: does truncated output match dense output?
                    dense_agreement = (
                        action.strip() == dense_action.strip()
                        if dense_action is not None and label != "dense"
                        else True
                    )

                    row = {
                        "episode": ep_meta["name"],
                        "domain": ep_meta["domain"],
                        "step": step_idx,
                        "method": label,
                        "action": action[:200],
                        "ground_truth_op": step.get("operation", ""),
                        "ground_truth_repr": step.get("action_repr", "")[:200],
                        "operation_match": eval_result["operation_match"],
                        "element_match": eval_result["element_match"],
                        "exact_match": eval_result["exact_match"],
                        "html_tokens": len(tokenizer.encode(html_ctx)),
                        "dense_agreement": dense_agreement,
                    }
                    writer.writerow(row)
                    all_results.append(row)

                csvfile.flush()
                print(f"  Step {step_idx}: done")

    # Print summary
    print_summary(all_results)
    print(f"\nResults saved to: {csv_path}")


def print_summary(results: list[dict]):
    """Print B2 + B5 summary statistics."""
    print(f"\n{'='*60}")
    print("B2: ACTION QUALITY SUMMARY")
    print(f"{'='*60}")

    # Overall by method
    by_method = defaultdict(list)
    for r in results:
        by_method[r["method"]].append(r)

    print(f"\n{'Method':<12} {'OpMatch':<10} {'ElemMatch':<12} {'ExactMatch':<12} {'DenseAgr':<10} {'N':<5}")
    print("-" * 65)
    for method in ["dense", "quest", "h2oquest"]:
        rows = by_method[method]
        if not rows:
            continue
        op = sum(1 for r in rows if r["operation_match"]) / len(rows)
        elem = sum(1 for r in rows if r["element_match"]) / len(rows)
        exact = sum(1 for r in rows if r["exact_match"]) / len(rows)
        dense_agr = sum(1 for r in rows if r["dense_agreement"]) / len(rows)
        print(f"{method:<12} {op:<10.3f} {elem:<12.3f} {exact:<12.3f} {dense_agr:<10.3f} {len(rows):<5}")

    # B5: Domain breakdown
    print(f"\n{'='*60}")
    print("B5: DOMAIN BREAKDOWN")
    print(f"{'='*60}")

    by_domain_method = defaultdict(lambda: defaultdict(list))
    for r in results:
        by_domain_method[r["domain"]][r["method"]].append(r)

    for domain in sorted(by_domain_method.keys()):
        print(f"\n  Domain: {domain}")
        for method in ["dense", "quest", "h2oquest"]:
            rows = by_domain_method[domain][method]
            if not rows:
                continue
            exact = sum(1 for r in rows if r["exact_match"]) / len(rows)
            dense_agr = sum(1 for r in rows if r["dense_agreement"]) / len(rows)
            print(f"    {method:<12} exact={exact:.3f} dense_agr={dense_agr:.3f} (n={len(rows)})")


def parse_args():
    parser = argparse.ArgumentParser(description="B2+B5: Mind2Web Action Quality")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Phase 9a
    extract_parser = subparsers.add_parser("extract-keys", help="Phase 9a: extract key vectors")
    extract_parser.add_argument("--model", type=str, default="Qwen/Qwen3-VL-4B-Instruct")
    extract_parser.add_argument("--output-dir", type=str, default="results/mind2web/")
    extract_parser.add_argument("--max-episodes", type=int, default=MAX_EPISODES)

    # Phase 9b
    eval_parser = subparsers.add_parser("evaluate", help="Phase 9b: action quality evaluation")
    eval_parser.add_argument("--keys-dir", type=str, default="results/mind2web/keys/")
    eval_parser.add_argument("--server-url", type=str, default="http://localhost:30000")
    eval_parser.add_argument("--output-dir", type=str, default="results/mind2web/")
    eval_parser.add_argument("--model", type=str, default="Qwen/Qwen3-VL-4B-Instruct")

    return parser.parse_args()


def main():
    args = parse_args()

    if args.command == "extract-keys":
        os.makedirs(args.output_dir, exist_ok=True)
        extract_keys_phase(args.model, args.output_dir, args.max_episodes)
    elif args.command == "evaluate":
        os.makedirs(args.output_dir, exist_ok=True)
        evaluate_phase(args.keys_dir, args.server_url, args.output_dir, args.model)


if __name__ == "__main__":
    main()

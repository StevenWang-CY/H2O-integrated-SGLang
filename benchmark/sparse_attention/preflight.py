"""Pre-flight verification for H2OQuest page selection benchmark.

Run before any benchmark to catch configuration mismatches, missing
dependencies, and API incompatibilities early.

Usage:
    python benchmark/sparse_attention/preflight.py

Checks:
    1. Model config (head counts, layers, head_dim, token IDs)
    2. Processor apply_chat_template return type (fix M1)
    3. Mind2Web pos_candidates format inspection (fix M2)
    4. _pending_fresh behavior in actual code (fix I2)
    5. FP8 server health (optional, skip if not running)
    6. GPU memory headroom for bf16 model + gradient oracle
"""

import os
import sys


def verify_model_config(model_name: str = "Qwen/Qwen3-VL-4B-Instruct") -> bool:
    """Check model architecture matches expected Qwen3-VL-4B config."""
    print(f"\n[1/6] Model config: {model_name}")
    try:
        from transformers import AutoConfig

        config = AutoConfig.from_pretrained(model_name)
        text_cfg = config.text_config

        checks = [
            ("num_attention_heads", text_cfg.num_attention_heads, 32),
            ("num_key_value_heads", text_cfg.num_key_value_heads, 8),
            ("num_hidden_layers", text_cfg.num_hidden_layers, 36),
            ("head_dim", text_cfg.head_dim, 128),
        ]
        # image_token_id may be on the top-level config
        img_token = getattr(config, "image_token_id", None)
        if img_token is not None:
            checks.append(("image_token_id", img_token, 151655))

        all_ok = True
        for name, actual, expected in checks:
            status = "OK" if actual == expected else "MISMATCH"
            if actual != expected:
                all_ok = False
            print(f"  {name}: {actual} (expected {expected}) [{status}]")

        gqa_group = text_cfg.num_attention_heads // text_cfg.num_key_value_heads
        print(f"  GQA group size: {gqa_group}")
        print(f"  -> {'PASS' if all_ok else 'FAIL'}")
        return all_ok
    except Exception as e:
        print(f"  ERROR: {e}")
        return False


def verify_processor(model_name: str = "Qwen/Qwen3-VL-4B-Instruct") -> bool:
    """Check apply_chat_template return type (fix M1)."""
    print(f"\n[2/6] Processor return type")
    try:
        from transformers import AutoProcessor

        processor = AutoProcessor.from_pretrained(model_name)

        # Test with minimal input
        messages = [{"role": "user", "content": [{"type": "text", "text": "test"}]}]
        result = processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True, return_tensors="pt"
        )
        result_type = type(result).__name__

        # Check what we get back
        if hasattr(result, "input_ids"):
            input_ids = result.input_ids
            print(f"  Return type: {result_type} (has .input_ids)")
            print(f"  input_ids shape: {input_ids.shape}")
        elif isinstance(result, dict) and "input_ids" in result:
            input_ids = result["input_ids"]
            print(f"  Return type: dict (has 'input_ids' key)")
            print(f"  input_ids shape: {input_ids.shape}")
        elif isinstance(result, list):
            print(f"  Return type: list (token IDs directly), len={len(result)}")
        else:
            print(f"  Return type: {result_type}")
            print(f"  WARNING: Unexpected type, inspect manually")

        print(f"  -> PASS")
        return True
    except Exception as e:
        print(f"  ERROR: {e}")
        return False


def verify_mind2web() -> bool:
    """Inspect Mind2Web pos_candidates format (fix M2)."""
    print(f"\n[3/6] Mind2Web dataset format")
    try:
        from datasets import load_dataset

        hf_home = os.environ.get("HF_HOME", os.path.expanduser("~/hf_models"))
        print(f"  HF_HOME: {hf_home}")

        ds = load_dataset(
            "osunlp/Multimodal-Mind2Web",
            split="test_task",
            streaming=True,
            cache_dir=hf_home,
        )
        row = next(iter(ds))

        print(f"  Available columns: {list(row.keys())[:15]}...")
        print(f"  pos_candidates type: {type(row.get('pos_candidates', 'N/A'))}")
        pos_cand = row.get("pos_candidates", "")
        print(f"  pos_candidates sample (200 chars): {str(pos_cand)[:200]}")
        print(f"  operation: {row.get('operation', 'N/A')}")
        print(f"  value: {row.get('value', 'N/A')}")
        if "domain" in row:
            print(f"  domain: {row.get('domain', 'N/A')}")
        if "action_repr" in row:
            print(f"  action_repr: {str(row.get('action_repr', ''))[:200]}")
        if "target_action_repr" in row:
            print(f"  target_action_repr: {str(row.get('target_action_repr', ''))[:200]}")

        print(f"  -> PASS (inspect output to adapt evaluate_action)")
        return True
    except Exception as e:
        print(f"  ERROR: {e}")
        print(f"  (Mind2Web may require authentication or specific HF token)")
        return False


def verify_pending_fresh_behavior() -> bool:
    """Check _pending_fresh uses torch.cat (not overwrite) in actual code (fix I2)."""
    print(f"\n[4/6] _pending_fresh accumulation behavior")
    try:
        import inspect

        from sglang.srt.mem_cache.sparsity.algorithms.h2oquest_algorithm import (
            H2OQuestAlgorithm,
        )

        source = inspect.getsource(H2OQuestAlgorithm._retrieve_page_scores)

        if "torch.cat" in source and "prev_pages" in source:
            print(f"  _pending_fresh: CONCATENATES across requests (torch.cat)")
            print(f"  -> PASS (matches scoring_adapter assumption)")
            return True
        elif "self._pending_fresh[group_id] =" in source and "torch.cat" not in source:
            print(f"  _pending_fresh: OVERWRITES per request (last-request-wins)")
            print(f"  WARNING: scoring_adapter may need adjustment")
            return False
        else:
            print(f"  Could not determine behavior from source")
            print(f"  Relevant snippet:")
            # Find the _pending_fresh section
            for i, line in enumerate(source.split("\n")):
                if "_pending_fresh" in line:
                    print(f"    {line.strip()}")
            return True
    except ImportError:
        print(f"  SGLang not installed locally — verify on remote before running")
        print(f"  -> SKIP")
        return True
    except Exception as e:
        print(f"  ERROR: {e}")
        return False


def verify_server_health(
    server_url: str = "http://localhost:30000",
) -> bool:
    """Check if FP8 server is reachable (optional — skip if not running)."""
    print(f"\n[5/6] FP8 server health: {server_url}")
    try:
        import requests

        resp = requests.get(f"{server_url}/health", timeout=5)
        print(f"  Status: {resp.status_code}")
        if resp.status_code == 200:
            print(f"  -> PASS")
            return True
        else:
            print(f"  -> FAIL (unexpected status)")
            return False
    except ImportError:
        print(f"  requests not installed")
        return False
    except Exception:
        print(f"  Server not running (expected if Phase 3 not started)")
        print(f"  -> SKIP")
        return True


def verify_gpu_memory() -> bool:
    """Check GPU memory headroom for bf16 model + gradient oracle."""
    print(f"\n[6/6] GPU memory")
    try:
        import torch

        if not torch.cuda.is_available():
            print(f"  No CUDA GPU available")
            print(f"  -> SKIP (CPU-only phases still work)")
            return True

        props = torch.cuda.get_device_properties(0)
        total_gb = props.total_memory / 1e9
        model_bf16_gb = 4e9 * 2 / 1e9  # 4B params * 2 bytes bf16
        headroom_gb = total_gb - model_bf16_gb

        print(f"  GPU: {props.name}")
        print(f"  Total VRAM: {total_gb:.1f} GB")
        print(f"  Model bf16: {model_bf16_gb:.1f} GB")
        print(f"  Headroom: {headroom_gb:.1f} GB")
        print(f"  SM version: {props.major}.{props.minor}")

        if headroom_gb < 4:
            print(f"  WARNING: Tight for gradient oracle at 8K tokens")
            print(f"  Consider capping sequences at 4K or using gradient checkpointing")
        if headroom_gb < 2:
            print(f"  WARNING: May not fit gradient oracle at all")

        # Check if FP8 server might be using the GPU
        allocated_gb = torch.cuda.memory_allocated(0) / 1e9
        if allocated_gb > 0.5:
            print(f"  WARNING: {allocated_gb:.1f} GB already allocated on GPU")
            print(f"  Ensure FP8 server is stopped before Phase 2")

        print(f"  -> PASS")
        return True
    except Exception as e:
        print(f"  ERROR: {e}")
        return False


def verify_all(
    model_name: str = "Qwen/Qwen3-VL-4B-Instruct",
    server_url: str = "http://localhost:30000",
) -> bool:
    """Run all pre-flight checks. Returns True if all critical checks pass."""
    print("=" * 60)
    print("PRE-FLIGHT VERIFICATION")
    print("=" * 60)

    results = {}
    results["model_config"] = verify_model_config(model_name)
    results["processor"] = verify_processor(model_name)
    results["mind2web"] = verify_mind2web()
    results["pending_fresh"] = verify_pending_fresh_behavior()
    results["server"] = verify_server_health(server_url)
    results["gpu"] = verify_gpu_memory()

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    all_pass = True
    for name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"  {name}: {status}")
        if not passed:
            all_pass = False

    print(f"\n{'ALL CHECKS PASSED' if all_pass else 'SOME CHECKS FAILED'}")
    return all_pass


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Pre-flight verification")
    parser.add_argument(
        "--model", type=str, default="Qwen/Qwen3-VL-4B-Instruct"
    )
    parser.add_argument(
        "--server-url", type=str, default="http://localhost:30000"
    )
    args = parser.parse_args()
    ok = verify_all(args.model, args.server_url)
    sys.exit(0 if ok else 1)

"""
Research benchmark: Comparative evaluation of Dense vs Quest vs H2OQuest.

Sweeps across algorithms, sparsity ratios, and evaluation tasks.
Uses fa3 (FlashAttention 3) backend — the ONLY backend that supports
sparse page selection via attention_begin.

IMPORTANT: Do NOT use "flashinfer" for either --attention-backend or
the JSON "backend" field. FlashInfer silently disables sparse page selection.
The only valid JSON backends are "fa3" and "flashattention".

Usage:
  python benchmark/sparse_attention/run_comparison.py \
    --model Qwen/Qwen3-VL-4B-Instruct \
    --output-dir results/sparse_comparison/

  python benchmark/sparse_attention/run_comparison.py \
    --model Qwen/Qwen3-VL-4B-Instruct --quick \
    --output-dir results/quick_check/
"""

import argparse
import csv
import json
import os
import signal
import subprocess
import sys
import time
from datetime import datetime
from types import SimpleNamespace

_current_proc = None


def _signal_handler(sig, frame):
    if _current_proc is not None:
        kill_server(_current_proc)
    sys.exit(1)


signal.signal(signal.SIGINT, _signal_handler)
signal.signal(signal.SIGTERM, _signal_handler)


def parse_args():
    parser = argparse.ArgumentParser(description="Sparse attention comparison benchmark")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="results/sparse_comparison/")
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=30000)
    parser.add_argument("--algorithms", nargs="+", default=["dense", "quest", "h2oquest"])
    parser.add_argument("--sparsity-ratios", nargs="+", type=float, default=[0.1, 0.2, 0.3, 0.5, 0.7])
    parser.add_argument("--eval-tasks", nargs="+", default=["mmlu"])
    parser.add_argument("--num-examples", type=int, default=100)
    parser.add_argument("--num-threads", type=int, default=16)
    parser.add_argument("--server-timeout", type=int, default=900)
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--quick", action="store_true")
    return parser.parse_args()


def build_server_cmd(args, algorithm, sparsity_ratio):
    """Build SGLang server launch command.

    All configs use fa3 backend and page-size 16 for consistent comparison.
    Dense baseline uses the same page layout as sparse configs.
    """
    cmd = [
        sys.executable, "-m", "sglang.launch_server",
        "--model-path", args.model,
        "--host", args.host,
        "--port", str(args.port),
        "--attention-backend", "fa3",
        "--disable-cuda-graph",
        "--page-size", "16",
        "--mem-fraction-static", "0.45",
        "--tp", str(args.tp),
    ]

    if algorithm != "dense":
        sparse_config = {
            "algorithm": algorithm,
            "backend": "flashinfer",
            "min_sparse_prompt_len": 2048,
            "sparsity_ratio": sparsity_ratio,
            "num_recent_pages": 32,
        }
        if algorithm == "h2oquest":
            sparse_config.update({
                "score_decay": 0.95,
                "initial_alpha": 1.0,
                "alpha_decay": 0.9,
                "num_sink_pages": 2,
                "protect_vision_tokens": True,
                "image_token_id": 151655,
                "video_token_id": 151656,
                "score_layer_group_size": 4,
            })
        cmd.extend([
            "--hierarchical-sparse-attention-extra-config",
            json.dumps(sparse_config),
        ])

    return cmd


def build_env():
    """Build subprocess environment with correct HF_HOME (absolute path)."""
    env = os.environ.copy()
    env["HF_HOME"] = os.path.expanduser("~/hf_models")
    return env


def wait_for_server(base_url, timeout=900):
    import requests
    start = time.time()
    while time.time() - start < timeout:
        try:
            resp = requests.get(f"{base_url}/health", timeout=5)
            if resp.status_code == 200:
                return True
        except (requests.ConnectionError, requests.Timeout):
            pass
        time.sleep(5)
    return False


def verify_server_log(log_path, algorithm):
    """Check server log for silent failures. MUST be called after server is ready."""
    try:
        with open(log_path) as f:
            content = f.read()
        if "FlashInfer backend does not support attention_begin" in content:
            print(f"  [FATAL] Sparse page selection NOT active for {algorithm}!")
            print(f"  The FlashInfer backend was used instead of fa3.")
            print(f"  Fix: ensure --attention-backend fa3 and 'backend': 'fa3' in JSON.")
            return False
        if "ValueError" in content and "Unknown attention backend" in content:
            print(f"  [FATAL] Invalid backend in sparse config JSON.")
            return False
    except FileNotFoundError:
        pass
    return True


def run_eval_task(base_url, model, task, num_examples, num_threads):
    from sglang.test.run_eval import run_eval
    eval_args = SimpleNamespace(
        base_url=base_url, model=model, eval_name=task,
        num_examples=num_examples, num_threads=num_threads,
    )
    try:
        return run_eval(eval_args)
    except Exception as e:
        print(f"  [ERROR] Eval failed: {e}")
        return {"score": -1.0, "error": str(e)}


def measure_latency(base_url, prompt_lengths=None):
    import requests
    if prompt_lengths is None:
        prompt_lengths = [512, 1024, 2048, 4096, 8192]
    results = {}
    for plen in prompt_lengths:
        prompt = "Hello " * (plen // 2)
        start = time.time()
        try:
            resp = requests.post(
                f"{base_url}/generate",
                json={"text": prompt, "sampling_params": {"temperature": 0.0, "max_new_tokens": 50}},
                timeout=120,
            )
            elapsed = time.time() - start
            if resp.status_code == 200:
                output = resp.json().get("text", "")
                num_tokens = len(output.split())
                results[plen] = {
                    "total_time_ms": elapsed * 1000,
                    "output_tokens": num_tokens,
                    "tpot_ms": (elapsed * 1000) / max(num_tokens, 1),
                }
            else:
                results[plen] = {"error": f"HTTP {resp.status_code}"}
        except Exception as e:
            results[plen] = {"error": str(e)}
    return results


def kill_server(proc):
    if proc is None:
        return
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
    except (ProcessLookupError, PermissionError, OSError):
        pass
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except (ProcessLookupError, PermissionError, OSError):
            pass


def main():
    global _current_proc
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    if args.quick:
        args.sparsity_ratios = [0.3, 0.5, 0.7]
        args.num_examples = min(args.num_examples, 20)
        print("[Quick mode] ratios=[0.3, 0.5, 0.7], num_examples capped at 20")

    base_url = f"http://{args.host}:{args.port}"
    env = build_env()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(args.output_dir, f"comparison_{timestamp}.csv")

    fieldnames = [
        "algorithm", "sparsity_ratio", "task", "accuracy",
        "latency_512_ms", "latency_1024_ms", "latency_2048_ms",
        "latency_4096_ms", "latency_8192_ms",
        "timestamp",
    ]

    with open(csv_path, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for algorithm in args.algorithms:
            ratios = [0.0] if algorithm == "dense" else args.sparsity_ratios

            for ratio in ratios:
                print(f"\n{'='*60}")
                print(f"Algorithm: {algorithm}, Sparsity Ratio: {ratio}")
                print(f"{'='*60}")

                cmd = build_server_cmd(args, algorithm, ratio)
                print(f"  Command: {' '.join(cmd[:6])}...")

                log_name = f"server_{algorithm}_s{ratio}.log"
                log_path = os.path.join(args.output_dir, log_name)
                log_file = open(log_path, "w")

                proc = subprocess.Popen(
                    cmd, stdout=log_file, stderr=subprocess.STDOUT,
                    preexec_fn=os.setsid, env=env,
                )
                _current_proc = proc

                try:
                    if not wait_for_server(base_url, args.server_timeout):
                        print(f"  [ERROR] Server failed to start in {args.server_timeout}s. See {log_path}")
                        continue

                    # Verify sparse is actually active (not silently disabled)
                    log_file.flush()
                    if not verify_server_log(log_path, algorithm):
                        print(f"  [ABORT] Skipping {algorithm} s={ratio} — sparse not active.")
                        continue

                    print("  Server ready. Sparse verification passed.")

                    for task in args.eval_tasks:
                        print(f"  Running {task} ({args.num_examples} examples)...")
                        metrics = run_eval_task(
                            base_url, args.model, task,
                            args.num_examples, args.num_threads,
                        )
                        accuracy = metrics.get("score", -1.0)
                        print(f"  {task} score: {accuracy:.4f}")

                        latency = measure_latency(base_url)
                        for plen, lat in latency.items():
                            if "error" not in lat:
                                print(f"    latency@{plen}: {lat['total_time_ms']:.1f}ms")

                        row = {
                            "algorithm": algorithm,
                            "sparsity_ratio": ratio,
                            "task": task,
                            "accuracy": accuracy,
                            "latency_512_ms": latency.get(512, {}).get("total_time_ms", -1),
                            "latency_1024_ms": latency.get(1024, {}).get("total_time_ms", -1),
                            "latency_2048_ms": latency.get(2048, {}).get("total_time_ms", -1),
                            "latency_4096_ms": latency.get(4096, {}).get("total_time_ms", -1),
                            "latency_8192_ms": latency.get(8192, {}).get("total_time_ms", -1),
                            "timestamp": datetime.now().isoformat(),
                        }
                        writer.writerow(row)
                        csvfile.flush()

                finally:
                    print("  Shutting down server...")
                    kill_server(proc)
                    _current_proc = None
                    log_file.close()
                    time.sleep(5)

    print(f"\nResults saved to: {csv_path}")
    print(f"Server logs in: {args.output_dir}/server_*.log")


if __name__ == "__main__":
    main()

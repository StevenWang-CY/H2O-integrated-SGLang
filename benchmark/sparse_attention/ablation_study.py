"""
H2OQuest hyperparameter ablation study.

Sweeps one hyperparameter at a time while holding others at default values.
Generates CSV for ablation heatmaps and sensitivity analysis.

Usage:
  python benchmark/sparse_attention/ablation_study.py \
    --model Qwen/Qwen2.5-0.5B-Instruct \
    --output-dir results/ablation/
"""

import argparse
import csv
import json
import os
import signal
import subprocess
import sys
import time
from copy import deepcopy
from datetime import datetime
from types import SimpleNamespace

# Global reference for signal handling
_current_proc = None


def _signal_handler(sig, frame):
    if _current_proc is not None:
        kill_server(_current_proc)
    sys.exit(1)


signal.signal(signal.SIGINT, _signal_handler)
signal.signal(signal.SIGTERM, _signal_handler)

# Default H2OQuest configuration
DEFAULT_CONFIG = {
    "algorithm": "h2oquest",
    "backend": "flashattention",
    "min_sparse_prompt_len": 2048,
    "sparsity_ratio": 0.5,
    "num_recent_pages": 32,
    "score_decay": 0.95,
    "initial_alpha": 1.0,
    "alpha_decay": 0.9,
    "num_sink_pages": 2,
    "score_layer_group_size": 4,
    "protect_vision_tokens": True,
    "image_token_id": 151655,
    "video_token_id": 151656,
}

ABLATION_SWEEPS = {
    "score_decay": [0.8, 0.9, 0.95, 0.99],
    "alpha_decay": [0.5, 0.7, 0.9, 0.95],
    "num_sink_pages": [0, 1, 2, 4],
    "score_layer_group_size": [1, 2, 4, 8],
    "sparsity_ratio": [0.1, 0.3, 0.5, 0.7, 0.9],
}


def parse_args():
    parser = argparse.ArgumentParser(description="H2OQuest ablation study")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="results/ablation/")
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=30000)
    parser.add_argument("--eval-task", type=str, default="mmlu")
    parser.add_argument("--num-examples", type=int, default=100)
    parser.add_argument("--num-threads", type=int, default=16)
    parser.add_argument("--server-timeout", type=int, default=900)
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument(
        "--sweep-params", nargs="+", default=list(ABLATION_SWEEPS.keys()),
    )
    return parser.parse_args()


def launch_server(args, sparse_config, log_path):
    """Launch SGLang server. Returns (proc, base_url) or (None, None) on failure."""
    global _current_proc
    base_url = f"http://{args.host}:{args.port}"
    env = os.environ.copy()
    env["HF_HOME"] = os.path.expanduser("~/hf_models")

    cmd = [
        sys.executable, "-m", "sglang.launch_server",
        "--model-path", args.model,
        "--host", args.host,
        "--port", str(args.port),
        "--attention-backend", "fa3",
        "--disable-cuda-graph",
        "--page-size", "16",
        "--quantization", "fp8",
        "--mem-fraction-static", "0.5",
        "--tp", str(args.tp),
        "--hierarchical-sparse-attention-extra-config", json.dumps(sparse_config),
    ]

    log_file = open(log_path, "w")
    proc = subprocess.Popen(
        cmd, stdout=log_file, stderr=subprocess.STDOUT,
        preexec_fn=os.setsid, env=env,
    )
    _current_proc = proc

    import requests
    start = time.time()
    while time.time() - start < args.server_timeout:
        try:
            resp = requests.get(f"{base_url}/health", timeout=5)
            if resp.status_code == 200:
                return proc, base_url, log_file
        except (requests.ConnectionError, requests.Timeout):
            pass
        time.sleep(5)

    kill_server(proc)
    log_file.close()
    _current_proc = None
    return None, None, None


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


def run_eval(base_url, model, task, num_examples, num_threads):
    from sglang.test.run_eval import run_eval as _run_eval
    eval_args = SimpleNamespace(
        base_url=base_url, model=model, eval_name=task,
        num_examples=num_examples, num_threads=num_threads,
    )
    try:
        return _run_eval(eval_args)
    except Exception as e:
        return {"score": -1.0, "error": str(e)}


def main():
    global _current_proc
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(args.output_dir, f"ablation_{timestamp}.csv")

    fieldnames = ["sweep_param", "sweep_value", "task", "accuracy", "full_config", "timestamp"]

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
                config = deepcopy(DEFAULT_CONFIG)
                config[param_name] = value

                print(f"\n  {param_name}={value}")
                log_path = os.path.join(args.output_dir, f"server_{param_name}_{value}.log")
                proc, base_url, log_file = launch_server(args, config, log_path)

                if proc is None:
                    print(f"  [ERROR] Server failed to start. See {log_path}")
                    writer.writerow({
                        "sweep_param": param_name, "sweep_value": value,
                        "task": args.eval_task, "accuracy": -1.0,
                        "full_config": json.dumps(config),
                        "timestamp": datetime.now().isoformat(),
                    })
                    csvfile.flush()
                    continue

                try:
                    metrics = run_eval(
                        base_url, args.model, args.eval_task,
                        args.num_examples, args.num_threads,
                    )
                    accuracy = metrics.get("score", -1.0)
                    print(f"  Score: {accuracy:.4f}")

                    writer.writerow({
                        "sweep_param": param_name, "sweep_value": value,
                        "task": args.eval_task, "accuracy": accuracy,
                        "full_config": json.dumps(config),
                        "timestamp": datetime.now().isoformat(),
                    })
                    csvfile.flush()
                finally:
                    kill_server(proc)
                    _current_proc = None
                    if log_file:
                        log_file.close()
                    time.sleep(5)

    print(f"\nAblation results saved to: {csv_path}")
    print(f"Server logs in: {args.output_dir}/server_*.log")


if __name__ == "__main__":
    main()

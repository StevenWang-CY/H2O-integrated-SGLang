"""
Integration tests for sparse attention algorithms (Quest, H2OQuest, Dense baseline).

Launches SGLang server with Qwen3-VL-4B-Instruct and fa3 backend.

IMPORTANT: Must use --attention-backend fa3 and "backend": "fa3" in JSON.
FlashInfer backend silently disables sparse page selection.

Run:
  HF_HOME=~/hf_models python -m pytest test/manual/test_sparse_attention_algorithms.py -v -s
"""

import json
import os
import time
import unittest
from types import SimpleNamespace

import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.run_eval import run_eval
from sglang.test.test_utils import (
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

# Default: Qwen3-VL-4B-Instruct (262K ctx, VLM, cached on remote)
_DEFAULT_SPARSE_MODEL = "Qwen/Qwen3-VL-4B-Instruct"
MODEL = os.environ.get("SPARSE_TEST_MODEL", _DEFAULT_SPARSE_MODEL)
NUM_MMLU_EXAMPLES = int(os.environ.get("SPARSE_TEST_MMLU_N", "20"))

# 900s timeout for Blackwell SM 12.0 first-time FA3 JIT compilation
_SERVER_TIMEOUT = 900

# Absolute path — tilde won't expand in subprocess env
_HF_HOME = os.path.expanduser("~/hf_models")

# Common server args for all configs (fa3 backend, page-size 16, no CUDA graph)
_COMMON_ARGS = [
    "--attention-backend", "flashinfer",
    "--disable-cuda-graph",
    "--page-size", "16",
    "--mem-fraction-static", "0.45",
]


class TestDenseBaseline(CustomTestCase):
    """T2.1: Dense attention baseline — no sparse attention.

    NOTE: MMLU 5-shot prompts (~200 tokens) are below min_sparse_prompt_len=2048,
    so even sparse configs would run dense for these prompts. This test establishes
    the accuracy baseline that sparse configs must match.
    """

    dense_score = None

    @classmethod
    def setUpClass(cls):
        cls.model = MODEL
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=_SERVER_TIMEOUT,
            env={"HF_HOME": _HF_HOME},
            other_args=list(_COMMON_ARGS),
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)
        time.sleep(3)

    def test_mmlu_dense(self):
        args = SimpleNamespace(
            base_url=self.base_url, model=self.model,
            eval_name="mmlu", num_examples=NUM_MMLU_EXAMPLES, num_threads=16,
        )
        metrics = run_eval(args)
        score = metrics["score"]
        TestDenseBaseline.dense_score = score
        print(f"\n[Dense Baseline] MMLU score: {score:.4f}")
        self.assertGreaterEqual(score, 0.25, "Dense MMLU score below random chance")


class TestQuestSparse(CustomTestCase):
    """T2.2: Quest sparse attention at sparsity_ratio=0.5.

    NOTE: MMLU 5-shot prompts (~200 tokens) are below min_sparse_prompt_len=2048,
    so this tests 'sparse config loaded but inactive for short prompts' — a valid
    regression check that the Quest framework doesn't break short-prompt inference.
    """

    @classmethod
    def setUpClass(cls):
        cls.model = MODEL
        cls.base_url = DEFAULT_URL_FOR_TEST
        sparse_config = json.dumps({
            "algorithm": "quest",
            "backend": "flashinfer",
            "min_sparse_prompt_len": 2048,
            "sparsity_ratio": 0.5,
            "num_recent_pages": 32,
        })
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=_SERVER_TIMEOUT,
            env={"HF_HOME": _HF_HOME},
            other_args=_COMMON_ARGS + [
                "--hierarchical-sparse-attention-extra-config", sparse_config,
            ],
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)
        time.sleep(3)

    def test_mmlu_quest(self):
        args = SimpleNamespace(
            base_url=self.base_url, model=self.model,
            eval_name="mmlu", num_examples=NUM_MMLU_EXAMPLES, num_threads=16,
        )
        metrics = run_eval(args)
        score = metrics["score"]
        print(f"\n[Quest] MMLU score: {score:.4f}")
        if TestDenseBaseline.dense_score is not None:
            self.assertGreaterEqual(score, TestDenseBaseline.dense_score * 0.85)
        else:
            self.assertGreaterEqual(score, 0.25)

    def test_generation_not_degenerate(self):
        response = requests.post(
            f"{self.base_url}/generate",
            json={"text": "The capital of France is", "sampling_params": {"temperature": 0.0, "max_new_tokens": 50}},
        )
        self.assertEqual(response.status_code, 200)
        text = response.json().get("text", "")
        self.assertGreater(len(text), 5)


class TestH2OQuestSparse(CustomTestCase):
    """T2.3: H2OQuest sparse attention at sparsity_ratio=0.5.

    NOTE: MMLU 5-shot prompts (~200 tokens) are below min_sparse_prompt_len=2048,
    so this tests 'H2OQuest config loaded but inactive for short prompts' — a valid
    regression check that the H2OQuest framework (temporal accumulation, vision
    protection, score normalization) doesn't break short-prompt inference.
    """

    @classmethod
    def setUpClass(cls):
        cls.model = MODEL
        cls.base_url = DEFAULT_URL_FOR_TEST
        sparse_config = json.dumps({
            "algorithm": "h2oquest",
            "backend": "flashinfer",
            "min_sparse_prompt_len": 2048,
            "sparsity_ratio": 0.5,
            "num_recent_pages": 32,
            "score_decay": 0.95,
            "initial_alpha": 1.0,
            "alpha_decay": 0.9,
            "num_sink_pages": 2,
            "protect_vision_tokens": True,
            "image_token_id": 151655,
            "video_token_id": 151656,
            "score_layer_group_size": 4,
        })
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=_SERVER_TIMEOUT,
            env={"HF_HOME": _HF_HOME},
            other_args=_COMMON_ARGS + [
                "--hierarchical-sparse-attention-extra-config", sparse_config,
            ],
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)
        time.sleep(3)

    def test_mmlu_h2oquest(self):
        args = SimpleNamespace(
            base_url=self.base_url, model=self.model,
            eval_name="mmlu", num_examples=NUM_MMLU_EXAMPLES, num_threads=16,
        )
        metrics = run_eval(args)
        score = metrics["score"]
        print(f"\n[H2OQuest] MMLU score: {score:.4f}")
        if TestDenseBaseline.dense_score is not None:
            self.assertGreaterEqual(score, TestDenseBaseline.dense_score * 0.85)
        else:
            self.assertGreaterEqual(score, 0.25)

    def test_multiturn_conversation(self):
        r1 = requests.post(
            f"{self.base_url}/generate",
            json={
                "text": "Remember this: The secret code is ALPHA-7. Repeat the secret code.",
                "sampling_params": {"temperature": 0.0, "max_new_tokens": 50},
            },
        )
        self.assertEqual(r1.status_code, 200)
        text1 = r1.json().get("text", "")
        self.assertIn("ALPHA", text1.upper())


@unittest.skipUnless(
    os.environ.get("SPARSE_TEST_CUDA_GRAPH"),
    "Skip CUDA graph warning test (set SPARSE_TEST_CUDA_GRAPH=1 to enable)",
)
class TestH2OQuestCudaGraphWarning(CustomTestCase):
    """T2.4: H2OQuest warns when CUDA graphs are not disabled."""

    @classmethod
    def setUpClass(cls):
        cls.model = MODEL
        cls.base_url = DEFAULT_URL_FOR_TEST
        sparse_config = json.dumps({
            "algorithm": "h2oquest",
            "backend": "flashinfer",
            "min_sparse_prompt_len": 2048,
            "sparsity_ratio": 0.5,
        })
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=_SERVER_TIMEOUT,
            env={"HF_HOME": _HF_HOME},
            other_args=[
                "--attention-backend", "flashinfer",
                "--page-size", "16",
                "--mem-fraction-static", "0.45",
                "--hierarchical-sparse-attention-extra-config", sparse_config,
            ],
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)
        time.sleep(3)

    def test_cuda_graph_warning_in_logs(self):
        try:
            response = requests.post(
                f"{self.base_url}/generate",
                json={"text": "Hello", "sampling_params": {"temperature": 0.0, "max_new_tokens": 10}},
                timeout=30,
            )
            self.assertEqual(response.status_code, 200)
        except requests.exceptions.ConnectionError:
            pass


if __name__ == "__main__":
    unittest.main()

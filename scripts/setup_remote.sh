#!/bin/bash
# Setup: rsync H2O-integrated-SGLang to remote, validate.
#
# Usage: ./scripts/setup_remote.sh [REMOTE]

set -e

REMOTE="${1:-wangcy07@gray.cis.upenn.edu}"
REMOTE_CODE="h2o_sglang"
REMOTE_VENV="sglang_my"
LOCAL_DIR="$(cd "$(dirname "$0")/.." && pwd)"

echo "============================================"
echo "Setup: $REMOTE:~/$REMOTE_CODE"
echo "============================================"

echo ""
echo "=== Syncing H2O-integrated-SGLang ==="
rsync -az --delete \
  --exclude '.git' \
  --exclude '__pycache__' \
  --exclude '*.pyc' \
  --exclude '.venv' \
  --exclude 'sgl-kernel' \
  --exclude '3rdparty' \
  --exclude 'docker' \
  --exclude 'assets' \
  "$LOCAL_DIR/" \
  "$REMOTE:~/$REMOTE_CODE/"
echo "  Done."

echo ""
echo "=== Installing (editable, no-deps) ==="
ssh "$REMOTE" "source ~/$REMOTE_VENV/.venv/bin/activate && \
  cd ~/$REMOTE_CODE/python && \
  pip install -e . --no-deps -q 2>&1 | tail -3"

echo ""
echo "=== Pre-flight checks ==="
ssh "$REMOTE" "source ~/$REMOTE_VENV/.venv/bin/activate && python -c '
import sglang; print(f\"  sglang: {sglang.__file__}\")

# FA3 available
from sgl_kernel.flash_attn import flash_attn_varlen_func
print(\"  FA3 backend: OK\")

# H2OQuest
from sglang.srt.mem_cache.sparsity.algorithms.h2oquest_algorithm import H2OQuestAlgorithm
print(\"  H2OQuestAlgorithm: OK\")

# Server args
from sglang.srt.server_args import ServerArgs
assert hasattr(ServerArgs, \"hierarchical_sparse_attention_extra_config\")
print(\"  server_args: OK\")

# Qwen3VL processor
from sglang.srt.models.qwen3_vl import Qwen3VLForConditionalGeneration
print(\"  Qwen3VL model: OK\")

# Model cached
import os
model_dir = os.path.expanduser(\"~/hf_models/hub/models--Qwen--Qwen3-VL-4B-Instruct\")
assert os.path.isdir(model_dir), f\"Model not cached at {model_dir}\"
print(\"  Qwen3-VL-4B model cached: OK\")

# CUDA 12.8
import subprocess
result = subprocess.run([\"nvcc\", \"--version\"], capture_output=True, text=True)
assert \"12.8\" in result.stdout, \"nvcc is not 12.8\"
print(\"  nvcc 12.8: OK\")

print(\"  === ALL PRE-FLIGHT CHECKS PASSED ===\")
'"

echo ""
echo "=== Tier 1 smoke test ==="
ssh "$REMOTE" "source ~/$REMOTE_VENV/.venv/bin/activate && cd ~/$REMOTE_CODE && \
  python -m pytest test/unit/test_sparse_algorithms.py --timeout=120 -q 2>&1 | tail -3"

echo ""
echo "=== GPU ==="
ssh "$REMOTE" "nvidia-smi --query-gpu=name,memory.total --format=csv,noheader"

echo ""
echo "============================================"
echo "Setup complete."
echo "============================================"

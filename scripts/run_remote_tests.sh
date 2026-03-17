#!/bin/bash
# Run sparse attention test tiers on the remote GPU machine.
#
# Usage:
#   ./scripts/run_remote_tests.sh --tier1
#   ./scripts/run_remote_tests.sh --tier2
#   ./scripts/run_remote_tests.sh --tier3
#   ./scripts/run_remote_tests.sh --all

set -e

REMOTE="${REMOTE:-wangcy07@gray.cis.upenn.edu}"
REMOTE_DIR="h2o_sglang"
VENV_DIR="sglang_my"
MODEL="${MODEL:-Qwen/Qwen3-VL-4B-Instruct}"
HF_HOME="/home/wangcy07/hf_models"

RUN_TIER1=false
RUN_TIER2=false
RUN_TIER3=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --tier1) RUN_TIER1=true; shift ;;
        --tier2) RUN_TIER2=true; shift ;;
        --tier3) RUN_TIER3=true; shift ;;
        --all) RUN_TIER1=true; RUN_TIER2=true; RUN_TIER3=true; shift ;;
        --model) MODEL="$2"; shift 2 ;;
        --remote) REMOTE="$2"; shift 2 ;;
        *) echo "Unknown flag: $1"; exit 1 ;;
    esac
done

if ! $RUN_TIER1 && ! $RUN_TIER2 && ! $RUN_TIER3; then
    RUN_TIER1=true
fi

echo "============================================"
echo "Remote: $REMOTE"
echo "Model:  $MODEL"
echo "Code:   ~/$REMOTE_DIR"
echo "Tiers:  T1=$RUN_TIER1 T2=$RUN_TIER2 T3=$RUN_TIER3"
echo "============================================"

ACTIVATE="source ~/$VENV_DIR/.venv/bin/activate && cd ~/$REMOTE_DIR"

if $RUN_TIER1; then
    echo ""
    echo "=== Tier 1: Unit Tests ==="
    ssh "$REMOTE" "$ACTIVATE && \
      python -m pytest test/unit/test_sparse_algorithms.py -v --timeout=120 2>&1"
    echo "=== Tier 1 complete ==="
fi

if $RUN_TIER2; then
    echo ""
    echo "=== Tier 2: Integration Tests (model: $MODEL) ==="
    ssh -o ServerAliveInterval=10 -o ServerAliveCountMax=200 "$REMOTE" "$ACTIVATE && \
      HF_HOME=$HF_HOME SPARSE_TEST_MODEL='$MODEL' \
      python -m pytest test/manual/test_sparse_attention_algorithms.py -v -s --timeout=1200 2>&1"
    echo "=== Tier 2 complete ==="
fi

if $RUN_TIER3; then
    echo ""
    echo "=== Tier 3: Benchmarks (model: $MODEL) ==="
    ssh -o ServerAliveInterval=10 -o ServerAliveCountMax=200 "$REMOTE" "$ACTIVATE && \
      mkdir -p results/sparse_comparison && \
      HF_HOME=$HF_HOME python benchmark/sparse_attention/run_comparison.py \
        --model '$MODEL' --quick \
        --output-dir results/sparse_comparison/ 2>&1"
    echo ""
    echo "=== Collecting results ==="
    mkdir -p results_remote
    scp -r "$REMOTE:~/$REMOTE_DIR/results/" results_remote/ 2>/dev/null || echo "(no results)"
    echo "=== Tier 3 complete ==="
fi

echo ""
echo "============================================"
echo "All requested tiers finished."
echo "============================================"

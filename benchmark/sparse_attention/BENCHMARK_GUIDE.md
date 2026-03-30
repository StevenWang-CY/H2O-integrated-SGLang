# H2OQuest Page Selection Quality Benchmark Suite

## 1. Introduction

### Motivation

H2OQuest is a hybrid sparse attention algorithm that combines **Quest** (bounding-box page scoring) with **H2O** (temporal score accumulation and decay) for efficient long-context inference in browser agent serving. The core hypothesis is:

> **H2OQuest's stateful accumulation retains historically-important KV cache pages across multi-turn browser sessions, improving page selection quality over Quest's stateless approach --- particularly at "return steps" where the agent revisits previously-seen content.**

This benchmark suite provides rigorous, offline evaluation of that hypothesis. It measures whether H2OQuest selects better pages than Quest at the **algorithm level**, independent of the inference engine.

### Why Offline Benchmarks?

The target hardware (NVIDIA RTX 5060 Ti, SM 12.0) lacks support for FlashAttention-3/4, which means sparse attention cannot be activated at inference time on this GPU. These benchmarks circumvent the hardware limitation by:

1. **Extracting the exact scoring math** from the SGLang algorithm implementations into standalone PyTorch functions (no SGLang dependency).
2. **Extracting key vectors** from a bf16 model via forward hooks (O(seq_len) memory per layer).
3. **Simulating sparse attention** via context truncation --- keeping only tokens from selected pages and sending to an FP8 server for action quality evaluation.

### Target Model

All benchmarks are configured for **Qwen3-VL-4B-Instruct**:

| Parameter | Value |
|-----------|-------|
| Query heads | 32 |
| KV heads | 8 |
| GQA group size | 4 |
| Layers | 36 |
| Head dimension | 128 |
| Page size | 16 tokens |
| Score layer | 17 (middle layer) |

---

## 2. Algorithm Description

### Quest (Baseline)

Quest is a stateless bounding-box page scoring algorithm. For each page, it stores per-dimension key minimums (`k_min`) and maximums (`k_max`). The page score is an upper bound on `max(q . k)` for any key `k` in the page:

```
score_d = q_d * k_max_d   if q_d >= 0
          q_d * k_min_d   if q_d <  0
page_score = sum over heads and dims of score_d
```

**GQA handling:** Quest uses **mean aggregation** --- it averages query heads within each KV group before scoring. This produces a single consensus query per group.

**Properties:** Stateless (no memory across decode steps), fast, but forgets previously-important pages when the topic shifts and returns.

### H2OQuest (Proposed)

H2OQuest extends Quest with temporal score accumulation inspired by H2O (Heavy-Hitter Oracle). It maintains a persistent accumulated score buffer across decode steps.

**Fresh scoring:** Uses the same bounding-box math as Quest, but with **max aggregation** GQA --- retaining a page if ANY query head within the group deems it important. This is more conservative (keeps more relevant pages).

**Alpha blending with normalization:**
1. Compute fresh Quest bounding-box score (max-agg GQA)
2. Inject `float_max` for sink pages (first N pages that anchor attention)
3. Normalize both fresh and accumulated scores to [0, 1], excluding `+inf` pages from min/max computation
4. Re-inject 1.0 for sink pages after normalization
5. Blend: `score = alpha * fresh_norm + (1 - alpha) * acc_norm`

**Alpha decay:** `alpha = initial_alpha * (alpha_decay ^ decode_step)`. Alpha starts at 1.0 (pure fresh scoring) and decays toward 0.0 (accumulated history dominates) as more tokens are generated within a turn. Alpha resets to `initial_alpha` at each new prefill (new turn).

**Accumulation update:** After scoring, the buffer is updated:
```
accumulated *= score_decay          # Global decay
accumulated += fresh_raw_scores     # Add RAW (pre-normalization) fresh scores
```

All accumulated buffers use **float32** (float16 overflows after ~5 decode steps since Quest scores sum to ~10,000 per step and float16 max = 65,504).

### Top-K Page Selection

Matches `base_algorithm.py retrieve_topk`:
1. Recent pages (last `num_recent_pages`) are always included
2. Recent pages are masked to `-inf` in scores
3. `history_pages = max(num_pages - num_recent_pages, 0)`
4. `k = max(int(history_pages * sparsity_ratio), 1)`
5. Top-k selected from history region
6. Final selection = recent pages + top-k history pages

Sink pages are **not** force-included at the selection level. In H2OQuest, sinks are retained via score injection (`float_max` -> normalized to 1.0), which naturally places them in top-k. Quest has no sink handling, matching the actual implementation.

### Key Differences (Quest vs H2OQuest)

| Aspect | Quest | H2OQuest |
|--------|-------|----------|
| GQA aggregation | Mean (consensus) | Max (any-head-votes) |
| State | Stateless | Accumulated buffer (float32) |
| Sink injection | None | float_max before normalization |
| Score normalization | None | Min-max to [0,1] |
| Alpha blending | N/A | alpha * fresh + (1-alpha) * accumulated |
| Temporal memory | None | Exponential decay + accumulation |

**Important:** Quest (mean-agg) and H2OQuest (max-agg) scores are NOT directly comparable in magnitude. All benchmarks compare selected page **sets** (recall, Jaccard), never raw score values.

---

## 3. Benchmark Descriptions

### B1: Oracle Recall (`benchmark_oracle_recall.py`)

**Goal:** Compare page selections against a gradient-based ground truth oracle.

**Method:**
- The gradient oracle computes `d(logit_target) / d(key_vectors)` via backward hooks on `k_proj` across all 36 layers. The L2 norm of the gradient at each key position indicates how much that position influences the model's next-token prediction. Within each page, the max gradient magnitude is taken, then max across layers.
- For each multi-turn synthetic browser episode (shopping -> travel -> return to shopping), key vectors and last-token queries are extracted from the bf16 model in a single forward pass.
- Quest and H2OQuest page selections are compared against the oracle's top-k important pages.

**Metric:** `recall = |selected ∩ oracle_topk| / oracle_budget`

**Expected results:**
- H2OQuest: 65--75% oracle recall at return steps
- Quest: 40--50% oracle recall at return steps
- Gap appears after alpha ramps down (decode step > 5)

**Resources:** GPU required (~14 GB VRAM). Sequences capped at 8K tokens.

### B2 + B5: Mind2Web Action Quality + Domain Breakdown (`benchmark_mind2web.py`)

**Goal:** Measure whether H2OQuest's page selection preserves the tokens needed for correct action prediction on real web tasks.

**Method:**
- Uses the [Mind2Web](https://huggingface.co/datasets/osunlp/Multimodal-Mind2Web) `test_task` split (~50 multi-step web episodes across domains: travel, shopping, social, etc.)
- **Phase 9a** (bf16 model): Extract key vectors and queries for each episode step, save to disk as `.pt` files. Unload model.
- **Phase 9b** (FP8 server): Load saved keys, compute Quest and H2OQuest page selections on CPU, truncate HTML to selected pages, send three versions (dense, Quest-truncated, H2OQuest-truncated) to the FP8 server, evaluate predicted actions against ground truth.
- B5 groups B2 results by Mind2Web `domain` field (no additional inference).

**Metrics:**
- `operation_match`: ground truth operation type appears in prediction
- `element_match`: key terms from target action description appear in prediction
- `exact_match`: both operation and element match
- `dense_agreement`: truncated output matches dense (full-context) output

**Expected results:**
- H2OQuest-truncated matches dense output 75--85% of the time
- Quest-truncated matches dense output 60--70%

**Note:** Context truncation is **stricter** than real sparse attention (non-selected tokens are completely absent, not just deprioritized). If H2OQuest-truncated outperforms Quest-truncated under this constraint, the real advantage would be even larger.

### B3: Selection Stability (`benchmark_stability.py`)

**Goal:** Measure how stable page selections are across multi-turn episodes, particularly at return steps.

**Method:**
- Generates **synthetic random key vectors** (no GPU or model needed) simulating multi-turn browser sessions with topic shifts:
  - Turns 0--1: Topic A distribution
  - Turns 2--3: Topic B distribution (shift)
  - Turn 4: Topic A again (return step)
- Simulates 20 decode steps per turn with intermediate accumulated buffer updates (matching the actual system where each generated token triggers an accumulation update).
- Sweeps over `score_decay` in {0.8, 0.9, 0.95, 0.99} and `alpha_decay` in {0.5, 0.7, 0.9, 0.95} (16 configurations total).

**Metrics:**
- `jaccard_prev`: Jaccard similarity of page selection between consecutive turns
- `jaccard_t0`: Jaccard similarity of page selection compared to turn 0 (critical at return steps)
- `method_overlap`: Jaccard between Quest and H2OQuest selections at the same step

**Expected results:**
- H2OQuest `jaccard_t0` > 0.80 at return steps
- Quest `jaccard_t0` < 0.60 at return steps
- H2OQuest shows smoother selection transitions

### B4: Offline Ablation (`benchmark_ablation_offline.py`)

**Goal:** Measure how H2OQuest hyperparameters affect retention of historically-important pages.

**Method:**
- Same synthetic key vector infrastructure as B3.
- "Historically important" = pages that were in Quest's top-k at earlier steps in the trajectory (not oracle importance). This directly tests whether H2OQuest's accumulation retains pages that Quest would forget.
- Sweeps one parameter at a time from defaults:
  - `alpha_decay` in {0.5, 0.7, 0.9, 0.95}
  - `score_decay` in {0.8, 0.9, 0.95, 0.99}
  - `num_sink_pages` in {0, 1, 2, 4}
- Reports metrics at multiple decode positions within each turn (alpha checkpoints: token 1, 5, 10, 20) to show the alpha ramp-up curve.
- Accumulated buffer is properly updated between checkpoints (each intermediate decode step performs decay + fresh score addition).

**Metric:** `historical_recall = |current_selection ∩ quest_history| / |quest_history|`

**Expected results:**
- `alpha_decay=0.7`: largest gap (+0.25 recall delta)
- `alpha_decay=0.95`: smallest gap (+0.05)
- Alpha ramp-up curve shows recall delta increasing as alpha decays toward 0

---

## 4. File Inventory

### Core Library Files

| File | Purpose | Dependencies |
|------|---------|--------------|
| `scoring_adapter.py` | Standalone Quest + H2OQuest scoring math | `torch` only |
| `extract_keys.py` | Key/query extraction via forward hooks on bf16 model | `torch`, `transformers` |
| `gradient_oracle.py` | Gradient-based page importance oracle | `torch`, `scipy` |
| `context_truncation.py` | Prompt truncation and action evaluation | `openai` (for server calls) |
| `preflight.py` | Pre-flight verification of all dependencies | various |

### Benchmark Scripts

| File | Benchmark | GPU Required | Estimated Runtime |
|------|-----------|-------------|-------------------|
| `benchmark_stability.py` | B3: Selection Stability | No (CPU only) | ~30 min |
| `benchmark_ablation_offline.py` | B4: Offline Ablation | No (CPU only) | ~1 hr |
| `benchmark_oracle_recall.py` | B1: Oracle Recall | Yes (~14 GB VRAM) | ~2--3 hrs |
| `benchmark_mind2web.py` | B2+B5: Action Quality | Yes (two phases) | ~4--6 hrs total |

### Pre-existing Files (Not Part of This Suite)

| File | Purpose |
|------|---------|
| `run_comparison.py` | Server-based Dense vs Quest vs H2OQuest comparison (requires FA3) |
| `ablation_study.py` | Server-based hyperparameter sweeps (requires FA3) |
| `analyze_results.py` | Statistical analysis and visualization of server-based results |

---

## 5. Divergences from SGLang Implementation

The offline adapter intentionally differs from the SGLang runtime in several documented ways:

| Aspect | SGLang Runtime | Offline Adapter | Impact |
|--------|---------------|-----------------|--------|
| Accumulation | `scatter_add_` (deterministic with shared pages) | Simple `+=` | Identical for B=1 (no shared pages in offline mode) |
| Key vectors | Post-RoPE (rotation applied inside attention module) | Pre-RoPE (captured from `k_proj` output) | Both algorithms see the same pre-rotation keys, so relative comparison is valid |
| Multi-request batching | Supports B>1 with RadixAttention page sharing | B=1 only | Offline benchmarks are single-request |
| Vision token protection | Injects `float_max` for image/video token pages | Not implemented | Synthetic prompts contain only text |
| `_pending_fresh` | Concatenates across batch requests via `torch.cat` | Single-request, no concatenation needed | No divergence for B=1 |

---

## 6. Running the Benchmarks

### Prerequisites

```bash
pip install torch transformers datasets scipy openai
```

Set `HF_HOME` to your Hugging Face model cache directory:
```bash
export HF_HOME=~/hf_models  # or your preferred location
```

### Step 0: Pre-Flight Verification

Always run preflight checks before any benchmark:

```bash
python benchmark/sparse_attention/preflight.py \
    --model Qwen/Qwen3-VL-4B-Instruct
```

This verifies:
1. Model architecture matches expected config (32 q_heads, 8 kv_heads, 36 layers, 128 head_dim)
2. Processor `apply_chat_template` return type is handled correctly
3. Mind2Web dataset is accessible and format is inspected
4. `_pending_fresh` behavior in actual H2OQuest code matches assumptions
5. FP8 server health (optional --- skips if not running)
6. GPU memory headroom for bf16 model + gradient oracle

### Step 1: CPU-Only Benchmarks (B3 + B4)

These can run immediately, in parallel, with no GPU:

```bash
# B3: Selection Stability (~30 min)
python benchmark/sparse_attention/benchmark_stability.py \
    --output-dir results/stability/ \
    --num-episodes 50

# B4: Offline Ablation (~1 hr)
python benchmark/sparse_attention/benchmark_ablation_offline.py \
    --output-dir results/ablation_offline/ \
    --num-episodes 30
```

**B3 options:**
- `--score-decays 0.8 0.9 0.95 0.99` --- score decay values to sweep
- `--alpha-decays 0.5 0.7 0.9 0.95` --- alpha decay values to sweep

**B4 options:**
- `--sweep-params alpha_decay score_decay num_sink_pages` --- which parameters to ablate

### Step 2: GPU Benchmarks --- Oracle Recall (B1)

**Ensure the FP8 server is NOT running** (GPU memory exclusivity).

```bash
python benchmark/sparse_attention/benchmark_oracle_recall.py \
    --model Qwen/Qwen3-VL-4B-Instruct \
    --output-dir results/oracle_recall/ \
    --num-episodes 10 \
    --max-tokens 8192
```

**Options:**
- `--gradient-checkpointing` --- enable gradient checkpointing (slower but uses less VRAM; use if OOM at 8K tokens)
- `--max-tokens 4096` --- reduce sequence length if VRAM is tight

**Memory profile:** bf16 model (~8 GB) + activations (~4 GB at 8K) + gradients (~2 GB) = ~14 GB. Fits in 16 GB but tight.

The model is automatically unloaded after completion to free GPU memory for subsequent phases.

### Step 3: GPU Benchmarks --- Mind2Web (B2 + B5)

This benchmark runs in two phases due to GPU memory exclusivity between the bf16 model and FP8 server.

**Phase 9a: Extract key vectors** (bf16 model, no server):

```bash
python benchmark/sparse_attention/benchmark_mind2web.py extract-keys \
    --model Qwen/Qwen3-VL-4B-Instruct \
    --output-dir results/mind2web/ \
    --max-episodes 50
```

This saves key vectors and queries as `.pt` files in `results/mind2web/keys/`, then unloads the bf16 model.

**Phase 9b: Evaluate action quality** (FP8 server, no bf16 model):

First, start the FP8 server (separate terminal):
```bash
python -m sglang.launch_server \
    --model Qwen/Qwen3-VL-4B-Instruct \
    --quantization fp8 \
    --port 30000
```

Then run evaluation:
```bash
python benchmark/sparse_attention/benchmark_mind2web.py evaluate \
    --keys-dir results/mind2web/keys/ \
    --server-url http://localhost:30000 \
    --output-dir results/mind2web/
```

---

## 7. Execution Order and GPU State

| Phase | Benchmark | Time | GPU State |
|-------|-----------|------|-----------|
| 0 | Preflight verification | 5 min | Idle |
| 1a | B3: Selection Stability | 30 min | Idle (CPU only) |
| 1b | B4: Offline Ablation | 1 hr | Idle (CPU only) |
| 2 | B1: Oracle Recall | 2--3 hrs | bf16 model loaded (~14 GB) |
| 3a | B2 Phase 9a: Key extraction | 1--2 hrs | bf16 model loaded (~10 GB) |
| --- | Unload bf16 model, start FP8 server | 5 min | Transition |
| 3b | B2 Phase 9b: Action quality evaluation | 3--4 hrs | FP8 server (~6 GB) |

Phases 1a and 1b can run in parallel with each other and with Phase 0. Phases 2 and 3a share the bf16 model load and can be run sequentially in the same session. Phase 3b requires the FP8 server, which is mutually exclusive with the bf16 model on a 16 GB GPU.

---

## 8. Output Format

All benchmarks produce timestamped CSV files in their respective output directories.

### B1 Output (`oracle_recall_YYYYMMDD_HHMMSS.csv`)

| Column | Description |
|--------|-------------|
| `episode` | Episode index |
| `turn` | Turn index within episode |
| `context_tokens` | Sequence length in tokens |
| `num_pages` | Number of KV cache pages |
| `budget` | Oracle budget (number of oracle top-k pages) |
| `is_return_step` | Whether this turn revisits a previous topic |
| `quest_recall` | Quest's recall against gradient oracle top-k |
| `h2oquest_recall` | H2OQuest's recall against gradient oracle top-k |
| `delta` | `h2oquest_recall - quest_recall` |

### B2+B5 Output (`mind2web_YYYYMMDD_HHMMSS.csv`)

| Column | Description |
|--------|-------------|
| `episode` | Episode annotation ID |
| `domain` | Mind2Web domain (travel, shopping, etc.) |
| `step` | Step index within episode |
| `method` | One of: `dense`, `quest`, `h2oquest` |
| `action` | Model's predicted action (truncated to 200 chars) |
| `ground_truth_op` | Ground truth operation type |
| `ground_truth_repr` | Ground truth action description |
| `operation_match` | Whether predicted op type matches ground truth |
| `element_match` | Whether predicted element matches ground truth |
| `exact_match` | `operation_match AND element_match` |
| `html_tokens` | Number of tokens in the HTML context sent |
| `dense_agreement` | Whether truncated output matches dense output |

### B3 Output (`stability_YYYYMMDD_HHMMSS.csv`)

| Column | Description |
|--------|-------------|
| `score_decay` | Score decay hyperparameter |
| `alpha_decay` | Alpha decay hyperparameter |
| `episode` | Episode index |
| `turn` | Turn index |
| `is_return_step` | Whether this is the return-to-topic-A turn |
| `quest_jaccard_prev` | Quest Jaccard with previous turn's selection |
| `h2oquest_jaccard_prev` | H2OQuest Jaccard with previous turn's selection |
| `quest_jaccard_t0` | Quest Jaccard with turn 0 selection |
| `h2oquest_jaccard_t0` | H2OQuest Jaccard with turn 0 selection |
| `method_overlap` | Jaccard between Quest and H2OQuest at same step |
| `quest_num_selected` | Number of pages Quest selected |
| `h2oquest_num_selected` | Number of pages H2OQuest selected |

### B4 Output (`ablation_offline_YYYYMMDD_HHMMSS.csv`)

| Column | Description |
|--------|-------------|
| `sweep_param` | Parameter being ablated |
| `sweep_value` | Current value of the swept parameter |
| `episode` | Episode index |
| `turn` | Turn index |
| `decode_step` | Decode token index within the turn |
| `alpha` | Effective alpha at this decode step |
| `is_return_step` | Whether this is the return turn |
| `quest_historical_recall` | Fraction of historically-important pages Quest retains |
| `h2oquest_historical_recall` | Fraction H2OQuest retains |
| `recall_delta` | `h2oquest_historical_recall - quest_historical_recall` |

---

## 9. Interpreting Results

### What "Good" Looks Like

| Benchmark | Quest (Expected) | H2OQuest (Expected) | Key Insight |
|-----------|-------------------|----------------------|-------------|
| B1 Oracle Recall (return steps) | 40--50% | 65--75% | Accumulated history guides selection toward truly important pages |
| B2 Dense Agreement | 60--70% | 75--85% | Better page selection preserves action-relevant tokens |
| B3 Jaccard(t0) at return | < 0.60 | > 0.80 | Temporal memory retains turn-0 pages through topic shift |
| B4 Historical Recall (alpha_decay=0.7) | baseline | +0.25 delta | Moderate decay is optimal; too fast forgets, too slow ignores fresh |

### What to Watch For

- **B1 at early turns (turn 0):** H2OQuest and Quest should be nearly identical (no accumulated history yet, alpha=1.0).
- **B3 at turn 2--3 (topic shift):** Both methods should show low Jaccard with turn 0 (expected --- different topic).
- **B4 alpha ramp-up:** At decode_step=0, alpha=1.0, so H2OQuest equals pure fresh scoring. The delta should grow as alpha decays.
- **B2 dense baseline:** Dense should always have the highest action quality. If Quest or H2OQuest sometimes beats dense, inspect for noise.
- **Float32 overflow:** If accumulated scores exceed ~1e30 after many steps, check the `score_decay` configuration.

### Statistical Considerations

- B3 and B4 use 50 and 30 episodes respectively for statistical power. Report means with standard errors.
- B1 uses 10 episodes due to GPU cost. Consider bootstrap confidence intervals.
- B2 uses ~50 Mind2Web episodes. Group by domain (B5) to identify whether gains are uniform or domain-specific.
- All benchmarks use seeded random number generators for reproducibility.

---

## 10. Troubleshooting

### OOM on Gradient Oracle (B1)

Reduce sequence length or enable gradient checkpointing:
```bash
python benchmark/sparse_attention/benchmark_oracle_recall.py \
    --max-tokens 4096 \
    --gradient-checkpointing
```

### FP8 Server Already Using GPU (B1/B2 Phase 9a)

The bf16 model (~8 GB) and FP8 server (~6 GB) cannot coexist on a 16 GB GPU. Stop the FP8 server before running B1 or B2 Phase 9a:
```bash
# Kill the server process, then run the bf16 benchmark
```

### Mind2Web Download Issues

The dataset requires a Hugging Face token for some splits:
```bash
huggingface-cli login
```

Or set the token via environment variable:
```bash
export HF_TOKEN=hf_xxxxx
```

### Preflight Check Failures

- **Model config mismatch:** Verify you are using `Qwen/Qwen3-VL-4B-Instruct`, not a different variant.
- **GPU memory warning:** Ensure no other GPU processes are running. Use `nvidia-smi` to check.
- **SGLang not installed:** Check 4 (_pending_fresh behavior) requires SGLang to be importable. It will skip gracefully if not available --- verify manually on the deployment machine.

---

## 11. Extending the Suite

### Adding a New Scoring Algorithm

1. Implement the scoring function in `scoring_adapter.py` following the pattern of `quest_score_mean_agg` or `quest_score_max_agg`.
2. Add it to the benchmark scripts' scoring section alongside Quest and H2OQuest.
3. No changes needed to `extract_keys.py`, `gradient_oracle.py`, or `select_topk` --- these are algorithm-agnostic.

### Changing the Model

1. Update the constants at the top of each benchmark file (`NUM_KV_HEADS`, `GQA_GROUP_SIZE`, `HEAD_DIM`, `SCORE_LAYER`).
2. Run `preflight.py` with `--model your/model-name` to verify the architecture.
3. Adjust `SCORE_LAYER` to the middle layer of the new model (layer_count // 2).

### Adding More Ablation Parameters

1. Add the parameter and its sweep values to `ABLATION_SWEEPS` in `benchmark_ablation_offline.py`.
2. Add the corresponding logic in the sweep loop (similar to `num_sink_pages` handling).

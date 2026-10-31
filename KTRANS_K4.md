# K4: LLM-based Urban Indicator Knowledge Representation

**Goal:** Controlled comparison of text representation, while using the
unchanged K3-A normalized spatial expert bank and the same source-task LOTO.
This PR adds a Qwen3 text **embedding model**, not an online Qwen3-VL
inference path. Text is encoded once, then cached.

## Environment
Use your project Python with `transformers>=4.51.0`; PyTorch already
configured for ROCm is reused. Do NOT replace the ROCm torch installation.
Qwen's official implementation uses the final non-padding token of the
last hidden state, followed by L2 normalization. We use that method.

## Step 1: precompute K4 embeddings

```bash
git pull origin main
python -m unittest discover -s tests -p 'test_ktrans_k4.py' -v
python -m py_compile run_ktrans.py urban_transfer/ktrans.py urban_transfer/ktrans_text.py scripts/precompute_llm_task_embeddings.py

python -u scripts/precompute_llm_task_embeddings.py \
  --task-profile short --text-model-id Qwen/Qwen3-Embedding-0.6B

python -u scripts/precompute_llm_task_embeddings.py \
  --task-profile structured --text-model-id Qwen/Qwen3-Embedding-0.6B
```

The cached task vectors are fingerprinted by full task text + model ID.
Training does not load Qwen weights, only these small .pt dictionaries.

## Step 2: one-fold comparisons

```bash
# T1: original short text, but Qwen3 encoder
python -u run_ktrans.py train \
  --expert-bank --normalized-experts --task-encoder qwen3 \
  --task-profile short --heldout-task digital_total_cell_intensity \
  --limit 5000 --epochs 30 --patience 0 --batch-size 128 \
  --output runs/k4_t1_short_qwen

# T2: structured task knowledge, Qwen3 encoder
python -u run_ktrans.py train \
  --expert-bank --normalized-experts --task-encoder qwen3 \
  --task-profile structured --heldout-task digital_total_cell_intensity \
  --limit 5000 --epochs 30 --patience 0 --batch-size 128 \
  --output runs/k4_t2_structured_qwen
```

T0 is existing K3-A with RemoteCLIP text and short descriptions.

## Step 3: LOTO

```bash
python -u scripts/run_ktrans_loto.py --mode all \
  --normalized-experts --task-encoder qwen3 --task-profile short \
  --runs-root runs/ktrans_k4_t1_loto

python -u scripts/run_ktrans_loto.py --mode all \
  --normalized-experts --task-encoder qwen3 --task-profile structured \
  --runs-root runs/ktrans_k4_t2_loto
```

## Optional T3 reviewed LLM profiles

Provide a JSON dictionary mapping **all ten indicator keys** to at least
50-character research-reviewed profile strings. Check descriptions against
the actual dataset computation code. The repository does NOT auto-generate
domain claims from an LLM or silently treat speculative causal factors as
verified truth.

```bash
python -u scripts/precompute_llm_task_embeddings.py \
  --task-profile llm --llm-profile-file profiles_reviewed.json

python -u run_ktrans.py train --expert-bank --normalized-experts \
  --task-encoder qwen3 --task-profile llm \
  --llm-profile-file profiles_reviewed.json \
  --heldout-task digital_total_cell_intensity \
  --limit 5000 --epochs 30 --patience 0 \
  --output runs/k4_t3_reviewed
```

**Leakage rules:** no target-label means, heldout test statistics,
test-derived error analysis, or unseen-city features in task profiles.
The pseudo-heldout source task's definition is allowed, but its labels
must never enter its training fold. Meta-validation is development data
used for checkpoint selection, not a statistically independent final test.

## Interpretation

Compare per-fold **Pearson, R2, MSE, Bias and prediction std**, not only
macro MSE. Existing results indicate a tradeoff between spatial correlation
and numeric output scale; a stronger language encoder does not necessarily
solve numerical calibration or learn predictive transferability.

# K7-M6: frozen MLLM hidden-state guided source-expert routing

## Goal

Rejoin the original proposal: **satellite image + indicator definition +
source/target mechanism text -> frozen MLLM hidden states -> trainable
cross-attention expert router -> continuous urban-indicator regression.**

MLLM: `Qwen/Qwen2.5-VL-3B-Instruct`, frozen.
Student: RemoteCLIP image features + geographic context + two learned
source experts (Population, Food). The router cross-attends to **8
contextualized hidden tokens** of the image/indicator-prompt MLLM
forward pass, **without decoding or generating any answer text**.

Target-aware label-free training: on each SOURCE training image we
precompute 3 distinct Qwen-VL forwards, one each with the Population,
Food, and Education definitions plus permitted directed transfer texts.
A frozen **image-conditioned teacher** uses hidden-space cosine
similarities between Education and each source prompt to derive two soft
weights **per image**. Target router outputs on that source image are
KL-distilled toward the dynamic teacher. Population/Food regression
labels train two source experts; **zero Education labels** are used for
optimization. Teacher scores are imperfect proxy similarities, not
known transfer utility. MLLM weights are never finetuned.

The target text's participation in train-time gradients prevents
"Education appears only at inference" mismatch, unlike M3. It does
not guarantee that the MLLM hidden states are task-meaningful, or that
router learning carries across tasks.

## One small run on MI250

This experiment starts with only **128 SOURCE train rows** and
**64 + 64 Education pseudo-validation rows** to avoid paying for
tens of thousands of Qwen-VL forwards. **This sample is deliberately
not comparable numerically with the full M1–M5 results**, and has high
sampling variability. The split remains seed42, city-isolated UCUI,
from the same source train.csv. After validating the model plumbing,
scale the *same experiment* rather than creating architectural variants.

Source and target label policies:
- From train.csv: select Population/Food train rows in source-seen
  cities, and separate Education SCUI/UCUI pseudo-dev rows.
- Source training images get THREE MLLM prompts. Education pseudo-dev
  images get Education prompt only.
- No Education pseudo-dev row participates in optimization.
- Source-only training labels are used for regression and scale
  calibration; **no label is included in any MLLM prompt**.
- No official final-test files are opened.

### Step 1. Precompute frozen MLLM hidden states (resumable)

```bash
git pull --ff-only origin main
python -m unittest discover -s tests -p 'test_k7_m6_hidden_router.py' -v

python -u scripts/cache_k7_m6_hidden.py \
  --source-csv /scratch/project_462001391/xiyanxin/Urban_experience/codes_BaseStation/city_indicator_splits/train.csv \
  --image-root /flash/project_462001391/xiyanxin/Urban_experience/output_files \
  --feature-dir runs/m3_features \
  --tensor-cache runs/k7/e1_e2_v1/_source_tensors.pt \
  --model-id Qwen/Qwen2.5-VL-3B-Instruct \
  --train-limit 128 --eval-limit 64 --seed 42 \
  --n-tokens 8 --max-pixels 100352 \
  --run-dir runs/k7/m6_mllm_hidden_seed42
```

This requires `transformers` with Qwen2.5-VL support and the model
checkpoint in HF cache or reachable from the server. GPU is required.
For better ROCm compatibility the model uses regular eager attention,
not mandatory CUDA FlashAttention. Keep the same arguments on restart:
the SQLite cache resumes missing image/task pairs without discarding
already completed embeddings. First inference is much more expensive
than one lightweight regression model training.

### Step 2. Train EXACTLY one student

```bash
python -u scripts/run_k7_m6_student.py \
  --run-dir runs/k7/m6_mllm_hidden_seed42 \
  --text-cache-dir runs/ktrans_text \
  --seed 42 --epochs 20 --batch-size 64 \
  2>&1 | tee runs/k7/m6_mllm_hidden_console.log
```

The runner refuses to overwrite existing model or metrics. Outputs:
`manifest.json`, `hidden.sqlite3`, `model.pt`, `metrics.csv`,
`config.json`, and the tee log. Only one trained student. Output:

```text
[k7-m6] SCUI MSE=... Pearson=... Ratio=... PopWt=... FoodWt=...
[k7-m6] UCUI MSE=... Pearson=... Ratio=... PopWt=... FoodWt=...
```

The MLLM is frozen and runs only during hidden precompute. It is not
jointly trained or asked to output text, JSON, weights, or numbers.

### Important limitations

This is a **small proof-of-mechanism** run, not a final metric
comparison. The student mixes source-expert features on a normalized
shared target scale inferred from source-only task descriptions. A
frozen MLLM similarity teacher does not constitute ground-truth
migration quality. If a positive result appears, a subsequent frozen
checkpoint evaluation with image-prompt mismatches or text-only
hidden states can isolate image-conditioned knowledge before making a
knowledge-effect claim.

The full Qwen2.5-VL checkpoint is separate from Qwen3-Embedding-0.6B
and RemoteCLIP. It is **not already cached merely because those other
embeddings exist**.

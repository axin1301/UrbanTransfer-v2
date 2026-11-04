# K7-D: source-only multimodal expert transfer prototype

K7-A/B established that frozen Qwen3-VL-8B visual proxy scores vary
with task and scene. K7-C gray images yielded valid JSON yet **all
24 blank task×tile outputs had null values** for every visual cue,
relevance and confidence score. This is a necessary caution signal,
not human-verified visual fidelity or proof of transfer benefit.

K7-D advances to a **source-only development experiment** that
tests whether cached multi-modal knowledge can improve *source-task
pseudo-transfer* while preserving the existing K5-A global route.

## Design

Frozen:
- K4 normalized RemoteCLIP expert bank and its source-only moments
- K5-A global learned transfer prior and source-only pseudo-target routes
- Qwen task embeddings, and Qwen3-VL-8B-Instruct (BF16/SDPA)

For image tile `r` and indicator `t`, retrieve versioned cached
VLM evidence. Every evidence vector has six cues, three score types
(visible, relevance, confidence) and corresponding 18 missingness
indicators: 36 dimensions total. Null scores are **not** confused with
true ordinal zero. Image–Task relevance additionally subtracts the
conceptual Text-Only prior to isolate a candidate image-dependent
component. Router capacity is identical for every knowledge ablation.

```
spatial weights = (1-g) * frozen_K5A_prior + g * learned_knowledge_route
```

The baseline K5-A variant does not train a router. All three K7
ablations train the same small source-only network:
- `text_only`: conceptual task knowledge, no observed visual cues
- `image_only`: image morphology, no task-conditioned relevance
- `image_task`: task-conditioned image cues/relevance/confidence,
  with text-only relevance subtraction

On outer source-task examples, the current source expert is **masked**
before computing mixture weights. The frozen K4 shared trunk is
previously trained on all outer source labels: **this is not
independent inner-fold cross-fitting**. K5-A nested checkpoints are
not reused for each pseudo-target row here. Consequently K7-D
is a quick mechanistic **DEV prototype**, not valid proof of
cross-fitted expert reliability or a conclusive novel-task claim.

Both training and eval use the same frozen standardized expert bank.
For outer heldout tasks, output scale follows the unchanged K4
semantic source moments. For heldout-city seen source tasks,
source-task moments are known and unchanged.

No outer heldout task labels enter optimizer, prompt construction,
model selection or checkpoint selection. All trained routers are
saved at the **last** epoch. The existing K4 `best.pt` has already
been meta-selected, so all outer meta_SCUI/meta_UCSI/meta_UCUI
scores are explicitly DEV-only.

## Exact steps on the ROCm MI250X server

Run from project root with the environment that already ran K7-B/C.

### 1. Obtain the actual Education K4 checkpoint

Do not assume an unverified `runs/k4_focused/...` path. Recover the
K4 path from the saved K5-B2 composer:

```bash
git pull origin main
python -m py_compile scripts/prepare_ktrans_k7d.py scripts/cache_ktrans_k7d.py scripts/run_ktrans_k7d.py
python -m unittest discover -s tests -p 'test_ktrans_k7d.py' -v

CKPT=$(python - <<'PY'
import torch
x=torch.load('runs/k5b2_education/composer.pt',map_location='cpu',weights_only=False)
print(x['checkpoint'])
PY
)
echo "K4 checkpoint: $CKPT"
test -f "$CKPT"
```

Adjust the composer path to the actual saved `composer.pt` if needed.

### 2. Freeze a deterministic city×task-balanced manifest

Use 512 rows **per split** as a modest first trial.
The manifest includes exact dataset indexes and image paths, and
**no numeric labels**. Set the limit to 0 only when ready to perform
full precompute (which may take many VLM generations).

```bash
python -u scripts/prepare_ktrans_k7d.py \
  --checkpoint "$CKPT" \
  --max-rows-per-partition 512 \
  --seed 42 \
  --output runs/k7d_education/manifest.json
```

The K4 expert bank and K5-A protocol remain exactly unchanged.
Manifest validates train contains only five outer source tasks and
excludes heldout cities. `meta_SCUI`, `meta_UCSI` and `meta_UCUI`
are sampled deterministically without using target values.

### 3. Inspect cache work estimate, then smoke-test VLM

```bash
python -u scripts/cache_ktrans_k7d.py \\
  --manifest runs/k7d_education/manifest.json \\
  --output runs/k7d_education/evidence.jsonl \\
  --reuse-cache runs/k7b_pilot/evidence.jsonl \\
  --dry-run
```

```bash
python -u scripts/cache_ktrans_k7d.py \
  --manifest runs/k7d_education/manifest.json \
  --output runs/k7d_education/evidence.jsonl \
  --reuse-cache runs/k7b_pilot/evidence.jsonl \
  --batch-size 4 \
  --max-pixels 262144 \
  --max-new-tokens 440 \
  --max-new-jobs 8
```

### 4. Complete VLM evidence cache

```bash
python -u scripts/cache_ktrans_k7d.py \
  --manifest runs/k7d_education/manifest.json \
  --output runs/k7d_education/evidence.jsonl \
  --reuse-cache runs/k7b_pilot/evidence.jsonl \
  --batch-size 4 \
  --max-pixels 262144 \
  --max-new-tokens 440
```

The exact number of VLM jobs is printed before model loading. It
deduplicates identical image-only tile observations, per-image/task
observations and task-only priors. Prior K7-B records are reused
only when their **full versioned prompt identity** matches.

This can be expensive for hundreds of samples; use GPU batch size
4 first, then try 8 if memory permits. HIP OOM halves batch size.
The JSONL cache is append-only and resumes after interruption.
A successfully parsed job need never be computed again.

### 5. Train fixed-epoch source-only ablations

```bash
python -u scripts/run_ktrans_k7d.py \
  --checkpoint "$CKPT" \
  --k5a-protocol runs/k5a_education/protocol.json \
  --manifest runs/k7d_education/manifest.json \
  --knowledge-cache runs/k7d_education/evidence.jsonl \
  --reuse-cache runs/k7b_pilot/evidence.jsonl \
  --output runs/k7d_education/ablation \
  --epochs 12 \
  --batch-size 256 \
  --seed 42
```

Training **requires 100% valid VLM knowledge coverage for the frozen
manifest**. Missing, invalid, misaligned prompt or image-task records
fail closed: no partial/8-tile shortcut and no imputation from labels.

Results:
- `runs/k7d_education/ablation/evaluation.csv`:
  `k5a`, `text_only`, `image_only`, `image_task` ×
  meta_SCUI / meta_UCSI / meta_UCUI, with MSE, Pearson, bias and gate stats.
- `protocol.json`: source isolation and DEV limitations.
- `text_only_last.pt`, `image_only_last.pt`,
  `image_task_last.pt`: fixed-epoch lightweight routers.

Repeat the exact three stages with Nightlight and Digital Cell K4/K5-A
checkpoints and independently generated manifests. Do **not**
choose architecture or adjust hyperparameters based on meta-UCUI
comparisons; doing so would reuse the development split for selection.

## Interpretation

A multimodal route beating K5-A on one heldout DEV task does not
establish generalizable transfer. The meaningful next scientific
step is independent **inner pseudo-task expert cross-fitting** and
heldout-city validation, plus multiple seeds, confidence intervals,
paired prediction differences, source-only ablation and eventually
pre-registered, untouched target-task evaluation.

The VLM can suggest potential spatial proxies but cannot directly
observe true education access, nightlight radiance or cell tower
intensity from daytime aerial tiles. No target labels are ever
generated by the model for training.

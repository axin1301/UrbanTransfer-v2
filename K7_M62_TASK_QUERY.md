# K7 M6.2 — Task-Queried Multimodal Expert Routing

## Why this revision

M6.1's cached Qwen-VL tokens **do contain task-dependent signals**:
Definition/Relation RAW relative differences were 0.26–0.86 on
the same source images, but after Student's shared-query attention
the knowledge cosine was ~0.9999 and router changes ~0.001.
This pointed to **fusion suppressing task semantics** more than
a shortage of MLLM hidden-state dimensions.

M6.2 does NOT repeat MLLM inference. It reads the existing M6.1
`64×2048` hidden tokens (32 visual / 16 definition / 16 relation)
with strict existing content keys and source-only manifest.

New trainable student:
- Separate Definition and Relation projections, mean pooling and
  a learned **Task Query**. The task query is never a single shared
  free parameter across indicators.
- Attention `Q=task query, K/V=Qwen image tokens`.
- A fused feature preserves `task query`, `attended visual`,
  and their multiplicative interaction; the original task query
  additionally has an **explicit residual path** into both expert
  router logits and expert channel gates.
- Two specialized Population/Food expert feature branches continue
  using RemoteCLIP tile vectors + four geographic context values.
- One regression head maps the feature-level expert mixture to a
  standardized continuous prediction.

Fixed **one Student**, **seed 42**, **20 epochs**, **batch size 128**,
the same small train=128, SCUI=64, UCUI=64 source-derived
pseudo-validation rows as M6.1.

Training objective:
1. Source Population/Food log1p normalized MSE (full two-source
   unmasked router).
2. Source expert-specific MSE on the matching supervised task,
   weighted 0.3.
3. On the **same SOURCE training image** pass both source indicator
   definitions, classify Population vs Food from each task query.
   This adds explicit task identity separation with source task
   IDs, not target labels; weighted 0.2. This is an auxiliary task,
   NOT evidence of knowledge transfer by itself.
4. For the **Education definition applied to SOURCE training images**,
   align its fused knowledge representation weakly to the frozen
   RemoteCLIP image vector, weighted 0.05. Only positive cosine
   alignment is used; no erroneous negative pairs when the same
   tile occurs under multiple supervised indicator rows.

No direct Education target regression labels, Education dev images
or held city image features enter training. Source text-only scale
calibration comes from the unchanged `calibrated_moments` method.
Do NOT use final official test CSVs. All evaluation remains
post-optimization on pseudo-dev carved from the source train CSV.

**No guaranteed gains**: the target image-alignment loss can still
encourage generic visual features. The important test is whether the
indicator-conditioned **Task Query** and output actually change
on the *same* image when source/target indicator text changes.

## On MI250 / ROCm

Pull the new source and run CPU unit tests:

```bash
git pull --ff-only origin main
python -m unittest discover -s tests \
  -p 'test_k7_m62_task_query.py' -v
```

**NO Qwen-VL cache command is needed.** The existing complete M6.1
cache directory is read-only. If your M6.1 cache lives somewhere else,
pass its location in `--cache-dir`.

Run exactly one M6.2 fit:

```bash
python -u scripts/run_k7_m62_task_query.py \
  --cache-dir runs/k7/m61_structured_seed42 \
  --output runs/k7/m62_task_query_seed42 \
  --text-cache-dir runs/ktrans_text \
  --seed 42 --epochs 20 --batch-size 128 \
  2>&1 | tee runs/k7/m62_task_query_console.log
```

The new output directory MUST NOT exist. Existing M6/M6.1
cache/checkpoints/metrics are untouched. Output files in the new
directory: `model.pt`, `config.json`, `metrics.csv`,
`same_image_prompt_audit.json`.

### Expected terminal diagnostics

```text
[k7-m62] SCUI MSE=... Pearson=... Ratio=... PopWt=... PopWtStd=... MismatchChange=...
[k7-m62] UCUI MSE=... Pearson=... Ratio=... PopWt=... PopWtStd=... MismatchChange=...
[k7-m62] population_count__vs__education_accessibility task_query_cos=... router_abs_delta=... pred_abs_delta=...
[k7-m62] food_accessibility__vs__education_accessibility task_query_cos=... router_abs_delta=... pred_abs_delta=...
[k7-m62] population_count__vs__food_accessibility task_query_cos=... router_abs_delta=... pred_abs_delta=...
```

Same-image task comparisons use source TRAIN image rows only and
no target labels. Swapped-image MLLM inputs are a frozen-checkpoint
sensitivity audit. Nonzero response to task definitions does NOT
by itself prove useful semantic transfer; predictive quality and
held city performance must be assessed separately.

### Scientific precautions

This task-aware training setup knows the Education **description**
ahead of fitting but never its labels; describe it as target-aware,
zero-label transfer, not task-definition-unseen zero-shot.
Evaluation is SCUI/UCUI pseudo-dev only, NOT official final test.
Results with 128 source training rows are not comparable to large
M1–M5 baselines without matching the sample.
No hyperparameter search or checkpoint selection on Education pseudo
labels should be performed.

# K7 M6.2 frozen calibration decomposition + attention-value audit

## Purpose

M6.2 froze the trained student and previously found:
- Normal UCUI pseudo-dev: MSE 2.087, Pearson 0.640.
- Zero direct task query: UCUI MSE 1.802, Pearson 0.624.
- Zero fused direct path: UCUI MSE 1.950, Pearson 0.642.
- Zero attention output: UCUI MSE 2.174, Pearson 0.628.
- Attention JS divergences vary by source indicator and image.

To identify **why MSE changes despite similar Pearson**, report the
exact three-part error decomposition of each frozen intervention:

```text
MSE = (mean(pred) - mean(y))^2
    + (std(pred)  - std(y))^2
    + 2 * (std(pred)*std(y) - covariance(pred,y))
```

This is the population-statistics identity using `ddof=0`, with
no fitting or calibration adjustment. The last term equals
`2*std(pred)*std(y)*(1-Pearson)` where both standard deviations are
positive. The script asserts the reconstructed MSE matches direct
MSE to numerical precision. It also records prediction/true means,
standard deviations, signed mean error, and Pearson for each
SCUI/UCUI condition.

For the SAME 128 **source training images**, compare
Population/Food/Education (no labels read for this analysis) at
three actual trained Student stages:

- **Task Query**, computed from cached Definition + Relation.
- **Attention Output**, the weighted sum of the 32 visual value
  tokens under that task query, after the MHA module's output projection.
- **Fusion Output**, the trained combination of
  `[query, attention_output, query*attention_output]`.

For each pair and stage, show cosine similarity and relative L2
difference. These quantities diagnose information loss across
stages, not geographical explainability or semantic correctness.

## Run (CPU only; no MLLM or new training)

```bash
git pull --ff-only origin main
python -m unittest discover -s tests \
  -p 'test_k7_m62_calibration_attention.py' -v

python -u scripts/audit_k7_m62_calibration_attention.py \
  --run-dir runs/k7/m62_task_query_seed42 \
  --cache-dir runs/k7/m61_structured_seed42 \
  --batch-size 32 --device cpu
```

Reads existing `model.pt` and M6.1 `hidden.sqlite3` and source
tensor cache. No official final test files are touched.
The original M6.2 SCUI/UCUI pseudo-dev Education labels from the
**source train CSV** are used for score decomposition only, strictly
post-fit. The checkpoint, metrics and SQLite caches are untouched.

The report is written to
`runs/k7/m62_task_query_seed42/frozen_calibration_attention_audit.json`.
The script refuses to overwrite this file; for repeat experiments
provide a fresh `--output path/to/new.json`.

## Interpret cautiously

- Larger mean-bias-squared term points to a calibration offset.
- Larger std-mismatch-squared term points to a prediction dispersion
  mismatch, not necessarily incorrect spatial ranking.
- Larger correlation term reflects incomplete agreement of spatial
  ranking/amplitude in the MSE identity; these parts are arithmetic
  descriptors, **not independent causal error sources**.
- If attention-output vectors change very little when source indicator
  changes, despite nontrivial attention JS, the visual VALUE path
  could be suppressing task-specific information.
- If attention outputs differ but fusion outputs converge, the
  trained fusion pathway is likely suppressing task-conditioned
  signals.
- In all cases, zero-path interventions are off-manifold. Never
  select or recalibrate a new model based on Education pseudo-dev
  ablation metrics, or describe those scores as an unbiased final
  test estimate.

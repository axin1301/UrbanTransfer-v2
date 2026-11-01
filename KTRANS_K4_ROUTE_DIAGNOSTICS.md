# K4 paired short-versus-structured diagnostics (no training)

This study tests whether task-profile differences mainly alter prediction
means/scales or alter spatial patterns. It does **not** use evaluation labels
for calibration and never reads official target test sets.

For the same held-out source task and the same split, the script loads two
independently trained K4 normalized-expert checkpoints. It exports:

- `expert_routes.csv`: short/structured semantic expert weights, and
  source TRAIN-only moments.
- `paired_predictions.csv`: paired prediction Pearson correlation, means,
  mean shift, standard deviation ratio, best diagnostic affine fit and
  residual std. The affine fit is computed only to describe the relationship
  between two sets of predictions, **not** to calibrate a deployable model.

Run from repository root:

```bash
git pull origin main
python -m py_compile scripts/diagnose_k4_profiles.py

for TASK in digital_modernization_tmi education_accessibility nighttime_light_intensity; do
  python -u scripts/diagnose_k4_profiles.py \
    --short-checkpoint "runs/k4_focused/short_${TASK}/best.pt" \
    --structured-checkpoint "runs/k4_focused/structured_${TASK}/best.pt" \
    --batch-size 1024 \
    --output "runs/k4_route_analysis/${TASK}"
done

python -u scripts/diagnose_k4_profiles.py \
  --short-checkpoint runs/k4_t1_short_qwen8b/best.pt \
  --structured-checkpoint runs/k4_t2_structured_qwen8b/best.pt \
  --output runs/k4_route_analysis/digital_total_cell_intensity
```

The script validates same held-out task, training CSV, sampled row order,
city split and source-expert task list before computing paired metrics.
The correlation of **two predictions** is not their respective correlation
with labels; compare original diagnostics separately. A high paired
correlation with large mean shift suggests output scaling rather than
spatial discrimination changed. It cannot establish causality because
the models were trained separately.

Print output with three-decimal formatting. No additional training required.

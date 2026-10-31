# K-TransMoE: prediction diagnostics

Compare K1 and K2-A **best checkpoints** on the exact internal meta-SCUI,
meta-UCSI and meta-UCUI partitions used for their original development runs.
No external target split is opened.

## Run

```bash
git pull origin main
python -m unittest discover -s tests -p test_ktrans_diagnostics.py -v

python -u scripts/analyze_ktrans_predictions.py \
  --checkpoint runs/k2a_baseline_digital_total_cell_intensity/best.pt \
  --checkpoint runs/k2a_scale_digital_total_cell_intensity/best.pt \
  --feature-dir runs/m3_features \
  --text-cache-dir runs/ktrans_text_cache \
  --batch-size 256 \
  --output runs/ktrans_diagnostics/digital_total_cell_intensity \
  --save-predictions
```

Outputs: `diagnostics.csv` and `diagnostics.json`, and optional
`predictions_<index>_<modelname>.csv` with per-row city, task, target, prediction.

## Metrics

- `target_mean`, `pred_mean`, `target_std`, `pred_std`: distribution statistics
- `bias`: mean(prediction - target)
- `pearson`: within-partition spatial Pearson correlation (null if constant)
- `mse`, `mae`, `r2`: regression metrics
- `oracle_mean_mse`: MSE of the partition's **true label mean**, equal to target variance; diagnostic oracle **not a deployable baseline**.
- `pred_mean_constant_mse`: MSE of constant prediction equal to the **model's own prediction mean**.
- `spatial_gain_vs_pred_mean_constant`: pred_mean_constant_mse - model_mse; positive means model's spatial deviations improve on its own constant prediction.

Use `--checkpoint` multiple times to compare configurations. To compare six
held-out tasks, repeat the command with the task string in checkpoint paths
and aggregate the CSVs.

These are **development meta-validation** metrics; checkpoint selection
has already used these partitions. They must not be treated as independent
unbiased tests.

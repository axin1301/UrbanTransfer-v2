# K6.1 Loss Prediction-Level Diagnostics

A **read-only, inference-only** paired comparison of four saved
`k61loss_<task>_<mode>/gate_last.pt` checkpoints for the *same*
heldout partition. Loads existing K4 and K5-B2 weights, replays
K5-A anchor and the frozen regional router with Adapter OFF, and
measures where source-only auxiliary losses do or do not affect final
predictions.

For shared fixed expert outputs:
```
pred_mode - pred_mse =
    (gate_mode - gate_mse) * (pred_regional - pred_global)
```
The script checks this identity to within float32 tolerance.

- `route_gap_std`: standard deviation of regional minus global
  predictions. High means the two routes differ in spatial pattern;
  zero means gate changes cannot alter predictions.
- `route_gap_mean_abs`: average absolute route prediction difference.
- `gate_delta_mean_abs`: how much the auxiliary loss shifts the
  deployed gate relative to MSE.
- `pred_delta_mean_abs`: average absolute numerical impact on
  deployed predictions; unlike parameter norms, this measures effect
  at the output.
- `identity_error`: numerical sanity check on the exact interpolation
  formula.

No outer heldout task labels are loaded into a loss, optimizer,
checkpoint selection process, or any reported diagnostic. This is
development-set *prediction* analysis, not a performance test; the
underlying K4 checkpoint was previously meta-selected.

## Commands (run at repo root)

```bash
git pull origin main
python -m py_compile scripts/diagnose_ktrans_loss_predictions.py
python -m unittest discover -s tests -p 'test_ktrans_loss_predictions.py' -v

python -u scripts/diagnose_ktrans_loss_predictions.py \
  --run-dir runs \
  --task education \
  --protocol meta_UCUI \
  --output runs/k61loss_education_diagnostics
```

After all four runs for another task are complete, substitute
`--task nighttime` or `--task digital_total_cell` and use a
corresponding output folder. Do not retrain checkpoints for this
diagnostic.

This report does not determine the next architecture by itself.
If regional-vs-global predictions have little variation, gate tuning
has limited headroom. If routes differ but gate changes and predictions
remain very small, inspect supervised gradients and gate capacity.
A gate difference with a large prediction delta but no label-metric
improvement suggests the chosen correction is not consistently
beneficial. Strict K6.2-B reliability supervision still needs
independently trained inner regional routers.

# K7-H: label-free mean-preserving routing diagnostic

K7-G found that **Image–Task UCUI** candidate route decreased
centered residual error but increased squared mean bias, so we now
test a strictly label-free, **transductive** prediction adjustment.

## Exact operation

For a predeclared cohort `B` containing **every sampled row sharing
the same manifest partition and task**, the frozen K5-A and frozen
K7-D candidate predictions are `p0(x)` and `pg(x)`. Correct only
the mean of their difference:

```
d(x)        = pg(x) - p0(x)
mean_d(B)   = sum_{x in B} d(x) / |B|     # NO target values
p_fixed(x)  = pg(x) - mean_d(B)
```

**No indicator labels are read to construct the correction**; labels
are accessed only later to report errors. All three ablations and
Gate modes (`fixed_zero`, `trained`, `fixed_one`) use identical
cohorts and frozen checkpoints. A per-task group is essential for
`train` and `meta_UCSI`, where task scales are heterogeneous.

Mathematical invariants within each cohort:
- `mean(p_fixed)=mean(p0)`
- `bias(p_fixed)=bias(p0)` given *any* ground-truth y
- `var(p_fixed-y)=var(pg-y)`
- hence `MSE(p_fixed)-MSE(p0)` equals the **centered residual MSE**
  difference between the original VLM route and K5-A on that cohort

UCUI and SCUI each contain one heldout task, so their pooled
mean-preserved difference equals the corresponding K7-G centered
residual difference (up to floating-point/rounding). **This prediction
is an algebraic expectation, not independent empirical validation.**

## Run one seed, no new 8B inference or training

```bash
git pull origin main
python -m py_compile scripts/run_ktrans_k7h_mean_preservation.py urban_transfer/ktrans_k7h.py
python -m unittest discover -s tests -p 'test_ktrans_k7h_mean_preservation.py' -v

python -u scripts/run_ktrans_k7h_mean_preservation.py \
 --checkpoint "$K4_CKPT" \
 --k5a-protocol "$K5A_PROTOCOL" \
 --manifest runs/k7d_education/manifest_complete_cases.json \
 --knowledge-cache runs/k7d_education/evidence.jsonl \
 --reuse-cache runs/k7b_pilot/evidence.jsonl \
 --routers-dir runs/k7d_education/single_seed42 \
 --output runs/k7h_education/seed42 \
 --seed 42
```

Unset checkpoint paths can be recovered from
`runs/k7d_education/single_seed42/image_task_last.pt` as in K7-G.

Output:
- `mean_preservation.csv`: per-variant, per-partition and per-Gate
  original/mean-preserved paired error decomposition.
- `per_task_metrics.csv`: per-task error decomposition for all four
  partitions; per-task bias preservation is asserted.
- `cohort_audit.csv`: each task cohort's sample count and **label-free**
  mean correction, plus original/baseline/corrected prediction means.
- `protocol.json`: inputs, cohort policy and methodological cautions.

Inspect UCUI:

```bash
python - <<'PY'
import csv
p='runs/k7h_education/seed42/mean_preservation.csv'
for row in csv.DictReader(open(p)):
    if row['protocol']=='meta_UCUI' and row['variant']=='image_task':
        print(row['gate_mode'],row['correction'],
              'MSE='+format(float(row['mse']),'.3f'),
              'dMSE='+format(float(row['delta_mse_vs_k5a']),'+.3f'),
              'dBias2='+format(float(row['delta_bias_squared']),'+.3f'),
              'dCentered='+format(float(row['delta_centered_mse']),'+.3f'))
PY
```

## Scientific interpretation and deployment constraint

This is **posthoc DEV research**. The group is the full
`Complete-Case` partition/task cohort: it **requires all unlabeled
predictions for that batch** and can change when batches or urban
geography change. This is a transductive deployment assumption,
not an ordinary per-tile predictor. No target labels enter
correction, but the idea itself was motivated by previously
observed UCUI errors. Therefore do not claim an unbiased UCUI
improvement, select Gate/mean-preservation based on this UCUI, or
compare to full-sample K5-A as if they were matched. K4 was
meta-selected and is not independently cross-fitted.

If this approach is later formalized, define a realistic reference
cohort before inspecting test labels (e.g. the entire unlabeled
target-region prediction population) and evaluate on new
independent heldout tasks and cities.

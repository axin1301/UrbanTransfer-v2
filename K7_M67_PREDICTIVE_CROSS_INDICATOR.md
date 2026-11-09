# M6.7 Neighbor Cross-Indicator Predictive Context — Source-City DEV

## Main question

Does the source-trained prediction of an *other* indicator B at physical
neighbor tiles help predict target indicator A at the current tile,
beyond neighboring images and the predicted value of indicator A alone?

This is meaningfully different from M6.6's definition/relationship
text prototypes: the new context contains **numerical predictions of
source Population and Food**, never a real value from a validation tile.

## Label-leakage protocol

1. Reserve exactly the previous M6.5 DEV source cities; same
   leak-free `dev_visual_anchor.pt`.
2. Fit 4-fold GroupKFold source ridge nuisance predictors
   (StandardScaler + Ridge alpha=100) on RemoteCLIP+geo features, using
   **only Population/Food labels from M6.5 source-fit city rows**.
3. For all images/rows in a source-fit city, assign predictions from
   the fold model fitted on **other source-fit cities**, even if the row
   is not itself in the supervised source training partition.
   This prevents same-city source label shortcuts through neighbors.
4. For all other cities (including completely held source cities),
   predict with ridge regressors fitted on the entire source-fit
   subset only. Neither held-city label is used for auxiliary fit.
5. Predicted values are in each source task's training-standardized
   units (clipped to [-5,5]); they are not neighbor ground truth.
   There are no Education predicted values or Education outcome labels.
6. Models still use the same real K=8 physical RemoteCLIP neighbor
   statistics as M6.5, with physical-tile-deduplicated neighbor graph.

Cross-fit per-city predictions are cached in
`runs/k7/m67_oof_source_city.pt` with source label hash and
protocol metadata; they never modify source CSV, training cache, or
the original M6.5 checkpoints.

## Fixed paired student ablation

Three 20-epoch runs, identical randomly initialized student weights,
model parameter count, frozen M6.5 DEV visual anchor, source-only loss,
source-city training split, seed42 batch sequence, and source Qwen cache:

| Arm | Neighbor prediction channels (for target A) |
|---|---|
| `cross_indicator` | [predicted A at neighbor, predicted other B] |
| `same_indicator` | [predicted A, predicted A] |
| `no_predictive` | [0,0]; keeps distance, visual features, architecture |

All three arms use the same score-token encoder and attention;
only neighbor indicator predictive input differs. The model
computes a bounded residual on the frozen visual prediction.
This is a **source-task mechanism** comparison only, not an
independent validation of unseen Education transfer, which would
require a separately designed B-only protocol.

## Run — one Python command, no sbatch

From repository root **in an allocated GPU session**:

```bash
git pull --ff-only origin main
python -m unittest discover -s tests \
  -p 'test_k7_m67_predictive_cross.py' -v

python -u scripts/run_k7_m67_predictive_cross_dev.py \
  --full-dir runs/k7/m62_full_seed42 \
  --dev-dir runs/k7/m65_neighborhood_dev_seed42 \
  --neighbor-cache runs/k7/m65_neighbors_k8.pt \
  --graph-cache runs/k7/m66_physical_graph_k8.pt \
  --oof-cache runs/k7/m67_oof_source_city.pt \
  --output runs/k7/m67_predictive_cross_dev_seed42 \
  --epochs 20 --batch-size 256 --device cuda
```

The source-fold ridge preprocessing runs on CPU once. All three student
arms then train sequentially on the GPU, **60 model-epochs** total.
No Qwen-VL re-encoding; no visual anchor retraining. Original
M6.5/M6.6 checkpoint files are left untouched.

Each finished epoch writes an atomic checkpoint in its arm directory.
After a crash, rerun the identical command to continue.
Completed arms are skipped.

## Results (all displayed floating values 3dp)

```bash
cat runs/k7/m67_predictive_cross_dev_seed42/REPORT.md
cat runs/k7/m67_predictive_cross_dev_seed42/SUMMARY.csv
```

Interpretation:

- If `cross_indicator` improves both MSE and Pearson compared with
  `same_indicator` and `no_predictive`, this supports cross-source
  predictive spatial information.
- If only one source task improves, report task dependence.
- Same neighbor graph, visual statistics, capacity and initial weights
  across arms; only predicted-score channels vary.
- Because these are single-seed source-city DEV comparisons, not
  independent held-out target evaluation, avoid general transfer
  claims or Education-based model selection.

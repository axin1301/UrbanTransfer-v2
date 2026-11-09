# M6.5 paired source-city 3-arm training ablation

**One command**, no Slurm wrapper. Source-only Pop/Food city-held
validation; the official Education/test outcomes are never used. The
original Dev/Final runs are immutable.

Compare **independently trained** real K=8 neighborhoods, all-zero
neighborhoods, and same-city physical-tile mean context. All arms use
the **same initialized trainable state** copied from a common
seed42 model, identical architecture and trainable parameter count,
same `dev_visual_anchor.pt`, source train rows, optimizer, fixed 20 epochs,
batch256, fixed seed42 shuffle order, source-only losses, and frozen
Qwen SQLite hidden features. The only controlled difference is
which neighbor tensor is supplied **during both fit and inference**.

Unlike the old single-seed no-neighbor retraining experiment, the
three NEW arms are precisely initialization-matched *to each other*.
They are not claimed to share the original (already finished) Dev M6.5
initialization. This is one-seed paired design, not an across-seed
generalization guarantee.

City-mean neighbors are computed using physical-tile-deduplicated
unlabeled imagery within each city, from the same source CSV
transductive imagery scope as the real KNN features. No row
indicator labels are used to construct neighborhoods.

## Run

From the repository root in an already allocated GPU session:

```bash
git pull --ff-only origin main
python -m unittest discover -s tests -p 'test_k7_m65_paired_train_ablation.py' -v

python -u scripts/run_k7_m65_paired_train_ablation.py \
  --full-dir runs/k7/m62_full_seed42 \
  --dev-dir runs/k7/m65_neighborhood_dev_seed42 \
  --neighbor-cache runs/k7/m65_neighbors_k8.pt \
  --output runs/k7/m65_paired_ablation_dev_seed42 \
  --epochs 20 --batch-size 256 --device cuda
```

The script trains three 20-epoch students, or **60 model-epochs** total.
It DOES NOT retrain the DEV visual anchor and DOES NOT run new Qwen
encoding. Training is sequential on one allocated GPU.

Each arm has `checkpoint_last.pt` with its own fixed run signature.
Restart the exact same command after interruption to resume the
unfinished arm and skip completed arms. Do not launch two concurrent
instances pointing to the same output path.

After completion:

```bash
cat runs/k7/m65_paired_ablation_dev_seed42/REPORT.md
cat runs/k7/m65_paired_ablation_dev_seed42/SUMMARY.csv
```

Per-arm file layout:
`real/checkpoint_last.pt`, `real/metrics.csv`, `zero/...`,
`city_mean/...`, plus root `REPORT.md` and `SUMMARY.csv`.

All displayed metrics are 3 decimals; model training and metric
calculations are not rounded.

Interpretation of report:
- For MSE **negative** `real-minus-arm` means real neighborhoods
  performed better.
- For Pearson **positive** `real-minus-arm` means real performed
  better.
- Examine both tasks. Existing earlier frozen perturbation experiments
  cannot be used as a substitute for this paired retraining ablation.
- This remains source city development validation on a limited set of
  cities, not the independent official test.

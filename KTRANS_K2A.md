# K2-A task-conditioned scale calibration

This is a **controlled ablation** of the existing K0+K1 meta-protocol.
`--scale-head` adds a small semantic-conditioned affine output layer.
There is no use of held-out task labels or test statistics in the scale head.
Without the flag, the previous K1 model is unchanged.

## Set up and test

```bash
git pull origin main
python -m unittest discover -s tests -p 'test_ktrans_scale.py' -v
python -m py_compile run_ktrans.py urban_transfer/ktrans.py urban_transfer/ktrans_scale.py
```

## One fold first

```bash
for VARIANT in baseline scale; do
  EXTRA=()
  if [ "$VARIANT" = scale ]; then EXTRA=(--scale-head); fi
  python -u run_ktrans.py train --task-encoder remoteclip \
    --heldout-task digital_total_cell_intensity \
    --limit 5000 --batch-size 128 --epochs 30 --patience 0 \
    --meta-city-fraction 0.2 --seed 42 \
    --output "runs/k2a_${VARIANT}_digital_total_cell_intensity" "${EXTRA[@]}"
done
```

All 30 epochs are recorded (no early stopping). `best.pt` is still selected
by the average of the three meta-MSE scores, consistent with K1. These meta
partitions are development data and are not independent test splits.

## Six held-out source tasks (LOTO)

```bash
TASKS=(population_count nighttime_light_intensity education_accessibility food_accessibility digital_modernization_tmi digital_total_cell_intensity)
for TASK in "${TASKS[@]}"; do
  for VARIANT in baseline scale; do
    EXTRA=()
    if [ "$VARIANT" = scale ]; then EXTRA=(--scale-head); fi
    python -u run_ktrans.py train --task-encoder remoteclip \
      --heldout-task "$TASK" \
      --limit 5000 --batch-size 128 --epochs 30 --patience 0 \
      --meta-city-fraction 0.2 --seed 42 \
      --output "runs/k2a_${VARIANT}_${TASK}" "${EXTRA[@]}"
  done
done
```

Compare `history.jsonl` and report **all six tasks**, rather than a single fold.
All scalar metrics on the terminal use three decimals. Checkpoint selection
still uses full numerical precision.

### Important caveat

The original K0 meta-training uses a stratified 5k sample and chooses checkpoints
based on development meta-validation. This experiment is exploratory; it does
not establish performance on unseen external SCUI/UCSI/UCUI tests. A single
fixed heldout task with large mean can dominate MSE; compare per-fold results
and use additional development folds before drawing model-selection conclusions.

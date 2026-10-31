# K3-A: normalized spatial expert composition

**Controlled K2-B1 comparison.** Use the same frozen RemoteCLIP imagery, source-task semantic vectors, and fixed Semantic/Uniform/Random routing strategies. The new optional `--normalized-experts` flag changes expert targets/outputs:

- Each source expert learns its own task in **standardized units**.
- Source task moments (mean and population std of log1p target) are computed from meta-TRAIN ONLY: known tasks and known cities. Heldout tasks/cities never contribute moments.
- Seen task inference uses its own training-task moments to return predictions in log1p units.
- Unseen task inference combines experts' standardized spatial patterns, and restores scale using the **same router-weighted source means/stds**. This is an intentionally simple source-only zero-shot calibration hypothesis, not target-scale ground truth and not performance-distilled routing.

There is no new RemoteCLIP image precompute. The old K2-B1 behavior remains the default.

## Quick start

```bash
git pull origin main
python -m unittest discover -s tests -p test_ktrans_normalized.py -v
python -m py_compile run_ktrans.py urban_transfer/ktrans.py urban_transfer/ktrans_normalized.py
python -u run_ktrans.py train \
  --expert-bank --normalized-experts --task-encoder remoteclip \
  --heldout-task digital_total_cell_intensity \
  --limit 5000 --meta-city-fraction 0.2 --seed 42 \
  --batch-size 128 --epochs 30 --patience 0 \
  --output runs/k3a_digital_total_cell_intensity
```

## Diagnose same trained checkpoint with three routes

```bash
CKPT=runs/k3a_digital_total_cell_intensity/best.pt
for MODE in semantic uniform random; do
  python -u scripts/analyze_ktrans_predictions.py \
    --checkpoint "$CKPT" --routing-mode "$MODE" --routing-seed 42 \
    --output "runs/k3a_diagnostics/digital_total_cell_intensity/$MODE"
done
```

## Full six-task LOTO with 5 random seeds

```bash
python -u scripts/run_ktrans_loto.py --mode all --normalized-experts \
  --runs-root runs/ktrans_k3a_loto --limit 5000 --epochs 30 \
  --batch-size 128 --eval-batch-size 256
```

Compare with `runs/ktrans_loto/summary/summary.csv` from K2-B1; K3-A writes
`runs/ktrans_k3a_loto/summary/summary.csv`.

**Crucial caveat:** This module does NOT know the absolute target indicator
scale. Its zero-shot reconstruction interpolates source moments. Improvement
is not guaranteed, and the internal meta partitions were used for checkpoint
selection. Always report individual heldout-task R², Pearson, bias and std
alongside aggregate results.

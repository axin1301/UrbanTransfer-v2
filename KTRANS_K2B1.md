# K2-B1: fixed semantic routing over source-task spatial experts

This is the first **compositional expert-transfer** ablation for K-TransMoE,
not the full paper method. Unlike M3's anonymous MoE experts, K2-B1 trains
one regression expert per **supervised source task**. The shared spatial
trunk consumes frozen cached RemoteCLIP image features plus urban context.

For seen tasks, only the task-matched expert contributes to the supervised
prediction. For a held-out task, fixed softmax weights computed from
**frozen RemoteCLIP task-text cosine similarity** combine the source experts.
The held-out task's labels never update the trunk or any expert; the text
features are frozen. The softmax temperature is configurable.

Important limitation: task-specific experts learn *absolute log1p target
scales*, and naive mixture can retain scale bias; B1 tests whether semantic
expert composition improves spatial Pearson before introducing calibration.

## Setup

```bash
git pull origin main
python -m unittest discover -s tests -p 'test_ktrans_experts.py' -v
python -m py_compile run_ktrans.py urban_transfer/ktrans.py urban_transfer/ktrans_experts.py
```

## Single held-out task: apples-to-apples meta experiment

```bash
python -u run_ktrans.py train \
  --expert-bank --task-encoder remoteclip --router-temperature 0.15 \
  --heldout-task digital_total_cell_intensity \
  --limit 5000 --meta-city-fraction 0.2 \
  --batch-size 128 --eval-batch-size 256 \
  --epochs 30 --patience 0 --seed 42 \
  --output runs/k2b1_digital_total_cell_intensity
```

Use the same data split as K1 and K2-A. Contrast the **full epoch curves**;
K1/K2-A and K2-B1 have different model capacities, so this is a mechanism
comparison, not matched-parameter proof.

## Diagnostic predictions

```bash
python -u scripts/analyze_ktrans_predictions.py \
  --checkpoint runs/k2a_baseline_digital_total_cell_intensity/best.pt \
  --checkpoint runs/k2a_scale_digital_total_cell_intensity/best.pt \
  --checkpoint runs/k2b1_digital_total_cell_intensity/best.pt \
  --batch-size 256 --output runs/k2b1_diagnostics
```

Inspect `pearson`, `bias`, `pred_std`, `mse`, and
`spatial_gain_vs_pred_mean_constant` for meta-SCUI and meta-UCUI.
The output is from **internal development** meta partitions: best.pt is
selected using their losses, not an independent test.

## LOTO rotation

```bash
TASKS=(population_count nighttime_light_intensity education_accessibility food_accessibility digital_modernization_tmi digital_total_cell_intensity)
for TASK in "${TASKS[@]}"; do
  python -u run_ktrans.py train \
    --expert-bank --task-encoder remoteclip --router-temperature 0.15 \
    --heldout-task "$TASK" --limit 5000 --meta-city-fraction 0.2 \
    --batch-size 128 --epochs 30 --patience 0 --seed 42 \
    --output "runs/k2b1_${TASK}"
done
```

No Qwen model or uncached image encoder is used during training; existing
RemoteCLIP image and text SQLite/PT cache is reused. Fixed routing is
deterministic and does not use expert-validation errors or target labels.
This is **not** PHATGOOSE or a Hyper-X implementation yet.

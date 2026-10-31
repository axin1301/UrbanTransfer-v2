# K-TransMoE prototype K0 + K1

This is the **first, deliberately limited prototype** for
**K-TransMoE: Knowledge-Guided Compositional Expert Transfer for Generalizable Urban Prediction**.
It implements meta-validation + semantic task embeddings only, NOT yet
hypernetworks, post-hoc expert banks, LLM knowledge prompting, reward-distilled routing
or task-aware scale calibration.

## What it does

- Reuses the M3 **frozen RemoteCLIP image embedding SQLite store**; no PNG decoding in training.
- K1 compares `--task-encoder hash` (M3-style nonsemantic reference) vs
  `--task-encoder remoteclip` (frozen pretrained RemoteCLIP text encoder).
- Text features are computed once for the ten permitted indicator descriptions
  and cached in `runs/ktrans_text_cache/task_text_<hash>.pt`.
- K0 takes a **deterministically stratified** subset of the original **training CSV only**.
  With `--heldout-task`, remove *every* label of that source task from optimization.
  With `--meta-city-fraction`, hold out entire source cities.
- On each epoch, report `meta_SCUI` (heldout task, training cities),
  `meta_UCSI` (trained tasks, heldout cities),
  `meta_UCUI` (heldout task, heldout cities).
- Select `best.pt` on averaged **meta-development** MSE.
  **Important:** this reuses heldout-task labels for *model selection*.
  These are development/validation numbers, not unbiased final scores.
- Does **not** use the externally supplied SCUI/UCSI/UCUI test CSVs.

## Setup
```bash
git pull origin main
python -m unittest discover -s tests -p test_ktrans.py -v
```

The train CSV location and image root are already defaulted in the CLI.
RemoteCLIP visual features must exist in `runs/m3_features`.

## First quick experiment: fixed pseudo-heldout source task

Run a directly comparable pair with the same seed, stratum-balanced sample,
heldout task, and heldout source cities:

```bash
for ENCODER in hash remoteclip; do
  python -u run_ktrans.py train \
    --task-encoder "$ENCODER" \
    --heldout-task digital_total_cell_intensity \
    --limit 5000 --meta-city-fraction 0.2 \
    --batch-size 128 --epochs 30 --patience 6 --seed 42 \
    --output "runs/ktrans_k1_${ENCODER}"
done
```

`remoteclip` task encoding loads the pretrained model once the first time;
subsequent runs reuse the text feature cache. The vision model is never
rerun on images.

Show the best epoch of each run:

```bash
python - <<'PY'
import json
from pathlib import Path
for encoder in ['hash','remoteclip']:
    p=Path(f'runs/ktrans_k1_{encoder}/history.jsonl')
    rows=[json.loads(s) for s in p.read_text().splitlines() if s.strip()]
    best=min(rows,key=lambda x:x['best_meta_mse'])
    print(f"{encoder:12} epoch={best['epoch']:2} meta={best['best_meta_mse']:.3f} "
          + ' '.join(f"{s}={best['meta_validation'][s]['overall']['mse']:.3f}"
                   for s in ['meta_SCUI','meta_UCSI','meta_UCUI']))
PY
```

## Rotation

Because there are only six training tasks, repeat with different `--heldout-task`
values instead of drawing claims from one holdout. At the final research stage,
ensure model choices are frozen before running external protocol tests.

## Limitations

This prototype retains the M3 MoE experts and a single regression head,
does not yet implement the full compositional framework, and uses only
existing urban task definitions, not LLM-generated world knowledge. The
semantic text encoder itself is **frozen RemoteCLIP**, not Qwen3-VL.
Both the hidden-state store and original M2/M3 scripts remain unchanged.
For repeated training at 128k rows, an in-memory feature dataset is a later
performance optimization.

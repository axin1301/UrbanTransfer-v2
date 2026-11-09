# M6.5 zero-neighborhood ablation (source-held-city only)

## What is controlled

One **additional 20-epoch training run** tests whether physically meaningful
per-tile neighborhood information is useful, rather than just adding
network capacity. The new control uses:

- exactly the same `NeighborhoodCityIndicatorResidual` architecture
  and number of trainable parameters as M6.5 Dev;
- exactly the same source-city split, frozen `dev_visual_anchor.pt`,
  Population/Food-only source stats, loss formula, AdamW settings,
  20 epochs, seed42 permutation, batch256, and cached Qwen-VL tokens;
- **all-zero [N,3*D+2] neighborhood tensor** in both training and evaluation,
  so no image-neighbor signal is available. Neighborhood encoder parameters
  remain trainable: this isolates *information content* instead of capacity.

This is distinct from passing zeros only at inference, which would
produce a distribution shift in an already trained model.

The source-held-city validation contains ONLY Population and Food,
not the original held-city UCSI or Education. No SCUI/UCUI/Education
label is accessed by this ablation.

**Initialization caveat:** the original M6.5 Dev model was initialized
after first training a new visual anchor, consuming additional RNG
state, while the control starts with explicit seed42 initialization.
The fits have matched training settings/architecture but **not exactly
identical initial trainable weights**; differences are single-seed
associations, not definitive causal attribution. An exact
initialization-matched ablation would require rerunning both variants
from a new common initialization.

## Run

```bash
git pull --ff-only origin main
python -m unittest discover -s tests \
  -p 'test_k7_m65_zero_neighbor_ablation.py' -v

python -u scripts/run_k7_m65_zero_neighbor_ablation.py \
  --full-dir runs/k7/m62_full_seed42 \
  --m65-dev-dir runs/k7/m65_neighborhood_dev_seed42 \
  --output runs/k7/m65_ablation_zero_neighborhood_dev_seed42 \
  --epochs 20 --batch-size 256 --device cuda
```

No visual anchor retraining. No Qwen-VL encoding. No KNN feature
construction needed; zero neighborhood tensor is built in CPU RAM.
Training still runs 20 source-only epochs and reads cached tokens.

At each completed epoch an atomic `checkpoint_last.pt` is saved.
Restart the **same** command after preemption to resume. Never reuse
the original M6.5 Dev or Final output directory.

Outputs:
- `metrics.csv`: zero-neighborhood control held-city MSE/Pearson/bias
- `comparison.csv`: original real-neighborhood Dev vs control
  (real minus zero) on exactly the same source-held-city tasks
- `config.json`: source city split, anchor and method identity

Original real-neighborhood source Dev benchmark:

| Task | MSE | Pearson |
|---|---:|---:|
| Population | 3.2723 | 0.7307 |
| Food | 1.0283 | 0.6159 |

For MSE, a **negative** (real minus zero) indicates the true
neighborhood model is better. For Pearson, **positive** indicates
it is better. Assess both source tasks and do not tune the
ablation using UCUI. The final M6.5 UCUI results remain
development-only and were not used to fit this control.

No official test files are opened.

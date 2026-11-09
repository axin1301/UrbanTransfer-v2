# M6.5 frozen Dev neighborhood intervention (no retraining)

The prior matched-capacity zero-input **retrained** control on held-source
cities did not show consistent MSE benefit from real neighborhood features.
This small follow-up is a **frozen-model sensitivity analysis** to see
whether a fitted M6.5 model uses the correctly paired spatial context.

Three predetermined input conditions evaluated with the SAME
`runs/k7/m65_neighborhood_dev_seed42/m65.pt`:

1. `real`: original true K=8 same-city tile-neighborhood.
2. `zero`: zeros with the same [N,3*D+2] shape.
3. `city_permuted`: deterministic within-city derangement of entire
   neighbor vectors among distinct physical tiles. Population and Food
   rows of the same tile receive the same reassigned neighbor vector.
   The city-level neighborhood distribution is preserved, while the
   local tile-neighbor correspondence is broken.

There is **no network training**, no checkpoint selection, no
Education targets, and no official test file access. Only
`dev_held_city_population_count` and
`dev_held_city_food_accessibility` are evaluated. Before
interventions, `real` predictions must reproduce the saved original
Dev metrics (tolerance 1e-4).

Run from repo root in an allocated GPU job:

```bash
git pull --ff-only origin main
python -m unittest discover -s tests \
  -p 'test_k7_m65_dev_intervention.py' -v

python -u scripts/run_k7_m65_dev_intervention.py \
  --full-dir runs/k7/m62_full_seed42 \
  --dev-dir runs/k7/m65_neighborhood_dev_seed42 \
  --neighbor-cache runs/k7/m65_neighbors_k8.pt \
  --output runs/k7/m65_dev_neighborhood_intervention \
  --batch-size 256 --device cuda
```

Outputs:
- `metrics.csv`: six rows (2 source tasks × 3 conditions) including
  MSE, Pearson, bias, prediction standard deviation, expert weight.
- `comparison.csv`: real-minus-perturbed metrics.
- `config.json`: full protocol and provenance.

How to read:
- In `comparison.csv`, for MSE, **negative real-minus-perturbed**
  means true neighbors are better. For Pearson, **positive** means better.
- If the `city_permuted` input substantially worsens accuracy, this is
  *consistent with* useful local spatial correspondence, but is not
  conclusive because permutation changes joint input statistics.
- If accuracy is unchanged, model may largely ignore neighborhood
  details; if perturbed improves accuracy, neighbor context may be
  counterproductive on that held-city split.
- The zero condition is particularly out-of-distribution: do not
  interpret performance change as an unbiased causal effect.
- Compare with the previous **retrained** no-neighbor ablation before
  making any strong mechanism claim.

This is a fast frozen inference experiment, not another 20-epoch fit.
Do not run on the final UCUI Education labels, which have already been
inspected repeatedly.

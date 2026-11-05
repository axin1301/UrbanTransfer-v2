# K7 fixed-protocol cross-city and cross-indicator comparison

## Scope and experimental integrity

This is the next **verification** step after the frozen M6.2
mechanistic diagnostics, not an M6.3 architecture search.

Use the **same source-only** M6.1 manifest and City×Indicator fold:
- 128 source-labelled training rows (Population + Food, seen cities).
- 64 SCUI: Education in seen cities.
- 64 UCUI: Education in held cities.
- **64 NEW UCSI**: Population/Food rows in the *same* held cities,
  sampled deterministically from the **source** `train.csv` by
  held-city × source-indicator round robin, seed42.
- There is **no use** of `seen_city_test.csv`,
  `unseen_city_seen_indicator_test.csv` or `unseen_city_test.csv`.
- UCSI labels and Education pseudo-dev labels are read for **scoring
  after fitting only**, never as model-training, scaling, checkpoint
  selection, or hyperparameter selection inputs.

For UCSI, we **cannot reuse** the M6.1 512-entry Qwen hidden-state
cache: that cache does not contain held-city Population/Food images.
The new UCSI cache script computes only the 64 additional
Qwen2.5-VL image+indicator hidden states, in a **new SQLite DB**;
M6.1's original `hidden.sqlite3` remains untouched.
The new cache has deterministic manifest / sqlite resume semantics.

Compare three models:
1. `visual_only`: RemoteCLIP tile vector + 4 geo context features,
   with two source experts/router; NO task embedding or Qwen-VL.
2. `text_router`: same frozen RemoteCLIP/geo, plus frozen Qwen3
   0.6B task embedding conditioning the two expert router + gate.
   Does NOT use Qwen2.5-VL image-conditioned hidden states.
3. `m62`: **exact frozen existing checkpoint**, task-query
   cross-attention over cached Qwen2.5-VL hidden tokens.
   **Not retrained**.

Both new baseline fits use **the same original 128 source rows**,
seed42, epochs20, batch128, hidden128, AdamW lr.001,
matching-source expert auxiliary MSE weighted .3, and the same
original M6.2 checkpoint's TRAIN-ONLY source standardization.
Education mean/std are the unchanged source-text similarity
estimates from that same checkpoint. No target labels used
to fit scalers.

**Interpretation caution:** The two controls share training data,
source scaling and core two-expert supervised losses, but M6.2 also
had source-task separation and unlabeled Education-image alignment
losses, so this is **not** a fully parameter-/objective-matched
single-factor ablation. Differences are indicative, not causal proof
of the MLLM component. M6.2 SCUI/UCUI pseudo-dev labels were already
inspected during architecture development. Do not claim those
results are independent holdout estimates, even though labels
were never used for gradients.

## Commands (ROCm MI250)

First pull and run CPU tests:

```bash
git pull --ff-only origin main
python -m unittest discover -s tests \
  -p 'test_k7_m62_fixed_comparison.py' -v
```

**Step A — cache UCSI only.** Runs frozen Qwen2.5-VL on the new
held-city Population/Food images (about 64 new image-task forwards).
For reproducibility, existing M6.1 cache must be complete:

```bash
python -u scripts/cache_k7_m62_ucsi.py \
  --cache-dir runs/k7/m61_structured_seed42 \
  --run-dir runs/k7/m62_fixed_comparison_seed42 \
  --limit 64 --seed 42
```

Writes
`runs/k7/m62_fixed_comparison_seed42/ucsi_manifest.json` and
`ucsi_hidden.sqlite3`. Supports safe resume after interruption.

**Step B — two baseline fits + unchanged M6.2 evaluation**:

```bash
python -u scripts/run_k7_m62_fixed_comparison.py \
  --cache-dir runs/k7/m61_structured_seed42 \
  --m62-dir runs/k7/m62_task_query_seed42 \
  --comparison-dir runs/k7/m62_fixed_comparison_seed42 \
  --text-cache-dir runs/ktrans_text \
  --output runs/k7/m62_fixed_baselines_seed42 \
  --seed 42 --epochs 20 --batch-size 128 \
  --device cuda
```

Two new baseline checkpoints only; original M6.2 checkpoint remains
unchanged. The output folder must **not exist** at launch. Reports:

- `runs/k7/m62_fixed_baselines_seed42/visual_only.pt`
- `runs/k7/m62_fixed_baselines_seed42/text_router.pt`
- `runs/k7/m62_fixed_baselines_seed42/metrics.csv`
- `runs/k7/m62_fixed_baselines_seed42/config.json`

The script reports **all three splits** SCUI, UCSI and UCUI
in exactly the same units: MSE and MAE in log1p, Pearson, and MSE
ratio to source-only constant mean baseline. For UCSI this ratio
uses the matching **source task's** train-only mean, not the
Education target mean. This is essential because UCSI contains
two known source tasks with different label scales.

This initial fixed protocol uses only a small 128 source train
sample; it should not be compared numerically to large M1–M5
experiments with different source training volumes.

## Analysis guidelines

- If M6.2 improves UCUI over both controls, with comparable
  source/held-city UCSI, it offers **supporting evidence** for
  additional benefit from Qwen-VL image/indicator-conditioned hidden
  states, though not proof of a single mechanism.
- If text_router performs similarly on UCUI, much of M6.2's
  advantage may be due to the indicator-conditioned routing, rather
  than deeper image+text MLLM fusion.
- If UCSI performance fails badly while UCUI looks good, the
  city-generalization story is incomplete: report both source tasks
  separately and examine scale/city effects before changing the
  architecture.
- One seed and 64 pseudo-dev observations per split cannot establish
  robust rank ordering. Future work should use pre-fixed independent
  source validation cities, multiple seeds, and larger source
  training volume. Avoid selecting models from repeated inspection
  of Education pseudo-dev metrics.

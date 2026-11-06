# K7 M6.3 — City×Indicator Residual Transfer (FULL source)

## Motivation and fixed protocol

M6.2 FULL source development results:

| Partition | Visual-only MSE | Text Router MSE | M6.2 MSE |
| --- | ---: | ---: | ---: |
| source_val | 2.3614 | 0.9123 | 0.8506 |
| SCUI | 1.0757 | 0.7567 | 0.6761 |
| UCSI | 2.4417 | 1.6689 | 1.6662 |
| UCUI | **1.7026** | 2.0881 | 2.0142 |

Task definition clearly conditions M6.2, but the target
Education+held-city UCUI split favored the task-independent
Visual-only student. M6.3 is one *prespecified architectural
hypothesis*, not a claimed UCUI improvement.

**Reuse the EXACT full-scale M6.2 fixed manifest and frozen
Qwen2.5-VL SQLite cache.** New architecture gets no new Education
labels, final test inputs, target fitting, target scale calibration,
or MLLM inference.

## Architecture

1. **Visual-only anchor** — load
   `runs/k7/m62_full_models_seed42/visual_only.pt`, verify it
   matches the same full manifest and seed42, and freeze ALL
   its weights. This exactly preserves the original RemoteCLIP
   + 4 geographic features + 2 source experts.
2. **City/spatial encoder** — learn an image-local representation
   of the same cached RemoteCLIP satellite embedding and 4
   available geography features. No city ID memorization and
   no additional external city statistics are assumed. It
   represents **local urban spatial context**, not measured
   multi-tile city-wide morphology.
3. **Indicator encoder** — extract the target Task Query from
   actual frozen Qwen-VL Definition [16] and Relation [16]
   last-layer states.
4. **City × Indicator query** — form a query from the spatial
   representation, task query, and their product.
5. **Centered visual attention** — project the 32 actual
   frozen Qwen-VL visual tokens, center by their per-image
   token mean, cross-attend with the city×indicator query,
   and explicitly retain the uncentered global visual
   component through a separate projection.
6. **Bounded residual experts** — predict a learned adjustment
   to the frozen visual baseline through 2 residual experts,
   image/task-conditioned routing and gate. Residual magnitude
   is provably bounded to **0.5 standardized source-task
   output units**: `y_std = y_visual_std + Δ`, `|Δ|<=0.5`.
   The residual scale is initialized exactly to zero:
   initial M6.3 predictions are bitwise equal to Visual-only.

This isn't guaranteed to preserve the anchor's UCUI quality
after optimization: the residual is source-trained and its
effect on an unseen target is still unknown. It is NOT an
oracle that chooses the better model at inference.

## Training

One run with same fixed full manifest and source rows, seed42,
20 epochs, batch128, hidden128, heads4, AdamW lr.001.
The Visual-only anchor is FROZEN, and only City×Indicator,
attention, source-task classification, source-image-alignment
and residual parameters are optimized.

- Pop/Food source main regression MSE.
- matching source residual-expert loss, weighted 0.3.
- source task identity CE, weighted 0.2.
- Education-description-on-**source-image** positive
  RemoteCLIP alignment, weighted 0.05, no Education labels.
- bounded residual squared penalty, weighted 0.02.

Education mean/std are **copied unchanged** from the original
Visual-only full checkpoint, derived from Population/Food training
stats and frozen Qwen3 text similarity. No UCUI-dependent calibration.
No Education labels are used for gradients/checkpoint/scale.
The Source validation split is scored AFTER fixed 20 epochs.
SCUI, UCSI and UCUI are also scored only post-fit.
Neither of the official `*test.csv` files is opened.

**Methodological caveat:** The visual backbone is pretrained and
frozen, so M6.3 uses two stages of source-only training. Compare
its extra training cost and capacity accordingly. It is not a
parameter/loss-matched ablation of M6.2.

## Commands

On MI250 with cloned GitHub repo:

```bash
git pull --ff-only origin main

python -m unittest discover -s tests \
  -p 'test_k7_m63_city_indicator.py' -v
```

**Train + evaluate directly — no separate audit, no Qwen inference:**

```bash
python -u scripts/run_k7_m63_full.py \
  --full-dir runs/k7/m62_full_seed42 \
  --reference-dir runs/k7/m62_full_models_seed42 \
  --output runs/k7/m63_city_indicator_full_seed42 \
  --seed 42 --epochs 20 --batch-size 128 \
  --device cuda
```

The script requires existing M6.2 full cache/manifest, full
Visual-only checkpoint and source-only shared tensor features.
It refuses overwriting the output directory and checks anchor
checkpoint/manifest identity.

Outputs:

- `runs/k7/m63_city_indicator_full_seed42/m63.pt`
- `runs/k7/m63_city_indicator_full_seed42/metrics.csv`
- `runs/k7/m63_city_indicator_full_seed42/config.json`

Prints MSE(log1p), Pearson, Ratio, average transfer magnitude
and Visual-only anchor MSE in source_val, SCUI, UCSI, UCUI
and both source-indicator-specific UCSI partitions.

## Results interpretation

Compare M6.3 UCUI against 1.7026 (Visual-only), 2.0881
(Text Router), and 2.0142 (M6.2), but also compare Pearson
(Visual-only 0.6281, Text Router 0.5873, M6.2 0.5914).
The decisive question is whether adding city×indicator-conditioned
MLLM residuals *improves UCUI beyond its own visual anchor*
without harming Population/Food transfer. If not, the new
structural hypothesis has failed in this development setting.

The development Education labels have been repeatedly examined,
so any architecture improvements suggested by these numbers
are development findings only — not independent confirmatory
results. No official tests should be accessed until the final
method and comparisons are locked.

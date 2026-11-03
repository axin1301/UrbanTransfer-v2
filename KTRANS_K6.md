# K6: Unified K-TransMoE Development Pipeline

## Goal and scientific boundary

K6 is a **fixed composition and evaluation framework**, not a claim that
every historical component independently improves MSE. It combines:

- Frozen K4 Qwen3-8B Structured Knowledge + five normalized source experts;
- K5-A **source-only nested-LOTO transfer prior**;
- K5-B2 region × task dynamic expert composition and stable low-rank adapter,
  optimized on outer source-task labels only, with own-expert masked;
- K5-C source-only ridge scale and K5-D explicit measurement-aware scale.

The **predeclared full K6** uses a source-only transfer prior as a logit
correction to K5-B2 dynamic weights:

```
alpha_K6(r,t) = normalize(alpha_region(r,t) * alpha_transfer(t)
                           / alpha_semantic(t))
```

and source-only K5-D measurement-aware task-scale estimates:

```
prediction = mu_K5D(t) + std_K5D(t) * sum_j alpha_K6(r,t,j)
                                     * expert_j(adapted_hidden(r,t))
```

No target label moments, target validation feedback or task-wise
post-hoc variant selection in either formula. This is a **fixed, testable
design choice**, not tuned to a particular outer task.

All split reconstruction and checkpoint selection caveats still apply:
underlying K4 `best.pt` was selected on meta-validation labels. These
are exploratory DEV results; unbiased benchmark claims require
target-label-blind final evaluation. Also, K5-B2's experts were trained
on source labels, so its leave-own-expert-out loss is a source-supervised
surrogate rather than strict independently trained nested experts.

## Per-fold zero-training unified evaluation

The existing Digital Total Cell checkpoints already provide every
component:

```bash
git pull origin main
python -m py_compile urban_transfer/ktrans_k6.py \
  scripts/run_ktrans_k6.py scripts/run_ktrans_k6_suite.py
python -m unittest discover -s tests -p 'test_ktrans_k6.py' -v

python -u scripts/run_ktrans_k6.py \
  --composer-checkpoint runs/k5b2_digital_total_cell/composer.pt \
  --k5a-protocol runs/k5a_digital_total_cell/protocol.json \
  --output runs/k6_digital_total_cell \
  --eval-batch-size 1024
```

The ten **fixed variants** are:

| ID | Spatial route | Adapter | Scale |
| --- | --- | --- | --- |
| k4 | semantic | off | semantic |
| k5a | transfer only | off | semantic |
| router_only | region dynamic | off | semantic |
| adapter_only | semantic | on | semantic |
| k5b2 | region dynamic | on | semantic |
| k5c | region dynamic | on | learned K5-C |
| k5d | region dynamic | on | measurement K5-D |
| k6_spatial | region dynamic + K5-A transfer | on | semantic |
| k6_scale_only | K5-A transfer | off | K5-D |
| k6_full | region dynamic + K5-A transfer | on | K5-D |

All protocols `meta_SCUI`, `meta_UCSI` and `meta_UCUI` are evaluated
with MSE, RMSE, MAE, R², Pearson and Bias. For **seen-source** Meta-UCSI,
inference explicitly uses the frozen task-specific K4 expert (same for
all variants); this avoids misrepresenting an unsupervised task
transfer extension as an improvement to supervised seen-task inference.

`evaluation.csv` has 30 rows (10 variants × 3 protocols). Print only
selected rows to avoid terminal flooding:

```bash
python - <<'PY'
import pandas as pd
d=pd.read_csv('runs/k6_digital_total_cell/evaluation.csv')
d=d[d.variant.isin(['k4','k5a','k6_spatial','k6_scale_only','k6_full'])]
print(d[['variant','protocol','mse','r2','pearson','bias']].round(3).to_string(index=False))
PY
```

## Multi-fold orchestration

Create a JSON manifest with one entry per outer heldout source task.
Example **single fold**:

```json
{
  "digital_total_cell_intensity": {
    "k4_checkpoint": "runs/k4_t2_structured_qwen8b/best.pt",
    "k5a_dir": "runs/k5a_digital_total_cell",
    "k5b2_dir": "runs/k5b2_digital_total_cell"
  }
}
```

Save it as `runs/k6_manifest.json`, then run:

```bash
python -u scripts/run_ktrans_k6_suite.py \
  --manifest runs/k6_manifest.json \
  --output runs/k6_suite
```

For more folds, add entries to the manifest with the corresponding K4
Structured checkpoint and output directories. Missing K5-A or K5-B2
artifacts can optionally be trained with `--train-missing`; this may
execute **five inner K5-A model fits plus one K5-B2 fit per outer task**,
so do not use it without planning the compute budget.

For a full six-fold study, populate **all six source tasks** and pass
`--require-all-six`. The runner fails on missing checkpoints rather than
quietly dropping tasks and writes `aggregate.csv` and per-fold
`evaluation.csv`.

## Evaluation/reporting principles

- Do not select a different variant for each heldout task based on
  meta-target validation labels.
- Show the complete task-level metric table, including crowd-sourced
  cellular indicators, with clear source-reliability caveats.
- Macro averages of raw MSE mix scales and should be supplemented
  with task-normalized MSE, R², Pearson and per-task breakdown.
- These meta partitions are development splits, not official unseen
  target-indicator test partitions.
- Always audit `protocol.json` and K5-A `protocol.json` to verify
  matched checkpoints, source expert ordering, and no target statistics.

## Focused factorial interpretation

The two additional inference-only variants isolate previously entangled
components without re-training. Compare `k5a` vs `k6_spatial` to measure
the region router + stable adapter under *the same frozen semantic scale*.
Compare `k5a` vs `k6_scale_only` to measure K5-D scale changes with
*identical K5-A spatial predictions*. Compare `k6_spatial` vs `k6_full`
to measure scale effects on the combined spatial field. Because all scales
are positive task-level affine transforms, paired Pearson values should be
identical to reporting precision. K6's extra changes are inference-only
ablation of jointly trained components, not isolated retrained models.
Do not select a per-task winner using outer validation labels.

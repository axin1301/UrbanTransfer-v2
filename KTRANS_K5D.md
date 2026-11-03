# K5-D: Measurement-Aware Knowledge Calibration

Purpose: test whether **audited indicator measurement definitions**, rather
than task text embedding cosine alone, improve zero-shot **scale inference**.
This is not a claim that all indicators must improve.

The source-only task ontology is in `urban_transfer/measurement_knowledge.py`:
two separate attributes encode (1) urban phenomenon domain and (2) measurement
mechanism. The descriptors come from task descriptions, NEVER from evaluation
labels or target moments. In particular `digital_modernization_tmi` is marked
as a modernization index of *uncertain methodology*, not tower counts.

For source-only leave-one-task-out pseudo target t, compute:
- semantic mixture of other source-task moments,
- measurement-aware mixture, with modest fixed additive compatibility logit
  bonuses 0.7 for the same domain and 0.5 for the same mechanism.

Fit a **single scalar** blend coefficient in [0,1] by regularized source-only
LOTO scale prediction error. Each held-out source pseudo task is removed
from the input stats but used as the supervised response (allowed source label
moment). The *outer* unseen indicator's labels or moments are NEVER used to
train or select this blend.

K5-B2 spatial outputs are frozen. Evaluate 3 variants on the same predictions:
`k5b2_semantic_scale`, `k5c_learned_scale`,
`k5d_measurement_scale`. Positive std keeps Pearson identical.

## Focused pilot

```bash
git pull origin main
python -m py_compile urban_transfer/measurement_knowledge.py \
  urban_transfer/ktrans_k5d.py scripts/run_ktrans_k5c.py
python -m unittest discover -s tests -p 'test_ktrans_k5d.py' -v

python -u scripts/run_ktrans_k5c.py \
  --composer-checkpoint runs/k5b2_digital_total_cell/composer.pt \
  --measurement-aware --output runs/k5d_digital_total_cell \
  --eval-batch-size 1024
```

Compact terminal results:
```bash
python - <<'PY'
import pandas as pd
d=pd.read_csv('runs/k5d_digital_total_cell/evaluation.csv')
print(d[['variant','protocol','mse','r2','pearson','bias']].round(3).to_string(index=False))
PY
```

## Limitations and research integrity

- Only five source tasks are available in each outer fold, so the scale
  generalization estimate has very high uncertainty.
- The ontology is **hand-audited method knowledge**, not a pretrained
  classifier or evidence that exact measurement methods are verified for
  every indicator; specific telecom methodology remains uncertain.
- Fixed bonuses and LOTO shrinkage are declared design choices, not
  tuned against outer meta validation or final target tests.
- Outer K4 `best.pt` was already selected by meta labels: all analyses are
  **development experiments**. Never use true target test labels or moments
  to choose the best scale variant task by task.
- Model-level gains are judged with prespecified metrics and full
  task coverage; report regressions as well as improvements.

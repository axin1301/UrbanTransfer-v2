# K5-C: Independent Knowledge-Guided Scale Router

Prototype that adds the missing **independent task-scale estimation** while
keeping the trained K5-B2 dynamic spatial router, frozen experts and stable
adapter entirely unchanged.

Source-only leave-one-task-out supervision learns a shrinkage ridge
correction to semantic source-moment mixtures. For each source task t,
the calibrator's input excludes t's own means/stds, and only the other
four source tasks provide text similarities and statistics. The target
t moments are the supervised output. This produces five task examples:
the learner is intentionally low capacity and may overfit.

At deployment, compute source-text similarities for the truly unseen task,
then predict its mean/std from **five source tasks**. No target labels,
observed target moments, test scores or external test CSVs enter fitting.
Compare unchanged spatial predictions under frozen semantic scale versus
new independent scale estimates.

Note: In K5-C, the scale router is **task-conditioned**, not region-specific.
It is NOT the final cross-region independent-scale system; the sample size
of five tasks is a fundamental limit. Source stats are from outer Meta-Train.
The outer K4 best.pt was selected via development meta metrics, so all
results remain exploratory, not independent test results.

## First focused run

```bash
git pull origin main
python -m py_compile urban_transfer/ktrans_k5c.py scripts/run_ktrans_k5c.py
python -m unittest discover -s tests -p 'test_ktrans_k5c.py' -v

python -u scripts/run_ktrans_k5c.py \
  --composer-checkpoint runs/k5b2_digital_total_cell/composer.pt \
  --output runs/k5c_digital_total_cell \
  --eval-batch-size 1024
```

Print four lines:

```bash
python - <<'PY'
import pandas as pd
x=pd.read_csv('runs/k5c_digital_total_cell/evaluation.csv')
print(x[['variant','protocol','mse','r2','pearson','bias']].round(3).to_string(index=False))
PY
```

The spatial predictions are identical between scale variants; Pearson
should agree to 0.001 because both scale standard deviations are positive.
A difference indicates a bug. The baseline must match the earlier K5-B2
checkpoint metrics before drawing conclusions.

Potential next experiments: validate on education and nighttime light
without changing ridge regularization or using outer meta-labels to select
a particular task-specific scale variant. Report negative findings too.

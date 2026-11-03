# K6.1-A: Transferability-Anchored Regional Gate

## Hypothesis

K5-A (source-only nested pseudo-task transfer supervision) gives a strong
global prior. K5-B2's Region × Task Router can sometimes improve a target
but may hurt other tasks. Instead of forcing the original K6 multiplicative
composition, K6.1-A adds a **conservative, source-only-trained gate**:

```
route(region, target) = (1 - g) * k5a_transfer(target)
                        + g * k5b2_regional(region, target)
g = sigmoid(Linear(task_signature, geographic_context,
                   route_disagreement, regional_entropy))
```

The gate is a single linear layer with zero weights and bias -2
(initial average g=0.119). It is optimized by standardized source-only
pseudo-task MSE plus 0.010 mean(g²) regularization. No outer heldout task
labels or meta-dev metrics enter its optimizer or checkpoint selection.

For each source pseudo-target t during gate training:
- K5-A's **fixed ridge coefficients** generate the global transfer
  route to the **other four** source experts only.
- K5-B2's **frozen** region router excludes expert t as well.
- K5-B2 Adapter is **OFF**, so we isolate dynamic routing.
- The target standardized response uses only source task t's training
  examples and source-task training-only normalization moments.

For a truly unseen heldout task at inference:
- K5-A's saved transfer weights are the global anchor to all five
  available source experts.
- The trained gate conditionally interpolates with K5-B2 region weights.
- Scale is **always** frozen K4 semantic mean and std, so this stage
  measures spatial routing without scale interference.

Training the gate on the outer source data is NOT strict nested
cross-fitting because the frozen experts were already source-supervised.
Prior K4 best.pt also came from meta checkpoint selection: these are
development-only diagnostics, not independent final test results.

## Single fold on previously generated checkpoints

```bash
git pull origin main
python -m py_compile urban_transfer/ktrans_k61.py scripts/run_ktrans_k61a.py
python -m unittest discover -s tests -p 'test_ktrans_k61.py' -v

python -u scripts/run_ktrans_k61a.py \
  --composer-checkpoint runs/k5b2_education/composer.pt \
  --k5a-protocol runs/k5a_education/protocol.json \
  --output runs/k61a_education \
  --epochs 12 --batch-size 512 --eval-batch-size 1024
```

Repeat unchanged for the other two already trained folds:

```bash
python -u scripts/run_ktrans_k61a.py \
  --composer-checkpoint runs/k5b2_nighttime/composer.pt \
  --k5a-protocol runs/k5a_nighttime/protocol.json \
  --output runs/k61a_nighttime \
  --epochs 12 --batch-size 512 --eval-batch-size 1024

python -u scripts/run_ktrans_k61a.py \
  --composer-checkpoint runs/k5b2_digital_total_cell/composer.pt \
  --k5a-protocol runs/k5a_digital_total_cell/protocol.json \
  --output runs/k61a_digital_total_cell \
  --epochs 12 --batch-size 512 --eval-batch-size 1024
```

Each run produces source-only `gate_last.pt` plus
`evaluation.csv` (3 model variants × 2 meta protocols).
Gate percentiles are printed to spot collapse to zero or one.

The matched inference-only variants are:
- `k5a`: K5-A transfer prior, no Adapter, semantic scale
- `regional_only`: K5-B2 region router, no Adapter, semantic scale
- `k61a_gate`: learned anchored mixture, no Adapter, semantic scale

The next stage, K6.1-B, may test the stable Adapter **only if this
gate has demonstrated a useful nontrivial behavior** across tasks.
A near-zero gate with K5-A-like performance is a meaningful negative
result, not proof of adaptive region specialization.

## Three-fold summary

```bash
python - <<'PY'
import pandas as pd
from pathlib import Path
names={'Education':'runs/k61a_education/evaluation.csv',
       'Nightlight':'runs/k61a_nighttime/evaluation.csv',
       'Digital Cell':'runs/k61a_digital_total_cell/evaluation.csv'}
rows=[]
for task,path in names.items():
    d=pd.read_csv(path)
    d=d[d.protocol.eq('meta_UCUI')].copy()
    d.insert(0,'task',task);rows.append(d)
print(pd.concat(rows)[['task','variant','mse','r2','pearson','bias']]
      .round(3).to_string(index=False))
PY
```

All three folds are development results; report every task and do not tune
gate penalty or initial bias based on these outer heldout evaluations.

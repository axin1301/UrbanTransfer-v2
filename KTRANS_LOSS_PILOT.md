# K6.1-L: Transfer-Oriented Auxiliary Loss Pilot

**Goal:** measure whether spatially meaningful supervision improves
zero-shot task/city transfer without changing model architecture, K5-A
expert transfer prior, K5-B2 frozen regional router or semantic scale.
This intentionally trains **only** the K6.1-A confidence gate. It does
not retrain the frozen expert bank; full-model gains are not implied.

The four predeclared modes are:

| Mode | Training loss |
| --- | --- |
| `mse` | standardized MSE + existing gate² penalty |
| `corr` | MSE + 0.200 × mean task-grouped (1−Pearson) + gate penalty |
| `rank` | MSE + 0.100 × same-task/same-city pairwise softplus ranking + gate penalty |
| `combined` | MSE + 0.200 × Pearson + 0.100 × ranking + gate penalty |

No target task/meta-test labels enter any objective. Pearson is
calculated separately per task for sufficiently populated source-only
mini-batch groups (at least eight examples), skipping constant-target
groups. Pairwise ranking uses within-task, within-city low/high
label pairs, requiring a standardized target difference above 0.200.
This avoids comparing different indicators' units or unrelated cities.
When groups are too sparse the auxiliary loss contributes zero and
training falls back to MSE, which is reported as a limitation.

The weights are pilot values fixed **before** inspecting new
development labels; do not optimize them task-by-task on outer meta
SCUI/UCUI. The existing K6.1-A source pseudo-target training still
uses a bank previously trained on source labels. This is **not** strict
nested source crossfitting; K6.2-A and K6.2-B address that.

## Run a controlled Education experiment

```bash
git pull origin main
python -m py_compile urban_transfer/ktrans_losses.py scripts/run_ktrans_k61a.py
python -m unittest discover -s tests -p 'test_ktrans_losses.py' -v
python -m unittest discover -s tests -p 'test_ktrans_k61.py' -v

for mode in mse corr rank combined; do
  python -u scripts/run_ktrans_k61a.py \
    --composer-checkpoint runs/k5b2_education/composer.pt \
    --k5a-protocol runs/k5a_education/protocol.json \
    --output "runs/k61loss_education_${mode}" \
    --loss-mode "${mode}" --corr-weight 0.2 --rank-weight 0.1 \
    --epochs 12 --batch-size 512 --eval-batch-size 1024 \
    --seed 42
done
```

Print Meta-UCUI paired comparison:

```bash
python - <<'PY'
import pandas as pd
frames=[]
for mode in ('mse','corr','rank','combined'):
    d=pd.read_csv(f'runs/k61loss_education_{mode}/evaluation.csv')
    d=d[(d.protocol=='meta_UCUI')&(d.variant=='k61a_gate')].copy()
    d.insert(0,'loss_mode',mode)
    frames.append(d)
print(pd.concat(frames)[['loss_mode','mse','r2','pearson','bias']]
      .round(3).to_string(index=False))
PY
```

## Run the same four modes on other existing folds

Repeat the exact loop with the following path triples, **without**
changing hyperparameters:
- Nightlight: `runs/k5b2_nighttime/composer.pt`,
  `runs/k5a_nighttime/protocol.json`,
  `runs/k61loss_nighttime_${mode}`
- Digital Cell: `runs/k5b2_digital_total_cell/composer.pt`,
  `runs/k5a_digital_total_cell/protocol.json`,
  `runs/k61loss_digital_total_cell_${mode}`

Document every mode and heldout task, including negative outcomes.
For a publishable conclusion, eventually run all six outer source
tasks with prespecified analysis, disclose that outer K4 `best.pt`
was previously meta-selected, and keep final target-task tests
completely untouched by tuning or model selection.

Potential failure modes:
- Pearson ignores affine scale; MSE remains necessary.
- Ranking pairs in large city/task batches can be noisy and uneven.
- Small groups contribute no auxiliary loss.
- Improved source-task training loss or high gate variance does not
  establish heldout task generalization.

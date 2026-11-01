# K5-B: Region × Task Compositional Routing (small pilot)

## Research role

This is an intentionally small implementation of **knowledge-conditioned,
region-adaptive expert routing plus residual adaptation**. It is a step
toward the original Hyper-X/HINT/Hypter + PHATGOOSE/ZOOTER vision, **not yet
a full dynamic hypernetwork or trained independent Scale Router**.

The outer K4 Structured Qwen3 normalized expert bank is **frozen**. Only
a lightweight region × task route network and a rank-8 residual adapter
on its shared spatial hidden states are optimized. Input task signatures
are Qwen3 cosine similarities with source tasks. Region features combine
location context and frozen shared image features.

During training, examples are exclusively from the **outer train** set
(five available tasks in seen cities); each labeled sample's own source
expert is MASKED so the composer learns inter-task transfer rather than
trivially reusing the already-supervised expert. Source-only target label
moments are used to standardize training losses. The outer held-out task
and held-out cities never enter the optimizer. At outer prediction time,
all available experts may contribute; *scale routing stays frozen semantic*.

The source experts have already been trained on the available source
task labels, so this is **source-supervised episodic training**, not fully
independent nested expert training. Do not describe it as a strict
nested-LOTO assessment. In this pilot the base K4 `best.pt` was selected
using meta labels; results are development-only, not final testing.

## One focused experiment: Digital Total Cell

```bash
git pull origin main
python -m py_compile urban_transfer/ktrans_k5b.py scripts/run_ktrans_k5b.py
python -m unittest discover -s tests -p 'test_ktrans_k5b.py' -v

python -u scripts/run_ktrans_k5b.py \
  --checkpoint runs/k4_t2_structured_qwen8b/best.pt \
  --output runs/k5b_digital_total_cell \
  --epochs 20 --batch-size 512 --eval-batch-size 1024 \
  --lr 0.001 --rank 8
```

The script reports **K4 initial and K5-B** Meta-SCUI/UCUI MSE,
Pearson, and Bias, with the learned region-averaged route for each
protocol; output `evaluation.csv`, `composer.pt`, `protocol.json`.
Check that the K4 initial metric matches the prior K4 Structured
semantic/semantic baseline before interpreting any improvements.

## Print four compact rows

```bash
python - <<'PY'
import pandas as pd
d=pd.read_csv('runs/k5b_digital_total_cell/evaluation.csv')
print(d[['route','protocol','mse','r2','pearson','bias']].round(3).to_string(index=False))
PY
```

## Interpretation

This pilot changes **both spatial routing and a low-rank shared-hidden
adapter**. It does not isolate their effects. Follow-up ablations should
disable one mechanism at a time; never treat local increases/decreases
as a final method verdict. Report results task by task and keep actual
SCUI/UCSI/UCUI test labels completely isolated.

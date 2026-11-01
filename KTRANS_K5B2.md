# K5-B2: Stable Region × Task Adapter

A focused alternative to K5-B, not a new grid search. Reuse the same
frozen K4 Structured Qwen3-8B expert checkpoint and source-only training
partition. Preserve the learned Region × Task Router and introduce:

- bounded, rank-8 hidden residual: `adapter_cap * tanh(residual / adapter_cap)`
  before the existing `adapter_strength=0.1` scaling;
- expert consistency loss: average squared difference between all
  frozen experts' outputs on adapted vs original hidden representations,
  using **source-only** batches;
- backward-compatible opt-in flags so K5-B runs remain unchanged.

This limits distortion of frozen expert predictions rather than promising
the Adapter will necessarily improve every indicator.

```bash
git pull origin main
python -m py_compile urban_transfer/ktrans_k5b.py scripts/run_ktrans_k5b.py
python -m unittest discover -s tests -p 'test_ktrans_k5b*.py' -v

python -u scripts/run_ktrans_k5b.py \
  --checkpoint runs/k4_t2_structured_qwen8b/best.pt \
  --output runs/k5b2_digital_total_cell \
  --stable-adapter --adapter-cap 0.5 --preserve-weight 1.0 \
  --epochs 20 --batch-size 512 --eval-batch-size 1024 \
  --lr 0.001 --rank 8

python -u scripts/ablate_ktrans_k5b.py \
  --composer-checkpoint runs/k5b2_digital_total_cell/composer.pt \
  --eval-batch-size 1024
```

`evaluation.csv` contains K4 versus K5-B2 Meta-SCUI and Meta-UCUI.
`ablation/evaluation.csv` contains the four inference-only combinations
of dynamic router and adapter (not separately trained variants).

Same outer development-set checkpoint caveat as K5-B: Meta heldout labels
were used when selecting the underlying K4 `best.pt`, but they do not
enter K5-B2 optimizer or consistency penalty. No true target task labels
used. Fixed default regularization and cap are **proposed pilot values**;
do not tune them using the outer heldout labels.

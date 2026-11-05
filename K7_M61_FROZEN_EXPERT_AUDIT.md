# M6.1 Frozen Expert + Same-Image Indicator Audit

## Purpose

Disambiguate the M6.1 population-heavy route (SCUI/UCUI population
weight ~0.96) **without any further fits or Qwen-VL computations**.

1. Use cached Education Hidden States for SCUI/UCUI pseudo-validation
   images and frozen `model.pt`. For each sample report Education
   MSE/Pearson for four inference conditions:
   `learned`, `population_only`, `food_only`, `equal_mix`.
   The overrides occur **before the shared nonlinear prediction
   head**. Averaging two expert scalar predictions would be wrong.
2. On the exact source-training image subset (128 rows), keep
   RemoteCLIP and geographic context fixed and swap only the
   **task-conditioned MLLM Hidden States** among Population, Food,
   Education. Report pairwise absolute changes in Population router
   weight, standardized prediction, and cosine of the *student's
   projected knowledge representation*. Never inspect source or
   target labels for these comparisons.

This code uses the **existing M6.1 cache and checkpoint**; it does not
download checkpoints, perform MLLM forward passes, train new models,
select hyperparameters, or read official test CSV files. Education
pseudo-dev labels are loaded from the *source train.csv* for reporting
metrics only, after frozen predictions are produced.

## Run

```bash
git pull --ff-only origin main
python -m unittest discover -s tests -p 'test_k7_m61_frozen_audit.py' -v
python -u scripts/audit_k7_m61_frozen_experts.py \
  --run-dir runs/k7/m61_structured_seed42 \
  --batch-size 64 --device cpu
```

Optional `--device cuda` for ROCm, though GPU is not needed for the
small existing subset. Defaults to CPU.

Output:
`runs/k7/m61_structured_seed42/frozen_expert_indicator_audit.json`.
Existing file is never overwritten (specify a new `--output` if
necessary). No M1-M6 or M6.1 model, cache, metric, or config files are
changed.

## Interpretation

- If `food_only` outperforms `population_only` under the same
  frozen Education Hidden States while `learned` routes ~96% to Pop,
  this supports a routing failure hypothesis on the inspected pseudo-dev
  split; it does not validate selecting a new routing weight by target
  labels.
- If Food is worse, artificial balancing may be harmful.
- If the same-source-image Pop/Food/Education prompt changes
  substantially change routing and prediction, MLLM features provide
  **task conditioning**. That is not proof of correct transfer or
  helpful semantic reasoning.
- If the student knowledge cosines and routes barely differ even for
  same-image task swaps, the structured tokens have not delivered
  sufficient usable task-specific information to the student.
- Differences in metrics on the small 64-row pseudo-validation sets
  are noisy. The checkpoint must remain frozen and unseen City/Indicator
  official final tests untouched.

The diagnosis is a **post-training model audit**, not a performance
selection procedure. Avoid retroactively choosing the best override
from Education dev metrics and reporting it as a zero-shot method.

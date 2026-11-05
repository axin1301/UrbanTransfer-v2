# K7 M6.2: frozen visual-token ablation

## Why

M6.2 improved UCUI pseudo-dev MSE/Pearson, and SAME-IMAGE indicator
definitions changed its task query and router substantially. However,
mismatching **all** image-conditioned MLLM tokens changed prediction
by only ~0.012–0.013 log1p units, leaving open whether visual
information is actually used by its Task-Queried Attention.

This audit loads **the existing M6.2 checkpoint** and **M6.1 64x2048
structured frozen Qwen-VL hidden cache**; it does not run Qwen-VL again,
retrain any model, or change parameters.

For SCUI and UCUI Education pseudo-dev image rows, hold fixed:

- Original frozen RemoteCLIP image feature.
- Original four geographic context values.
- Original Education **Definition (16)** and **Relation (16)** MLLM tokens.

Vary only the first **32 Visual tokens**:

1. `matched`: original cached image visual tokens.
2. `swap_visual`: visual chunk from the NEXT pseudo-dev image in the
   same partition, cyclically across the full partition; original
   Definition/Relation chunks remain untouched.
3. `zero_visual`: replace ONLY visual chunk with zeros.

Report MSE, Pearson, ratio to the source-only constant baseline,
mean absolute log1p prediction change and mean absolute Population
router weight change relative to `matched`. Full distributions
(mean/std/min/max) are stored in JSON.

Important: visual-token swaps and zeroes produce **counterfactual
hidden-state inputs**, not valid holistic MLLM forwards.
The later definition/relation states are themselves contextualized
and can contain image information. Therefore zeroing the early visual
group does **not** remove all image information from the MLLM; nor
does an out-of-distribution intervention demonstrate correct causal
visual reasoning. A near-zero result suggests the *separately
extracted visual group* is weakly used, but does not establish that
all joint multimodal knowledge is ignored.

## Run (CPU sufficient)

```bash
git pull --ff-only origin main
python -m unittest discover -s tests \
  -p 'test_k7_m62_visual_token_audit.py' -v
python -u scripts/audit_k7_m62_visual_tokens.py \
  --run-dir runs/k7/m62_task_query_seed42 \
  --cache-dir runs/k7/m61_structured_seed42 \
  --batch-size 32 --device cpu
```

Outputs `runs/k7/m62_task_query_seed42/frozen_visual_token_audit.json`
and concise console results for SCUI/UCUI. Existing file is never
overwritten; use a new `--output` path if you intentionally run a
separate diagnostic. The script reads pseudo-dev labels from the
source `train.csv` **only for reporting**, after the checkpoint is
frozen. Official final test CSVs are never used.

## Interpret

If `swap_visual` and `zero_visual` barely change prediction and
router, the trained model is likely dominated by the explicit
Definition/Relation task-query path and RemoteCLIP image features.
This suggests improving visual cross-attention **within the Student**,
not automatically enlarging the Qwen-VL model.

If interventions change predictions appreciably, check whether
the changes hurt pseudo-dev MSE/Pearson or are unstructured noise.
Nonzero sensitivity alone is not evidence of useful visual transfer.
Do not select another model or tune interventions using SCUI/UCUI
labels; this is a diagnostic, not a model-selection sweep.

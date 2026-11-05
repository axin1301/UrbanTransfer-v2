# M6.2 frozen attention + direct-query bypass audit

## Goal

M6.2 improved UCUI pseudo-dev regression and developed clear same-image
indicator-conditioning, but changing Qwen's Visual tokens across images
barely changed predictions. Determine whether the trained model (a)
learns nontrivial task/image-specific attention distributions and
(b) uses the task-query direct bypass more heavily than attended image
information.

**This is read-only**: loads the trained M6.2 `model.pt` and M6.1
`hidden.sqlite3`; no additional Qwen-VL forwards, no retraining,
no checkpoint selection, no official final test files.

## ROCm / CPU run

```bash
git pull --ff-only origin main

python -m unittest discover -s tests \
  -p 'test_k7_m62_attention_mechanism.py' -v

python -u scripts/audit_k7_m62_attention_mechanism.py \
  --run-dir runs/k7/m62_task_query_seed42 \
  --cache-dir runs/k7/m61_structured_seed42 \
  --batch-size 32 \
  --device cpu
```

The script refuses to overwrite the report. Output:
`runs/k7/m62_task_query_seed42/frozen_mechanism_audit.json`.
Specify a fresh `--output` file to rerun.

## Frozen diagnostics

**Attention distribution** — use real trained M6.2
`nn.MultiheadAttention` with `need_weights=True` and
`average_attn_weights=False`. The diagnostic reconstructs the
original frozen forward, asserts all predictions, routes, queries
and fused representations agree within 1e-5.

- `normalized_head_entropy` ranges from 0 (one-token attention) to
  1 (uniform attention). Mean across attention heads per image.
- `same image [Population vs Education, Food vs Education,
  Population vs Food] attention JS`: keeps the source tile fixed and
  changes the task-conditioned Qwen hidden tokens. Pairwise
  Jensen–Shannon divergence in nats.
- `different image same indicator attention JS`: compare each
  source training tile with the next tile under exactly the same
  indicator prompt. Token positions are compared by index, which is
  not guaranteed to represent aligned geographical regions.

**Controlled direct-path ablation**, on SCUI/UCUI pseudo-dev:

- `normal`: original cross-attended model, with numerical parity check.
- `zero_query_direct`: set only the direct `q` half of the
  `[q, fused]` router/gate input to zero. The fused half is computed
  as usual and **still contains query semantics**.
- `zero_fused_direct`: set only the `fused` half to zero. The
  explicit query half remains intact.
- `zero_attention_output`: keep the task query intact; recompute
  the frozen fusion module on `[q,0,0]` instead of
  `[q,attended,q*attended]`. This isolates the attention-output
  route through fusion while preserving the query direct path.

Report for each mode: MSE, Pearson, relative MSE to source constant
baseline, mean abs prediction change in log1p target units and
Population router-weight change. Output includes distributions
(mean/std/min/max). None of the interventions produces a new trained
model, and no Education label is used in training (the checkpoint is
already frozen).

### Interpretation

Attention entropies near 1 and small task/image JS differences suggest
the model spreads attention almost uniformly over cached visual
positions. However, a uniform attention output may still encode
useful visual context through values. Also, attention variation is
not a causal explanation or a spatial saliency map.

If suppressing the direct query has a greater effect than suppressing
the fused/attention pathway, the current route is plausibly
task-text dominated. Conversely, if attention suppression damages
prediction materially, the student uses that pathway. These are
**off-manifold** zero interventions: magnitude comparisons depend on
learned projection scales, so they are clues not formal attributions.

Qwen visual token states precede task definitions in the causal
sequence, while later Definition/Relation token states can themselves
already contain image information. Thus this experiment does not
isolate *all* MLLM image knowledge, and no ablation proves semantic
correctness.

Education pseudo-dev values come only from `train.csv` and are used
for post-fit reporting. Don't tune additional models or hyperparameters
against this audit.

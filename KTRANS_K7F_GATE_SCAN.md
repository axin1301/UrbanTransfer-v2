# K7-F fixed-gate counterfactual (DEV mechanism diagnostic only)

## Compatibility with K4

K4 loads frozen RemoteCLIP image features and cached Qwen3 task text
embeddings to produce standardized source-expert spatial predictions,
with source-task-only moments. K7-D does **not** concatenate VLM outputs
to those image/text embeddings or claim that Qwen3-VL 36d cues inhabit
the RemoteCLIP/Qwen3 embedding space. It uses VLM-derived visual
morphology, conceptual task relevance and missing-value masks *only to
adjust expert weights* around the unchanged K5-A transfer route.

This is structurally compatible with the K4 frozen expert-bank design.
It is not proof of semantic alignment: VLM relevance scores are
indirect *hypotheses*, not measured correlations or target labels.
Train/eval K7-D checkpoint provenance is checked. Source-only router
training masks each pseudo-heldout source task's own expert. K4
meta-selection and source-label reused features mean all findings are
**development**, not independent test results.

## Fixed-gate experiment, no retraining

The source-trained Router candidate `candidate(x)` and K5-A anchor
`prior(x)` are held fixed. For the predeclared values
`0, .05, .118, .25, .50, .75, 1` compute:

```
w_g(x) = (1-g) * prior(x) + g * candidate(x)
```

The trained samplewise gate is included as an additional reference.
Rows and frozen K4/K5-A components are identical across all values.
Report paired MSE/Pearson and changes relative to `g=0` on the
same complete-case subset. Posthoc MC/meta UCUI numbers are exploratory:
do not choose best `g` on UCUI and cite that as an unbiased result.

## Run seed 42

```bash
git pull origin main
python -m unittest discover -s tests -p 'test_ktrans_k7f_gate_scan.py' -v
python -u scripts/run_ktrans_k7f_gate_scan.py \
 --checkpoint "$K4_CKPT" \
 --k5a-protocol "$K5A_PROTOCOL" \
 --manifest runs/k7d_education/manifest_complete_cases.json \
 --knowledge-cache runs/k7d_education/evidence.jsonl \
 --reuse-cache runs/k7b_pilot/evidence.jsonl \
 --routers-dir runs/k7d_education/single_seed42 \
 --output runs/k7f_education/seed42 \
 --seed 42
```

The script reads the existing K4 bank and K7-D routers and never
loads Qwen3-VL, optimizes the gate or writes training checkpoints.
Output: `gate_scan.csv` (3 partitions ×3 VLM variants ×8 gate
modes), and `protocol.json`. Compare the `trained` rows against
K7-D `evaluation.csv` for exact-run consistency.

## How to interpret

- Candidate route wins for higher fixed gates on heldout DEV: suggests
  conservative gate may inhibit useful candidate routing, **not**
  evidence of independent generalization or a safe gate-selection rule.
- Larger fixed gates consistently worsen UCUI: the candidate
  route direction is suspect; increasing gate alone is unlikely to help.
- Predictions barely move despite high fixed gates: frozen source
  experts are functionally similar or differences cancel.

Do not change K4 checkpoint/manifest while interpreting the curve.

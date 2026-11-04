# P2-D-VLM — Can task-conditioned Qwen3-VL relevance rescue source cross-task expert competence transfer?

**Motivation:** P2-B2 / P2-C2B show genuine cue information supports
supervised same-source-task SCUI→UCUI, but source pseudo-task LOPO
competence routing is worse than static expert/task baselines.

**Important implementation discovery:** P2-B1's `task_cue_expert`
uses IMAGE_TASK `visual_cues` and `evidence_confidence`, and
TEXT_ONLY source task `evidence_relevance`, but did not use
the actual IMAGE_TASK **`evidence_relevance`** output. That block
explicitly asks Qwen3-VL whether six observable morphology cues
are relevant to the *named task for this specific image*.
P2-D-VLM isolates this previously omitted source of conditionally
task-specific evidence, before changing expertise loss supervision,
model capacity or source expert architecture.

## Run on HPC (CPU only; requires existing P2-C2B results)

```bash
git pull origin main
python -m unittest discover -s tests -p 'test_ktrans_p2d_vlm_relevance.py' -v

OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 \
python -u scripts/run_ktrans_p2d_vlm_relevance.py \
  --original-input runs/ktrans_competence/education_all_outer_train_dev \
  --stable-input runs/ktrans_competence/p2c2b_stable_source_dev \
  --knowledge-cache runs/ktrans_competence/full_source_evidence.jsonl \
  --output runs/ktrans_competence/p2d_vlm_source_dev \
  --alpha 10 --temperature 0.5 --seed 42
```

Source-only manifest and exact text-only Qwen3-VL prompt/key identity are
read from original empirical source evidence. Cache sources default to
existing full source VLM cache, `runs/k7b_pilot/evidence.jsonl` and
`runs/k7d_education/evidence.jsonl`. No new VLM inference, no expert
retraining, no GPU and no target Education labels.

The script checks that original and stable banks use exactly the SAME
pseudo-task / split / tile / city / named expert / image / VLM knowledge
/ source label pairs. Only expert predictions and their squared errors
may differ. For the stable bank it additionally validates that a
named source expert has the same full-model hash across all pseudo folds.

## Seven predeclared modes

| Mode | Task-specific information | Control question |
|---|---|---|
| `task_expert` | P2-B1 TEXT_ONLY prior × named source expert | Prior-only baseline |
| `task_cue_expert` | Original P2-B1 text_only + visual × confidence × expert | Prior knowledge baseline |
| `vlm_task_relevance` | Actual IMAGE_TASK relevance × image cues/confidence × expert | Does direct task-conditioned VLM output help? |
| `vlm_task_residual` | Actual IMAGE_TASK relevance - TEXT_ONLY relevance, confidence-gated with scene | Does image-specific deviation from task conceptual prior help? |
| `vlm_task_full` | Both direct and residual with all original VLM masks | Does combined VLM evidence help? |
| `permuted_vlm_task_relevance` | Same as direct but shuffle relevance + mask across images of SAME task/split | Is tile–VLM relevance pairing informative? |
| `permuted_vlm_task_full` | Same as full but shuffled image-conditioned relevance | Does true VLM relevance outperform marginal information? |

These seven are **fixed before examining P2-D outcomes**.
All modes use the EXACT original P2-B1 relative target, fixed
Ridge alpha 10, softmax temperature .5, globally named expert
one-hot and four source expert paired predictions. We do not
perform hyperparameter sweeps or selection on source UCUI.

**Permutation detail:** shuffle only IMAGE_TASK `evidence_relevance`
and its *observation mask*, separately within each source
pseudo-task / SCUI or UCUI split, independently for SCUI train
and evaluation. Preserve task descriptions, image cue presence,
VLM confidence, source experts, predictions, city/label and
loss. This tests the incremental value of IMAGE_TASK relevance,
not the entire VLM knowledge signal.

## Two evaluation routes per bank

- `cross`: Train on OTHER source pseudo-task SCUI, evaluate
  heldout pseudo-task SCUI and UCUI. Source pseudo task LOPO.
- `same`: Train on THIS pseudo-task SCUI labels and evaluate
  THIS task UCUI. SUPERVISED cross-city diagnostic, NOT zero-shot.

Absolute MSE cannot be directly compared between original neural
bank and stable independent Ridge bank due to architectural and
output scaling differences; within-bank ratios against its uniform
mixture are descriptive.

## Outputs

For each `original_k5a/` and `stable_ridge/`:
- `cross_macro_metrics.csv`: source-task macro Top1, pairwise,
  blend MSE and uniform ratio for each mode/split.
- `cross_task_metrics.csv`: task-by-task results for diagnosing
  outlier task directions.
- `same_macro_ucui.csv`, `same_task_ucui_metrics.csv`:
  supervised source-task city transfer sanity check.
- `vlm_relevance_signal_audit.csv`: for six cues × each
  task/split, fraction observed, number of distinct image-task
  relevance scores, relevance variance, absolute deviation from
  text_only conceptual prior, fraction identical to text_only.
- Shared `protocol.json` describing cached VLM keys, exact
  evidence alignment, source isolation and exploratory status.

Read the log in this order:

1. `evidence_relevance_variable_cues` and average
   `image_task - text_only`: if nearly zero there may be no
   additional task-specific spatial information in this schema.
2. SOURCE-LOPO UCUI `vlm_task_relevance` and `vlm_task_residual`
   vs `task_expert`, `task_cue_expert` and shuffled controls,
   across BOTH source expert architectures.
3. Same-task UCUI check: the new feature must meaningfully encode
   a region/task-relevant capability effect before claiming its
   failure/success as cross-task transfer.
4. Per-task source LOPO results to check whether any macro gains
   hide regressions on four of five source tasks.

## Scientific limitation

This is **exploratory POSTHOC DEV**; outer K4 best.pt was
meta-selected on Education, and source UCUI has been inspected.
Five source tasks are not sufficient for confirmatory
Hypernetwork-generalization claims. Even if IMAGE_TASK relevance
improves scores, it is a coarse (six ordinal cues) *proxy
relevance*, not an empirically calibrated competence label. A
VLM can say a road cue is plausible for mobility without proving
that the existing roads expert is a good predictor.

If direct task-conditioned relevance still fails, the next VLM
experiment should ask for **task/region-specific predicted expert
transfer evidence with fixed named expert descriptions**, using
source-only pseudo-task supervision to ground/calibrate these
judgments. That requires NEW task×expert prompts and strict source
provenance; do not fabricate those signals from the current cache.

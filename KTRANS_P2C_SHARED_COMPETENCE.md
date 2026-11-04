# K-TransMoE P2-C1/C2 — source expert competence sharing and identity audit

P2-B2 showed real VLM knowledge enables **same pseudo-task SCUI→UCUI**
expert ordering and blending (pairwise .684, macro MSE ratio .948),
but cross-pseudo-task transfer failed (pairwise .408, ratio 1.019).
Before designing a task-conditioned Hypernetwork, first verify if
source tasks share a consistent named-expert competence structure and
whether the experts are actually functionally comparable across LOTO
folds. Both parts are **read-only CPU audits**.

## Commands

```bash
git pull origin main
python -m unittest discover -s tests -p "test_ktrans_p2c_shared_structure.py" -v

python -u scripts/diagnose_ktrans_p2c_shared_structure.py \
  --input runs/ktrans_competence/education_all_outer_train_dev \
  --k5a-protocol "$K5A_PROTOCOL" \
  --manifest runs/ktrans_competence/full_source_coverage/source_train_manifest.json \
  --knowledge-cache runs/ktrans_competence/full_source_evidence.jsonl \
  --reuse-cache runs/k7b_pilot/evidence.jsonl \
  --extra-reuse-cache runs/k7d_education/evidence.jsonl \
  --output runs/ktrans_competence/p2c_education_source_dev \
  --min-group 8
```

`$K5A_PROTOCOL` must point to the original five-source-task nested
K5-A `protocol.json`; inner `last.pt` checkpoints are located
beneath its parent folder. No GPU, VLM inference, model training or
outer Education target labels.

## P2-C1: task × expert × cue competence tensor

For task `t`, named expert `k` and cue `c`, compute:

```text
D[t,k,c] = (mean(high-cue within-tile relative expert loss)
            - mean(low-cue within-tile relative expert loss))
           / (median SCUI expert-loss span for task t)
```

- High/low cut from original task-specific SCUI ordinal visual-cue
  median; strict inequalities. Median ties and missing VLM evidence
  are excluded from contrast and separately counted in counts.
- Normalize by a strictly **SCUI-only** per-task expert-loss span.
  This reduces across-task label/MSE scale domination but does not
  remove all task-specific training/city effects.
- `D[t,t,c]` is **MISSING**, never zero. A pseudo-task is not included
  in its own inner expert bank. Unobserved high/low contrasts remain
  missing too. No tensor factorization or pseudo-label imputation.
- For two pseudo-tasks, only the THREE common named experts × six
  cues can be compared; cosine/sign agreement use common actually
  observed coordinates, not missing values.
- Low-rank descriptive SVD uses only 3-task × 2-common-expert ×
  fully observed cue blocks; report observed rank1 and rank2 energy
  fractions but **do not** treat the small complete-block rank as a
  global 5×30 ability tensor estimate or evidence of generalizing to
  an unseen task.
- Source task knowledge vectors use exact validated five existing
  `text_only` Qwen3-VL cues; task/task cosine can be compared
  descriptively with source expert competence cosine. There are
  only ten distinct task pairs, highly dependent; no p-values.

Outputs:

- `cue_effects.csv` — observed, task-scale-normalized high-minus-low
  relative error effect per named expert and morphology cue.
- `task_pair_ability_similarity.csv` — pairwise task competence
  cosine/sign agreement on COMMON observed experts; text cue cosine.
- `masked_pair_svd.csv` — three-task complete-block rank energy
  diagnostic (historical filename, uses triples).
- `text_ability_similarity.csv` — descriptive correlation between
  conceptual task similarity and observed task competence similarity.
- `protocol.json` — frozen threshold/normalization choices and
  exact text cache provenance.

## P2-C2: same expert name does NOT mean same trained model

Use the original nested K5-A source-only CSV and five LOTO
`inner_models/<pseudo>/last.pt` checkpoints; never load target
`best.pt` or evaluation target labels.

For each fold:
- Check the heldout pseudo-task, exactly four named source experts,
  normalized bank settings, same source-only nested CSV path,
  source moments and checkpoint epoch.
- Reconstruct permitted supervised rows: all non-pseudo source-task
  rows outside that fold's inner heldout city set. Audit counts,
  city coverage and overlap/Jaccard of source training records.
- Compare expert head state tensors by **named source task**, not
  fold-local `experts.0`, across the ten pairs of folds. Each pair
  shares three named experts, so the audit contains 30 comparisons.
- Separately compare the `shared.*` encoder tensors in each fold.
  Report cosine, exact parameter identity and relative L2 drift.
  Parameter cosine cannot independently prove functional equivalence
  because the upstream shared trunk and input distributions also
  differ.

Outputs:

- `checkpoint_fold_metadata.csv`
- `named_expert_parameter_shift.csv`
- `shared_trunk_parameter_shift.csv`
- `fold_supervised_source_counts.csv`
- `fold_training_overlap.csv`

## Interpretation / decisions

1. Strong competence similarity on shared named experts **and**
   comparable provenance suggests there may be shareable
   task-conditioned expert structure, warranting low-rank modeling.
   Be careful: task pairs are dependent and few.
2. If source task semantics do NOT explain task-to-task competence
   similarity, a text-only Hypernetwork is not supported by this
   evidence; richer task knowledge or an alternative representation
   may be necessary.
3. Large inter-fold parameter drift for the same named expert and
   shared trunk creates a confound. Before treating `p_k` as a
   fixed reusable expert capability prototype, prepare a frozen,
   task-identity-stable source expert bank and repeat controls.
4. No architecture or hyperparameters are tuned here. A positive
   masked SVD result or weight cosine alone does not demonstrate
   zero-shot transfer.

**All outputs are POSTHOC DEV.** K4 outer `best.pt` was
Education-meta-selected. Inner SCUI was used in K5-A selection, and
inner UCUI was already repeatedly inspected. This diagnostic is not
independent final cross-city/cross-task performance evidence.

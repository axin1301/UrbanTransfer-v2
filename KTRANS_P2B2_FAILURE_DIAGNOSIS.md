# K-TransMoE P2-B2 — Diagnose conditional-knowledge prediction failure

P2-B1 combined VLM cue + task + expert Ridge underperformed task-only
and uniform blending on heldout source-task UCUI. P2-B0's 92/96
conditional sign agreement is descriptive, not equivalent to
learnable out-of-task expert ranking. P2-B2 separates **cross-city**
vs **cross-task** transfer of the conditional expert signal and
quantifies rank ambiguity, signed residual dependence and the
achievable fixed-source-expert baseline.

## Commands (CPU)

```bash
git pull origin main
python -m unittest discover -s tests -p 'test_ktrans_p2b2_failures.py' -v

python -u scripts/diagnose_ktrans_p2b2_failures.py \
  --input runs/ktrans_competence/education_all_outer_train_dev \
  --manifest runs/ktrans_competence/full_source_coverage/source_train_manifest.json \
  --knowledge-cache runs/ktrans_competence/full_source_evidence.jsonl \
  --reuse-cache runs/k7b_pilot/evidence.jsonl \
  --extra-reuse-cache runs/k7d_education/evidence.jsonl \
  --output runs/ktrans_competence/p2b2_education_source_dev \
  --ridge-alpha 10 --temperature 0.5 \
  --near-tie-fraction 0.05 --seed 42
```

Reuses P2-B1 frozen model, VLM prompt/cache key validation, bounded
paired source expert loss target and predetermined hyperparameters.
No Qwen3-VL calls, retraining source experts or outer Education
labels. Source expert predictions/labels are needed ONLY for
training/evaluating source competence diagnostics, not deployed
region features.

## Two transfer routes, SAME UCUI evaluation examples

- `same_task_scui_to_ucui`: Fit on the given **source pseudo-task's
  own SCUI labels**, test on its UCUI. This is a *supervised same-task
  unseen-city diagnostic*, NOT zero-shot across tasks.
- `other_tasks_scui_to_ucui`: Fit on the other FOUR source
  pseudo-tasks' SCUI labels, test on the heldout pseudo-task UCUI.
  This is a pseudo-task cross-transfer diagnostic.

Both routes compare the four P2-B1 baselines:
`task_expert`, `cue_expert`, `task_cue_expert`,
`permuted_task_cue_expert`. For each task/route/model the EXACT
same UCUI tile/city identities are verified, and their expert
prediction banks are the SAME pretrained inner pseudo-task bank.
SCUI and UCUI city sets are explicitly checked as disjoint for
every pseudo task.

No hyperparameter search. `--ridge-alpha 10` and
`--temperature .5` are identical to P2-B1.

## Outputs and what to inspect

- `same_vs_cross_macro.csv`: per route/model equal-source-task
  macro Top-1, pairwise accuracy, MSE, and **task-normalized soft
  blend MSE / uniform blend MSE** (less than 1 helps).
- `same_vs_cross_task_ucui.csv`: per pseudo task and route; essential
  to identify tasks that fail to transfer.
- `per_tile_ucui_metrics.csv`: task-tile-level predicted blend,
  label-dependent hindsight error metrics, exact evaluation keys.
- `expert_structure.csv`: fixed best expert chosen on that task's
  SCUI aggregate MSE and scored separately on UCUI; per partition
  oracle pointwise best expert (hindsight only), uniform MSE, fraction
  of near-tied top experts and MSE spread.
- `expert_residual_covariance.csv`: per task and split, pairs of
  **signed** expert prediction errors, their covariance and Pearson
  correlation, and mean error product. Signed residuals matter for
  soft-ensemble complementarity; correlated experts may make a good
  ranker a poor blender.
- `expert_rank_stability.csv`: task-level SCUI vs UCUI aggregate
  best-expert stability and UCUI attainable fixed-SCUI baseline.
- `protocol.json`: label boundaries, provenance and caveats.

Near-tie definition: `(second-lowest squared error - lowest error)
 / (max expert error - min expert error) <= .05`. Samples with
 all-equal expert errors are excluded from near-tie percentage,
 reported separately via `n_informative_tile_spans`.

## Interpretation, NOT a claim of independent evaluation

1. If **same-task** cues beat same-task `task_expert` and permuted
   controls on UCUI but **cross-task** cues do not, the bottleneck is
   transferring the cue–expert competence map across tasks.
2. If neither route's real cues beat controls, the P2-B0 grouped
   association is not yet enough for per-tile capability prediction;
   investigate expert uncertainty, VLM cue resolution and objective.
3. If ranking improves but blending does not, investigate signed
   residual covariances and adapt the ensemble objective, not just
   score→softmax temperature.
4. SCUI fixed-best vs UCUI fixed-best and oracle gaps reveal how
   much gains a robust frozen prior can plausibly capture. The
   per-tile Oracle is never available in zero-shot inference.

**Protocol warning:** P2-B2 is deliberately POSTHOC DEV (UCUI already
inspected, original K4 best.pt selected using Education meta labels).
The same-task route uses task labels on SCUI by design; it is *not*
zero-shot cross-task. Also, named inner pseudo-fold experts are
separately retrained, so a name is not a shared frozen weight set
across tasks. These source-only checks do not certify final
independent UCUI performance or paper-level generalization.

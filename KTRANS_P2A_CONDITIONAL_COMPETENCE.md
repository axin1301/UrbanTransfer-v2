# K-TransMoE P2-A: empirical conditional expert competence (SOURCE-ONLY)

P2-A is a CPU-only, read-only diagnostic of **source-task pseudo-LOTO**
expert transfer evidence. It DOES NOT train a new router, fit any
prediction parameters, call an 8B VLM, or evaluate outer Education
labels. The input is the complete 13,348-row source competence table
already built from five inner LOTO folds.

## Run

```bash
git pull origin main
python -m unittest discover -s tests -p 'test_ktrans_p2a_competence.py' -v

python -u scripts/diagnose_ktrans_p2a_competence.py \
  --input runs/ktrans_competence/education_all_outer_train_dev \
  --output runs/ktrans_competence/p2a_education_source_dev \
  --min-group 8
```

`--min-group 8` is a predeclared descriptive filter (minimum **eight
distinct source pseudo-task tiles** in EACH low/high cue bin for each
task/split); it is not optimized on UCUI metrics.

## Files

- `task_expert_ranking.csv`: per pseudo-task × inner split × expert
  aggregate MSE, mean **within-tile relative** loss, fractional Top-1
  winner rate (ties share credit), and MSE rank.
- `pairwise_wins.csv`: for each source expert pair in the SAME tile,
  expert-A wins, ties, and paired squared-error difference.
- `cue_conditional_contrasts.csv`: for each pseudo-task × split ×
  observed six-cue morphology × expert, SCUI-frozen median threshold,
  counts below, above, tied and missing, then high-minus-low mean
  within-tile relative squared error (only when both bins adequate).
- `city_expert_profiles.csv`: per pseudo-task × split × city × expert,
  sample count and mean within-tile relative squared error; city
  correlations and uncertainty are NOT inferred from tile counts.
- `scui_ucui_consistency.csv`: whether the signed high-minus-low
  expert-relative-loss contrast agrees between SCUI and UCUI for the
  same pseudo task, cue and named expert; inadequate bins excluded.
- `protocol.json`: full provenance, sample counts, frozen SCUI
  thresholds and scientific warnings.

## Interpretation

For a single (pseudo task t, tile i), the four expert losses are
`e[t,i,k] = (pred[t,i,k] - y[t,i])**2`; relative loss is
`e[t,i,k] - mean_k(e[t,i,k])`. This removes the *within-tile*
common loss magnitude from expert comparisons; it does NOT normalize
the result across tasks. All comparisons are stratified by the same
pseudo task and inner split, not pooled across distinct target units.

`visual_cues` is a task-conditioned VLM-produced ordinal scene
estimate, NOT human-verified or causal urban morphology. The
`evidence_relevance` and `evidence_confidence` scores are retained
in source records; in this first diagnostic we stratify only by six
**observed visual cue scores**, to avoid confusing task relevance
with actual visual presence. Missing cues remain missing. A
task-specific median is computed ONLY from SCUI samples. UCUI
samples reuse that frozen cut; ties (common with ordinal four-level
scoring) are excluded from high-vs-low comparisons and reported.
Too-small groups have no contrast.

Descriptive consistency is NOT a learned competence predictor.
A high/low contrast is neither a causal effect of roads/buildings
nor evidence that VLM knowledge can select an expert online. City
clusters and heavily overlapping tiles may further reduce effective
sample size. Later P2-B should ask if cue-conditioned signals predict
heldout-city named-expert pairwise wins above **task-only** and
**permuted-cue** controls, and measure variance across pseudo tasks
and cities before accepting a new architecture.

## Strict labels and evaluation limits

- This entire dataset comes from SOURCE pseudo-task labels only,
  never outer Education or truly heldout target task labels.
- The old K4 `best.pt` was selected using Education meta data.
  Therefore this is still **development-only**, not strict independent
  zero-shot evaluation.
- Inner SCUI already informed K5-A ridge transfer scores; inner
  UCUI has unseen inner cities but becomes a development diagnostic
  as soon as its results guide new model choices.
- Do not count four expert rows as four independent spatial samples.
  Do not tune cue thresholds, min-group or architecture using
  UCUI results and claim an unbiased test.

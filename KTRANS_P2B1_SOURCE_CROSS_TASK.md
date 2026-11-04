# K-TransMoE P2-B1 — low-capacity SOURCE task–region–expert competence prediction

This is the **first predictive check** after P2-A/P2-B0 descriptive diagnostics.
It is deliberately a low-capacity Ridge model, not a new deployed
Router. Source pseudo-task labels from the already exported inner
LOTO expert predictions are the **only** supervised training signal.

## Run (CPU only)

```bash
git pull origin main
python -m unittest discover -s tests -p 'test_ktrans_p2b1_source_lopo.py' -v

python -u scripts/run_ktrans_p2b1_source_lopo.py \
  --input runs/ktrans_competence/education_all_outer_train_dev \
  --manifest runs/ktrans_competence/full_source_coverage/source_train_manifest.json \
  --knowledge-cache runs/ktrans_competence/full_source_evidence.jsonl \
  --reuse-cache runs/k7b_pilot/evidence.jsonl \
  --extra-reuse-cache runs/k7d_education/evidence.jsonl \
  --output runs/ktrans_competence/p2b1_education_source_dev \
  --ridge-alpha 10 --temperature 0.5 --seed 42
```

No GPU inference and no checkpoint fitting. Requires exact source
task `text_only` cache records already observed to be complete (5/5).
The command checks model ID, prompt, version, max-pixels, task and
cache hash against the full source-only canonical manifest. An outer
Education text-only record is NEVER requested here.

## Cross-task protocol

For every heldout pseudo-task `t` among the five SOURCE tasks:
1. Train Ridge ONLY from **other four pseudo-task SCUI** tile-level
   expert losses. The task `t` contributes no labels to that fit.
2. Freeze the model and evaluate the withheld task SCUI and withheld
   task UCUI separately. UCUI is also city-unseen for the INNER source
   experts. Neither SCUI/UCUI of the withheld task selects hyperparameters.
3. Use one-hot **globally named** expert identities (five source names)
   instead of inner-bank position, which changes by pseudo task.
4. Task semantic features are six source task-specific conceptual
   `text_only` cue relevance scores. They are independent of target
   labels and do not rely on a pseudo-task ID lookup classifier.
5. Image features: six observed morphology scores and their six masks;
   combined features add cue confidence and conceptual relevance
   interaction. All 36D source `image_task` evidence values are loaded
   through the existing evidence exporter.

Supervised Ridge targets are **within-tile centered loss / within-tile
expert-loss range**, a bounded target that does not let tasks with
larger original squared-error magnitude dominate the fit. The four
label-dependent losses are used ONLY in source-side Ridge fitting and
posthoc source-side evaluation; no losses are used to produce a heldout
tile's competence features or weight predictions. For all-equal expert
losses, target score is zero.

Ridge penalty is FIXED to 10; expert-score softmax temperature FIXED
to 0.5. Neither is tuned on UCUI. This prototype offers no claim that
these constants are optimal.

## Baselines

- `task_expert`: frozen source task-only text conceptual relevance +
  globally named expert. No regional image knowledge.
- `cue_expert`: VLM regional morphology + expert. No explicit
  task-only descriptor interaction (its image_task cues still
  originate from a task-conditioned VLM prompt; NOT a genuinely
  task-free visual input).
- `task_cue_expert`: task conceptual relevance, observed VLM
  morphology/confidence and their expert-conditioned interaction.
- `permuted_task_cue_expert`: same architecture as the combined
  model but randomly exchanges complete VLM knowledge blocks
  within source pseudo-task×partition in both training and
  evaluation, retaining task descriptor, expert error, split and
  city identity. Seeded, exploratory negative control.

All four variants train the SAME source SCUI-only folds. They report
per pseudo-task/partition metrics and equal-weight macro averages
over the five SOURCE pseudo tasks. Expert rows are NOT independent
units: each heldout source pseudo-task tile has four experts.

## Files and evaluation

- `source_task_metrics.csv`: heldout source pseudo-task SCUI and
  UCUI, n_tiles/cities, Top-1 tie-fractional correctness, paired
  expert-order accuracy, selected expert regret and MSE, oracle
  single-expert MSE (posthoc unavailable upper benchmark), Ridge
  soft-mixture prediction MSE and uniform-mixture MSE.
- `macro_metrics.csv`: equal-task macro Top-1, pairwise accuracy,
  regret, raw MSE, plus **per-task soft-mixture MSE / uniform MSE
  macro ratio**, important because pseudo tasks have different
  indicator error scales. Ratios <1 are better than uniform.
- `tile_predictions.csv`: reproducible source task tile-level
  diagnostics; includes pseudo-task ground truth so do not feed it
  to a deployment inference service.
- `protocol.json`: provenance, withheld pseudo-task cross-fitting,
  fixed hyperparameters and scientific caveats.

**Interpretation:** Improved heldout source pseudo-task UCUI ranking
and blending against `task_expert` AND
`permuted_task_cue_expert` would support predictive added value
from real region knowledge. Pure descriptive sign agreement from
P2-A/P2-B0 does not suffice.

## Known limits: this is DEV-only, not an independent test

- This program was specified after viewing source UCUI; no UCUI
  results should be used to tune alpha, temperature, feature recipe,
  or select the best model as if UCUI were independent.
- The outer K4 `best.pt` was meta-selected on Education labels.
  No independent new-task UCUI result is certified.
- Every INNER pseudo-task bank was trained separately on four source
  tasks; a named expert here does not necessarily have the same
  parameters as the corresponding frozen expert in outer K4.
  Positive predictive results do NOT automatically transfer to the
  outer K4 bank.
- City clusters can share image morphology; the source-task LOTO
  folds support exploratory cross-task validation, not causal
  identification or strict independent test significance.
- A learned soft blend of expert PREDICTIONS can have different MSE
  behavior from choosing the single best expert; report both, and
  check per-source-task MSE ratios, not just pooled MSE.

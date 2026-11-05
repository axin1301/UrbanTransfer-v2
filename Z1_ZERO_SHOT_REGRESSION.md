# Z1 — Zero-shot semantic regression pilot

## Research hypothesis

Instead of K-TransMoE's separate predictor for each source indicator plus
an expert-combination rule for unseen indicators, learn a **shared function
conditioned on indicator semantics**. This is inspired by CoCoOp's conditional
prompts, MaPLe's visual-language interaction and generalized zero-shot
learning's class-description transfer. Unlike CV classification, our outputs
are continuous log1p urban indicators and may not have identifiable target
units. The code is an exploratory **source-only** pilot, not a claim of
independent zero-shot test performance.

**No old K4/K6 or P2-E1 files or caches are changed.**

## B1/B2/B3 architecture

- **B1:** frozen RemoteCLIP image + 4 spatial context values; learned shared
  visual trunk, frozen Qwen3 indicator embedding, learned semantic FiLM, and
  a single shared regression head.
- **B2:** B1 plus visual-instance-conditioned modification of indicator
  semantics, implementing the *idea* of CoCoOp rather than fine-tuning CLIP.
- **B3:** B2 with a lightweight low-rank, semantic-generated regression
  component instead of B1/B2's fixed shared MLP head.

These are controlled architectural steps. They do **not** reproduce
MaPLe's two-encoder prompt injection, nor do they train Qwen or RemoteCLIP.
A one-hot task-ID baseline is intentionally **not** treated as zero-shot.

## Leakage control and results

For each of the six registered source indicators, hold that task out
**entirely** and hold the same deterministic source-city subset out.
Train B1/B2/B3 only on other indicators in other cities. Evaluate:
- SCUI: held-out source pseudo-indicator / train cities
- UCSI: trained indicators / held-out source cities
- UCUI: held-out pseudo-indicator / held-out cities

All models use **fixed epochs, seed, hidden size, learning rate and city
partition**, without UCUI-based checkpoint selection. Source-only means and
standard deviations normalize training targets. A held-out indicator's
log1p scale is estimated *without its labels*, using semantic similarity
to source-indicator means/standard deviations. This **scale transfer is
unverified** and could be a major obstacle to absolute value prediction.

Metrics (written for every fold and model) are: log1p MSE, MAE, Pearson,
source-training-only constant-mean baseline MSE, and relative MSE. The
macro file averages tasks unweighted. The run writes an exact config,
fold metrics, macro metrics and checkpoints for all fitted models.

No final target tasks (healthcare, mobility, social, LTE) are evaluated
in this phase. These source pseudo-task DEV folds must not be claimed
to constitute an independent unseen-indicator benchmark. Architecture
selection based on them requires subsequent fresh, predefined final
evaluation and a comparison with existing K4/K6 metrics under precisely
the same splits and data budget.

## HPC procedure

First make sure the source CSV contains **all six registered source
indicators** in at least three source cities and that the SQLite
RemoteCLIP feature store already includes these images. Find your actual
data/cache paths locally; do NOT guess from previous runs.

### Update server and test (no GPU)

\`\`\`bash
git status --short
git branch --show-current
git pull --ff-only origin main

python -m unittest discover -s tests -p 'test_z1_pilot.py' -v
\`\`\`

### Prepare frozen semantics (only if not already cached)

Z1 defaults to **Qwen3-Embedding-0.6B** text-only vectors (not
Qwen3-VL). If the selected model/profile embeddings are already in the
cache, no Qwen inference will be invoked. Otherwise, on a machine with
sufficient resources, generate the ten task vectors **once**:

\`\`\`bash
python - <<'PY'
from urban_transfer.ktrans_text import qwen_task_vectors
qwen_task_vectors(root='runs/ktrans_text',
                  model_id='Qwen/Qwen3-Embedding-0.6B',
                  profile='short', batch_size=2)
PY
\`\`\`

For an existing **different** Qwen embedding cache, pass the exact
\`--text-model\`, \`--text-profile\`, and \`--text-cache-dir\` used to
generate it. A missing cache fails closed; Z1 never silently replaces
Qwen vectors with hashed text.

### Optional tiny architecture smoke test

The deliberately poor deterministic hash text encoder is allowed only
with explicit \`--semantic-backend hash --allow-hash-smoke\` and must
**never** be used to claim research performance.

\`\`\`bash
python -u scripts/run_z1_pilot.py \
  --source-csv /PATH/TO/SOURCE.csv \
  --image-root /PATH/TO/IMAGES \
  --feature-dir /PATH/TO/REMOTECLIP_FEATURE_DIR \
  --semantic-backend hash --allow-hash-smoke \
  --limit 300 --epochs 1 --batch-size 32 --cpu \
  --output runs/z1/hash_smoke_seed42
\`\`\`

If the CSV contains an explicit image path column, add
\`--image-column COLUMN_NAME\`. The smoke limit must cover all
city×indicator strata and enough rows to form each training fold.

### Full frozen-encoder experiment

\`\`\`bash
python -u scripts/run_z1_pilot.py \
  --source-csv /PATH/TO/SOURCE.csv \
  --image-root /PATH/TO/IMAGES \
  --feature-dir /PATH/TO/REMOTECLIP_FEATURE_DIR \
  --text-cache-dir runs/ktrans_text \
  --text-model Qwen/Qwen3-Embedding-0.6B \
  --text-profile short \
  --modes b1 b2 b3 --epochs 20 --batch-size 128 \
  --seed 42 --city-fraction 0.25 \
  --output runs/z1/qwen06b_seed42_v1
\`\`\`

The Z1 model heads are small; with cached frozen features they generally
do not require loading Qwen3-VL or RemoteCLIP on the GPU. To force CPU,
add \`--cpu\`. Do **not** reuse a nonempty Z1 output directory; experiments
fail closed rather than overwrite existing results.

### Read compact results (three decimal places)

\`\`\`bash
python - <<'PY'
import csv
from pathlib import Path
p=Path('runs/z1/qwen06b_seed42_v1/macro_metrics.csv')
with p.open() as f:
    for r in csv.DictReader(f):
        print(f"{r['mode']:3} {r['partition']:4} "
              f"MSE={float(r['mean_mse_log1p']):.3f} "
              f"Pearson={float(r['mean_pearson']):.3f} "
              f"MSE/source_mean={float(r['mean_relative_mse']):.3f}")
PY
\`\`\`

Interpretation:
- UCUI is the **primary diagnostic**, but it is not independently held out
  from the researcher's ongoing model design.
- A lower UCUI log1p MSE is better; lower relative MSE means the model
  improves upon its matched source-only scale/mean prediction.
- Pearson measures spatial association, not numeric calibration.
- If B2/B3 improve UCUI only on one source pseudo-task, do not claim a
  general unseen-indicator effect.
- This pilot cannot verify true four-target UCUI transfer until an
  evaluation protocol is frozen and run separately.

## Record-keeping

Always keep the run's \`config.json\`, \`fold_metrics.csv\`,
\`macro_metrics.csv\`, and all \`model_*.pt\` checkpoints together.
Use a fresh output directory per seed/model/ablation. Report the exact
Git commit, semantic embedding model/cache, source dataset identity,
feature encoder, seed, splits, and fixed training budget. No results
are claimed until actually run on the HPC server.

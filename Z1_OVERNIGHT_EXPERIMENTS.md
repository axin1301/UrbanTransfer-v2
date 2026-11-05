# Z1 overnight: definition, relation, negative controls, regional conditioning

## Purpose and scope

This is the first unattended **representation-learning** suite following the
Z1 B1/B2/B3 short-definition baseline. It evaluates whether richer frozen
Qwen3 text descriptions and explicit *indicator-pair relation messages* can
improve the **unseen-indicator** regressor. This is NOT K4 expert routing and
does not read Qwen expert competence scores or any final target-task labels.

The 0.6B text embedding model is used only to build frozen task and paired
relation vectors. Small Z1 PyTorch models train on cached RemoteCLIP features.
No 8B or 32B VLM decoding or RemoteCLIP image encoding is needed.

## The 10 controlled conditions

| ID | Source indicator semantics | Relation binding | Image-conditioned semantics | Head |
|---|---|---|---|---|
| short_b1 | short definition | no | no | B1 shared |
| structured_b1 | definition + measurement + drivers | no | no | B1 shared |
| structured_b2 | structured | no | yes | B2 shared |
| structured_b3 | structured | no | yes | B3 generated low-rank |
| relation_b1 | structured | real indicator pair descriptions | no | B1 shared |
| relation_shuffle_b1 | structured | permuted pair-to-source mapping | no | B1 shared |
| relation_b2 | structured | real pair descriptions | yes | B2 shared |
| relation_shuffle_b2 | structured | permuted pair-to-source mapping | yes | B2 shared |
| relation_b3 | structured | real pair descriptions | yes | B3 generated |
| relation_shuffle_b3 | structured | permuted pair-to-source mapping | yes | B3 generated |

For each pseudo-held-out source task, its labels and all held-city labels are
excluded from training; training task relation messages reference **only
the other training source tasks**. The held-out task's definition is allowed
as the zero-shot input, but neither its labels nor moments are fitted.

Pair descriptions are built from the repo's indicator definitions and
**unverified** conceptual relationship registry. Qwen3-Embedding-0.6B
encodes each *target/source* relation text, after which each relation vector
is **bound to the corresponding source semantic embedding** before pooling.
This avoids a mathematically invalid control in which averaging relations
would make within-target card shuffling have zero effect.

This experiment does not establish empirically verified causal
indicator relations. Raw units/formulas (especially TMI) remain unverified.

## Compute budget and resume

For each variant and seed, six source indicators are held out in turn;
each fold trains a new model for 20 fixed epochs. The default 10 variants
× 6 folds = **60 model trainings per seed**, or 120 for seeds 42 and 43.
There is no UCUI checkpoint selection. All splits use the same held-city
selection for a given seed.

Each completed fold immediately writes:
- \`<heldout_task>_metrics.json\` (full precision, three partitions)
- \`<heldout_task>_model.pt\` (checkpoint; do not edit manually)

The outer driver writes:
- \`sweep_macro.csv\` — 3 rows per completed variant/seed
- \`REPORT.md\` — 3-decimal UCUI summary
- \`job_status.json\` — completed/failed task status
- \`<variant>_seed<seed>/console.log\` — full logs/progress
- \`<variant>_seed<seed>/fold_metrics.csv\` and \`macro_metrics.csv\`

Rerunning **identical arguments in the same output directory** skips
completed folds. Changed arguments are rejected by a frozen config
fingerprint. Independent failed variants are recorded, and later variants
continue. Underlying scheduler allocation limits still apply.

## Server: update and test

From the repository root, on \`main\` and with no local edits to overwrite:

\`\`\`bash
git status --short
git branch --show-current
git pull --ff-only origin main
python -m unittest discover -s tests -p 'test_z1_relation.py' -v
python -m unittest discover -s tests -p 'test_z1_pilot.py' -v
\`\`\`

The previously created short embedding cache is reusable; only structured
task descriptions and 30 ordered source-indicator relationship descriptions
will need new one-time embedding computation.

Locate the existing frozen image feature cache:

\`\`\`bash
find runs -name remoteclip_features.sqlite3 -print
\`\`\`

Set \`FEATURE_DIR\` to the **directory containing** the SQLite file. If
more than one cache is found, use the exact RemoteCLIP model used for Z1.

## One-command overnight run

Your known dataset paths:

\`\`\`bash
export SOURCE_CSV="/scratch/project_462001391/xiyanxin/Urban_experience/codes_BaseStation/city_indicator_splits/train.csv"
export IMAGE_ROOT="/flash/project_462001391/xiyanxin/Urban_experience/output_files"
export FEATURE_DIR="/REPLACE/WITH/REAL/REMOTECLIP/CACHE/DIRECTORY"

python -u scripts/run_z1_overnight.py \
  --source-csv "$SOURCE_CSV" \
  --image-root "$IMAGE_ROOT" \
  --feature-dir "$FEATURE_DIR" \
  --text-cache-dir runs/ktrans_text \
  --relation-cache-dir runs/ktrans_text \
  --text-model Qwen/Qwen3-Embedding-0.6B \
  --epochs 20 --batch-size 128 \
  --seeds 42 43 \
  --output runs/z1/overnight_relation_v1 \
  2>&1 | tee runs/z1_overnight_console.log
\`\`\`

Before launching, verify your current HPC compute allocation has access to
the features and sufficient memory to run 0.6B text embedding precompute.
On SLURM, prefer submitting this command in an **allocated batch script**
with your site-specific partition, account and walltime settings. Merely
using \`nohup\` outside a persistent allocation does not keep GPUs allocated
after a session ends. If you already have a persistent authorized compute
allocation, use \`tmux\` or the site's job scheduler as appropriate.

If you want a shorter first pass, use \`--seeds 42\` (60 trainings).
After the caches are built, reruns can add \`--skip-prepare\`, but only
if the cached short, structured and 30 ordered relation embeddings exist.
New seeds always use new model directories; prior results are not overwritten.

## Read the next-morning result

\`\`\`bash
cat runs/z1/overnight_relation_v1/REPORT.md
cat runs/z1/overnight_relation_v1/job_status.json
\`\`\`

Reports are formatted to three decimals. Raw CSV/JSON retains full numeric
precision. The primary diagnostic is **UCUI**, but all results are source
pseudo-task DEV folds. Do not claim the true four-target UCUI improvement
or select a final method using these already inspected DEV outcomes.

A clear next step is comparing the relation variants to same-capacity
permuted variants, checking the *per-indicator* results, and confirming
improvement in Pearson **and** log1p MSE. The optional label-fitted affine
oracle diagnostic, stored with \`ORACLE_NOT_ZERO_SHOT\`, uses evaluation labels
and must NEVER be treated as zero-shot performance.

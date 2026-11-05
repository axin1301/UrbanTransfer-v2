# K7 E1/E2 — Knowledge-conditioned hypernetwork adapters

This is an initial **controlled, executable** K7 prototype, not a claim of
ICDE-grade novelty or a reproduction of Hyper-X, HINT, Hypter, PHATGOOSE,
ZOOTER or LLaMP.

## Architecture and scientific question

Frozen RemoteCLIP ViT-B-32 + 4 context values -> shared visual trunk
\(h_i\). Frozen Qwen3-Embedding-0.6B short task definition -> \(e_t\).

**E1** encodes \(e_t\) into task state \(q_t\), then uses a hypernetwork
to generate low-rank visual adapter factors \(A_t,B_t\) for the shared
trunk; prediction uses the resulting adapted features plus a task bias.

**E2** retains the same hypernetwork/visual backbone. It additionally
encodes directed target/source relation descriptions
\(r_{t\leftarrow s}\), binds each to its eligible source task identity,
pools across eligible source tasks, and fuses this into \(q_t\) *before*
generating \(A_{t},B_{t}\).

**E2 shuffle** rotates the relation message-to-source assignment while
preserving task descriptions and the held-out source set. This is a
falsifiable negative control, not a random-text placeholder.

The E1/E2 comparison **holds numeric scale transfer constant**:
both use the identical short Qwen3 semantic vectors to estimate unseen
task means and standard deviations, avoiding the main confound in the
earlier Z1 short-vs-structured comparison.

The current 3 conditions are:
- \`k7_e1\` — source-task-definition generated adapter.
- \`k7_e2\` — same with directed relation-conditioned generated adapter.
- \`k7_e2_shuffle\` — same with deliberately mismatched source/relation pairs.

No separate source experts, reward-supervised routing, adaptive
Region × Task adapter bank, patch-level interaction, or learned physical
measurement-scale head is present yet. These belong to K7 E3/E4 after
the basic E1/E2 mechanism is evaluated.

## Protocol

Each seed runs 3 conditions × 6 pseudo-held-out source tasks = **18
trainings**, with 20 fixed epochs per training. Using two seeds gives
**36 trainings** / **720 epochs**.

For each fold, train on source cities and the *other five* source
indicators only. UCUI = held-out source cities × held-out indicator.
All numerical label moments use the current training folds only.
Relation source nodes exclude the pseudo-held-out task; only its
definition and externally prepared conceptual relation descriptions
are available at prediction time. No official four-target test CSV
is loaded, and no UCUI labels are used for checkpoint selection.

These are six-source pseudo-target **development** results; not
independent final true-unseen-indicator performance.

## HPC run commands

From repository root:

\`\`\`bash
git status --short
git branch --show-current
git pull --ff-only origin main

python -m unittest discover -s tests -p 'test_k7_hyper.py' -v
python -m unittest discover -s tests -p 'test_z1_relation.py' -v
\`\`\`

Check the existing frozen RemoteCLIP cache first:

\`\`\`bash
ls -lh runs/m3_features/remoteclip_features.sqlite3
\`\`\`

Then start **inside an existing scheduler-allocated HPC job** or wrap
this command in a site-specific Slurm batch script with appropriate
account, partition, memory, GPUs and walltime:

\`\`\`bash
mkdir -p runs
python -u scripts/run_k7_overnight.py \
  --source-csv /scratch/project_462001391/xiyanxin/Urban_experience/codes_BaseStation/city_indicator_splits/train.csv \
  --image-root /flash/project_462001391/xiyanxin/Urban_experience/output_files \
  --feature-dir runs/m3_features \
  --remoteclip-model ViT-B-32 \
  --text-cache-dir runs/ktrans_text \
  --relation-cache-dir runs/ktrans_text \
  --text-model Qwen/Qwen3-Embedding-0.6B \
  --epochs 20 --batch-size 128 \
  --seeds 42 43 \
  --output runs/k7/e1_e2_v1 \
  2>&1 | tee runs/k7_e1_e2_console.log
\`\`\`

The existing short and relation embedding caches are reused. Missing
Qwen3-Embedding-0.6B relation embeddings are prepared once. No Qwen3-VL
or RemoteCLIP image inference is invoked in the training loop.

To resume an interrupted sweep, run **the exact same command**.
Existing completed folds are skipped if their configuration fingerprint
matches. Variant-specific logs are in:

\`runs/k7/e1_e2_v1/<variant>_seed<seed>/console.log\`

Progress bars show condition and per-fold epoch; loss and printed
metrics use **three decimal places**. Raw CSV/JSON retain full precision.

After completion:

\`\`\`bash
cat runs/k7/e1_e2_v1/REPORT.md
cat runs/k7/e1_e2_v1/job_status.json

python scripts/analyze_z1_overnight.py \
  --input runs/k7/e1_e2_v1 \
  --output runs/k7/e1_e2_v1/DIAGNOSTICS.md
\`\`\`

The existing diagnostic script's named Z1 pair comparisons do not
match K7 variants (the pair table will have zero matched folds); its
per-condition and per-task tables are still valid. Interpret fitted
affine-oracle scores solely as **posthoc diagnostics**, not deployment
scores.

## Interpretation

The first required comparison is \`k7_e2\` vs \`k7_e1\`.
The second is \`k7_e2\` vs \`k7_e2_shuffle\`.
Check UCUI MSE, Pearson and source-mean ratio, *per held-out task and
per seed*. A convincing signal requires more than an aggregate gain
driven by a single pseudo-target. Do not use the official test sets
to develop or tune the method.

Failure at this stage does not invalidate the full K7 research idea:
these runs specifically test *relation-conditioned parameter generation*
with global frozen image features, not the later utility-calibrated
expert composition or spatially conditional knowledge alignment.

# K7-M2: two directed relation-specific hyper-adapters (one training run)

Goal: replace mean pooled text knowledge with **source-identity-preserving**
visual transfer. Fixed source labels: \`population_count\` and
\`food_accessibility\`. Fixed unseen DEV validation task:
\`education_accessibility\`. Fixed seed: **42**. No 6-fold sweep.

## Model

For each tile, build **two separate** directed relation tokens:
- Population→Education: [source identity, target description, directed
  mechanism card, is_self=0].
- Food→Education: analogous token, kept separate.

A task-conditioned text encoder projects each relation token, and a
hypernetwork generates **two separate low-rank visual adapters plus two
FiLM channel masks**. Two source-private visual experts then yield adapted
visual features. A query derived from the tile and held-task description
attends over these source-specific visual features; the weighted sum is
read out once as a normalized regression prediction. The model **never
averages the text vectors**, and does not simply average two expert output
predictions.

For train samples of Population and Food, each source has two paths:
- A neutral self-path (with explicit self-edge flag, no fabricated self
  relation text).
- The other-source→current-task directed relation card.

During training, the self-path is randomly removed on 50% of rows, so
the model must also learn cross-source transfer. The same two-path
architecture therefore gets supervision and attention gradients even
though the unseen Education indicator has no training labels.

This is an exploratory architectural mechanism, **not evidence that
the causal/spatial hypotheses in Qwen text are true**.

## GPU single run

\`\`\`bash
git pull --ff-only origin main
python -m unittest discover -s tests -p 'test_k7_m2.py' -v

mkdir -p runs/k7
python -u scripts/run_k7_m2.py \
  --source-csv /scratch/project_462001391/xiyanxin/Urban_experience/codes_BaseStation/city_indicator_splits/train.csv \
  --image-root /flash/project_462001391/xiyanxin/Urban_experience/output_files \
  --feature-dir runs/m3_features \
  --tensor-cache runs/k7/e1_e2_v1/_source_tensors.pt \
  --text-cache-dir runs/ktrans_text \
  --text-model Qwen/Qwen3-Embedding-0.6B \
  --seed 42 --train-limit 20000 --epochs 20 --batch-size 128 \
  --output runs/k7/m2_directed_fusion_seed42 \
  2>&1 | tee runs/k7/m2_console.log
\`\`\`

Reuses existing short Qwen vectors, the **four already encoded mechanism
cards** and the shared RemoteCLIP tensor cache. Only one new model is
trained. The pilot uses City×Indicator stratified source-only sampling,
fixed cities and fixed budgets, and runs in ROCm using PyTorch's
\`cuda\` interface. Existing M1 results are never overwritten.

Summary is two lines:
\`\`\`text
[k7-m2] SCUI MSE=... Pearson=... Ratio=... PopAttn=... FoodAttn=...
[k7-m2] UCUI MSE=... Pearson=... Ratio=... PopAttn=... FoodAttn=...
\`\`\`

Here attention averages describe which of the **two source-specific
visual pathways** was used; they do not establish causal explanation or
text correctness. Compare against M1 on the identical split, which had
SCUI MSE 1.142 / Pearson .821 / Ratio .572 and
UCUI MSE 2.598 / Pearson .614 / Ratio 1.204.

The model has not been benchmarked on the real held-out four-indicator
final-test data. Do not use official target files for this pilot.

# K7 E3: empirical transfer-utility calibrated routing

## One research question (single seed)

**Does Qwen-derived directed indicator relationship knowledge predict
the *actual* value of reusing a source-task adapter?**

The existing E1/E2 experiment found that relation-conditioned parameter
generation did not outperform shuffled relations. E3 tests a different
mechanism: first estimate real transfer reward on training-only indicators,
then fit a tiny reward router to relation knowledge, then reuse frozen
K7-E1 source-conditioned adapter pathways.

Only two conditions are evaluated:
- **k7_e3**: source-task knowledge + correctly paired relation descriptions
- **k7_e3_shuffle**: identical data, rewards and E1 weights, but rotate the
  source-to-relation text pairing.

The direct, zero-shot task adapter is always a third *internal option* with
fixed no-transfer utility of zero, not a separate trained experimental arm.
The two groups are evaluated **in the same process** so their inner-model
randomness and empirical reward labels are identical.

## Cross-fitting / leakage controls

For each of 6 source pseudo-held-out indicators:
1. Reconstruct the existing E1 outer split (same seed=42, same city fraction).
2. Read the existing E1 fold checkpoint; never refit the full adapter bank.
3. Split the current outer training cities again into inner-training cities
   and inner-reward-validation cities; no held-out indicator or held-out
   outer city appears in either.
4. Train **one** E1 inner model on only the five allowed training indicators
   in inner-training cities, for fixed 20 epochs.
5. Evaluate its source-conditioned adapters on other seen indicators in
   inner-validation cities. Reward is the reduction in *standardized target
   MSE* relative to the target's direct E1 adapter. These rewards are labels
   available only during this internal source-to-source cross-fitting.
6. Fit an L2-regularized 6-feature ridge utility predictor on the 20
   eligible directed source-to-source pairs (5 targets × 4 sources).
7. Predict rewards for candidate source→pseudo-heldout target pairs from
   Qwen relation vectors. Mix the **existing frozen E1 model** outputs
   according to a softmax of predicted utility, including a no-transfer
   direct-target adapter option.

Zero-shot output-scale moments are exactly the same short-embedding,
source-only prior as in E1. Source task relation hypotheses remain
unverified. No heldout target/city label is used for routing, scale fitting,
checkpoint selection or training.

E3 reports SCUI and UCUI only. UCSI involves five known indicators per
fold and would require separate task-conditioned router outputs, so this
diagnostic deliberately omits it rather than reporting a misleading result.

## Computational budget

**One seed (42), six inner model fits total, each 20 epochs.**
No repeated source-expert training and no new Qwen/RemoteCLIP inference.
Relation and short definition vectors are read from the existing caches.
The source data is read from the existing shared Tensor cache and placed
on MI250 GPU. Per-fold JSON is saved immediately; reruns skip completed
folds. Full CSV precision is retained; console numbers use three decimals.
This run is a fixed-hyperparameter development experiment, not tuning.

## One-command run (project root, inside MI250 allocation)

\`\`\`bash
git pull --ff-only origin main

python -m unittest discover -s tests -p 'test_k7_e3.py' -v
python -m unittest discover -s tests -p 'test_k7_hyper.py' -v

mkdir -p runs/k7
python -u scripts/run_k7_e3.py \
  --source-csv /scratch/project_462001391/xiyanxin/Urban_experience/codes_BaseStation/city_indicator_splits/train.csv \
  --image-root /flash/project_462001391/xiyanxin/Urban_experience/output_files \
  --feature-dir runs/m3_features \
  --tensor-cache runs/k7/e1_e2_v1/_source_tensors.pt \
  --e1-root runs/k7/e1_e2_v1/k7_e1_seed42 \
  --text-cache-dir runs/ktrans_text \
  --relation-cache-dir runs/ktrans_text \
  --epochs 20 --batch-size 128 --seed 42 \
  --output runs/k7/e3_seed42 \
  2>&1 | tee runs/k7/e3_seed42.log
\`\`\`

The command prints **two** final UCUI summary rows. Raw files:
\`runs/k7/e3_seed42/macro_metrics.csv\`,
\`runs/k7/e3_seed42/fold_metrics.csv\` and per-fold
\`*_metrics.json\`. To rerun after interruption, use the same command
without deleting these results.

The experiment is meaningful if **k7_e3** consistently outperforms
**k7_e3_shuffle**, ideally while also improving over the already-measured
K7-E1 baseline (seed42 UCUI: MSE 8.791, Pearson 0.439, Ratio 1.090).
One seed and six pseudo-task folds do not support formal significance
claims. Keep all true four-target CSVs entirely outside development.

If inner training city stratification does not retain every source task,
the script fails closed; do not silently mix outer held-out cities into
reward fitting.

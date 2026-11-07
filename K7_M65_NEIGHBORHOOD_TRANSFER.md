# K7 M6.5 — Neighborhood-aware City×Indicator Transfer

## Purpose

M6.3 full source-development UCUI achieved Pearson 0.6726, but
UCUI MSE 1.8520 remained worse than Visual-only 1.7026.
M6.4 posthoc source-only affine calibration increased UCUI MSE
to 1.9529. M6.5 **tests a different, prespecified spatial
representation hypothesis** — it does not attempt target-label
calibration or claim improvement before the run.

## Data and representation

Reuse exactly the existing `m62_full_seed42/full_manifest.json`,
frozen RemoteCLIP tensor cache and Qwen-VL `hidden.sqlite3`.
Construct label-free neighborhood features once from source-CSV
imagery using physical tiles keyed by `(city,image_name)`.

For each tile, use **up to 8 geographic nearest physical tiles
from the same city excluding itself**. Compute neighbors'
RemoteCLIP embedding mean, per-channel std and tile-versus-
neighborhood difference; append mean neighbor distance and
neighbor count fraction. Shape: `[N,3*D+2]`.

- No label or indicator ID is used to build neighbors.
- No city name is input to the trainable city model.
- Deduplicate same physical tile across indicator rows.
- **Transductive imagery context:** source-CSV images from all
  partitions can act as unlabeled neighbors. Their numeric
  labels never enter neighborhood construction. Not appropriate
  to call this a strictly inductive inference setup.
- No official test CSV used.

The new encoder processes neighborhood features with a
trainable projection and local-neighborhood gate, augments
the M6.3 City×Indicator query, and retains the two bounded
residual experts and frozen two-expert visual anchor.

## Experimental protocol

**Two independent stages, not UCUI-based hyperparameter tuning.**

**Dev stage:** within the ORIGINAL `train` split, reserve
20% of its cities at seed 42+65, holding complete cities out.
Train a **fresh** visual-only anchor using source Population/
Food labels from the remaining cities, then train the M6.5
neighborhood student on exactly those cities. Report **only**
Population and Food metrics on reserved source cities, with no
SCUI/UCSI/UCUI evaluation. This avoids the leakage that would
occur if a visual anchor pretrained on these held-dev cities
were used for city validation.

**Final stage:** use the original full-source Visual-only
anchor already trained on the complete original source
`train` split. Fit M6.5 on all of that split with the same
fixed k=8, epochs=20, batch256 and source-only loss schedule.
Then score source_val, SCUI, UCSI and UCUI in the final
epoch. Do not select weights on Education data.

The dev and final students are distinct fits; no checkpoint
transfer from dev. The final original source anchor is fixed,
not finetuned. Existing M6.3 results remain untouched.

**Important:** source-city dev here validates newly trained
source-only components on new cities within the original
training-city pool; the neighboring imagery is still accessible
without labels. It does not simulate an unseen indicator.

## Direct HPC commands

Run once, from repository root:

```bash
git pull --ff-only origin main

python -m unittest discover -s tests \
  -p 'test_k7_m65_neighborhood.py' -v
```

### 1 — Source city-development experiment

```bash
python -u scripts/run_k7_m65_full.py \
  --stage dev \
  --full-dir runs/k7/m62_full_seed42 \
  --reference-dir runs/k7/m62_full_models_seed42 \
  --neighbor-cache runs/k7/m65_neighbors_k8.pt \
  --output runs/k7/m65_neighborhood_dev_seed42 \
  --k 8 --epochs 20 --batch-size 256 --device cuda
```

The output contains `dev_visual_anchor.pt` and
`checkpoint_last.pt` for independent preemption-safe
resumption after anchor training. Resume by re-running the
**identical command**. If interrupted DURING the initial
visual-only anchor training, that incomplete anchor training
must restart; after the anchor is saved, it is reused.
The M6.5 student saves after every complete epoch.

Review `runs/k7/m65_neighborhood_dev_seed42/metrics.csv`:
Pop/Food source held-city MSE, Pearson and bias. This provides
the **source-only city generalization** evidence. The stage
does NOT print Education results.

### 2 — Final source-only full fit, independent output

After dev evidence is recorded and fixed model settings are
accepted using source-task evidence alone:

```bash
python -u scripts/run_k7_m65_full.py \
  --stage final \
  --full-dir runs/k7/m62_full_seed42 \
  --reference-dir runs/k7/m62_full_models_seed42 \
  --neighbor-cache runs/k7/m65_neighbors_k8.pt \
  --output runs/k7/m65_neighborhood_final_seed42 \
  --k 8 --epochs 20 --batch-size 256 --device cuda
```

This stage produces `m65.pt`, `metrics.csv`,
`config.json` and a resumable `checkpoint_last.pt`.
It uses **no** target task labels for fit, optimizer,
checkpoint or selection. Target Education labels are read
only during after-training descriptive scoring.

## Interpretation

Main outcome: held-source-city Pop/Food errors, then fixed
final UCUI Pearson/MSE relative to Visual-only 0.6281/1.7026
and M6.3 0.6726/1.8520. SCUI and per-source UCSI also matter.
The existing Education dev split has been repeatedly inspected,
so the final UCUI numbers are **development observations**, not
independent confirmation. Do not query official final test
until the model/protocol has been frozen.

M6.5 is NOT loss/compute-matched to M6.3 with batch256:
dev trains a fresh visual anchor, and the spatial encoder
adds parameters. The final comparison should be presented
as an architectural follow-up, not causal proof of neighbors.
A future clean ablation should match source fitting budget.

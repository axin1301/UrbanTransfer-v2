# K7 M6.2 FULL-SOURCE cross-city × cross-indicator study

**Status: implementation only, not executed on user's MI250.**

This is a full-scale **DEVELOPMENT** benchmark from the SOURCE
`city_indicator_splits/train.csv` — **NOT an official final-test
evaluation**. The three official file paths supplied by the user:

- `seen_city_test.csv`
- `unseen_city_seen_indicator_test.csv`
- `unseen_city_test.csv`

are deliberately **never opened or provided as CLI inputs** in
this stage. Keep them sealed until the architecture, training settings,
model-selection protocol and comparisons have been frozen. They
are not renamed or substituted for development data.

## Protocol

Fixed seed42, randomly held 25% of SOURCE cities (distinct from
the official test-city split). Of the seen-city Population/Food
rows, reserve 10% of **source image groups** per city for
source-only validation; **whole tiles** are grouped to avoid a
Population row of a tile in fit and Food row of the same tile
in source validation. Train on every remaining seen-city
Population/Food row — NO 128 limit.

Development partitions:

- `train`: all eligible seen-city source task rows after source val.
- `source_val`: source-only seen-city held-out image groups,
  never used for fitting/checkpoint choice in this fixed run.
- `SCUI`: **all** Education rows from seen source cities.
- `UCSI`: **all** Population/Food rows from held source cities.
- `UCUI`: **all** Education rows from held source cities.

The SOURCE CSV may also contain other urban indicators. Those
are **excluded** from the M6.2 two-expert experiment, not silently
converted into training data.

Models:
- `visual_only`: RemoteCLIP+geography, two experts; no text.
- `text_router`: RemoteCLIP+geography plus frozen Qwen3 task
  embeddings, two experts; no Qwen-VL multimodal representation.
- `m62`: frozen Qwen2.5-VL image/definition/relation tokens,
  task-query Cross-Attention and two source experts.

**All three are newly trained on exactly the same full source
training rows**, fixed epochs20, batch128, hidden128,
AdamW lr1e-3. Standardization is computed exclusively from
Pop/Food source **training** labels, including Education prediction
scale derived from source-task Qwen3 text similarities. There is
no Education label in the optimizer, loss, scaling, checkpoint
selection or training control flow.

M6.2 also has source task classification (weight 0.2) and an
unlabeled Education-description/source-image positive alignment
(weight 0.05), as in the earlier pilot. Both controls have
matching-supervised-source-expert loss weight 0.3 but not
those two auxiliary losses. They are architecture comparisons,
NOT parameter-/objective-matched single-factor ablations.

The **source-val** scores and all SCUI/UCSI/UCUI scores are
computed after the fixed 20th epoch and used only for reporting.
The previously viewed 64-row pseudo-dev results cannot be
treated as fresh independent confirmation. The full source
development benchmark may share images/cities with them.
Official test evaluation remains future one-shot work.

## Full-data commands on MI250 / ROCm

Use your cloned repo
`/pfs/lustrep4/scratch/project_462001163/xiyanxin/UrbanMoE/UrbanTransfer-v2`.

### A. Pull and test (CPU)

```bash
git pull --ff-only origin main
python -m unittest discover -s tests \
  -p 'test_k7_m62_full.py' -v
```

### B. Create full immutable manifest and estimate cost

```bash
python -u scripts/prepare_k7_m62_full.py \
  --train-csv /scratch/project_462001391/xiyanxin/Urban_experience/codes_BaseStation/city_indicator_splits/train.csv \
  --image-root /flash/project_462001391/xiyanxin/Urban_experience/output_files \
  --feature-dir runs/m3_features \
  --tensor-cache runs/k7/e1_e2_v1/_source_tensors.pt \
  --run-dir runs/k7/m62_full_seed42 \
  --seed 42 --city-fraction 0.25 --source-val-fraction 0.10
```

This refuses to overwrite an existing
`runs/k7/m62_full_seed42/full_manifest.json`. To intentionally
choose a different split, use a NEW directory.

### C. Exact unique-key cache audit — no GPU/MLLM loaded

```bash
python -u scripts/cache_k7_m62_full.py \
  --run-dir runs/k7/m62_full_seed42 \\
  --reuse-sqlite \\
    runs/k7/m61_structured_seed42/hidden.sqlite3 \\
    runs/k7/m62_fixed_comparison_seed42/ucsi_hidden.sqlite3 \\
  --audit-only
```

It prints `requests, unique_keys, cached, reusable, missing, raw_GiB`. Reuse paths must exist; omit any absent old SQLite file. Audit mode never copies hidden arrays or loads Qwen-VL.
Each Qwen2.5-VL cached sample costs **256 KiB** raw fp16
(hidden shape 64×2048). Exact unique content keys matter because
many indicator rows may refer to the same physical satellite tile.

**Stop here first and inspect `missing` and `raw_GiB`** before
allocating a long GPU job. Resource/time cost depends strongly on
the actual unique image count and MI250 performance; no runtime
has been measured by the assistant.

### D. Encode only missing unique keys (MI250)

```bash
python -u scripts/cache_k7_m62_full.py \
  --run-dir runs/k7/m62_full_seed42 \
  --reuse-sqlite \
    runs/k7/m61_structured_seed42/hidden.sqlite3 \
    runs/k7/m62_fixed_comparison_seed42/ucsi_hidden.sqlite3
```

Frozen Qwen2.5-VL-3B Instruct, BF16, eager attention; M6.1
[32 image,16 definition,16 relation] tokens. Uses a new
resumable content-addressed SQLite DB at
`runs/k7/m62_full_seed42/hidden.sqlite3`. Already complete
keys are skipped on resume. Original M6.1 and M6.2 caches
and checkpoints are untouched.

### E. Train three new full-data models / score development

```bash
python -u scripts/run_k7_m62_full.py \
  --run-dir runs/k7/m62_full_seed42 \
  --output runs/k7/m62_full_models_seed42 \
  --text-cache-dir runs/ktrans_text \
  --seed 42 --epochs 20 --batch-size 128 \
  --device cuda
```

Requires existing frozen Qwen3-Embedding-0.6B short-profile
vectors in `runs/ktrans_text`, the shared RemoteCLIP feature
tensor cache and successful completion of step D.

Student MLLM token batches are read from SQLite on demand
with a small CPU LRU; it does not materialize all MLLM
hidden states in GPU RAM. The new output directory MUST NOT
exist and will contain `visual_only.pt`,
`text_router.pt`, `m62.pt`,
`config.json`, and `metrics.csv`.

Metrics include source_val, SCUI, UCSI, UCUI, plus
`UCSI_population_count` and `UCSI_food_accessibility`
separately: MSE(log1p), MAE(log1p), Pearson, source-only
constant-baseline MSE ratio, route weights, predicted and
truth means.

## Key scientific cautions

- Do not compare absolute MSE to pilot directly as if the same sample
  or training data volume: splits/data volumes differ.
- A city split within the source CSV is **not** the official
  held-city split and cannot be represented as such.
- SCUI/UCUI labels are read for metrics **only after all 3 fits**.
- No source dev labels are used to select epochs in this first fixed
  run. Fixed epoch20 is predeclared, not necessarily optimal at scale.
- UCSI *pooled* Pearson can be inflated by differences in source-task
  means. Always inspect per-source UCSI Pearson/MSE.
- If M6.2 wins, that still does NOT prove the gains arise
  specifically from multimodal attention: its auxiliary objective
  differs from the text and visual controls.
- Once the protocol is stable, final official test should be run
  exactly once on a pre-frozen selected model. No official test
  CSV input is currently exposed in these scripts.

# K7 M6.1: structured MLLM hidden-state routing (single pilot)

M6 diagnostic showed Pop/Education and Food/Education cosine similarities
0.982/0.989, static teacher Pop weight ~0.493 (std 0.002), student
routing ~0.500 (std 0.001), and near-zero prediction changes from
mismatching the 8 final-prompt tokens. The original M6 source-supervised
forward also **masked out the other expert**, so supervised loss could
not train the mixture router. M6.1 addresses both issues.

## What actually changes

**Frozen Qwen2.5-VL-3B-Instruct forward, NO generated output:**
satellite image + target indicator definition + permitted directed
source-to-target transfer hypotheses. Locate real token spans in
processor-expanded `input_ids`, extract last-layer 2048-D hidden
vectors at explicit visual, indicator and relation sections:

- 32 actual image placeholder token positions.
- 16 indicator definition/measurement token positions.
- 16 relation-mechanism token positions.

The output per image/indicator is `[64,2048]`. These tokens are
deterministically subsampled from their true segment positions, **not**
the final 8 prompt tokens. Prompt boundaries must be located exactly,
and the script fails rather than quietly extracting incorrect tokens.

The model uses a learned-query cross-attention readout (query **cannot
copy RemoteCLIP inputs directly**); structured segment embeddings
distinguish visual/definition/relation chunks. The resulting knowledge
representation controls channel gates and two-source expert weights.
Source regression uses **both experts in its main loss**. Auxiliary
source-only regression directly trains the matching expert.

**Education target text DOES participate in training** on the *same
Population/Food source-training image tiles*, never Education target
labels or Education eval images. We train the Education-conditioned
MLLM cross-attention representation to align via in-batch contrastive
learning with each tile's **frozen RemoteCLIP image vector**, without
teacher-generated Population/Food pseudo-weights. This is image-text
alignment, not regression-label distillation; it may still be noisy and
may not guarantee learned transfer knowledge. MLLM and RemoteCLIP
encoders remain frozen.

Loss: source regression + 0.3 direct expert loss + 0.25
target-image contrastive alignment. All weights are fixed before
inspecting Education results. 20 epochs, seed42, **one trained student**.

## MI250 / ROCm steps (repository root)

Before first MLLM precomputation check your environment has a recent
Transformers supporting Qwen2.5-VL. Hugging Face weights must be
accessible or already cached. The code explicitly uses eager attention
for ROCm compatibility, and reserves a new M6.1 directory. Existing M6
8-token caches cannot be reused because the prompts/token extraction
are different. **Old M1–M6 files are never overwritten.**

```bash
git pull --ff-only origin main
python -m unittest discover -s tests -p 'test_k7_m61_structured_router.py' -v

python -u scripts/cache_k7_m61_structured.py \
  --source-csv /scratch/project_462001391/xiyanxin/Urban_experience/codes_BaseStation/city_indicator_splits/train.csv \
  --image-root /flash/project_462001391/xiyanxin/Urban_experience/output_files \
  --feature-dir runs/m3_features \
  --tensor-cache runs/k7/e1_e2_v1/_source_tensors.pt \
  --model-id Qwen/Qwen2.5-VL-3B-Instruct \
  --train-limit 128 --eval-limit 64 --seed 42 \
  --visual-tokens 32 --definition-tokens 16 --relation-tokens 16 \
  --max-pixels 100352 \
  --run-dir runs/k7/m61_structured_seed42 \
  2>&1 | tee runs/k7/m61_structured_cache.log
```

The cache task list is exactly 128 source rows × 3 task prompts,
plus 64+64 Education pseudo-development rows (with any repeated
image/task combinations deduplicated). It writes
`manifest.json` and a resumable `hidden.sqlite3`.

When cache is complete:

```bash
python -u scripts/run_k7_m61_student.py \
  --run-dir runs/k7/m61_structured_seed42 \
  --text-cache-dir runs/ktrans_text \
  --seed 42 --epochs 20 --batch-size 64 \
  2>&1 | tee runs/k7/m61_structured_console.log
```

The trained student is saved once as `model.pt`; metrics saved as
`metrics.csv`, config as `config.json`. It refuses to overwrite
training outputs from an existing run.

Prints SCUI/UCUI pseudo-development metrics: MSE, Pearson, source
mean-baseline ratio, PopWt/FoodWt, **PopWtStd** and
**MismatchChange** (mean absolute log1p prediction change after
cyclically mismatching Education MLLM tokens, keeping each row's
RemoteCLIP/geography input fixed). Output is not a final test:
`seen_city_test.csv`, `unseen_city_seen_indicator_test.csv` and
`unseen_city_test.csv` are never read.

### Interpret carefully

The M6.1 subset uses the same seed42 row selection and small
128+64+64 budget as M6; unlike M1–M5 full-size pseudo-dev metrics,
direct comparisons to M6 are more defensible but still noisy.

Nonzero PopWtStd and MismatchChange would indicate that the student
is **sensitive** to MLLM features. They do **not** establish beneficial,
causal, or correct knowledge transfer. A new frozen-checkpoint
comparison with matched wrong definitions and visual mismatches is
required before making a strong semantic claim.

The visual and relation token groups preserve raw Qwen hidden
vectors; they do **not** have guaranteed human-level interpretability,
nor do hidden-state cosines estimate real source transfer utility.

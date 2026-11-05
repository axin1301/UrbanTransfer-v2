# K7-M5 concept-grounded LoRA rank distillation

ONE model, ONE seed=42, source-only training and frozen text counterfactuals.

## Motivation

M3 routed nearly all LoRA mass to Population (0.998), and M4 faithfully
copied a static Qwen teacher source mixture (Population 0.386; Food 0.614).
M5 does not distill any source-mixture probabilities. Instead, a frozen
RemoteCLIP image/text alignment supplies weak visual concept supervision
for LoRA ranks, and Qwen relation-to-concept similarities teach which
rank groups to transfer.

**This pilot is NOT a claim that a particular visual concept can be
recognized perfectly from one satellite tile, or that Qwen provides
correct causal transfer knowledge.** Both supervision channels are noisy.

### Mechanism

The 8 LoRA ranks are paired into four predeclared concept groups:

- 0–1 residential/housing patterns
- 2–3 dense built-up settlement patterns
- 4–5 roads and intersections
- 6–7 commercial and service built environment

For 4 short concept phrases:
1. Frozen RemoteCLIP TEXT encoder produces 4 vectors in the same space
   as cached RemoteCLIP IMAGE features. Image/concept cosine scores on
   **source training images only** are used to weakly supervise
   source LoRA rank activations via a sigmoid MSE alignment.
2. Frozen Qwen3-Embedding-0.6B embeds the corresponding four concept
   descriptions in Qwen space. Cosine similarity between each Qwen
   Population→Education or Food→Education directed relation card and
   the four concept embeddings produces a per-source **concept/rank**
   teacher distribution. Target Education text participates *during*
   optimization, without any Education labels.
3. Student LoRA rank gates are matched to the relation-to-concept
   distribution using KL. The source LoRA matrices are trained with
   source indicator regression and source SELF supervision.

There is **NO learned 2-source mixing router** in M5. Both allowed
source paths receive equal coefficients (0.5/0.5), and relation
knowledge affects individual LoRA **rank gates**. This deliberately
removes the M3/M4 source-level routing collapse and separates it from
which LoRA directions are active. The output `PopWt/FoodWt` columns
will consequently remain exactly 0.500/0.500 by construction:
do not interpret them as learned or as evidence of successful routing.

The model uses the identical frozen RemoteCLIP tile tensor and
geographic context (longitude, latitude, area, constant); the code
only encodes EIGHT short concept texts (4 Qwen + 4 RemoteCLIP) on first
run and caches them. RemoteCLIP's own text tokenizer/weights are used
for text/visual alignment; **Qwen and RemoteCLIP embeddings are not
mixed in the same vector space**.

M5 does **not** fine-tune either foundation model. The mechanism cards
are the existing curated/hypothesized four directed relation cards, not
LLM-authored verified scientific facts.

## Run (MI250 GPU allocation, repo root)

```bash
git pull --ff-only origin main
python -m unittest discover -s tests -p 'test_k7_m5_concept_lora.py' -v

mkdir -p runs/k7
python -u scripts/run_k7_m5_concept_lora.py \
  --source-csv /scratch/project_462001391/xiyanxin/Urban_experience/codes_BaseStation/city_indicator_splits/train.csv \
  --image-root /flash/project_462001391/xiyanxin/Urban_experience/output_files \
  --feature-dir runs/m3_features \
  --tensor-cache runs/k7/e1_e2_v1/_source_tensors.pt \
  --text-cache-dir runs/ktrans_text \
  --text-model Qwen/Qwen3-Embedding-0.6B \
  --seed 42 --epochs 20 --train-limit 20000 --batch-size 128 --rank 8 \
  --output runs/k7/m5_concept_lora_seed42 \
  2>&1 | tee runs/k7/m5_concept_lora_console.log
```

Requires existing GPU open_clip / huggingface_hub packages (already
used when RemoteCLIP features were first precomputed). Frozen
RemoteCLIP checkpoint may need to be present in Hugging Face cache.
The first text-encoding run requires checkpoint access if not cached.
If initial concept anchor encoding fails, no model training starts.

All prior M1–M4 run directories and inputs are unchanged. A completed
M5 output path cannot be overwritten. M5 saves metrics.csv, model.pt,
and config.json in the M5 run directory.

### What to report

The console first prints the Qwen teacher's 4 concept probabilities for
each of Population and Food, then exactly six summary metrics:
SCUI / UCUI × correct / swapped / generic.

The SCUI/UCUI subsets are **pseudo-development** partitions from
source `train.csv`. The official final-test CSVs must not be read.

If correct relations outperform swapped/generic with stable Pearson
and MSE, that is preliminary evidence of task-specific text knowledge.
Because correct target text participates in KD whereas swapped/generic
do not, these are still **off-training-distribution** interventions;
they cannot alone establish semantic causality. If not, stop
escalating architecture complexity without stronger knowledge
supervision or additional independent source tasks.

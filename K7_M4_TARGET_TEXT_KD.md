# K7-M4: target-description-aware Qwen text distillation (single run)

## Research question

Can giving the model the **unseen Education task definition and the two
directed source-to-Education mechanism texts during training**, without
any Education labels, prevent the M3 source router collapsing to Population
and make learned LoRA composition more knowledge-dependent?

**Teacher:** a frozen Qwen3-Embedding-0.6B text model (cached embeddings),
NOT an instruction-generation LLM, and not a fine-tuned model. Teacher
computes a bounded soft allocation over the two sources from cosine
similarities among target description, source descriptions, and their
respective directed mechanism cards:

    score(source) = 0.7 cosine(directed_relation, target_description)
                  + 0.3 cosine(source_description, target_description)
    teacher_probs = 0.8 * softmax(score/0.15) + 0.1

These coefficients are **fixed pre-declared heuristics**, not measured
true knowledge utility or validated visual concepts. The teacher cannot
produce regression targets or verified rank-to-concept alignment.

**Student:** identical source-specific rank-gated LoRA architecture to
M3. Each supervised source batch supplies (i) source regression labels,
(ii) task-specific self LoRA training, and (iii) a separate forward pass
of those SAME source visual tiles with **Education description +
Population→Education and Food→Education edge tokens**. The latter is
optimized using KL(teacher || student attention) with fixed weight 0.5.
No Education images, Education labels, held-city tiles, or Education
target moments enter parameter updates. Geography is retained.

After this ONE fit, freeze the model and infer on internal source-only
SCUI/UCUI pseudo-development splits with correct/swapped/generic
texts, as in M3. This is not an independent final test.

## Launch on ROCm MI250 (repo root)

```bash
git pull --ff-only origin main
python -m unittest discover -s tests -p 'test_k7_m4_target_kd.py' -v

mkdir -p runs/k7
python -u scripts/run_k7_m4_target_kd.py \
  --source-csv /scratch/project_462001391/xiyanxin/Urban_experience/codes_BaseStation/city_indicator_splits/train.csv \
  --image-root /flash/project_462001391/xiyanxin/Urban_experience/output_files \
  --feature-dir runs/m3_features \
  --tensor-cache runs/k7/e1_e2_v1/_source_tensors.pt \
  --text-cache-dir runs/ktrans_text \
  --text-model Qwen/Qwen3-Embedding-0.6B \
  --seed 42 --epochs 20 --train-limit 20000 --batch-size 128 \
  --output runs/k7/m4_target_text_kd_seed42 \
  2>&1 | tee runs/k7/m4_target_text_kd_console.log
```

Fixed budget: **one model, 20 epochs, seed 42**. All cached inputs reused.
Refuses to overwrite an existing M4 output directory. Saves
`model.pt`, `config.json` (including teacher probabilities), and
`metrics.csv`. The console shows six rows for SCUI/UCUI and
correct/swapped/generic, plus the teacher's two probabilities.

## What evidence would mean

- **Less 0.998/0.002 collapse:** student has absorbed teacher routing
  preference. This is *not* proof either source actually helps prediction.
- **Correct better than swapped/generic:** preliminary relation text
  sensitivity, though the correct token was specifically trained into
  the teacher KL target and swapped/generic are off-distribution. Such
  results **cannot independently prove true semantic/causal knowledge**.
- **MSE/Pearson changes:** useful descriptive comparisons but not the
  primary M4 research claim.

The teacher distribution is task-level, not image-level: forcing student
probabilities toward it may suppress beneficial image-dependent routing.
The choice of temperature, KD weight and coefficients are fixed before
seeing Education validation results. Avoid retroactive hyperparameter
search. A robust knowledge-effect claim would ultimately require matched
training controls and a truly untouched evaluation indicator.

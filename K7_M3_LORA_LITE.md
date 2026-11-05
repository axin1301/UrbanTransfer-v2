# K7-M3 Lite: relation-gated source LoRA composition

**Exactly one MI250 training run, fixed seed=42, two supervised indicators.**

- Train: population_count, food_accessibility, source seen cities.
- DEV-only validation: education_accessibility (SCUI and UCUI from source train.csv).
- Reuse frozen RemoteCLIP shared tensor cache, Qwen short semantics, and
  four already embedded directed M1 relation cards.
- Source-only moments inferred exactly as in M1 and M2.
- Existing M1/M2 runs are not changed.

## Architecture

A single network contains two separately parameterized low-rank updates:
Population `B_pop A_pop` and Food `B_food A_food`. Both are learned in
**one** optimizer run, with task-specific SELF path supervision and source
cross-transfer training (self dropout). For Education inference, the
Population→Education and Food→Education Qwen embeddings are **not averaged**.
They each feed a directed edge encoder with source semantics, target
semantics and self-edge flag; the edge encoder predicts a vector of LoRA
rank gates. An image/target conditioned query assigns per-tile mixture
weights. The shared trunk receives RemoteCLIP + four geographic context
features (longitude, latitude, area, constant). The final adapted feature:

    h' = h + gain/r * sum_s alpha_s(image,target,relation)
             * B_s diag(g_s(source,target,relation)) A_s h

The readout and target-task semantic bias are shared. Source output heads
are **not** simply copied to the target. Zero Education labels are used
in parameter updates. The relation text's meaning is **not guaranteed**
to align with learned ranks; that is the hypothesis being tested.

## One command (from repository root within MI250 GPU allocation)

```bash
git pull --ff-only origin main
python -m unittest discover -s tests -p 'test_k7_m3_lite.py' -v

mkdir -p runs/k7
python -u scripts/run_k7_m3_lite.py \
  --source-csv /scratch/project_462001391/xiyanxin/Urban_experience/codes_BaseStation/city_indicator_splits/train.csv \
  --image-root /flash/project_462001391/xiyanxin/Urban_experience/output_files \
  --feature-dir runs/m3_features \
  --tensor-cache runs/k7/e1_e2_v1/_source_tensors.pt \
  --text-cache-dir runs/ktrans_text \
  --text-model Qwen/Qwen3-Embedding-0.6B \
  --seed 42 --train-limit 20000 --epochs 20 --batch-size 128 \
  --output runs/k7/m3_lora_seed42 \
  2>&1 | tee runs/k7/m3_lora_console.log
```

The script is GPU-only (ROCm uses the `cuda` PyTorch interface); it will
not silently train on CPU. At the end it shows six short rows:
SCUI and UCUI × correct/swapped/generic. No extra model fits are made;
the generic relation embeddings are read from the existing M2 audit cache,
or encoded once using frozen Qwen if absent. Outputs:
`runs/k7/m3_lora_seed42/metrics.csv`,
`model.pt`, and `config.json`.

## Interpretation

Existing identical-split baselines:
- M1: SCUI 1.142 MSE / .821 Pearson; UCUI 2.598 MSE / .614 Pearson.
- M2: SCUI 1.273 MSE / .788 Pearson; UCUI 2.407 MSE / .609 Pearson.

If M3 improves the main UCUI measure this suggests a potentially
useful architecture, **not proof of relation knowledge effectiveness**.
Separately, correct texts outperforming swapped and generic within the
same frozen M3 checkpoint would provide preliminary evidence that
the precise directed knowledge matters. Because generic/swapped texts
are out-of-training-distribution, a stronger claim eventually requires
training-time controls with matched distribution and independent tasks.

This is source-only DEV; do NOT run official final-test CSVs here.

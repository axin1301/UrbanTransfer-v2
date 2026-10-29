# Milestone 1 — Qwen3-VL + UrbanOpp train/evaluation

This is a research prototype, not a validated MI250X benchmark. The published code runs a frozen Qwen3-VL multimodal provider online for every tile–task example. A separate trainable CNN and latent Top-K expert predictor minimize MSE over source-task labels.

## Environment

Use your existing ROCm PyTorch environment if compatible. Do **not** indiscriminately reinstall torch, which can replace the ROCm build.
```bash
python scripts/check_existing_env.py
python scripts/rocm_check.py
python -m unittest discover -s tests -v
```
Install only needed additional dependencies (Transformers >= 4.57, Accelerate). Qwen model: `Qwen/Qwen3-VL-2B-Instruct`. The script expects GPU for actual Qwen. Using `--provider mock` is for smoke tests and **not** scientific experiments.

## Synthetic CPU smoke test

```bash
python scripts/make_smoke_data.py --out smoke_data
python train_m1.py --train-csv smoke_data/train.csv \
  --scui-csv smoke_data/scui.csv --ucsi-csv smoke_data/ucsi.csv \
  --ucui-csv smoke_data/ucui.csv --image-root smoke_data/images \
  --provider mock --epochs 1 --batch-size 2 --limit 4 --output runs/smoke
```

## Real data

UrbanOpp split CSVs should contain: `city`, `image_name`, `indicator_column`, `indicator_value_log1p`, `patch_lon`, `patch_lat`, and `patch_area_km2`. The loader supports an optional `image_path` CSV column.

```bash
python train_m1.py --train-csv /data/train.csv \
  --scui-csv /data/scui.csv --ucsi-csv /data/ucsi.csv \
  --ucui-csv /data/ucui.csv --image-column image_path \
  --provider qwen3vl --model-id Qwen/Qwen3-VL-2B-Instruct \
  --epochs 1 --batch-size 2 --limit 12 --output runs/qwen_m1
```

If image paths are not stored in CSV, use `--image-root /data/images`. No real satellite image files or Qwen weights are distributed in this repository.

## Protocol

The original distributed CSV files contain six source and four target indicators, different from the paper's stated 5+3. Train split city/indicator flags are checked, and SCUI/UCUI task overlap and UCSI/UCUI city overlap are forbidden.

## Limitations

- Qwen3-VL GPU generation on AMD MI250X is **not yet tested**.
- The initial visual encoder is a small CNN, not the pretrained RemoteCLIP backbone.
- Task representations are deterministic hash vectors, not pretrained text semantics.
- Online MLLM inference runs once per example; no batch optimization or knowledge cache.
- Every result produced by `--provider mock` is synthetic-only and not publication evidence.
- An empty/missing image path causes an error for actual Qwen inference.

Outputs: `last.pt`, `run_metadata.json`, `metrics.json`.


## M1.2 — Latent Qwen3-VL knowledge (recommended)

**Default provider is now `qwen3vl-latent`.** Instead of generating text/JSON, Qwen3-VL makes a single frozen multimodal forward pass on the real image, region and target task prompt. The resulting pooled hidden state passes through a **trainable projection** into the expert-routing knowledge space. This removes the brittle JSON schema constraint, and no output is interpreted as a calibrated numeric relevance score.

Use the same CLI as M1.1, but prefer:

```bash
python train_m1.py --train-csv /path/train.csv --scui-csv /path/scui.csv \
  --ucsi-csv /path/ucsi.csv --ucui-csv /path/ucui.csv \
  --image-root /flash/project_462001391/xiyanxin/Urban_experience/output_files \
  --provider qwen3vl-latent --model-id Qwen/Qwen3-VL-2B-Instruct \
  --limit 12 --epochs 1 --batch-size 2 --output runs/qwen_latent_smoke
```

`--provider qwen3vl` retains explicit JSON mode only for ablation. `--provider mock` is synthetic.

Test the new integration without downloading weights:

```bash
python -m unittest discover -s tests -p "test_latent.py" -v
```

**Validation status:** code committed, CPU-only mock latent tests provided, but not tested against actual Qwen3-VL model weights or MI250X. Online MLLM remains frozen, and its hidden-state projection, vision CNN, experts and regressors are trainable.

**Important scientific limitation:** these hidden states are not provably an explicit representation of geographic *knowledge*, and the current architecture does not guarantee transferability decisions. Follow-up controls should compare against random/frozen task-conditioned representations and no-LLM baselines, and add specialist/reward-supervised routing.

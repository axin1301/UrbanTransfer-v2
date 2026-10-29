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

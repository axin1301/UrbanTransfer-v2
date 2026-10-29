# UrbanTransfer v2

Independent research code for cross-city/cross-indicator urban opportunity prediction.

**Milestone 1.1:** online Qwen3-VL knowledge-guided expert routing, image-backed UrbanOpp CSV ingestion, train/eval, explicit image paths, ROCm checks.

This is a **prototype**: CPU synthetic smoke tests have passed. Real Qwen3-VL inference with AMD MI250X and UrbanOpp imagery remains unverified. The visual encoder is a small CNN and the task text vectorizer is a deterministic hash baseline—not publication-ready RemoteCLIP features.

## Quick start

Reuse your existing ROCm PyTorch environment. Run `python scripts/check_existing_env.py`, install only missing extras from `requirements-m1.txt` without blindly overwriting ROCm torch.

Test: `python -m unittest discover -s tests -v`

Run smoke data: `python scripts/make_smoke_data.py`, then see `MILESTONE1.md`.

Training with explicit image paths:

```bash
python train_m1.py --train-csv /data/train.csv --scui-csv /data/scui.csv \
  --ucsi-csv /data/ucsi.csv --ucui-csv /data/ucui.csv \
  --image-column image_path --provider qwen3vl \
  --model-id Qwen/Qwen3-VL-2B-Instruct --limit 12 --epochs 1
```

Use `--image-root /data/images` to resolve paths relative to a root. Never commit images, labels, checkpoints or tokens.

# Explicit image paths and existing ROCm environment

## Image path modes

**Explicit per-row path (preferred):** add an `image_path` column in each split CSV and pass `--image-column image_path`. Absolute paths are used as-is. Relative paths resolve against `--image-root` (if present), then against the directory containing the CSV. Missing specified paths raise an error.

**Image root:** without `--image-column`, pass `--image-root /path/to/images`; dataset searches `image_name`, `city/image_name`, and `city_query/image_name`.

The image path is an input file name, not a prediction feature. Do not submit the labels or image data to GitHub.

## Environment

Reuse existing AMD MI250X / ROCm PyTorch rather than reinstalling torch. Run:
```bash
python scripts/check_existing_env.py
python scripts/rocm_check.py
```
Qwen3-VL requires a compatible Transformers build; install only missing dependencies after checking. No MI250X runtime validation has been performed in this delivery.

## Test
```bash
python -m unittest discover -s tests -v
```

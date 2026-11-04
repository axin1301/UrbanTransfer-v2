# K7 Pilot: Frozen Qwen3-VL-8B task-conditioned visual knowledge

**Model:** `Qwen/Qwen3-VL-8B-Instruct` (BF16, SDPA attention).
Official docs: https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct
Minimum `transformers>=4.57.0` (per upstream Qwen3-VL docs).
Do not assume vLLM optimized ROCm kernels for MI300 run on MI250X.

We compare, for **the exact same aerial tile**:
- `image_only`: image plus generic visual observation prompt;
- `education_accessibility`: image plus structured task definition;
- `nighttime_light_intensity`: image plus structured task definition;
- `digital_total_cell_intensity`: image plus structured task definition.

All prompts request observable evidence only (roads, vegetation, built-up
morphology, etc.), with conservative task relevance and uncertainties.
The VLM is **never given indicator labels**, and is prohibited from
guessing true urban indicator values. We deliberately cache transparent
generated evidence for this feasibility stage, not arbitrary internal
hidden states or model predictions.

The script reads only `city`, `image_name` and optional
`city_query` columns from the K4 checkpoint's CSV. No indicator
value column is loaded. The text profiles are predeclared and sourced
from the current `urban_transfer/task_profiles.py`.

## Environment check on allocated MI250X GPU

```bash
git pull origin main
python - <<'PY'
import torch,transformers
print('torch:',torch.__version__,'HIP:',torch.version.hip)
print('transformers:',transformers.__version__)
print('GPU:',torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NOT AVAILABLE')
from transformers import Qwen3VLForConditionalGeneration,AutoProcessor
print('Qwen3-VL imports: OK')
PY

python -m py_compile scripts/run_ktrans_k7_vlm_pilot.py scripts/diagnose_ktrans_k7_vlm.py
python -m unittest discover -s tests -p 'test_ktrans_k7_vlm.py' -v
```

If Qwen3-VL imports are unavailable, install `transformers>=4.57.0`
and `accelerate` in **the same Python environment**. Prefer a
separate virtualenv or verify old K4/K5 environments still work.
Do not install flash-attn or assume high-end AMD-specific kernels.

### Smoke test (2 tiles × 4 conditions = 8 calls)

```bash
python -u scripts/run_ktrans_k7_vlm_pilot.py \
  --checkpoint runs/k4_focused/structured_education_accessibility/best.pt \
  --n-tiles 2 \
  --max-pixels 262144 \
  --max-new-tokens 320 \
  --output runs/k7_pilot/smoke.jsonl
```

### Main pilot (8 tiles × 4 conditions = 32 calls)

```bash
python -u scripts/run_ktrans_k7_vlm_pilot.py \
  --checkpoint runs/k4_focused/structured_education_accessibility/best.pt \
  --n-tiles 8 \
  --max-pixels 262144 \
  --max-new-tokens 320 \
  --output runs/k7_pilot/vlm_evidence.jsonl

python -u scripts/diagnose_ktrans_k7_vlm.py \
  --input runs/k7_pilot/vlm_evidence.jsonl \
  --output runs/k7_pilot/summary.csv
```

**Resume:** completed successfully parsed cache keys are skipped.
An interrupted JSONL retains completed lines; the same command
retries missing/invalid outputs. Cache identity contains tile path,
task, VLM model ID, prompt text and max_pixels. Use the same settings
for a valid resume. Beware concurrent processes writing to the same
JSONL (not supported). Other than the cache path, outputs are
entirely separate from K4/K5-A/K6 checkpoints.

Output `summary.csv` reports per-tile valid JSON conditions,
variation in task-relevance scores across task descriptions,
variation in visual cue scores and number of distinct scene
descriptions. **Task-selective variation is necessary but not sufficient
for useful transfer**: the prompt itself can induce relevance changes,
without the model having attended to the image. Subsequent tests
should include image-swapping and task-swapping controls and source-only
inner validation, before training K7 expert routing.

This pilot does NOT establish whether VLM improves predictions.
It is the representation feasibility/protocol phase, to be followed by
explicit controls: same image/different tasks, same task/different
images, no-image prompt, and aligned input/target city checks.

Image patch and indicator label alignment, collection years, and
structured text measurement definitions still require manual
data-generation methodology verification.

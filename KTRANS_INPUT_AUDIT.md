# Input and task-text audit for K-TransMoE

Read-only diagnostic. No training or target-label calibration.
Before changing K6.2 losses, inspect input provenance and the structure
of task embeddings. This checks:
- Frozen cached RemoteCLIP image features for nonfinite values, norm,
  dimensionality, variance, and duplicate rounded vectors;
- Geographic context distributions (longitude / 180, latitude / 90,
  log1p(area km²) / 10, constant intercept 1);
- Source-only log1p label ranges and train/heldout-city separation;
- Cached structured Qwen task vectors, nearest-neighbor similarities,
  semantic source route, and exact agreement with checkpoint routing;
- Duplicate (city, image_name, indicator) row identities.

The script never uses outer SCUI/UCSI/UCUI labels. It still loads
checkpoint-associated image caches and metadata, so it should run
on the same machine as the existing experiments.

## Run on three previously used folds

```bash
git pull origin main
python -m py_compile scripts/audit_ktrans_inputs.py
python -m unittest discover -s tests -p 'test_ktrans_input_audit.py' -v

python -u scripts/audit_ktrans_inputs.py \
  --checkpoint runs/k4_focused/structured_education_accessibility/best.pt \
  --output runs/input_audit_education

python -u scripts/audit_ktrans_inputs.py \
  --checkpoint runs/k4_focused/structured_nighttime_light_intensity/best.pt \
  --output runs/input_audit_nighttime

python -u scripts/audit_ktrans_inputs.py \
  --checkpoint runs/k4_focused/structured_digital_total_cell_intensity/best.pt \
  --output runs/input_audit_digital_total_cell
```

Each output directory has:
- `audit.json`: warnings, provenance and caveats;
- `input_partition_diagnostics.csv`: 3,000 first rows (maximum)
  per partition, image/context summary, source-only labels;
- `task_texts.json`: the exact text profile from which vectors
  were cached;
- `task_similarities.csv`: all ten text cosine similarities;
- `task_semantic_diagnostics.csv`: nearest text tasks and source routes.

The text profiles include hypotheses about spatial drivers and
measurement mechanisms. They must be checked against the actual
dataset-generation documentation (not inferred from predictive
performance). A checkpoint/cache match ensures consistency, not
that the task descriptions are semantically true.

Important limitations: matching PNG paths and cached features does
**not** establish that the tile imagery matches the label polygon,
CRS or time period. Review source CSV coordinates and original data
production separately. Similarity matrices do **not** by themselves
establish task transferability. K4 best.pt selection was meta-dev,
so reported downstream comparisons remain development-only.

## Send back concise reports

```bash
python - <<'PY'
import json
from pathlib import Path
for task in ('education','nighttime','digital_total_cell'):
    d=json.loads((Path('runs')/('input_audit_'+task)/'audit.json').read_text())
    print(task,'routing_max_error',f"{d['max_routing_consistency_error']:.8f}",
          'duplicates',d['duplicate_row_identities'])
    for warning in d['warnings']:
        print('  CHECK:',warning)
PY
```

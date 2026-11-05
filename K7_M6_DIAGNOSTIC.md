# K7 M6 frozen hidden-state / router audit

This runs **no optimizer steps**, performs **no MLLM forward passes**,
reads **no official final-test files**, and scores **no Education labels**.

It examines the exact source-training subset in the existing
`runs/k7/m6_mllm_hidden_seed42/manifest.json`. The three sets of
Qwen2.5-VL hidden states (Population, Food, Education) for each source
training image come from `hidden.sqlite3`.

It computes:
- Mean-token-pool cosine(Population, Education) and
  cosine(Food, Education), including mean/std/range and their difference.
- Teacher's image-specific two-source distribution from the EXACT same
  M6 teacher implementation and temperature.
- Frozen checkpoint's predicted Education router weights on those
  source training images.
- Teacher/student absolute weight gap and cross-image correlation
  (undefined if either variable is constant).
- Fixed RemoteCLIP image+geolocation intervention: cyclically swap
  the MLLM Education hidden-state samples only, preserving each row's
  image features and geography, then observe absolute prediction /
  router-weight changes. This is a sensitivity test, **not** a full
  causal isolation of semantic knowledge: the swapped hidden features
  may be out-of-distribution and also carry image information.

## Run from repo root

```bash
git pull --ff-only origin main
python -m unittest discover -s tests -p 'test_k7_m6_diagnostic.py' -v
python -u scripts/diagnose_k7_m6_hidden.py \
  --run-dir runs/k7/m6_mllm_hidden_seed42
```

Output: concise table with three-decimal statistics; full precision
in `runs/k7/m6_mllm_hidden_seed42/hidden_router_diagnostic.json`.
The diagnostic refuses to overwrite a previous output; use a distinct
`--output` path only if deliberately repeating the audit.

Interpretation:
- Both teacher and student Pop std ~= 0: embeddings / teacher may not
  provide much image-specific routing variation.
- Teacher Pop std large, student Pop std ~= 0: learned router collapse.
- Large swapped-hidden prediction differences: student uses MLLM
  representations (not proof this use is useful/correct).
- Near-zero swapped-hidden prediction differences: the student
  practically ignores the MLLM input in this local intervention.

This small pilot is **not** a direct metric comparison with M1–M5.
Do not tune architecture or hyperparameters based on Education dev labels.

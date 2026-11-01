# K5-A: Nested Transferability-Supervised Spatial Router (prototype)

This is a **small, hypothesis-driven** prototype, not a full learned dual
router or region-aware hypernetwork. It learns a low-capacity, 4-feature
ridge predictor of *source-expert spatial transfer Pearson* using only
source tasks available inside the outer LOTO training fold.

## Protocol

Given a K4-T2 Structured Qwen3-Embedding-8B outer checkpoint:
1. Reconstruct its exact outer `train` subset. The outer held-out
   indicator and outer held-out cities are **excluded** from the inner
   training dataset. Save this restricted dataset as `inner_source_only.csv`.
2. Rotate each of the five allowed training tasks as an **inner** pseudo-target.
   Train an inner normalized expert bank on the other four tasks, withholding
   ~20% of the remaining cities (seed 42).
3. Evaluate the four experts individually on the inner pseudo-target's
   SCUI examples. Its labels are allowed for **router supervision only**,
   not inner expert optimization.
4. Use inner `last.pt` (fixed training epoch), NOT `best.pt`
   (which was selected using heldout meta labels).
5. Fit regularized ridge from source/pseudo-target Qwen3 cosine similarity
   and source-only moments to spatial transfer Pearson (20 task-pair scores).
   No outer heldout label is used.
6. Apply the learned route to the outer checkpoint's five source experts.
   The **same original semantic scale route** is frozen for all comparisons.
   Compare learned spatial routing against Semantic and Uniform spatial
   routing on outer Meta-SCUI and Meta-UCUI.

The outer checkpoint itself was previously chosen using meta labels, so
all results are still *research development* metrics, not independent test
scores. This prototype explicitly avoids using any real target-indicator
test labels or target moments.

## Run one most informative outer task first

```bash
git pull origin main
python -m py_compile scripts/run_ktrans_k5a.py
python -m unittest discover -s tests -p 'test_ktrans_k5a.py' -v

python -u scripts/run_ktrans_k5a.py \
  --outer-checkpoint runs/k4_t2_structured_qwen8b/best.pt \
  --output runs/k5a_digital_total_cell \
  --epochs 30 --batch-size 512 --eval-batch-size 1024
```

This performs **five inner trainings** (four-source expert banks) and
one lightweight ridge fit. It writes `evaluation.csv`,
`pseudo_transfer_scores.csv`, `protocol.json`, and per-inner-fold train logs.
`--batch-size 512` changes optimization from the previous 128 runs;
the same inner setting is used for all five folds.

For education and nighttime light after inspection:

```bash
python -u scripts/run_ktrans_k5a.py \
  --outer-checkpoint runs/k4_focused/structured_education_accessibility/best.pt \
  --output runs/k5a_education --batch-size 512

python -u scripts/run_ktrans_k5a.py \
  --outer-checkpoint runs/k4_focused_bs512/structured_nighttime_light_intensity/best.pt \
  --output runs/k5a_nighttime --batch-size 512
```

Use the actual checkpoint location if different. No Qwen 8B inference occurs:
cached text vectors and cached RemoteCLIP imagery are reused.

## Caveats

- Only five inner pseudo-targets produce twenty transfer observations.
  This is deliberately tiny; ridge may not generalize. Do not tune penalty
  or temperature against outer target validation labels.
- The inner source training subset is smaller than the full source dataset.
  This tests *whether performance supervision helps*, not final SOTA.
- The current K5-A does not train a Scale Router or condition on region.
- It is important not to silently call an inner expert's `best.pt` a valid
  pseudo-target-independent checkpoint; here `last.pt` is used.

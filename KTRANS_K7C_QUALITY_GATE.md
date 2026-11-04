# K7-C final blank-image quality gate; K7-D readiness

This is a **no-training, no-GPU** read-only inspection of existing
K7-C intervention JSONL records. It tests the simplest necessary visual
grounding property: a uniform-gray image should not support positive
claims about buildings, roads, greenery or industrial facilities.

It explicitly distinguishes:
- `valid_json`: the structural parser succeeded;
- `*_nonnull`: number of 0-3 scores not equal to JSON null;
- `*_positive`: number of nonnull scores greater than zero.
- `null` is NOT the same as a zero score, and no original-vs-blank
  paired difference can be calculated when one side is null.

Conceptual `evidence_relevance` can remain nonnull for a blank image
without implying visual hallucination. Visual cue and evidence confidence
claims require review. Even all-null outputs do not establish that
the original-image descriptions are accurate.

## Commands

```bash
git pull origin main
python -m py_compile scripts/audit_ktrans_k7c_blank.py
python -m unittest discover -s tests -p 'test_ktrans_k7c_blank.py' -v

python -u scripts/audit_ktrans_k7c_blank.py \
  --input runs/k7c_pilot/interventions.jsonl \
  --output runs/k7c_pilot/quality_gate
```

Outputs:

- `runs/k7c_pilot/quality_gate/blank_summary.csv`: three task-level
  counts for each of the cue, relevance and confidence dictionaries.
- `runs/k7c_pilot/quality_gate/blank_human_review.csv`: all 24
  task × tile blank records, raw descriptions and columns for reviewer
  fidelity judgments.
- `runs/k7c_pilot/quality_gate/readiness.json`: automated flags and
  intentionally **PENDING_HUMAN_GROUNDING_REVIEW** status.

Use the existing:

```text
runs/k7c_pilot/diagnostics/paired_review.csv
runs/k7c_pilot/diagnostics/image_review/*.png
```

Inspect the real tile image next to the patch shuffle and gray
intervention, then mark:
1. Whether the original scene description is consistent with visible
   buildings/roads/greenery and industrial structures.
2. Whether the gray image produces unsupported visual claims.
3. Whether patch shuffle changes are consistent with disrupted layout
   (rather than merely degraded image quality).
4. Whether task-specific proxy reasoning is explicitly tentative and
   does not invent schools, base stations, nighttime radiance or labels.

Do not label an all-null blank output as a pass if original descriptions
are unsupported. Do not interpret perturbation score deltas as
predictive improvement.

## K7-D pre-training design (do not train until review)

- Freeze the existing RemoteCLIP/K4 expert bank, the K5-A transfer
  prior, and Qwen3-VL-8B.
- Version the K7-C-approved visual proxy rubric and cache structured
  image-only and image+task outputs by image bytes/identity, task,
  model, prompt version and image preprocessing configuration.
- Use batched offline VLM inference with resumable per-sample JSONL or
  SQLite cache; preserve NULL masks instead of zero imputation.
- Define source-only ablations: K5-A; +text-only prior;
  +image-only visual cues; +task-conditioned cues/relevance/confidence.
- Fit only on outer source-task training rows; for reliability losses,
  obtain **independent** inner-fold expert predictions and separately
  trained inner-fold regional router banks. Never use outer heldout
  task labels or meta-UCUI to select K7 hyperparameters.
- Once source-only inner validation supports K7, evaluate outer
  development splits and report MSE, Pearson, bias by task/protocol,
  clearly distinguishing dev from independent testing.

This patch produces **only** the blank quality-gate audit; it does not
implement or train a K7-D router.

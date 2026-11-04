# K7-C: Controlled Visual Grounding and Human Review

## Question and scientific limits

K7-B produced 8 image+task outputs per task, with image-vs-text
relevance changes (Education 0.583, Nightlight 0.688, Digital Cell
0.917) and across-image standard deviations (0.698, 0.723, 0.690).
Those statistics **do not** establish correct visual grounding.

This read-only follow-up uses the **exact same image-task prompt, frozen
Qwen3-VL-8B model, task profiles, and max_pixels** as K7-B. Only
the pixels change. No urban indicator labels are read or predicted.

Two *negative/intervention* conditions per image × task:

- `blank`: replace the real aerial scene with uniform middle gray,
  keeping image dimensions. This tests whether VLM reports strong
  visibly grounded cues when **no scene exists**.
- `patch_shuffle`: deterministically shuffle 4×4 spatial patches,
  preserving the pixel color inventory. This tests sensitivity to
  large-scale layout while keeping some local textures. It is not
  guaranteed to destroy all visual evidence.

Do not count simply swapping the identities of two already-seen images
under an identical prompt as an independent control: because there is
no city ID in the VLM prompt, the same model/image/text combination
would produce the same inference. K7-C instead performs true pixel
interventions.

For 8 tiles × 3 tasks × 2 interventions, **48 new VLM generations**
are required. Original K7-B `image_task` results are reused as the
aligned untreated baseline (24 outputs; no regeneration). On 2 tiles
the smoke test is **12 generations**.

## Reproducible MI250X commands

```bash
git pull origin main
python -m py_compile scripts/run_ktrans_k7c_grounding.py scripts/diagnose_ktrans_k7c_grounding.py
python -m unittest discover -s tests -p 'test_ktrans_k7c_grounding.py' -v
```

Smoke (2 tiles):

```bash
python -u scripts/run_ktrans_k7c_grounding.py \
  --manifest runs/k7_pilot/vlm_evidence.jsonl \
  --original runs/k7b_pilot/evidence.jsonl \
  --max-tiles 2 \
  --output runs/k7c_pilot/interventions.jsonl \
  --max-pixels 262144 \
  --max-new-tokens 440
```

Complete (8 tiles, reuse 12 cached inferences):

```bash
python -u scripts/run_ktrans_k7c_grounding.py \
  --manifest runs/k7_pilot/vlm_evidence.jsonl \
  --original runs/k7b_pilot/evidence.jsonl \
  --max-tiles 8 \
  --output runs/k7c_pilot/interventions.jsonl \
  --max-pixels 262144 \
  --max-new-tokens 440

python -u scripts/diagnose_ktrans_k7c_grounding.py \
  --manifest runs/k7_pilot/vlm_evidence.jsonl \
  --original runs/k7b_pilot/evidence.jsonl \
  --interventions runs/k7c_pilot/interventions.jsonl \
  --max-tiles 8 \
  --output runs/k7c_pilot/diagnostics
```

Outputs:

- `runs/k7c_pilot/interventions.jsonl`: checkpoint-free, append-only
  record cache; unique key includes actual intervention, exact prompt,
  tile, task, model and pixel budget. Supports resuming by the same command.
- `runs/k7c_pilot/diagnostics/task_summary.csv`: each task/intervention,
  number of matched valid pairs, mean absolute difference in original
  vs perturbed visual cues, conceptual relevance and evidence confidence.
- `runs/k7c_pilot/diagnostics/paired_review.csv`: per tile/task/
  intervention, **original** vs **perturbed** scene descriptions,
  explanations and limitations; empty columns for human fidelity review.
- `runs/k7c_pilot/diagnostics/image_review/*.png`: original,
  middle-gray blank and 4×4 patch shuffled tile side by side.

## Evaluation guidance

1. **First** check whether gray inputs produce invented roads,
   buildings or industrial facilities, or unjustifiably high visual
   evidence confidence. The rating for **conceptual** relevance could
   remain nonzero even on a blank image; this alone is not hallucination.
2. Inspect paired images and `paired_review.csv`. Mark whether
   original descriptions fit visibly present structures; whether
   perturbation descriptions acknowledge damaged structure; whether
   explanations are defensible. This is manual review, not automated
   ground truth. Do not silently fill missing ratings.
3. A high difference between original and shuffle is a sensitivity
   signal, not proof of accuracy. A small difference can be reasonable
   if useful local textures survive patch shuffling.
4. This is an 8-tile qualitative stress test; it cannot demonstrate
   new-task performance. Only after review should VLM proxy features
   be considered for source-only cross-fitted expert routing.

Existing K4/K5/K6 and K7-A/B results remain untouched.

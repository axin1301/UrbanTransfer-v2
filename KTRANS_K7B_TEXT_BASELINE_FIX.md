# K7-B: text-only conceptual proxy baseline correction

In the first K7-B run, all 8 image+task conditions per task produced
nonnull cue relevance. The text-only control returned all-null relevance
for Education and Digital Cell, while Nightlight returned six scores.
This is an instruction interpretation problem: without an image, the VLM
must NOT assess whether a cue is present, but it CAN assess whether a
hypothetical cue might be conceptually relevant to the task.

The text-only control now uses a **distinct system prompt** that describes
an urban-indicator methodology analyst working WITHOUT an image. Its
task prompt explicitly asks: *IF a visual cue were present in some
urban region, how plausibly would it serve as an indirect proxy?*

Text-only `visual_cues` and `evidence_confidence` MUST remain null;
`evidence_relevance` may have ordinal 0–3 values. This is
hypothesis-level task knowledge, not an actual visual observation or
measured indicator relationship.

The **image+task** and **image-only** prompts and system instructions
are unchanged. Cache identity includes the complete prompt, so the
new text-only protocol `k7b-text-conceptual-v2` triggers only **three**
new generations. The previous cached image records are preserved.

## Run on existing data

```bash
git pull origin main
python -m unittest discover -s tests -p 'test_ktrans_k7b_controls.py' -v

python -u scripts/run_ktrans_k7b_controls.py \
  --manifest runs/k7_pilot/vlm_evidence.jsonl \
  --max-tiles 8 \
  --output runs/k7b_pilot/evidence.jsonl \
  --max-pixels 262144 \
  --max-new-tokens 440

python -u scripts/diagnose_ktrans_k7b_controls.py \
  --input runs/k7b_pilot/evidence.jsonl \
  --output runs/k7b_pilot/diagnostics
```

Expected run prefix if all old image conditions remain cached:
`[k7b] tiles=8 tasks=3 jobs=35 cached_success=32 pending=3`.
The append-only cache will grow from 35 to 38 records. On subsequent
reruns all 35 current-protocol jobs should be cache hits.

Diagnostic tool reads cached records in file order; the newest
`text_only` entry per task supersedes the previous response for
paired comparisons. The old records remain auditable.

**Interpretation warning:** Text-only relevance is a model-generated
conceptual prior. Non-null relevance is not proof of empirical transfer.
Image-vs-text differences measure *model sensitivity* to visual input,
not prediction accuracy. There are only 8 tiles; next steps require
image grounding spot checks, negative controls and source-only
expert-transfer evaluation without heldout label model selection.

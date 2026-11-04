# K7-B: Task-conditioned visual proxy controls (Qwen3-VL-8B)

Frozen inference-only extension of K7-A. The previous 8-tile, 32-record
K7-A pilot produced valid JSON but every task-relevance cue was null.
Inspection of the harbor example suggests the VLM conflated lack of
direct label evidence with absence of useful *indirect* visual proxies.

This follow-up uses a **revised prompt**. It distinguishes:
- `visual_cues`: observations of built-up fabric, roads, residential
  appearance, vegetation, large facilities, industrial/commercial uses;
- `evidence_relevance`: whether each cue is a plausible *indirect*
  proxy for the named task (not an observed indicator value);
- `evidence_confidence`: actual visibility/strength of evidence;
- `proxy_reasoning`: brief auditable explanation of one cue;
- `limitations`: what imagery cannot support.

No generated target values are permitted. Prompt-based proxy relevance
is hypothesis-level knowledge, **not measured indicator correlation**.

## Controls

All conditions use exactly the same 8 K7-A image identities.

| Condition | Inputs | Diagnostic role |
| --- | --- | --- |
| `image_only` | image + task-free morphology prompt | scene/visual cue stability |
| `image_task` | image + structured task profile | same-image/different-task, same-task/different-image variation |
| `text_only` | structured task profile only, explicitly no image | task-semantic prior without visual grounding |

For 8 tiles and 3 source tasks, the experiment performs
8 × (1+3) + 3 = **35** VLM generations. Text-only conditions are
generated once per task, not artificially repeated for each tile.
Compared task prompts retain the same task definitions/scoring rubric,
with modality-specific visibility safeguards.

By default we test the *source* indicators Education, Nightlight, and
Digital Cell, not the truly heldout four target indicators. No labels
or checkpoint calibration enter this audit.

## On MI250X

```bash
git pull origin main
python -m py_compile scripts/run_ktrans_k7b_controls.py scripts/diagnose_ktrans_k7b_controls.py
python -m unittest discover -s tests -p 'test_ktrans_k7b_controls.py' -v
```

**Smoke test**: run the first two *existing* K7-A tiles, not random new
tiles. Uses 2 × 4 + 3 = 11 generations.

```bash
python -u scripts/run_ktrans_k7b_controls.py \
  --manifest runs/k7_pilot/vlm_evidence.jsonl \
  --max-tiles 2 \
  --output runs/k7b_pilot/evidence.jsonl \
  --max-pixels 262144 \
  --max-new-tokens 440
```

**Complete experiment**: same output cache, same options, 8 tiles.
Already completed records are skipped; only the additional 6 tiles
will be processed.

```bash
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

Outputs:
- `runs/k7b_pilot/evidence.jsonl`: append-only records with the
  original image identity, source task, model, prompt version, raw
  response and strictly parsed response.
- `runs/k7b_pilot/diagnostics/task_summary.csv`: per-task JSON
  coverage, fraction of nonnull relevance cues, mean absolute
  image-vs-text-only relevance score shift, across-image task-relevance
  standard deviation.
- `runs/k7b_pilot/diagnostics/tile_summary.csv`: within-tile
  cross-task relevance variation, visual cue consistency and paired
  text-only contrast.

Interpretation:
1. First verify full JSON parsing and nonnull `evidence_relevance`.
2. If task relevance differs across tasks but barely across images,
   the model may simply recite prompt-conditioned knowledge.
3. If relevance differs from the same-task no-image control and across
   images, there is evidence of visual sensitivity; this is still
   insufficient to establish *correct visual grounding* or superior
   urban prediction. Inspect raw explanations and representative
   imagery, then evaluate source-only transfer with fixed splits.
4. VLM evidence may be wrong, especially around invisible schools,
   base stations and nighttime radiance. Do not turn these judgments
   into labels or calibrate on unseen tasks.

No K4/K5-A/K6 checkpoints are changed, no optimizer is run, and the
K7-A cache is never overwritten. Resuming after interruptions skips
valid parsed records; invalid responses are reattempted on rerun.
Do not run simultaneous writers to the same JSONL.

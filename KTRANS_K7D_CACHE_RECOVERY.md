# K7-D Image-Only v2 cache recovery (after 3113/3289 complete)

## What happened

The overnight launcher correctly blocked training because 176 of 3,289
required frozen VLM evidence jobs were not valid. The reported examples
were Image-Only v2; inspect all 176 via the recovery audit instead of
assuming all share the same failure type.

**Do not delete `runs/k7d_education/evidence.jsonl`.** All valid
image+task and text-only records remain reusable, and already valid
Image-Only v2 examples are retained.

The v2 schema still has only `scene`, `visual_cues`,
`evidence_confidence`, `limitations`. Occasionally Qwen3-VL emits
additional fields such as `evidence_relevance` anyway. If, and only
if, the JSON itself is well-formed and every required visual cue and
confidence score is complete and valid 0..3/null, the recovery tool
projects out the unsolicited fields. Each recovered record keeps
the **original raw output** and stores explicit `parse_status`,
`discarded_extra_fields`, and `recovery_policy` metadata.

No inferred task relevance is carried into the router. We never
invent visual scores, turn unknown into zero, or recover truncated
JSON. Output formatting compliance and grounding fidelity are
different: this operation certifies only structural evidence quality.

## Start by recovering the cache WITHOUT GPU

From your repo root after updating:

```bash
git pull origin main
python -m unittest discover -s tests -p 'test_ktrans_k7d_cache_recovery.py' -v
python -m unittest discover -s tests -p 'test_ktrans_k7d_image_only_v2.py' -v

python -u scripts/repair_ktrans_k7d_image_only.py \
  --manifest runs/k7d_education/manifest.json \
  --cache runs/k7d_education/evidence.jsonl \
  --reuse-cache runs/k7b_pilot/evidence.jsonl \
  --dry-run
```

This prints counts of `already_valid`,
`projected_extra_fields`, `invalid_json`,
`invalid_visual_scores`, `unexpected_schema`, and `missing`.
The CSV `runs/k7d_education/image_only_recovery_audit.csv`
lists row-wise parse status and discarded extra keys.

To actually append repaired records, rerun without `--dry-run`:

```bash
python -u scripts/repair_ktrans_k7d_image_only.py \
  --manifest runs/k7d_education/manifest.json \
  --cache runs/k7d_education/evidence.jsonl \
  --reuse-cache runs/k7b_pilot/evidence.jsonl
```

Then inspect full cache coverage without allocating an 8B model:

```bash
python -u scripts/cache_ktrans_k7d.py \
  --manifest runs/k7d_education/manifest.json \
  --output runs/k7d_education/evidence.jsonl \
  --reuse-cache runs/k7b_pilot/evidence.jsonl \
  --dry-run
```

## Resume the same automatic experiment

You may restart `scripts/run_ktrans_k7d_unattended.py`
with **the same command** you used before, from a live MI250X
allocation (or Slurm batch job).

The enhanced launcher now automatically:
1. Repairs existing complete Image-Only v2 JSON with an explicit audit.
2. Preflights outstanding v2 images before launching the 8B batch.
3. Fills genuinely missing VLM cache jobs, keeping valid ones.
4. Repairs extra-field v2 outputs generated during that pass.
5. If any Image-Only jobs remain invalid, retries only those with
   batch size **1**, max-new-tokens at least **800**, leaving other
   cache jobs untouched.
6. Audits and again checks complete evidence coverage before
   starting the same 3-seed/4-variant K7-D router experiment.

If any output still fails validation, the pipeline stops and reports
the exact missing tile/condition, rather than treating invalid
evidence as usable. Failed GPU job results remain in the JSONL file
for inspection and targeted fixes.

**Important:** The earlier K4 checkpoint and K5-A protocol must match
the manifest; use the exact two paths in your original shell command
or its `pipeline_status.json` snapshot. The launcher does not
search checkpoints based on target labels.

No GPU inference or transfer training was executed by the assistant.
K7-D evaluation results remain DEV-only (the outer K4 trunk is not
independently inner cross-fitted).

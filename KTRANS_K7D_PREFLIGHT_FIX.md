# K7-D preflight recovery: do not abort on one malformed v2 response

Previously K7-D demanded every one of the first eight Image-Only v2
generations be valid. One INVALID response stopped the entire overnight
pipeline even if 7/8 were suitable. This is an overly strict batch
preflight requirement, not evidence that every sample is unusable.

Now the preflight:
- recovers eligible complete visual-only fields (CPU-side and auditable)
  before measuring progress;
- reports `first_pass_valid=X/8`;
- proceeds to bulk caching if **at least one** genuinely valid new
  Image-Only result was obtained, and later retries all remaining
  invalid images individually;
- if **zero** new results are valid, attempts a bounded, single-image
  retry with >=800 output tokens before bulk inference;
- stops before bulk inference and router training if both initial
  passes produce no valid evidence.

It still requires 100% valid, task-free image-only evidence before
training. It never supplies fake labels, invents missing visual scores,
or treats task relevance as visual knowledge.

## First inspect the current evidence, no GPU required

```bash
git pull origin main
python -m unittest discover -s tests -p 'test_ktrans_k7d_preflight_diagnostics.py' -v
python -u scripts/diagnose_ktrans_k7d_image_only.py \
 --manifest runs/k7d_education/manifest.json \
 --cache runs/k7d_education/evidence.jsonl \
 --output runs/k7d_education/image_only_failure_report.csv
```

The output distinguishes:
- `valid`: already usable
- `projectable`: complete strict visual scores with only known unwanted
  output fields (the recovery script can extract them with an audit)
- `invalid_json`: unable to parse a complete JSON object, often
  truncated output
- `invalid_visual_scores`: visual fields are present but incomplete,
  mis-keyed or nonordinal
- `unexpected_schema`: missing required fields or unknown added fields
- `never_generated`: model inference never created the required key

This script does not alter evidence or instantiate Qwen3-VL.

## Resume only after code update

Use the **same** `scripts/run_ktrans_k7d_unattended.py` command and
the same manifest/output/cache paths as before. The append-only
`runs/k7d_education/evidence.jsonl` will be reused. The launcher
automatically runs offline repair, bounded retry and full-cache audit,
then (only with 100% evidence) trains three fixed-seed source-only
transfer ablations. The previous `FAILED` status is overwritten at
the beginning of a new run.

Do not run two cache writers at once. Keep the MI250X allocation active
or submit via the correct HPC scheduler.

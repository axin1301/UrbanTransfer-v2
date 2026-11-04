# K7-D complete-case fallback when last VLM entries do not parse

For exploratory K7-D source-only DEV comparisons it is valid to exclude
manifest **rows** whose frozen VLM observations are not available.
The selection may depend only on metadata and VLM cache validity, never
on indicator values, performance, model predictions or the target labels.

The original 3289 VLM jobs are not the same as the manifest row count:
a missing task-free Image-Only record can affect multiple partition
rows if a tile is reused. Report exact numbers after filtering.

```bash
git pull origin main

python -m unittest discover -s tests -p 'test_ktrans_k7d_complete_cases.py' -v

python -u scripts/prepare_ktrans_k7d_complete_cases.py \
 --manifest runs/k7d_education/manifest.json \
 --cache runs/k7d_education/evidence.jsonl \
 --reuse-cache runs/k7b_pilot/evidence.jsonl \
 --output runs/k7d_education/manifest_complete_cases.json

python -u scripts/cache_ktrans_k7d.py \
 --manifest runs/k7d_education/manifest_complete_cases.json \
 --output runs/k7d_education/evidence.jsonl \
 --reuse-cache runs/k7b_pilot/evidence.jsonl --dry-run
```

The final command must show `pending=0`. The original manifest and
cache remain unchanged; `manifest_complete_cases.excluded.json`
audits each excluded row and its missing condition(s). If a partition
or source training task becomes empty, the filter fails closed.

Then rerun your existing on-node unattended command with **only**
`--manifest` changed to
`runs/k7d_education/manifest_complete_cases.json`; keep
`--output-dir runs/k7d_education` so previously computed valid
VLM cache remains available and the runner does not infer new
knowledge. The runner will evaluate K5-A, Text-Only, Image-Only and
Image–Task on exactly the same selected rows with seeds 42,43,44.
Do not run two cache writers concurrently.

Research caveats: the missingness mechanism may be image quality
dependent. Results are complete-case, source-only **DEV** comparisons
on a filtered evaluation set, and must not be compared numerically to
K5-A metrics obtained on the original full meta evaluation rows.
The outer K4 representation is not independently inner cross-fitted.

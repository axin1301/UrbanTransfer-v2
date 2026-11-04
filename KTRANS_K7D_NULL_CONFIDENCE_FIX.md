# K7-D whole-null confidence recovery

The recorded 175 Image-Only v2 `unexpected_schema` rows frequently
contain complete `visual_cues` but a top-level
`"evidence_confidence": null`. This means that the model abstains from
the entire confidence rating, not that the image is broken.

Image-Only v2 parsing now accepts a **present** whole-null confidence
field and normalizes it to six null confidence cues. The entire
confidence mask remains zero. No score is imputed, no target values
are read, no task relevance is introduced and the original raw output
is preserved unchanged. A **missing** confidence field still fails.

After `git pull origin main`:

```bash
python -m unittest discover -s tests -p 'test_ktrans_k7d_cache_recovery.py' -v
python -m unittest discover -s tests -p 'test_ktrans_k7d_image_only_v2.py' -v

python -u scripts/repair_ktrans_k7d_image_only.py \
  --manifest runs/k7d_education/manifest.json \
  --cache runs/k7d_education/evidence.jsonl \
  --dry-run

python -u scripts/repair_ktrans_k7d_image_only.py \
  --manifest runs/k7d_education/manifest.json \
  --cache runs/k7d_education/evidence.jsonl

python -u scripts/cache_ktrans_k7d.py \
  --manifest runs/k7d_education/manifest.json \
  --output runs/k7d_education/evidence.jsonl \
  --dry-run
```

Expect `expanded_null_confidence` in the repair summary.
The count cannot be predicted from five examples alone. When full
cache coverage is reached, rerun the exact existing overnight command
from a live GPU allocation; it will skip precompute and move on to
the three-seed K7-D ablation. Do not delete the cache or assume
all remaining records can be recovered.

This establishes structural compatibility only, not visual
ground-truth fidelity or improved transfer prediction.

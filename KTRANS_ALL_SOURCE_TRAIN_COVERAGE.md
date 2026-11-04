# P1-B: full outer SOURCE-TRAIN coverage planning

Goal: allow all outer training source-task rows to contribute to
empirical competence discovery, while preserving inner LOTO exclusion.
No Education outer pseudo-target/UCUI rows or outer heldout cities
belong in the expanded VLM manifest.

The plan intentionally separates three operations:
(1) CPU-only full train manifest + exact cache audit,
(2) optional GPU Qwen3-VL inference for missing jobs,
(3) frozen inner-Loto expert error export. You may stop after (1).

## Stage 1: enumerate ALL outer train tiles and audit cached VLM

```bash
git pull origin main
python -m unittest discover -s tests -p 'test_ktrans_source_coverage.py' -v

python -u scripts/plan_ktrans_full_source_coverage.py \
  --checkpoint "$K4_CKPT" \
  --existing-manifest runs/k7d_education/manifest_complete_cases.json \
  --knowledge-cache runs/k7d_education/evidence.jsonl \
  --reuse-cache runs/k7b_pilot/evidence.jsonl \
  --output runs/ktrans_competence/full_source_coverage
```

Outputs:
- `source_train_manifest.json`: every outer K4 `train` row (not just K7-D
  Complete-Case sampling), with tile-id derived from the exact resolved
  image path and canonical version `k7d-manifest-v1` for cache reuse.
  Deliberately **no** outer SCUI/UCSI/UCUI rows are included.
- `coverage_plan.json`: unique tiles, tile-task pairs, exact
  image_only/image_task/text_only prompt/model-key job totals, cache
  hits and outstanding counts.
- `source_task_coverage.csv`: original/expanded per-task row counts.
- `pending_jobs.jsonl`: unique outstanding jobs for budgeting,
  not an executable list. The canonical cache runner consumes
  `source_train_manifest.json` instead.

No new images are encoded by RemoteCLIP, no Qwen3-VL is loaded,
and no labels are used to decide which VLM jobs to create.

## Stage 2 (optional): fill remaining SOURCE-only cache

ONLY if stage 1's new-job count/cost is acceptable and a MI250X GPU is
allocated. Inspect dry-run BEFORE inference:

```bash
python -u scripts/cache_ktrans_k7d.py \
  --manifest runs/ktrans_competence/full_source_coverage/source_train_manifest.json \
  --output runs/ktrans_competence/full_source_evidence.jsonl \
  --reuse-cache runs/k7b_pilot/evidence.jsonl \
  --extra-reuse-cache runs/k7d_education/evidence.jsonl \
  --dry-run
```

To launch *one incremental 100-job segment* (not all at once),
remove `--dry-run` and add `--max-new-jobs 100 --batch-size 4`.
The runner is append-only and resumable; rerunning identifies missing
prompt-keyed records in the two historical sources plus the new
output. Do not alter the model/prompt/max-pixels between runs.

For empirical competence rows, only `image_task` evidence is required;
if GPU time is limited, Stage 2 can use
`--conditions image_task --max-new-jobs 100`. The full plan also
includes `image_only` and `text_only` for future controlled ablation,
but they are NOT required to export the current 36d `image_task`
competence evidence. A source-only task-only cache already exists for
many tasks.

## Stage 3: re-export empirical competence with all source rows

```bash
python -u scripts/build_ktrans_empirical_competence.py \
  --checkpoint "$K4_CKPT" \
  --k5a-protocol "$K5A_PROTOCOL" \
  --manifest runs/ktrans_competence/full_source_coverage/source_train_manifest.json \
  --knowledge-cache runs/ktrans_competence/full_source_evidence.jsonl \
  --reuse-cache runs/k7b_pilot/evidence.jsonl \
  --extra-reuse-cache runs/k7d_education/evidence.jsonl \
  --output runs/ktrans_competence/education_all_outer_train_dev \
  --batch-size 512
```

Unchanged restrictions:
- Only inner `meta_SCUI` and inner `meta_UCUI` pseudo-task rows
  generate competence supervision. Their inner-model `last.pt` has
  never trained on that pseudo-task's labels.
- All inner rows must belong to the FULL outer TRAIN image/task set;
  inner source-bank exactly excludes its pseudo target. In particular,
  using all outer train **does NOT mean treating every sample as an
  independently evaluated supervised expert**.
- The outer K4 `best.pt` provenance is still DEV-only. The script
  does not train/fine-tune K4 or an alignment router.
- SCUI/UCUI inner-city sets remain independent; do not mix them when
  training and validating a future competence model.
- Compute coverage separately for SCUI and UCUI and each source
  pseudo-task. Cached prompt VLM failures/missingness should not be
  silently replaced by a zero score.

## Resource and protocol notes

For all N train rows, the number of distinct VLM image_task jobs is
the number of unique (tile,task) pairs, which may be much smaller
than N if repeated images share tasks. The image_only count is unique
tiles; text_only count is number of source tasks. Actual values depend
on the user cluster's dataset/caches and are intentionally NOT guessed.

This is a transductive source-knowledge acquisition operation (source
train images only), not test-label access. All cache keys include VLM
model, prompt, version and max_pixels; the planner rejects stale or
mismatched records. Keep the original K7-D/H Complete-Case experiments
unchanged for reproducibility.

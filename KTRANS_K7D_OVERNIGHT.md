# K7-D overnight (on-node) transfer run: Image-Only v2 + 3-seed ablations

## Confirmed issue and fix

The original K7-D `image_only` prompt asked for a general object
including `evidence_relevance`. The 8B model frequently populated
that task-oriented field even when no task was supplied, causing
correctly formed JSON to fail the K7-B strict `image_only` parser.

**Image-Only v2 is a genuinely independent schema**, rather than an
instruction to fill relevance with null:

```json
{"scene":"...",
 "visual_cues":{"built_up":2,"roads":2,"residential":1,
                "vegetation":2,"large_facilities":1,
                "industrial_commercial":2},
 "evidence_confidence":{"built_up":2,"roads":2,"residential":1,
                        "vegetation":2,"large_facilities":1,
                        "industrial_commercial":2},
 "limitations":"..."}
```

It intentionally omits `evidence_relevance` and
`proxy_reasoning`. Its image-only prompt/version/cache key are new.
The decoder rejects the appearance of forbidden fields instead of
silently deleting generated task scores. The 36-element router input
fills the *unobserved* relevance component with null masks (zero data,
mask=0). Old cached `image_task` and `text_only` jobs preserve their
exact K7-B keys and are reused without recomputation.

## Launch a complete overnight experiment

**The assistant cannot launch or monitor your MI250X directly.**
The user must submit the following script **from within an active GPU
allocation**, or submit an equivalent job to the HPC scheduler. Merely
closing an interactive allocation can terminate work even if `nohup`
is used. Do not launch GPU inference from a login node.

From the repository root:

```bash
git pull origin main
python -m py_compile scripts/cache_ktrans_k7d.py scripts/run_ktrans_k7d_unattended.py scripts/run_ktrans_k7d.py
python -m unittest discover -s tests -p 'test_ktrans_k7d_image_only_v2.py' -v
python -m unittest discover -s tests -p 'test_ktrans_k7d.py' -v
```

Use actual existing files. For the currently active Education run,
replace the examples below with the verified paths on your server:

```bash
export K4_CKPT=/absolute/path/to/actual/K4/best.pt
export K5A_PROTOCOL=/absolute/path/to/actual/K5A/protocol.json
export K7D_MANIFEST=runs/k7d_education/manifest.json

test -f "$K4_CKPT" && test -f "$K5A_PROTOCOL" && test -f "$K7D_MANIFEST"
```

The full pipeline is ONE command:

```bash
mkdir -p runs/k7d_education

python -u scripts/run_ktrans_k7d_unattended.py \
  --checkpoint "$K4_CKPT" \
  --k5a-protocol "$K5A_PROTOCOL" \
  --manifest "$K7D_MANIFEST" \
  --output-dir runs/k7d_education \
  --reuse-cache runs/k7b_pilot/evidence.jsonl \
  --batch-size 8 \
  --preflight 8 \
  --seeds 42,43,44 \
  --epochs 12 \
  --max-pixels 262144 \
  --max-new-tokens 440
```

Use `--output-dir runs/k7d_education` to reuse your existing
`runs/k7d_education/evidence.jsonl` directly. The new v2 image-only
cache keys differ, while all valid image+task and text-only keys remain
unchanged. The old invalid records are retained for audit.

## For no-terminal-attention operation

On a compute node whose GPU allocation persists after logout,
use shell redirection (or a scheduler job script) for the same command:

```bash
mkdir -p runs/k7d_education
nohup python -u scripts/run_ktrans_k7d_unattended.py \
  --checkpoint "$K4_CKPT" \
  --k5a-protocol "$K5A_PROTOCOL" \
  --manifest "$K7D_MANIFEST" \
  --output-dir runs/k7d_education \
  --reuse-cache runs/k7b_pilot/evidence.jsonl \
  --batch-size 8 --preflight 8 --seeds 42,43,44 \
  --epochs 12 --max-pixels 262144 --max-new-tokens 440 \
  > runs/k7d_education/overnight.log 2>&1 &
echo "PID=$!"
```

**An allocation expiring ends this process; `nohup` cannot extend
an HPC scheduler reservation.** Use `sbatch` with your cluster's
actual GPU partition/time/allocation settings for reliable overnight
scheduling. Do not invent or assume scheduler resource settings.

## What the runner does automatically

1. Checks all required checkpoint/manifest paths.
2. Preflights **8 image-only v2** jobs (or fewer already-cached);
   aborts if any new image-only result fails strict schema parsing.
3. Reuses valid old `image_task`/`text_only` records; batches all
   remaining v2 image-only and incomplete image+task cache jobs.
4. Requires **100% valid evidence coverage** before training.
   Errors are reported; no silent imputation or label leakage.
5. Trains each source-only knowledge ablation at **three fixed seeds**
   42, 43 and 44, 12 epochs each; K5-A baseline is evaluated on
   identically sampled rows. Same model hyperparameters for all
   ablations, no best-epoch selection on heldout labels.
6. Saves per-seed predictions/metrics and a paired mean/std summary.

Files (with `--output-dir runs/k7d_education`):
- `runs/k7d_education/evidence.jsonl`: append-only cache, including
  old invalid records for audit;
- `runs/k7d_education/pipeline.log`: subprocess output and errors;
- `runs/k7d_education/pipeline_status.json`: RUNNING, FAILED,
  COMPLETED, stage and error on failure;
- `runs/k7d_education/seed_42/evaluation.csv` (also 43 and 44);
- `runs/k7d_education/summary.csv`: MSE/Pearson/Bias across seeds
  and paired MSE difference relative to K5-A.

Progress:

```bash
tail -f runs/k7d_education/overnight.log
cat runs/k7d_education/pipeline_status.json
cat runs/k7d_education/summary.csv
```

The metrics are **development** diagnostics, not independent transfer
proof. K4 has already been meta-selected and its shared representation
is not nested cross-fitted. A positive K7-D comparison must later be
tested on independently fitted nested source-expert banks.

# Unattended K-TransMoE six-fold experiments (approximately 8-hour session)

## Study design

Four model conditions:
- K2-B1: original scale, RemoteCLIP short task text
- K3-A: normalized experts, RemoteCLIP short task text
- K4-T1: normalized experts, Qwen3-Embedding-8B short task text
- K4-T2: normalized experts, Qwen3-Embedding-8B structured task text

Six source-task LOTO folds per condition, fixed seed 42, stratified 5,000
source train rows, 20% source cities held out, 30 epochs. On each trained
checkpoint, evaluate frozen spatial routes Semantic, Uniform, Random (seeds
41–45). For normalized expert checkpoints, also evaluate independent scale
routes (Semantic/Uniform) crossed with spatial routes (Semantic/Uniform).
**The dual routes do not use target labels or require retraining.**

No official target test labels are read. The internal meta partitions guide
checkpoint selection and must be called development results.

## Prerequisites

Qwen3-Embedding-8B **both** short and structured caches must be present.
Structured cache alone is not enough.

```bash
git pull origin main
python -m unittest discover -s tests -p 'test_ktrans_dual_routes.py' -v
python -m py_compile scripts/run_ktrans_overnight.py scripts/analyze_ktrans_predictions.py urban_transfer/ktrans_normalized.py
python -u scripts/precompute_llm_task_embeddings.py \
  --text-model-id Qwen/Qwen3-Embedding-8B --task-profile short --batch-size 1
python -u scripts/precompute_llm_task_embeddings.py \
  --text-model-id Qwen/Qwen3-Embedding-8B --task-profile structured --batch-size 1
```

## Single uninterrupted server run

Use the shell in the repository root. Existing checkpoints from earlier runs
can be analyzed separately; this experiment's run root is isolated to avoid
accidentally reusing checkpoints with different settings.

```bash
mkdir -p runs/ktrans_overnight
nohup python -u scripts/run_ktrans_overnight.py \
  --hours 7.75 --root runs/ktrans_overnight --limit 5000 \
  --epochs 30 --batch-size 128 --eval-batch-size 256 \
  > runs/ktrans_overnight/overnight.log 2>&1 < /dev/null &
echo "PID=$!"
```

Check progress:
```bash
tail -n 30 runs/ktrans_overnight/overnight.log
```

The runner has an elapsed-time budget, not a guarantee that every possible
configuration will finish within 8 hours. It attempts folds sequentially and
preserves existing best.pt and diagnostics.csv on resumption. If terminated
during a training fold, it restarts that fold from scratch on resumption unless
a finished best.pt exists. Because best.pt is saved after every epoch, an
interrupted fold's best.pt may be PARTIAL: inspect its
`meta_protocol.json` and `history.jsonl` before accepting it. For any
suspect partial fold, remove ONLY that fold's output directory and rerun.
Existing results are not overwritten silently.

Resume:
```bash
python -u scripts/run_ktrans_overnight.py --hours 7.75 \
  --root runs/ktrans_overnight
```

Summarize without GPU work:
```bash
python scripts/run_ktrans_overnight.py --only summarize --root runs/ktrans_overnight
```

Outputs: `aggregate/all_results.csv`, `aggregate/summary.csv`,
`job_history.json`, `failures.json`, and per-stage logs.

**Important:** partial aggregations include `folds_present`; compare models
only on matched complete folds. Never use `oracle_mean_mse` except as a
posthoc diagnostic. The default run could have 24 model trainings and more
than 200 diagnostic evaluations, so a 7.75-hour budget might not finish it
all. Prioritize the Qwen8B conditions by passing
`--conditions k4_t1_qwen8b_short k4_t2_qwen8b_structured`
if time is tight. K2-B1 and K3-A earlier six-fold summaries already exist.

## Interpretation

Report MSE/R2/Pearson/Bias to three decimals; inspect each held-out task and
target vs predicted standard deviations. The dual-scale route study is an
**inference ablation**, not a performance-trained router, nor a learned scale
hypernetwork. G1 Fourier geographic encoding, T3 generated-and-reviewed LLM
profiles, and nested performance-guided router are NOT implemented in this
batch, and should not be described as completed experiments.

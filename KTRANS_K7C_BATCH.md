# K7-C batched inference on MI250X

K7-C now batches frozen Qwen3-VL-8B image+task inputs for GPU
throughput, rather than calling `generate()` once per image.

- `--batch-size 4` default; try 2, 4 and 8 if memory permits.
- The tokenizer uses **left padding** for decoder-only generation.
  Output slicing removes the **common padded prompt width** from every
  generation. This avoids cross-sample output misalignment.
- Same image_task system prompt, task texts, intervention methods and
  cache identity as PR #44. **Batch size does not enter the cache key**.
- An OOM on batch size >1 halves the active batch and retries; OOM on
  batch size 1 propagates. Ordinary runtime errors are not swallowed.
- Each successful record remains append-only JSONL. Previously completed
  single-image runs are skipped.
- This is an efficiency improvement, not a new model/metric result.

## Commands from repository root

```bash
git pull origin main
python -m py_compile scripts/run_ktrans_k7c_grounding.py
python -m unittest discover -s tests -p 'test_ktrans_k7c_batch.py' -v
python -m unittest discover -s tests -p 'test_ktrans_k7c_grounding.py' -v

python -u scripts/run_ktrans_k7c_grounding.py \
  --manifest runs/k7_pilot/vlm_evidence.jsonl \
  --original runs/k7b_pilot/evidence.jsonl \
  --max-tiles 8 \
  --batch-size 4 \
  --output runs/k7c_pilot/interventions.jsonl \
  --max-pixels 262144 \
  --max-new-tokens 440

python -u scripts/diagnose_ktrans_k7c_grounding.py \
  --manifest runs/k7_pilot/vlm_evidence.jsonl \
  --original runs/k7b_pilot/evidence.jsonl \
  --interventions runs/k7c_pilot/interventions.jsonl \
  --output runs/k7c_pilot/diagnostics
```

The original 8 tiles × 3 source tasks × 2 interventions have **48**
new calls to amortize; `batch-size=4` corresponds to roughly 12
`generate()` batches in a fresh run, ignoring retries and incomplete
JSON. Actual wall-clock speedup depends on GPU utilization and memory;
no speedup or functional equivalence measurements have yet been run on
the user's MI250X. Batched padding can cause minor numeric generation
differences, so compare JSON validity and diagnostic metrics.

If some records are already cached, only pending combinations are
scheduled. If all 48 were completed before updating, there will be
zero new inference calls.

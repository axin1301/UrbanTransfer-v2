# M6.1 frozen grouped hidden-token audit

**Purpose**: Does Qwen2.5-VL already produce near-identical
Visual/Definition/Relation tokens for all indicators on the **same
image**, or is their variation lost inside the trained Student?

No finetuning, optimizer, MLLM forward pass, dev labels, target-city
imagery or official test files are used by this diagnostic.

On the original source supervised image subset recorded in the
M6.1 `manifest.json`, retrieve the three cached 64×2048
Qwen-VL hidden token groups per image: Population, Food, Education.
Compare each pair of tasks in three stages:

1. **RAW**: Qwen-VL last-layer Visual (32), Definition (16) and Relation
   (16) token chunks before Student processing.
2. **Projected**: the identical chunks passed through the frozen
   trained Student's LayerNorm + Linear 2048→128 token projection.
3. **Fused**: the final Student cross-attended knowledge vector and
   the resulting population router probabilities.

Both averaged-token cosine and normalized Frobenius differences are
reported for each group. Final fusion has cosine, normalized L2
distance and mean absolute router-weight change. Mean/std/min/max/median
of all values are in the JSON report.

## Run

```bash
git pull --ff-only origin main
python -m unittest discover -s tests \
  -p 'test_k7_m61_grouped_hidden_audit.py' -v
python -u scripts/audit_k7_m61_grouped_hidden.py \
  --run-dir runs/k7/m61_structured_seed42 \
  --batch-size 32
```

Output: `runs/k7/m61_structured_seed42/raw_grouped_hidden_audit.json`.
The script refuses to overwrite an existing report. All computation
is on CPU. `hidden.sqlite3`, `model.pt`, existing metrics, and
M6 caches are untouched.

### Interpretation

- RAW Definition/Relation differ, Projected/Fused become constant:
  learned Student suppresses task information; replace shared-query
  fusion with task-queried cross attention and task-separation training.
- RAW all groups near-identical in **both cosine and normalized
  difference**: inspect Qwen-VL prompt layout / layer / token extraction
  first, before training a larger Student.
- RAW visual group varies by task but text groups do not: inspect
  indicator/relation boundaries and the supplied prompts.
- The visual token hidden states are **contextualized**; do not assume
  that image-placeholder positions encode task definitions equally
  strongly. High cosine alone does not imply exact equality.
- Same-image source rows only: this is representation sensitivity,
  not evidence of useful or correct source-to-target transfer.

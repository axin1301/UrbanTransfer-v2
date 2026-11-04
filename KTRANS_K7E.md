# K7-E: frozen knowledge → expert route → prediction diagnosis

This stage is **read-only**: it does not run Qwen3-VL inference, update
the K4 expert bank, fit a router, change checkpoint selection or use
outer heldout labels for optimization. It replays the saved **K7-D
single-seed complete-case** rows exactly.

## Validate and run one seed

From the repository root, after `git pull origin main`:

```bash
python -m py_compile scripts/diagnose_ktrans_k7e.py urban_transfer/ktrans_k7e.py
python -m unittest discover -s tests -p 'test_ktrans_k7e.py' -v

python -u scripts/diagnose_ktrans_k7e.py \
  --checkpoint "$K4_CKPT" \
  --k5a-protocol "$K5A_PROTOCOL" \
  --manifest runs/k7d_education/manifest_complete_cases.json \
  --knowledge-cache runs/k7d_education/evidence.jsonl \
  --reuse-cache runs/k7b_pilot/evidence.jsonl \
  --routers-dir runs/k7d_education/single_seed42 \
  --output runs/k7e_education/seed42 \
  --seed 42
```

If shell variables `$K4_CKPT` and `$K5A_PROTOCOL` were lost, retrieve
them from the saved K7-D router metadata rather than guessing:

```bash
eval "$(python - <<'PY'
import torch,shlex
from pathlib import Path
state=torch.load('runs/k7d_education/single_seed42/image_task_last.pt',
                 map_location='cpu',weights_only=False)
for k,name in [('k4_checkpoint','K4_CKPT'),('k5a_protocol','K5A_PROTOCOL')]:
    p=Path(state[k]).resolve()
    if not p.is_file():raise SystemExit(f'Missing {k}: {p}')
    print('export '+name+'='+shlex.quote(str(p)))
PY
)"
```

The script refuses to run if router checkpoints do not belong to this
exact K4, K5-A, seed and manifest, or if any manifest evidence is
missing. Its output is **not** a replacement K7-D experiment.

## Diagnostics

Results in `runs/k7e_education/seed42/`:

- `summary.csv`: one row per partition and variant (12 rows)
- `expert_routes.csv`: source expert prior/candidate/final mean and
  samplewise standard deviation per partition and variant
- `protocol.json`: frozen provenance and scientific caveats

Interpret the following carefully:

| Metric | Interpretation |
| --- | --- |
| `knowledge_mean_feature_std` | variation in all 36 knowledge dimensions (including null masks) |
| `knowledge_within_task_std` | variation not explained merely by mixing task identities |
| `candidate_prior_l1_mean` | mean L1 distance between knowledge-conditioned candidate and K5-A prior |
| `final_prior_l1_mean` | actual mean expert-weight shift after multiplying by Gate |
| `gate_mean` and `gate_std` | global strength versus spatial variation |
| `route_expert_within_task_std` | within-task variation of resulting expert weights |
| `prediction_delta_std` | samplewise spread of predictions relative to frozen K5-A on the **same filtered rows** |
| `prediction_delta_abs_mean` | average absolute prediction change (not improvement) |
| `gate_knowledge_abs_corr_max` | largest descriptive absolute correlation of Gate with a varying knowledge feature; NaN if undefined |

At every row the route must obey
`final = (1-gate)*prior + gate*candidate`.
For L1 norms, `L1(final-prior)=gate*L1(candidate-prior)`.
The diagnostic checks this identity and checks exact K7-D Router forward
outputs. Large candidate changes with tiny final changes point to a
conservative Gate; tiny candidate changes despite varying inputs point
to an ineffective candidate route. Actual prediction gains cannot be
inferred from these descriptive route differences alone.

**Research limitation:** results are DEV-only; K4 was meta-selected,
not independently inner cross-fitted. This script does not make causal
claims about VLM knowledge grounding or independent target generalization.

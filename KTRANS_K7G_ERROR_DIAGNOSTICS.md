# K7-G: frozen error decomposition and source-to-target direction audit

This is a POSTHOC diagnostic of the existing seed-42 K7-D model and
Complete-Case evaluation rows. Reuse exactly the same K4 bank, K5-A
protocol, Router weights and cached Qwen3-VL knowledge. No training or
new Qwen3-VL generation.

## Questions

1. Is the UCUI MSE increase driven by squared bias or by the spatially
   centered residual (variance/covariance)?
2. Does the K7-D candidate route reduce residuals on pseudo-heldout
   SOURCE training rows but increase them on truly heldout target DEV
   rows?
3. Is the trained small Gate protective against the candidate's error
   direction, as opposed to just suppressing a useful candidate?

## Read-only command

```bash
git pull origin main
python -m py_compile scripts/run_ktrans_k7g_error_diagnostics.py urban_transfer/ktrans_k7g.py
python -m unittest discover -s tests -p 'test_ktrans_k7g_error_diagnostics.py' -v

python -u scripts/run_ktrans_k7g_error_diagnostics.py \
 --checkpoint "$K4_CKPT" \
 --k5a-protocol "$K5A_PROTOCOL" \
 --manifest runs/k7d_education/manifest_complete_cases.json \
 --knowledge-cache runs/k7d_education/evidence.jsonl \
 --reuse-cache runs/k7b_pilot/evidence.jsonl \
 --routers-dir runs/k7d_education/single_seed42 \
 --output runs/k7g_education/seed42 \
 --seed 42
```

Restore unset shell paths from the previous router checkpoint:

```bash
eval "$(python - <<'PY'
from pathlib import Path
import shlex,torch
x=torch.load('runs/k7d_education/single_seed42/image_task_last.pt',
             map_location='cpu',weights_only=False)
for k,name in [('k4_checkpoint','K4_CKPT'),('k5a_protocol','K5A_PROTOCOL')]:
    value=Path(x[k]).resolve()
    if not value.is_file():raise SystemExit('Missing '+str(value))
    print('export '+name+'='+shlex.quote(str(value)))
PY
)"
```

## Outputs

- `error_decomposition.csv`: four splits (train, SCUI, UCSI,
  UCUI), three variants, three modes (fixed 0, trained, fixed 1).
  Every row is paired against K5-A fixed 0 on that SAME split.
- `source_task_decomposition.csv`: per-source-task breakdown for
  router **train** and unseen-city/source-task UCSI, to avoid
  misleading pooled source-task moments.
- `transfer_direction.csv`: whether the signed MSE change on source
  training data agrees with each DEV split.
- `protocol.json`: provenance and limitations.

Exact identities (code asserts them):

```
MSE = bias^2 + mean((prediction - target - bias)^2)
ΔMSE_vs_K5A = 2 mean((K5A - target) (prediction - K5A))
              + mean((prediction - K5A)^2)
```

The first component of the paired identity shows whether the
perturbation aligns with the ORIGINAL prediction error. The second
component is nonnegative and penalizes route movement. Every value
is reported, not optimized.

`in_sample_oracle_mse_NOT_GENERALIZATION` fits an affine slope and
intercept using the **same evaluation labels**. It is ONLY an
error-analysis upper benchmark, not a valid prediction model. Do not
use it or the evaluated UCUI numbers to select a Gate, architecture,
scale correction or checkpoint and then claim independent transfer.

Source `train` metrics are **in-sample for both the K4 bank and K7-D
router**, so an apparent source gain is not cross-fitted generalization.
The K4 checkpoint was selected using meta DEV metrics. All target
partitions are DEV only, even though no heldout labels enter fitting
in this diagnostic.

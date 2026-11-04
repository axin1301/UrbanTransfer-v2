# K6.1 Loss Diagnostics (read only)

Diagnose why the four Education loss objectives produce nearly identical
Meta-UCUI scores. **Do not retrain any model or use heldout-task labels for
gradient diagnostics.**

## Run on the four already saved checkpoints

```bash
git pull origin main
python -m py_compile scripts/diagnose_ktrans_loss_gradients.py
python -m unittest discover -s tests -p 'test_ktrans_loss_diagnostics.py' -v

python -u scripts/diagnose_ktrans_loss_gradients.py \
  --runs-prefix runs/k61loss_education \
  --output runs/k61loss_education_diagnostics \
  --batches 4 --batch-size 512 --seed 42

python - <<'PY'
import pandas as pd
p='runs/k61loss_education_diagnostics/summary.csv'
d=pd.read_csv(p)
print(d[['loss_mode','gate_mean','gate_std','corr_groups',
         'rank_pairs','mse_grad_norm','weighted_corr_to_mse',
         'weighted_rank_to_mse','corr_cos_mse','rank_cos_mse',
         'expert_delta_abs_mean']].round(3).to_string(index=False))
PY
```

Outputs:
- `batch_gradients.csv`: one row per mode and batch, including raw
  loss values, gradient norms, coverage, weighted gradient-to-MSE ratios,
  cosine alignment and regional-vs-global prediction disagreement.
- `summary.csv`: means of the batch diagnostics.
- `protocol.json`: provenance, fixed source-only partition and seed.

Interpretation:
- `corr_groups` counts source task groups with at least eight examples
  and non-constant labels.
- `rank_pairs` counts valid within-task and within-city pairs with a
  standardized label gap above 0.200.
- `weighted_corr_to_mse`, `weighted_rank_to_mse` compare actual
  *optimizer-weighted* auxiliary gradients with MSE gradients, all
  measured with respect to **Gate parameters only**.
- `corr_cos_mse`, `rank_cos_mse` measure gradient agreement; NaN means
  at least one gradient was zero, not a positive or negative alignment.
- `route_l1_mean` and `expert_delta_abs_mean` quantify the extent
  to which changing the Gate can even change expert combination output.

Important limitations:
- These gradients measure *local* sensitivity of four saved Gate
  checkpoints, not epoch-by-epoch gradients during prior training.
- The gate parameterization itself limits the directions auxiliary losses
  can improve. Near-zero gradients do not prove the loss is intrinsically
  unhelpful to a jointly trained regional router or expert.
- The same underlying source experts were source-supervised; masking an
  expert is not strict nested crossfitting. Results are diagnostics, not
  unbiased cross-task transfer estimates.
- The script does not evaluate or read any outer heldout task rows.
- It assumes `runs/k61loss_education_{mse,corr,rank,combined}/gate_last.pt`
  were all trained with matching source checkpoints, epochs, seed and
  weights. Any mismatch stops with an error.

# K2-B1 routing ablations: SAME trained experts

Compare fixed frozen-text `semantic`, uniform averaging `uniform`, and
fixed-seed Dirichlet random weights `random` **without training again**.

```bash
git pull origin main
python -m unittest discover -s tests -p 'test_ktrans_routing_ablation.py' -v
CKPT=runs/k2b1_digital_total_cell_intensity/best.pt
for MODE in semantic uniform random; do
  python -u scripts/analyze_ktrans_predictions.py \
    --checkpoint "$CKPT" --routing-mode "$MODE" --routing-seed 42 \
    --output "runs/k2b1_routes/digital_total_cell_intensity/$MODE"
done
```

Print side-by-side:

```bash
python - <<'PY'
import pandas as pd
from pathlib import Path
root=Path('runs/k2b1_routes/digital_total_cell_intensity')
pieces=[]
for mode in ['semantic','uniform','random']:
    df=pd.read_csv(root/mode/'diagnostics.csv')
    df['route']=mode
    pieces.append(df)
all_df=pd.concat(pieces,ignore_index=True)
print(all_df[['route','protocol','mse','r2','bias','pearson','pred_std']].round(3).to_string(index=False))
PY
```

All routes use the same checkpoint and exactly the same image features,
source-task experts, source-city/heldout-task split and random seed. Only
the **unseen task** route is overridden. Seen source-task predictions
still use their own experts, so meta-UCSI should be **identical** across
the three ablations. Random is a single seeded draw, not a broad statistical
random baseline; repeat with additional seeds to estimate variability.

For the six-task LOTO, first train one K2-B1 checkpoint per source task
(as in `KTRANS_K2B1.md`), then run the three evaluation modes for
each checkpoint, changing `CKPT` and the output directory accordingly.
Don't read original external SCUI/UCSI/UCUI target labels for route
optimization. Meta metrics are development metrics and checkpoint selection
has already used them.

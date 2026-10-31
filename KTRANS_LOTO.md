# K2-B1 six-task LOTO + routing ablation suite

Reuses the frozen RemoteCLIP **image and task-text** caches. Each of the
six source indicators is held out once; a source-expert bank is trained
on the five remaining tasks. Each resulting checkpoint is evaluated using
**the same weights** with semantic, uniform, and five fixed-seed random
routes. The script skips existing checkpoints and routing CSVs.

Run from the repository root:

```bash
git pull origin main
python -m py_compile scripts/run_ktrans_loto.py
python -u scripts/run_ktrans_loto.py --mode all \
  --limit 5000 --epochs 30 --batch-size 128 --eval-batch-size 256
```

Individual phases:

```bash
python -u scripts/run_ktrans_loto.py --mode train
python -u scripts/run_ktrans_loto.py --mode evaluate
python scripts/run_ktrans_loto.py --mode summarize
```

The experiment writes `runs/ktrans_loto/summary/summary.csv` and
`runs/ktrans_loto/summary/all_runs.csv`. Random seeds default to
41,42,43,44,45. The summary first averages random seeds **inside each
held-out task**, then macro-averages the six tasks, with per-task standard
deviation for MSE. This avoids treating the five random draws as five
independent task folds.

**Interpretation:** the six heldout tasks have vastly different target
scales. Macro-MSE is dominated by high-variance tasks, so also inspect
macro-R2, Pearson, mean |Bias| and individual task rows. Selecting
`best.pt` based on the meta validation splits makes these *development*
results, not independent unbiased test scores. Do not use the official
held-out target indicator splits to optimize the router.

To rerun with different parameters, use a fresh `--runs-root` directory;
the suite deliberately reuses existing files.

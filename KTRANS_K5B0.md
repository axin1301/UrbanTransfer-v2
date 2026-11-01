# K5-B0 Complementarity Diagnostics

Use only K5-A nested inner `last.pt` checkpoints. **No new training**.

```bash
git pull origin main
python -m py_compile scripts/diagnose_k5b0_complementarity.py
python -m unittest discover -s tests -p 'test_ktrans_k5b0.py' -v
python -u scripts/diagnose_k5b0_complementarity.py \
  --k5a-dir runs/k5a_digital_total_cell \
  --pseudo-target education_accessibility \
  --batch-size 1024
```

Produces `complementarity/expert_contributions.csv` and
`pair_correlations.csv` under the requested K5-A directory.
Runs on the **inner pseudo-target Meta-SCUI** split, previously used to
produce K5-A transfer scores. No true target task test labels used.

`single_pearson` is an individual normalized expert's Pearson with pseudo
target. `delta_pearson` is the difference between the uniform average of
all four experts and the uniform average of the other three. A positive value
means adding that expert improved the combined spatial Pearson. All values
are strictly **diagnostic**, computed using pseudo-target evaluation labels,
and cannot be used for deployable target-specific expert selection.

Pair correlation is between two standardized expert prediction arrays;
`pair_pearson` is Pearson for their equal-weight average versus target.
A high pair correlation indicates possible redundancy, not necessarily
low complementarity. A high single-expert Pearson does not guarantee a
positive marginal contribution.

Since raw experts output standardized latent values and the pseudo target
uses a different label scale, **do not compare their uncalibrated MSE**;
this diagnostic intentionally reports only scale-invariant Pearson.

For all five pseudo-targets, omit `--pseudo-target`. The output prints
one summary line per task followed by detailed contributions. To avoid
scrolling, target one pseudo-task at a time.

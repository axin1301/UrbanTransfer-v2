# M6.5 DEV unattended frozen robustness suite

Source-only, non-training stability checks of local spatial neighborhood
correspondence. The existing M6.5 Dev checkpoint is **never modified**.
No target Education labels, target-score selection or official test files.

## Fixed tests

1. Real K=8 neighborhood (must reproduce existing M6.5 Dev metrics
   within 1e-4 in MSE/Pearson/bias).
2. Zero-context stress test (potential out-of-distribution).
3. City-mean neighborhood: one average physical-tile neighbor vector per
   held city; removes within-city tile context and keeps city-level signal.
4. 20 within-city random physical-tile derangements, seeds 10001–10020.
   Same physical image under both tasks gets the exact same remapped
   neighborhood. All original image, city, indicator, geo context and
   cached Qwen tokens stay fixed.

Metrics: Population/Food MSE, MAE, Pearson, bias, prediction std, router
weight, mean absolute change relative to real predictions.
Per-city differences in MSE and Pearson plus city counts are recorded.
Report includes average permutation metrics, one-sided permutation
rank summaries (not independent hypothesis-test p-values), and
2000-resample **city-cluster** bootstrap 95% intervals of the mean
perturbed-minus-real MSE (keeping original city sample weights).

**Interpretation:** positive delta MSE (perturbed-minus-real) indicates
real features improve fitted-model performance over this perturbation;
positive delta Pearson (real-minus-perturbed) indicates better correlation
with real features. Paired frozen-input perturbations are informative
but *not* retraining controls or proof of causal transferable information.
Bootstrap intervals with few cities may be wide/unstable.

## Run from repository root on allocated GPU

```bash
git pull --ff-only origin main
python -m unittest discover -s tests -p 'test_k7_m65_robustness_suite.py' -v

python -u scripts/run_k7_m65_robustness_suite.py \
  --full-dir runs/k7/m62_full_seed42 \
  --dev-dir runs/k7/m65_neighborhood_dev_seed42 \
  --neighbor-cache runs/k7/m65_neighbors_k8.pt \
  --output runs/k7/m65_dev_robustness_seed42 \
  --permutations 20 --seed-start 10001 \
  --batch-size 256 --device cuda \
  2>&1 | tee m65_robustness.log
```

### Truly unattended / Slurm batch job

Use the GPU queue and project/account rules already in force at your
site. **Do not run a heavy GPU job directly on a login node.**
Example job script in your repository root (modify `--gpus=1`,
partition, account, time as required by your specific HPC):

```bash
cat > m65_robustness.sbatch <<'SH'
#!/bin/bash
#SBATCH --job-name=m65-robust
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gpus=1
#SBATCH --output=m65_robustness_%j.log
set -euo pipefail
cd /pfs/lustrep4/scratch/project_462001163/xiyanxin/UrbanMoE/UrbanTransfer-v2
python -u scripts/run_k7_m65_robustness_suite.py \
  --full-dir runs/k7/m62_full_seed42 \
  --dev-dir runs/k7/m65_neighborhood_dev_seed42 \
  --neighbor-cache runs/k7/m65_neighbors_k8.pt \
  --output runs/k7/m65_dev_robustness_seed42 \
  --permutations 20 --seed-start 10001 \
  --batch-size 256 --device cuda
SH
sbatch m65_robustness.sbatch
```

If your GPU node requires a module/conda activation, insert the exact
same environment setup you used for M6.5 final training. Consult existing
job scripts for valid GPU resource directives.

### Resume

Every finished condition is saved atomically under
`runs/k7/m65_dev_robustness_seed42/predictions/*.npz`.
If the job stops for walltime or preemption, resubmit exactly the same
command/script. Completed conditions are logged as `HIT`.
The fixed config and checkpoint content hash prevent silently mixing
incompatible reruns. The final report is regenerated after all
conditions finish.

### Inspect on waking

```bash
cat runs/k7/m65_dev_robustness_seed42/REPORT.md
column -s, -t runs/k7/m65_dev_robustness_seed42/summary.csv
```

Outputs:
- `REPORT.md`: concise table with 3-decimal source-task results
- `summary.csv`: 3-decimal mode comparisons and uncertainty
- `per_city.csv`: 3-decimal paired per-city comparisons for all seeds
- `metrics.csv`: 3-decimal metrics for all conditions
- `predictions/*.npz`: FULL-PRECISION predictions (necessary for
  accurate downstream aggregation and restart)
- `config.json`: immutable source-only run provenance.

The original Dev / Final M6.5 checkpoints and original metrics are left
unchanged.

# K7 M6.4 — source-only post-hoc calibration of a frozen M6.3

## Question

Full development UCUI: M6.3 Pearson **0.6726** (best of four), but
MSE **1.8520** versus Visual-only **1.7026**. Its UCUI mean prediction
**2.6675** is higher than the UCUI true mean **1.8592**. These
Education development labels motivate the *general calibration
question* but are NEVER permitted to set coefficients, map the
target mean, pick the checkpoint or tune the M6.4 rule.

M6.4 is a **single prespecified zero-label calibration test**, not
a claim that it will fix the Education shift. It adds **no network
training** and requires no new Qwen-VL encoding.

## Source-only method

Load the completed M6.3 full checkpoint and freeze it completely.

For each source task Population and Food, use ONLY the M6.2
full-manifest source **train** rows and their normalized ground
truth values. Fit an affine mapping of frozen M6.3 standardized
outputs `z`:

```
OLS slope_t = cov(z, y_source_std) / var(z)
a_t = clip(1 + 0.5 * (OLS slope_t - 1), 0.75, 1.25)
b_t = 0.5 * (mean(y_source_std) - a_t * mean(z))
calibrated(z,t) = a_t * z + b_t
```

The fixed 50% shrink and positive bounded slopes are
conservative defaults, **not selected from UCUI or source_val**.
Because slopes are positive and there is only a task-wide
offset, this transform cannot change Pearson within one
indicator partition. It may change MSE. It **cannot correct**
an arbitrary unseen-city-specific target calibration bias.

For Education, compute Qwen3-Embedding-0.6B frozen task
similarities from the existing `runs/ktrans_text` cache:

```
weights = softmax([cos(Edu,Pop), cos(Edu,Food)] / 0.2)
a_Edu = weights[Pop] * a_Pop + weights[Food] * a_Food
b_Edu = weights[Pop] * b_Pop + weights[Food] * b_Food
```

No Education numeric labels, per-city information, output means
or Education test samples are used in this process. The
original source-only-imputed Education mean/std remain unchanged.

M6.4 **cannot improve UCUI Pearson by construction**: a fixed,
positive affine transform preserves correlation. Its only
scientific question is whether source-derived affine parameters
improve **UCUI MSE/bias** without requiring target labels.

## Run

```bash
git pull --ff-only origin main
python -m unittest discover -s tests \
  -p 'test_k7_m64_source_calibration.py' -v

python -u scripts/run_k7_m64_source_calibration.py \
  --full-dir runs/k7/m62_full_seed42 \
  --m63-dir runs/k7/m63_city_indicator_full_seed42_bs256 \
  --output runs/k7/m64_source_calibration_seed42 \
  --text-cache-dir runs/ktrans_text \
  --batch-size 256 --device cuda
```

M6.3 training is NOT repeated. The script runs one frozen
forward per source-train row for fitting and one per eval row
for paired original-vs-calibrated metrics (this still requires
cached Qwen-VL token reads, and can take nontrivial time).

Output:
- `runs/k7/m64_source_calibration_seed42/metrics.csv`
- `runs/k7/m64_source_calibration_seed42/calibration.json`
- `runs/k7/m64_source_calibration_seed42/config.json`

Shows original M6.3 and calibrated M6.4 side by side on
source_val (held out from calibration fitting), SCUI, UCSI,
UCUI, and separate UCSI Population/Food. Includes MSE, Pearson,
prediction mean, truth mean, mean bias, prediction std,
truth std and centered error MSE.

## Interpretation

If source-trained affine does NOT reduce UCUI MSE, that is a
useful negative result: Population/Food scalar calibration does
not necessarily transfer to unseen-city Education. Avoid
selecting a different temperature, intercept or correction
amount based on SCUI/UCUI Education feedback.

Source_val is untouched for fitting but, as a source-indicator
tile split within seen cities, is NOT a genuine unseen-city
and unseen-indicator model-selection substitute. SCUI/UCUI
remain development-only because their labels have already
been inspected repeatedly. Official test files remain sealed.

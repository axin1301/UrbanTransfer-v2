# K-TransMoE P2-C2B — fixed-identity independent source expert bank

P2-B2: same pseudo-task SCUI→UCUI competence prediction helps
(pairwise 0.684, macro blend ratio 0.948), while cross-pseudo-task
learning fails (pairwise 0.408, macro blend ratio 1.019). P2-C1
reported mean task-pair conditional loss-effect cosine ~-0.19, while
P2-C2 found 0/30 identical same-name source expert heads across
nested LOTO banks (mean parameter cosine 0.664). The **name** of a
source expert did not correspond to frozen weights between folds.

P2-C2B is a *mechanism isolation probe*, not an architectural
replacement for K4: train five INDEPENDENT, fixed-identity Ridge
source-task experts on **frozen cached RemoteCLIP + static spatial
context**, with one consistent inner source city split. Each expert
uses ONLY its own source task labels from inner seen cities, exactly
once; no shared trainable trunk, no pseudo-target conditional fitting.
All pseudo-task evaluations use this exact same set of five frozen
expert coefficients. Each pseudo-task excludes its own expert.

**Important confound:** Ridge experts are architecturally different
from K4's learned shared neural experts; a change in cross-task
transfer or MSE does NOT, by itself, identify the causal effect of
fixing expert identities. This is a cheap CPU diagnostic to test
whether a stable-bank prototype even supports task sharing.

## One-command source-only CPU run

At repo root on the HPC, after checking out updated `main`:

```bash
git pull origin main
bash scripts/run_ktrans_p2c2b_all.sh
```

The shell script runs the four relevant unit test suites first, then
the complete experiment. Set `P2C2B_INPUT` and `P2C2B_OUTPUT` to
override the defaults or pass additional CLI args (e.g.
`--feature-dir` or `--image-root`) to the shell script. Original
nested K5-A `protocol.json` and full source manifest paths are
recovered from the EXISTING
`runs/ktrans_competence/education_all_outer_train_dev/protocol.json`
and cross-checked. No invented `$K5A_PROTOCOL` path necessary.

Expected dependencies: NumPy, PyTorch, Pandas, PIL, and the same
frozen RemoteCLIP cached SQLite files previously used by K4/K5-A
(`runs/m3_features/remoteclip_features.sqlite3`). **No GPU
inference, RemoteCLIP image recomputation, Qwen3-VL calls, or
expert neural network training.**

Optionally run manually:

```bash
python -u scripts/run_ktrans_p2c2b_stable_bank.py \
  --input runs/ktrans_competence/education_all_outer_train_dev \
  --knowledge-cache runs/ktrans_competence/full_source_evidence.jsonl \
  --reuse-cache runs/k7b_pilot/evidence.jsonl \
  --extra-reuse-cache runs/k7d_education/evidence.jsonl \
  --output runs/ktrans_competence/p2c2b_stable_source_dev \
  --alpha 10 --competence-alpha 10 --temperature 0.5 \
  --seed 42 --min-group 8
```

The script validates COMPLETE source manifest membership, original
source label/visual knowledge alignment, original per-pseudo-task
SCUI/UCUI city split against a SINGLE common heldout set, outer
source isolation and all four named experts per tile. It refuses
missing cached feature vectors rather than silently fetching images
or computing new embeddings.

## What exactly is trained?

For each source task `k`, train one positive-regularization Ridge
on normalized **source-own** log1p labels in inner seen cities:

```text
x = [precomputed RemoteCLIP image embedding, lon, lat, log1p(area)]
z_k = (y_k - mean(y_k, train-k)) / std(y_k, train-k)
theta_k = ridge_fit(x, z_k; alpha=10, unpenalized bias)
```

The input representation is fixed, shared by the five models but
NEVER jointly supervised. Each expert `k` has ONE coefficient set,
ONE source-train normalization pair and ONE verified full-state SHA256
across all pseudo tasks. Freeze and persist the bank as
`stable_expert_bank.npz`; no pickle. This implementation does not
adopt the previous learned shared K4 trunk or its fold-dependent
source semantic routing.

When evaluating pseudo task `t` use all `k != t`. The target
indicator's own source expert is excluded **even though its own
source labels are present elsewhere in the overall diagnostic**.
All four available expert normalized predictions are converted to
one **common**, source-only target proxy scale using the uniform
mean of those four experts' train means/stds. This is not a learned
prediction of target-specific scale and is not directly comparable
in absolute MSE to K4/K5-A semantic output scale. Compare relative
routing benefit against a bank-specific uniform blend.

The source ability estimator is held fixed at P2-B1 architecture,
Ridge penalty 10, temperature 0.5; its supervision is exclusively
other source pseudo-task SCUI and its output is evaluated on
withheld source task SCUI/UCUI. The supplementary same-task
SCUI→UCUI analysis has in-task label supervision and is NOT a
zero-shot cross-task result.

## Outputs

- `stable_expert_training.csv`: source training counts, unseen count,
  full model hash and source-own moments.
- `stable_expert_bank.npz`: trained frozen coefficients + moments
  with roundtrip SHA256 integrity checking.
- `stable_evidence/competence_rows.jsonl`: identical original
  task-tile/expert/partition VLM evidence/labels, REPLACED
  predictions/squared errors and stable per-expert state hashes.
- `stable_p2b1_macro.csv`, `stable_p2b1_source_task.csv`:
  Task/Cue/Task-Cue/Permuted-Cue cross-task source-only results.
- `stable_p2b2_macro.csv`, `stable_p2b2_task_ucui.csv`:
  Same-task supervised city-only versus cross-task UCUI
  on identical source pseudo task-tile IDs.
- `stable_expert_structure.csv`, `stable_rank_stability.csv`,
  `stable_residual_covariance.csv`.
- `stable_cue_effects.csv`,
  `stable_task_pair_ability_similarity.csv`,
  `stable_masked_triplet_svd.csv`: same P2-C1
  masked conditional expert competence structural diagnostic,
  except expert model parameters are globally fixed.
- `protocol.json` and `experiment.log`: source provenance,
  current settings and all console summaries.

## Decision framework

1. Does **cross-task UCUI** real `task_cue_expert` improve over
   both `task_expert` and permuted controls on pairwise accuracy
   AND relative blend ratio? If not, stable expert identity alone
   does not rescue current alignment.
2. Is task-pair cue-effect similarity more positive than the old
   -0.19? If so, inspect per-task pair and cue masks; centered
   within-tile losses enforce negative correlations and can
   produce misleading global averages.
3. Does **same-task** supervised cue prediction still beat
   permutations? If not, the new Ridge source experts may be too
   similar or too weak to exhibit conditional specialization.
4. Do not tune alpha, feature recipes or routing temperature on UCUI
   results. Record negative outcomes, then plan a strict nested
   source-only selection protocol before final unseen-city/task
   claims.

**This is explicitly POSTHOC DEV.** The original outer K4
`best.pt` provenance was selected using Education meta labels.
Inner SCUI was already used for K5-A development, UCUI was inspected
in P2-Bx, and the same source tasks now train their own expert models.
The own pseudo-task head is excluded from its competence test and
other task experts never see its labels, but this is not an
independent source-task test or a confirmed strict zero-shot
generalization result.

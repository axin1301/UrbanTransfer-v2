# K-TransMoE empirical competence evidence, stage P1

**Research goal:** construct source-only, tile-level evidence for *conditional transfer competence*, not another Gate adjustment. No new router, no Qwen3-VL generation and no model training in this stage.

## Correctness and label isolation

For each of the five outer source tasks `t` (Education is the outer DEV heldout), load the corresponding **K5-A INNER** `inner_models/<t>/last.pt`. Its expert bank contains the other four source tasks; `t` itself is absent. Inspect only its `meta_SCUI` and `meta_UCUI` rows, which are pseudo-target samples with no inner-model supervised training labels.

Require:
- Every inner-model sample is a known outer `train` image/task identity. Reject any sample from the outer heldout task or cities.
- The nested CSV path equals the specific K5-A source-only training CSV. Inner expert list is exactly outer source tasks minus pseudo target.
- Only `last.pt`, never the inner `best.pt`. K4 outer `best.pt` is read only to validate outer task, sample identities, and provenance; never to generate competence predictions.
- Exact image/task matching to the **existing label-free K7-D Complete-Case manifest**; rows absent from that sample are counted as unmatched. Each selected row requires a valid prompt/model/condition-keyed `image_task` Qwen3-VL record. Missing records are counted, not generated/imputed.
- No target-task labels, outer target rows, inner pseudo-task expert or inner pseudo-task output scale labels contribute to predictions.

For a pseudo-heldout task `t`, use the INNER source-only semantic route `q_t` to construct `mu_t=sum_s q_t[s] mu_s` and `sigma_t=sum_s q_t[s] sigma_s`. Every source expert `s` predicts `p_{t,s}(x)=mu_t+sigma_t*z_s(x)`. Compare `p_{t,s}` to the **SOURCE pseudo-target** label `y_t(x)` to obtain `error_log1p` and `squared_error`. These are source-task competence signals; each expert receives the *same* task-scale transform, so source-expert comparison is well defined. We do **not** use evaluation labels to estimate the transform.

VLM features are the original 18 ordinal values and 18 presence masks from `image_task`; the source-only pseudo-task labels are written separately and never passed to the VLM.

## Run on the same seed-42 HPC setup

```bash
git pull origin main
python -m py_compile scripts/build_ktrans_empirical_competence.py urban_transfer/ktrans_empirical_competence.py
python -m unittest discover -s tests -p 'test_ktrans_empirical_competence.py' -v

python -u scripts/build_ktrans_empirical_competence.py \
 --checkpoint "$K4_CKPT" \
 --k5a-protocol "$K5A_PROTOCOL" \
 --manifest runs/k7d_education/manifest_complete_cases.json \
 --knowledge-cache runs/k7d_education/evidence.jsonl \
 --reuse-cache runs/k7b_pilot/evidence.jsonl \
 --output runs/ktrans_competence/education_source_dev \
 --batch-size 512
```

If `$K4_CKPT` or `$K5A_PROTOCOL` are unset, recover their paths from
`runs/k7d_education/single_seed42/image_task_last.pt` as documented
in `KTRANS_K7G_ERROR_DIAGNOSTICS.md`.

## Outputs

- `competence_rows.jsonl`: one record per (inner pseudo-task, split, matched tile, other source expert), with frozen per-expert standardized output, scale inferred from other SOURCE tasks, raw `log1p` pseudo-target SOURCE label/error, VLM values/masks, and source identity. Potentially sensitive path fields are local research metadata.
- `coverage.json`: total pseudo rows, matched K7-D train manifest, rows with valid `image_task` evidence, missing evidence, and final expert rows, separately for each pseudo task and inner partition. Never treat filtered coverage as representative without checking.
- `protocol.json`: inner checkpoint paths, excluded outer task, provenance, scale, and methodological caveats.

The proposed competence model must use `meta_SCUI` ONLY for source-side development and reserve inner `meta_UCUI` as a **source-task, unseen-inner-city test** of its conditional capability estimates. Neither is a truly independent outer target test. In particular, `meta_SCUI` already informed K5-A ridge fitting, so avoid describing it as independent of all model selection.

**Do not train a new expert router based on the outer Education UCUI results.** K4 `best.pt` is meta-selected, and any future strict evaluation requires an outer checkpoint selected without heldout task labels plus audit of preprocessing, source city coverage, and independent target testing.

## Next P2 (not implemented by this exporter)

Compare a cue-conditional empirical competence estimator against task-only reliability, random expert profiles, and the nearly-collinear six-cue conceptual proxy. Evaluate capability prediction itself on inner `meta_UCUI` before attempting new outer target tasks. Verify distinctiveness and stability of expert profiles across pseudo tasks; limited source tasks can make the competence model weakly identifiable.

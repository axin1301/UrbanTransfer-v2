# K-TransMoE: protocol reset and Knowledge–Expert Alignment foundation

## Research goal

Zero-shot transfer to a NEW urban indicator task and UNSEEN city from
supervised source tasks. Frozen RemoteCLIP features yield a shared
spatial representation and K4 source-specific experts. Task semantics
and satellite-scene evidence should select source experts according
to *documented competence*, not merely feed an unconstrained 36-d
MLP. Never train/select models on outer target-task labels.

## Provenance finding (P0)

`urban_transfer/ktrans.py` in K4 `run()` selects `best.pt` using
mean MSE over `meta_SCUI`, `meta_UCSI`, `meta_UCUI`. SCUI/UCUI contain
the outer heldout task labels. Thus old `best.pt` experiments are
**DEV** even when downstream K5-A/K7 stages fit only source rows.
K5-A fits ridge from INNER pseudo-heldout source-task transfer scores
and uses inner `last.pt` (good), but inherits outer best-epoch
selection if invoked with `best.pt`.

A frozen K4 `last.pt` is an alternative checkpoint candidate, **not
an automatic certificate of strict independence**: model/data
preprocessing, experiment choice, outer-task selection and heldout
label inspection still need an auditable protocol. Explicitly avoid
reusing source-trained K4 expert outputs on those exact same samples
as independent pseudo-task validation.

```bash
git pull origin main
python -m unittest discover -s tests -p 'test_ktrans_alignment_protocol.py' -v

python -u scripts/audit_ktrans_zero_shot_protocol.py \
 --checkpoint "$K4_CKPT" \
 --k5a-protocol "$K5A_PROTOCOL" \
 --manifest runs/k7d_education/manifest_complete_cases.json \
 --output runs/ktrans_protocol_audit/education_dev.json
```

This emits explicit `DEV_ONLY` or `CANDIDATE_SOURCE_ONLY`,
never `strict_protocol_certified=true`.

## Source-only expert-cue proxy (P1 exploratory)

A narrow first hypothesis, not yet the final knowledge-expert
alignment model:

1. From existing K5-A `pseudo_transfer_scores.csv`, obtain
   `Pearson(pseudo_target, source_expert)` for INNER source-task
   pseudo-heldout pairs. All tasks must be source tasks, and
   pseudo target must differ from source expert.
2. For each pseudo source task obtain **text_only** VLM conceptual
   six-cue relevance, with the exact prompt/version/cache key.
3. For each source expert build a cue proxy as the positive-transfer
   weighted mean of its pseudo-targets' conceptual six-cue relevance.
   Each expert needs at least two pseudo-task pairs with positive
   support; fail closed otherwise.
4. Given a task–image VLM observation, use visible cue × proxy
   relevance × confidence to form demand. Mask null scores
   (no ungrounded confidence); zero observed evidence falls back
   exactly to K5-A.
5. Cosine-match six-dimensional demand against source cue proxies,
   softmax scores, and mix with fixed K5-A prior.

**Important limitations:** The cue proxy is inferred from source-task
associations, not empirically proven sensitivity of an individual
expert to a cue. K5-A pseudo scores are source-ONLY but are themselves
derived from inner supervised models. Six cues from only a handful of
pseudo tasks are weakly identified; do not claim causal visual
expertise or independent performance. It is a *transparent baseline
for a more ambitious source-only capability learning module*, with
fixed parameters and no target-label tuning.

Generate only source profiles (NO outer target evaluation):

```bash
python -u scripts/prepare_ktrans_source_expert_profiles.py \
 --manifest runs/k7d_education/manifest_complete_cases.json \
 --k5a-protocol "$K5A_PROTOCOL" \
 --cache runs/k7d_education/evidence.jsonl \
 --reuse-cache runs/k7b_pilot/evidence.jsonl \
 --output runs/ktrans_alignment/source_expert_cue_proxy_dev.json
```

On legacy best.pt data, the file must be marked development-only,
not strict-zero-shot. No optimization is performed and no target
rows or target labels are read to construct this source profile.

## Next experiment, before a new outer evaluation

1. Reserve an independent outer task/city evaluation protocol that
   has not guided method selection; use source-only checkpoint
   selection, independent nested pseudo-transfer folds and documented
   fixed VLM prompts.
2. Measure expert competence **conditional on actual observable
   cues** using heldout source rows from independently trained inner
   models, rather than solely conceptual task-level relevance.
3. Compare fixed K5-A, random/permuted cue profiles, source-only
   cue alignment, task-only alignment, and image/task alignment with
   the exact same data subset, fixed hyperparameters and seeds.
4. Once the method is locked using source-only evidence, evaluate
   the truly heldout four target tasks and cross-city UCUI, reporting
   MSE/Pearson to three decimals, full coverage and label isolation.

The current Education K7-D/H history remains a record of exploratory
mechanism development, not an unbiased validation set.

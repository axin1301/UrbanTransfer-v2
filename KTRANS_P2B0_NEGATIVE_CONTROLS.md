# P2-B0 — Negative controls for SOURCE-ONLY conditional expert competence

P2-A found 92/96 matched SCUI/UCUI cue×expert conditional
relative-loss contrasts with the same sign, but this is NOT evidence
that VLM knowledge predicts transferable expertise unless cue/image
correspondence beats appropriate negative controls. P2-B0 adds
predeclared exploratory CPU diagnostics, using the **existing**
3,337 source pseudo-task tile units (13,348 paired expert rows).

## Reproduce

```bash
git pull origin main
python -m unittest discover -s tests -p 'test_ktrans_p2b0_negative_controls.py' -v

python -u scripts/diagnose_ktrans_p2b0_negative_controls.py \
  --input runs/ktrans_competence/education_all_outer_train_dev \
  --output runs/ktrans_competence/p2b0_education_source_dev \
  --min-group 8 --repeats 199 --seed 42
```

No checkpoint loading, no Qwen3-VL calls, no learning new
parameters, no outer Education task labels. This is purely descriptive
analysis of existing inner source pseudo-LOTO errors.

## Three independent checks

1. **Within-task/split cue permutation.** Within a specific source
   pseudo-task and SCUI or UCUI, shuffle the entire task-image VLM
   18-score + 18-mask block across samples. Preserve every expert
   prediction error, city identity, task identity and split. This
   destroys local knowledge alignment while preserving the marginal
   task cue distribution. It does not preserve city-specific cue
   distributions.
2. **Within-task/split/city cue permutation.** Same, but shuffle
   only within the same city. Thus city-level cue confounding is
   preserved, and only intra-city image–cue association is disrupted.
   Singleton city strata cannot be permuted and are counted.
3. **City-cluster bootstrap.** Resample entire cities WITH replacement,
   separately within each source pseudo-task × split; preserve
   complete tile groups and their paired experts. It assesses
   descriptive instability to city composition, not a confirmatory
   confidence interval for previously unseen cities.

All comparisons reuse the **original source SCUI medians** for each
task's observed visual cue. UCUI never defines thresholds.
Missing cue masks and ordinal median ties remain separate. A cue
contrast is reported only if both low/high groups have at least
`--min-group` independent task–tile units. Each named expert is
compared across SCUI and UCUI; nested expert/cue observations are
correlated.

## Outputs

- `observed_direction_details.csv`: the actual SCUI/UCUI signed cue
  contrasts and support counts.
- `effect_by_task_cue.csv`: cue/task level agreement and **absolute
  contrast magnitudes**, not just the direction signs.
- `negative_control_replicates.csv`: seeded 199 repeats per null mode,
  agreement, denominator, number of evidence swaps and strata.
- `negative_control_comparison.csv`: observed agreement, null mean,
  5th/95th percentiles and **descriptive Monte Carlo tail fraction**.
- `protocol.json`: reproducibility metadata and warnings.

## Decision rule for research direction (not a statistical test)

- If the null cue permutations retain comparable SCUI–UCUI agreement,
  the P2-A sign rate likely reflects baseline task/expert structure
  rather than useful image-conditioned expertise. It would be
  premature to fit a new knowledge routing model.
- If real cue alignment is consistently more informative than
  both null modes, examine effect sizes and city bootstrap stability.
  Particularly, a large `n_singleton_strata` means the
  within-city permutation is underpowered; report it explicitly.
- Even strong differences do not establish causal VLM alignment or
  out-of-distribution zero-shot improvement. A task-only baseline,
  paired prospective capability prediction and locked independent
  outer-task evaluation are still necessary.

**Scientific constraints:** This design was proposed after inspecting
inner UCUI results and is therefore POSTHOC DEV. Monte Carlo tail
fractions are NOT confirmatory p-values. Permutations may change
the eligible high/low bin counts; their agreement-rate denominators
are recorded rather than silently held constant. City bootstrap
percentiles must not be interpreted as 95% independent evaluation
confidence intervals. K4 outer `best.pt` remains meta-selected.

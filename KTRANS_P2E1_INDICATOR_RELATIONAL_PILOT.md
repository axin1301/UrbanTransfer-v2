# P2-E1: Indicator-relational, expert-aware Qwen3-VL Pilot

P2-D-VLM demonstrated that the existing `image_task.evidence_relevance`
has per-image variation (60/60 task/split/cue cells; mean deviation
from text-only prior 0.275), but source-task LOTO UCUI pairwise scores
were still below Task–Expert. The current question is **whether VLM
knowledge about target indicator measurement, units, relations to
source indicators and visible regional morphology** better predicts
which existing source expert will transfer.

## Crucial data provenance and units

The repo's `urban_transfer/protocol.py` and
`urban_transfer/task_profiles.py` describe all ten indicators at
a conceptual level and state that targets are modeled as `log1p`.
They do **NOT** establish the raw units, exact formulas or all
statistical denominators (particularly TMI). **No raw unit or exact
formula is invented.** The registry marks them `unverified`.
Expert output label transformation is stated separately from raw
measurement units. The VLM must not infer measured target or
expert error from the image.

Optionally supply a separately reviewed, source-referenced ten-task
JSON at `--reviewed-methodology` to fill exact units and formulas
AFTER method review. Each key must be a task ID, with exactly:
`raw_unit`, `exact_formula`, `raw_statistical_basis`,
`source_reference`. A missing source or incomplete task list fails
closed. **Changing this JSON changes the prompt hashes and must use a
NEW output directory.** Without a confirmed data dictionary, keep
the unverified defaults; do not guess.

Indicator relations are explicitly **hypotheses**:
- the accessibility family shares population-normalized
  distance-decay facility access descriptions but differs in
  facility types and taxonomy;
- telecom indicator group shares network infrastructure themes,
  but LTE/total/TMI subset or exact formula assertions are NOT
  assumed without an external methodology;
- other indicator pairs may share morphology proxy cues, not a
  demonstrated causal link or expert prediction compatibility.

These candidate relations are built from the existing source
descriptions, not target labels or observed expert losses.

## Design and arms

One fixed set of five source pseudo tasks (outer Education excluded),
with exactly four non-own candidate source experts per task.
Deterministic SHA256 task+split+tile ID sampling; no score/error/label
based selection. Default **four pilot tiles per source task and
partition**: 5 × 2 × 4 = 40 selected image/task records.

Four arms:
1. `definition_only`: task/source measurement definitions, units
   (unverified if unknown), source expert identities; no image,
   no asserted indicator relation.
2. `definition_relation`: plus conceptual task–source indicator
   relationship hypothesis; no image.
3. `definition_relation_region`: same plus the satellite image,
   region-specific expert compatibility/transfer risk hypotheses.
4. `permuted_relation_region`: identical image and expert identities,
   but candidate relationship descriptions rotated between experts.
   **The VLM is NOT told which arm is a negative control.**

10 no-image jobs + 80 image jobs = **90 Qwen3-VL-32B inference
jobs** at default sample size. A region image is submitted only
to the two region arms. Every job key hashes model, full prompt,
condition, tile, and pixel budget; cached outputs are validated
and appended without overriding previous valid responses.

Output JSON has four named experts, each with:
`compatibility` 0..3/null, `region_support` 0..3/null,
`transfer_risk` 0..3/null, `complementarity` description,
`uncertainty` 0..3 and `reason`. No observed accuracy or
numeric target labels. Strict schema parser rejects invalid
records; no filler/silent imputation.

**Important limitation:** for some targets, four
relationship hypotheses can all have the same general
`possible_shared_urban_proxy` type. Permuting these effectively
identical cards will NOT produce a useful negative control; inspect
per-task relation diversity rather than declaring an observed
zero delta evidence that indicator relationships do not matter.

## HPC commands

At repository root:

```bash
git pull origin main
python -m unittest discover -s tests -p 'test_ktrans_p2e1_indicator_pilot.py' -v
python -m unittest discover -s tests -p 'test_ktrans_p2e1_eval.py' -v

# CPU only; freezes all 90 exact model prompts/sample IDs and checks unknown units:
python -u scripts/run_ktrans_p2e1_indicator_pilot.py --dry-run

# With a real ROCm Qwen3-VL-32B multi-GPU allocation:
python -u scripts/run_ktrans_p2e1_indicator_pilot.py \
  --model Qwen/Qwen3-VL-32B-Instruct \
  --batch-size 8 \
  --max-new-jobs 16

# After inspecting first valid prompt JSON entries, finish remaining
# cache jobs using the SAME command without a max-new-jobs cap:
python -u scripts/run_ktrans_p2e1_indicator_pilot.py

# Requires all 90 jobs validated; otherwise fails closed:
python -u scripts/evaluate_ktrans_p2e1_indicator_pilot.py
```

Default:
`runs/ktrans_competence/p2e1_qwen3vl32b_source_dev/`

Files:
- `pilot_manifest.json`: frozen task/tile/image/source expert IDs,
  conceptual task specs, exact prompts and key hashes.
- `vlm_evidence.jsonl`: append-only valid/invalid raw VLM outputs
  + parsed expert hypotheses, exact provenance.
- `vlm_expert_hypotheses.csv`: source-only VLM descriptive scores.
- `protocol.json`: cache completeness and warnings.
- `p2e1_source_lopo_task.csv`,
  `p2e1_source_lopo_macro.csv`: read-only posthoc source
  task-heldout UCUI, with a fixed small Ridge calibrator trained on
  other source pseudo-task SCUI only.

The four prompt arms use **identical source tile/task pairs**
for image-required conditions, with static no-image comparisons.
No external target test data or true expert error is provided to
Qwen3-VL. For calibration, only observed SOURCE SCUI of OTHER tasks
is fitted to P2-B1 normalized expert regret; UCUI labels are used
only as descriptive outcome evaluation, never for prompt selection,
route training or hyperparameter tuning. Fixed alpha=10,
temperature=0.5.

## Interpretation

- Does adding task relations change the VLM expert-specific
  compatibility distribution compared to definition-only?
- Does adding the satellite image produce nontrivial
  `region_support` variance for a given task/expert?
- Does real relation+image beat relationship-shuffled image on
  heldout-source-task UCUI, while exceeding definition-only and
  Task–Expert baseline? If no, do not train Hyper-X/ZOOTER yet.
- Are uncertain/null judgments common? Treat that as useful
  epistemic information, not as failure to force a high score.
- Results with 4 tiles per task/split are severely underpowered and
  **not independent zero-shot validation**: K4 best.pt was
  Education-meta-selected, and source UCUI has been inspected.

If the pilot is promising, the next scientific step is new
source-fold/target-independent, pre-registered evaluation, not
immediate full-scale 3,337-tile VLM generation.

## Qwen3-VL-32B runtime and cache separation

As of P2-E1-32B, the inference default is
`Qwen/Qwen3-VL-32B-Instruct` (official model), starting
`--batch-size 8`, configurable `--batch-size 16`,
and `--max-new-tokens 1536`. The previous 8B pilot's
`runs/ktrans_competence/p2e1_indicator_relation_source_dev/`
is intentionally **not** reused: the 32B model is a different
experiment with a separate prompt-hashed manifest/cache/output folder.
Do NOT change the model in an existing manifest directory.

The batch runner never mixes image-bearing and text-only prompts,
keeps request-key output order, and reduces 16→8→4→2→1 (or
8→4→2→1) on GPU out-of-memory. Inability to load the 32B weights,
or an OOM at batch size one, still requires more available accelerator
memory or a different serving setup: batch backoff cannot solve model
weight memory exhaustion. Qwen3-VL-32B BF16 weights alone require
roughly 64 GiB before activations/vision overhead; ensure the
allocated ROCm GPU group can hold the full model. Use the same
HPC environment with a recent Qwen3-VL-compatible Transformers.

Robust JSON parsing now accepts fenced JSON or an explanatory prefix,
but **does not relax** expert identity, field count, 0..3 score ranges,
null/missingness or required four-expert constraints. Every invalid
response remains in the append-only JSONL with its raw text and
`parsed=null`; subsequent invocations retry invalid jobs. The longer
1536-token default reduces the risk of truncating the four expert
profiles but cannot guarantee valid generations.

For a first smoke test that reaches the multimodal arm, use
`--max-new-jobs 16 --batch-size 8`: the first ten jobs are
definition-only/definition+relation text prompts. A cap of 8
runs only text prompts and **does not** test image batching.

Recommended 32B commands:

```bash
git pull origin main
python -m unittest discover -s tests -p 'test_ktrans_p2e1_indicator_pilot.py' -v
python -u scripts/run_ktrans_p2e1_indicator_pilot.py --dry-run
# On a suitable multi-GPU ROCm allocation:
python -u scripts/run_ktrans_p2e1_indicator_pilot.py --batch-size 8 --max-new-jobs 16
# After checking parser success and resource usage:
python -u scripts/run_ktrans_p2e1_indicator_pilot.py --batch-size 16
# Only when all 32B jobs are complete:
python -u scripts/evaluate_ktrans_p2e1_indicator_pilot.py
```

Increasing the parameter count is a hypothesis, not proof of
better knowledge quality or improved cross-task transfer: compare
32B valid output rate, expert differentiation, region-specific
variation and source-LOPO performance against frozen baselines.

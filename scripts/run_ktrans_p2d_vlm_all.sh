#!/usr/bin/env bash
# P2-D-VLM: compare original and stable source experts with task-conditioned VLM relevance.
set -euo pipefail
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-4}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
OUT="${P2D_VLM_OUTPUT:-runs/ktrans_competence/p2d_vlm_source_dev}"
mkdir -p "$OUT"
python -m unittest discover -s tests -p 'test_ktrans_p2d_vlm_relevance.py' -v
python -u scripts/run_ktrans_p2d_vlm_relevance.py \
  --original-input runs/ktrans_competence/education_all_outer_train_dev \
  --stable-input runs/ktrans_competence/p2c2b_stable_source_dev \
  --knowledge-cache runs/ktrans_competence/full_source_evidence.jsonl \
  --output "$OUT" \
  --alpha 10 --temperature 0.5 --seed 42 "$@" 2>&1 | tee "$OUT/experiment.log"

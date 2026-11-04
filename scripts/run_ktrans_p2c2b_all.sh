#!/usr/bin/env bash
# P2-C2B full CPU sanity and mechanistic experiment (source-only DEV).
# Run at the UrbanTransfer-v2 repository root on the HPC login/CPU node.
set -euo pipefail
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-4}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
INPUT="${P2C2B_INPUT:-runs/ktrans_competence/education_all_outer_train_dev}"
OUT="${P2C2B_OUTPUT:-runs/ktrans_competence/p2c2b_stable_source_dev}"
mkdir -p "$OUT"
echo "[p2c2b-all] checking source-only unit tests"
python -m unittest discover -s tests -p 'test_ktrans_p2c2b_stable_bank.py' -v
python -m unittest discover -s tests -p 'test_ktrans_p2b1_source_lopo.py' -v
python -m unittest discover -s tests -p 'test_ktrans_p2b2_failures.py' -v
python -m unittest discover -s tests -p 'test_ktrans_p2c_shared_structure.py' -v
echo "[p2c2b-all] running frozen bank and paired diagnostics"
python -u scripts/run_ktrans_p2c2b_stable_bank.py \
  --input "$INPUT" \
  --knowledge-cache runs/ktrans_competence/full_source_evidence.jsonl \
  --reuse-cache runs/k7b_pilot/evidence.jsonl \
  --extra-reuse-cache runs/k7d_education/evidence.jsonl \
  --output "$OUT" \
  --alpha 10 \
  --competence-alpha 10 \
  --temperature 0.5 \
  --seed 42 \
  --min-group 8 \
  "$@" 2>&1 | tee "$OUT/experiment.log"
echo "[p2c2b-all] completed $OUT"

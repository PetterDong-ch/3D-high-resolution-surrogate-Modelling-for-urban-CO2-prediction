#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
DATA_ROOT="${V40_DATA_ROOT:-${ROOT}/external_data}"
JOBS_ROOT="${CAMDEN_JOBS_ROOT:-${DATA_ROOT}/raw/JOBS}"
CHECKPOINT="${CHECKPOINT:-${ROOT}/checkpoints/best_model.pt}"
OUT_DIR="${OUT_DIR:-${ROOT}/reproduced/v40_report_evaluation}"
RUNTIME="${ROOT}/reproduction"

mkdir -p "${OUT_DIR}" "${ROOT}/logs"
export PYTHONPATH="${RUNTIME}${PYLIBS_ROOT:+:${PYLIBS_ROOT}}"

"${PYTHON_BIN}" "${RUNTIME}/scripts/evaluate_v15_test_outputs.py" \
  --jobs-root "${JOBS_ROOT}" --checkpoint "${CHECKPOINT}" \
  --v13-sidecar-root "${DATA_ROOT}/v33_context_sidecar_camden_z001_010_no1112" \
  --v22-prev-sidecar-root "${DATA_ROOT}/v33_previous_co2_sidecar_camden_z001_010_no1112" \
  --v14-normalization-root "${DATA_ROOT}/v33_context_sidecar_camden_z001_010_no1112" \
  --out-dir "${OUT_DIR}" --samples-per-job 4 --visual-samples 40 \
  --visual-local-z 1,4,8,10 --fixed-z0 0 \
  --eval-layer-min 1 --eval-layer-max 10 --eval-min-layer-overlap 8 \
  --exclude-months 11,12 --save-delta-visuals \
  --delta-active-threshold 0.75 --inference-mode direct \
  2>&1 | tee "${ROOT}/logs/reproduce_camden_v40_evaluation.log"

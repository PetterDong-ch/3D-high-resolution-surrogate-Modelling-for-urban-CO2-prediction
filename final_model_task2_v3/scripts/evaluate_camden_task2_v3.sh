#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
PYLIBS_ROOT="${PYLIBS_ROOT:-}"
PROFILE_CACHE_ROOT="${PROFILE_CACHE_ROOT:-${ROOT}/external_data/camden_average_profile_cache}"
CHECKPOINT="${CHECKPOINT:-${ROOT}/checkpoints/camden_background_enhancement_best_model.pt}"
OUT_DIR="${OUT_DIR:-${ROOT}/evaluations/camden_background_enhancement_v3}"
LOG_PATH="${LOG_PATH:-${ROOT}/logs/evaluate_camden_task2_v3.log}"

mkdir -p "${ROOT}/logs" "${OUT_DIR}"

PYTHONPATH="${ROOT}/src${PYLIBS_ROOT:+:${PYLIBS_ROOT}}" "${PYTHON_BIN}" -m task2_v3.cli \
  --profile-cache-root "${PROFILE_CACHE_ROOT}" \
  --out-dir "${OUT_DIR}" \
  --eval-only \
  --checkpoint "${CHECKPOINT}" \
  --seed "${SEED:-123}" \
  2>&1 | tee "${LOG_PATH}"

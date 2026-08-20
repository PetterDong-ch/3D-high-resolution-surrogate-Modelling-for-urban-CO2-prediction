#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
PYLIBS_ROOT="${PYLIBS_ROOT:-}"

CACHE_ROOT="${CACHE_ROOT:-${ROOT}/generated/stage2_cache}"
PREV_SIDECAR_ROOT="${PREV_SIDECAR_ROOT:-${ROOT}/generated/previous_co2_sidecar}"
GLOBAL_SIDECAR_ROOT="${GLOBAL_SIDECAR_ROOT:-${ROOT}/generated/full_domain_context_sidecar}"
CHECKPOINT="${CHECKPOINT:-${ROOT}/checkpoints/stage2_best_model.pt}"
OUT_DIR="${OUT_DIR:-${ROOT}/evaluations/v6_stage2_best_model_val}"
LOG_PATH="${LOG_PATH:-${ROOT}/logs/evaluate_twostage_v6_stage2.log}"

mkdir -p "${ROOT}/logs" "${OUT_DIR}"

PYTHONPATH="${ROOT}/src:${ROOT}/runtime${PYLIBS_ROOT:+:${PYLIBS_ROOT}}" "${PYTHON_BIN}" "${ROOT}/scripts/evaluate_v40_with_stage1_met.py" \
  --cache-root "${CACHE_ROOT}" \
  --prev-sidecar-root "${PREV_SIDECAR_ROOT}" \
  --global-sidecar-root "${GLOBAL_SIDECAR_ROOT}" \
  --checkpoint "${CHECKPOINT}" \
  --out-dir "${OUT_DIR}" \
  --split val \
  --batch-size "${BATCH_SIZE:-1}" \
  --num-workers "${NUM_WORKERS:-2}" \
  --visual-samples "${VISUAL_SAMPLES:-12}" \
  --visual-local-z "${VISUAL_LOCAL_Z:-0,4,8}" \
  2>&1 | tee "${LOG_PATH}"

#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
PYLIBS_ROOT="${PYLIBS_ROOT:-}"

CACHE_ROOT="${CACHE_ROOT:-${ROOT}/generated/stage2_cache}"
PREV_SIDECAR_ROOT="${PREV_SIDECAR_ROOT:-${ROOT}/generated/previous_co2_sidecar}"
GLOBAL_SIDECAR_ROOT="${GLOBAL_SIDECAR_ROOT:-${ROOT}/generated/full_domain_context_sidecar}"
INIT_CHECKPOINT="${INIT_CHECKPOINT:-${ROOT}/checkpoints/v40_initialization.pt}"
OUT_DIR="${OUT_DIR:-${ROOT}/runs/v6_stage2_full_domain_context}"
LOG_PATH="${LOG_PATH:-${ROOT}/logs/train_twostage_v6_stage2.log}"

mkdir -p "${ROOT}/logs" "${OUT_DIR}"

PYTHONPATH="${ROOT}/src:${ROOT}/runtime${PYLIBS_ROOT:+:${PYLIBS_ROOT}}" "${PYTHON_BIN}" "${ROOT}/scripts/train_v40_with_stage1_met.py" \
  --cache-root "${CACHE_ROOT}" \
  --prev-sidecar-root "${PREV_SIDECAR_ROOT}" \
  --global-sidecar-root "${GLOBAL_SIDECAR_ROOT}" \
  --out-dir "${OUT_DIR}" \
  --init-checkpoint "${INIT_CHECKPOINT}" \
  --epochs "${EPOCHS:-110}" \
  --batch-size "${BATCH_SIZE:-1}" \
  --num-workers "${NUM_WORKERS:-2}" \
  --train-samples-per-epoch "${TRAIN_SAMPLES_PER_EPOCH:-512}" \
  --val-samples "${VAL_SAMPLES:-128}" \
  --base-channels 32 \
  --lr "${LR:-3e-5}" \
  --weight-decay 1e-4 \
  --lr-factor 0.5 \
  --lr-patience 7 \
  --min-lr 1e-6 \
  --early-stopping-patience 18 \
  --layer-min 1 \
  --layer-max 10 \
  --min-layer-overlap 8 \
  --dx 5.0 \
  --dy 5.0 \
  --dz 10.0 \
  --progress none \
  2>&1 | tee "${LOG_PATH}"

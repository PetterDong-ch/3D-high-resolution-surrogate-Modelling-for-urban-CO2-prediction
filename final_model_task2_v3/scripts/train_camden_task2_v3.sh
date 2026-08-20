#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
PYLIBS_ROOT="${PYLIBS_ROOT:-}"
PROFILE_CACHE_ROOT="${PROFILE_CACHE_ROOT:-${ROOT}/external_data/camden_average_profile_cache}"
OUT_DIR="${OUT_DIR:-${ROOT}/runs/camden_background_enhancement_v3}"
LOG_PATH="${LOG_PATH:-${ROOT}/logs/train_camden_task2_v3.log}"

mkdir -p "${ROOT}/logs" "${OUT_DIR}"

PYTHONPATH="${ROOT}/src${PYLIBS_ROOT:+:${PYLIBS_ROOT}}" "${PYTHON_BIN}" -m task2_v3.cli \
  --profile-cache-root "${PROFILE_CACHE_ROOT}" \
  --out-dir "${OUT_DIR}" \
  --epochs "${EPOCHS:-150}" \
  --batch-size "${BATCH_SIZE:-256}" \
  --num-workers "${NUM_WORKERS:-0}" \
  --train-samples-per-epoch "${TRAIN_SAMPLES_PER_EPOCH:-4096}" \
  --val-samples "${VAL_SAMPLES:-0}" \
  --base-channels "${BASE_CHANNELS:-64}" \
  --lr "${LR:-3e-4}" \
  --weight-decay "${WEIGHT_DECAY:-1e-4}" \
  --huber-delta "${HUBER_DELTA:-1.0}" \
  --profile-gradient-weight "${PROFILE_GRADIENT_WEIGHT:-0.05}" \
  --low-layer-alpha "${LOW_LAYER_ALPHA:-1.5}" \
  --low-layer-tau "${LOW_LAYER_TAU:-4.0}" \
  --valid-weight-power "${VALID_WEIGHT_POWER:-0.25}" \
  --target-min-std "${TARGET_MIN_STD:-0.2}" \
  --seed "${SEED:-123}" \
  2>&1 | tee "${LOG_PATH}"

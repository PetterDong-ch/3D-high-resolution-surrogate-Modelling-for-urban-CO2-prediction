#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
PYLIBS_ROOT="${PYLIBS_ROOT:-}"

CACHE_ROOT="${CACHE_ROOT:-${ROOT}/generated/stage2_cache}"
STAGE1_MANIFEST="${STAGE1_MANIFEST:-${ROOT}/generated/stage1_manifest/stage1_manifest.json}"
STAGE1_CHECKPOINT="${STAGE1_CHECKPOINT:-${ROOT}/checkpoints/stage1_local_fno_best_model.pt}"
JOBS_ROOT="${CAMDEN_JOBS_ROOT:-${ROOT}/external_data/camden/JOBS}"
OUT_ROOT="${OUT_ROOT:-${ROOT}/generated/full_domain_context_sidecar}"
LOG_PATH="${LOG_PATH:-${ROOT}/logs/build_full_domain_context_sidecar.log}"

mkdir -p "${ROOT}/logs" "${OUT_ROOT}"

args=(
  "${ROOT}/scripts/build_full_domain_stage1_context_sidecar.py"
  --cache-root "${CACHE_ROOT}"
  --stage1-manifest "${STAGE1_MANIFEST}"
  --stage1-checkpoint "${STAGE1_CHECKPOINT}"
  --jobs-root "${JOBS_ROOT}"
  --out-root "${OUT_ROOT}"
  --splits train,val
  --target-source av3d
  --context-size 80
  --full-h 800
  --full-w 800
  --tile-size "${TILE_SIZE:-256}"
  --tile-stride "${TILE_STRIDE:-256}"
  --shard-size "${SHARD_SIZE:-16}"
  --progress-every "${PROGRESS_EVERY:-10}"
)

if [[ "${OVERWRITE:-0}" == "1" ]]; then
  args+=(--overwrite)
fi

PYTHONPATH="${ROOT}/src${PYLIBS_ROOT:+:${PYLIBS_ROOT}}" "${PYTHON_BIN}" "${args[@]}" 2>&1 | tee "${LOG_PATH}"

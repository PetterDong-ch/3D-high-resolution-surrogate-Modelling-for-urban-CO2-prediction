#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
REGION="${REGION:-camden}"
SPATIAL_CACHE_ROOT="${SPATIAL_CACHE_ROOT:-${ROOT}/external_data/${REGION}_spatial_patch_cache}"
CONTEXT_SIDECAR_ROOT="${CONTEXT_SIDECAR_ROOT:-${ROOT}/external_data/${REGION}_context_sidecar}"
PREV_SIDECAR_ROOT="${PREV_SIDECAR_ROOT:-${ROOT}/external_data/${REGION}_previous_co2_sidecar}"
OUT_DIR="${OUT_DIR:-${ROOT}/generated/${REGION}_average_profile_cache}"

args=(
  scripts/build_average_profile_cache.py
  --cache-root "${SPATIAL_CACHE_ROOT}"
  --context-sidecar-root "${CONTEXT_SIDECAR_ROOT}"
  --prev-sidecar-root "${PREV_SIDECAR_ROOT}"
  --out-dir "${OUT_DIR}"
)
if [[ "${OVERWRITE:-0}" == "1" ]]; then
  args+=(--overwrite)
fi
"${PYTHON_BIN}" "${args[@]}"

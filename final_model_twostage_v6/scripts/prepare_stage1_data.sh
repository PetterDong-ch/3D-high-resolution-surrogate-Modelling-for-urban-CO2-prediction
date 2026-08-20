#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
PYLIBS_ROOT="${PYLIBS_ROOT:-}"
export PYTHONPATH="${ROOT}/src${PYLIBS_ROOT:+:${PYLIBS_ROOT}}"
export CAMDEN_JOBS_ROOT="${CAMDEN_JOBS_ROOT:-${ROOT}/external_data/camden/JOBS}"

"${PYTHON_BIN}" scripts/build_stage1_manifest.py \
  --config configs/stage1_manifest_camden.yaml \
  --out-dir generated/stage1_manifest

cache_args=(
  scripts/build_stage1_cache.py \
  --config configs/stage1_fno_camden.yaml \
  --manifest generated/stage1_manifest/stage1_manifest.json \
  --out-root generated/stage1_cache \
  --splits train val \
  --train-samples 5280 \
  --val-samples 1120 \
  --selection random \
  --seed 42 \
  --patch-h 256 \
  --patch-w 256 \
  --shard-size 8 \
  --dtype float16
)
if [[ "${OVERWRITE:-0}" == "1" ]]; then
  cache_args+=(--overwrite)
fi
"${PYTHON_BIN}" "${cache_args[@]}"

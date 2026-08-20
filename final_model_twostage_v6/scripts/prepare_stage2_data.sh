#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
PYLIBS_ROOT="${PYLIBS_ROOT:-}"
export PYTHONPATH="${ROOT}/src${PYLIBS_ROOT:+:${PYLIBS_ROOT}}"
export CAMDEN_JOBS_ROOT="${CAMDEN_JOBS_ROOT:-${ROOT}/external_data/camden/JOBS}"
METADATA_ROOT="${V40_COORDINATE_SIDECAR_ROOT:-${ROOT}/external_data/v40_coordinate_sidecar}"
STAGE1_CHECKPOINT="${STAGE1_CHECKPOINT:-${ROOT}/checkpoints/stage1_local_fno_best_model.pt}"

"${PYTHON_BIN}" scripts/build_stage2_manifest.py \
  --metadata-root "${METADATA_ROOT}" \
  --jobs-root "${CAMDEN_JOBS_ROOT}" \
  --out-dir generated/stage2_manifest \
  --patch-d 16 --patch-h 256 --patch-w 256

cache_args=(
  scripts/build_stage2_cache_from_stage1.py \
  --stage1-cache-root generated/stage1_cache \
  --stage1-checkpoint "${STAGE1_CHECKPOINT}" \
  --manifest generated/stage2_manifest/stage1_manifest.json \
  --out-root generated/stage2_cache \
  --splits train val \
  --train-samples 16384 \
  --val-samples 2048 \
  --selection random \
  --seed 42 \
  --patch-h 256 --patch-w 256 \
  --shard-size 4 \
  --dtype float16 \
  --target-mode bg_residual \
  --target-source av3d \
  --w-strategy predicted \
  --stats-layer-min 1 --stats-layer-max 10
)
if [[ "${OVERWRITE:-0}" == "1" ]]; then
  cache_args+=(--overwrite)
fi
"${PYTHON_BIN}" "${cache_args[@]}"

prev_args=(
  scripts/build_previous_co2_sidecar.py \
  --cache-root generated/stage2_cache \
  --jobs-root "${CAMDEN_JOBS_ROOT}" \
  --out-root generated/previous_co2_sidecar \
  --splits train,val \
  --shard-size 8 \
  --dx 5 --dy 5 --dz 10
)
if [[ "${OVERWRITE:-0}" == "1" ]]; then
  prev_args+=(--overwrite)
fi
"${PYTHON_BIN}" "${prev_args[@]}"

CACHE_ROOT="${ROOT}/generated/stage2_cache" \
STAGE1_MANIFEST="${ROOT}/generated/stage1_manifest/stage1_manifest.json" \
STAGE1_CHECKPOINT="${STAGE1_CHECKPOINT}" \
CAMDEN_JOBS_ROOT="${CAMDEN_JOBS_ROOT}" \
OUT_ROOT="${ROOT}/generated/full_domain_context_sidecar" \
OVERWRITE="${OVERWRITE:-0}" \
bash scripts/build_full_domain_context_sidecar.sh

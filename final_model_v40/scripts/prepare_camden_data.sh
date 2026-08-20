#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
DATA_ROOT="${V40_DATA_ROOT:-${ROOT}/external_data}"
JOBS_ROOT="${CAMDEN_JOBS_ROOT:-${DATA_ROOT}/raw/JOBS}"
TOPOGRAPHY_PATH="${TOPOGRAPHY_PATH:-${JOBS_ROOT}/cam07_175vm_topo_surf_N02.000.nc}"
RUNTIME="${ROOT}/reproduction"

mkdir -p "${DATA_ROOT}" "${ROOT}/logs"
export PYTHONPATH="${RUNTIME}${PYLIBS_ROOT:+:${PYLIBS_ROOT}}"

"${PYTHON_BIN}" "${RUNTIME}/scripts/prepare_patch_cache_v17.py" \
  --jobs-root "${JOBS_ROOT}" \
  --out-dir "${DATA_ROOT}/patch_cache_v33_camden_z001_010_no1112" \
  --metadata-out-dir "${DATA_ROOT}/coordinate_sidecar_v33_camden_z001_010_no1112" \
  --topography-path "${TOPOGRAPHY_PATH}" \
  --train-patches 16384 --val-patches 2048 --shard-size 8 \
  --patch-d 16 --patch-h 256 --patch-w 256 \
  --z-min-start 0 --z-max-start 3 --xy-stride 32 --time-bin-size 8 \
  --score-candidates-per-stratum 96 --split-modulo 10 --val-mod-value 0 \
  --exclude-months 11,12 --seed 3300 --progress-every 50 --overwrite \
  2>&1 | tee "${ROOT}/logs/prepare_camden_v40_cache.log"

"${PYTHON_BIN}" "${RUNTIME}/scripts/build_v13_sidecar.py" \
  --jobs-root "${JOBS_ROOT}" \
  --cache-root "${DATA_ROOT}/patch_cache_v33_camden_z001_010_no1112" \
  --metadata-root "${DATA_ROOT}/coordinate_sidecar_v33_camden_z001_010_no1112" \
  --out-dir "${DATA_ROOT}/v33_context_sidecar_camden_z001_010_no1112" \
  --topography-path "${TOPOGRAPHY_PATH}" --splits train,val \
  --global-size 80 --shard-size 8 --height-gate-decay-levels 30 \
  --stats-stride 16 --progress-every 100 \
  2>&1 | tee "${ROOT}/logs/build_camden_v40_context.log"

"${PYTHON_BIN}" "${RUNTIME}/scripts/build_v22_previous_co2_sidecar.py" \
  --jobs-root "${JOBS_ROOT}" \
  --cache-root "${DATA_ROOT}/patch_cache_v33_camden_z001_010_no1112" \
  --metadata-root "${DATA_ROOT}/coordinate_sidecar_v33_camden_z001_010_no1112" \
  --out-dir "${DATA_ROOT}/v33_previous_co2_sidecar_camden_z001_010_no1112" \
  --splits train,val --global-size 80 --shard-size 8 \
  --stats-stride 16 --progress-every 100 \
  2>&1 | tee "${ROOT}/logs/build_camden_v40_previous_co2.log"

"${PYTHON_BIN}" "${RUNTIME}/scripts/build_v30_texture_sidecar.py" \
  --cache-root "${DATA_ROOT}/patch_cache_v33_camden_z001_010_no1112" \
  --prev-sidecar-root "${DATA_ROOT}/v33_previous_co2_sidecar_camden_z001_010_no1112" \
  --metadata-root "${DATA_ROOT}/coordinate_sidecar_v33_camden_z001_010_no1112" \
  --out-dir "${DATA_ROOT}/v33_texture_sidecar_camden_z001_010_no1112" \
  --splits train,val --layer-min 1 --layer-max 10 --min-layer-overlap 8 \
  --exclude-months 11,12 --high-delta-threshold 3.0 \
  --z-gradient-weight 0.25 --norm-percentile 90.0 --score-clip 3.0 \
  --abs-weight 0.50 --std-weight 0.35 --gradient-weight 1.00 \
  --high-delta-fraction-weight 0.50 --sampler-alpha 1.50 \
  --min-weight 0.25 --max-weight 8.0 --log-every 500 \
  2>&1 | tee "${ROOT}/logs/build_camden_v40_texture.log"

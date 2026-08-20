#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
DATA_ROOT="${V40_DATA_ROOT:-${ROOT}/external_data}"
OUT_DIR="${OUT_DIR:-${ROOT}/reproduced/v40_training}"
RUNTIME="${ROOT}/reproduction"

KEEP_CHANNELS="emission_values,ls_forcing_right_CO2,u,v,p,theta,w,month_sin,month_cos,tod_sin,tod_cos,x_norm,y_norm,z_norm,prev_kc_CO2,prev_dCdx,prev_dCdy,prev_dCdz"

mkdir -p "${OUT_DIR}" "${ROOT}/logs"
export PYTHONPATH="${RUNTIME}${PYLIBS_ROOT:+:${PYLIBS_ROOT}}"

"${PYTHON_BIN}" "${RUNTIME}/scripts/train_3d_unet.py" \
  --cache-root "${DATA_ROOT}/patch_cache_v33_camden_z001_010_no1112" \
  --v13-sidecar-root "${DATA_ROOT}/v33_context_sidecar_camden_z001_010_no1112" \
  --v22-prev-sidecar-root "${DATA_ROOT}/v33_previous_co2_sidecar_camden_z001_010_no1112" \
  --v14-normalization-root "${DATA_ROOT}/v33_context_sidecar_camden_z001_010_no1112" \
  --metadata-root "${DATA_ROOT}/coordinate_sidecar_v33_camden_z001_010_no1112" \
  --texture-sidecar-root "${DATA_ROOT}/v33_texture_sidecar_camden_z001_010_no1112" \
  --v14-layer-min 1 --v14-layer-max 10 --v14-min-layer-overlap 8 \
  --exclude-months 11,12 --v13-global-sample-size 64 \
  --model-variant v38_event_texture_context_v7 \
  --height-gate-surface --height-gate-decay-levels 30 \
  --surface-channel-indices 7,8,9,10,11 \
  --v28-advection-features --v28-advection-dx 5.0 --v28-advection-dy 5.0 \
  --v28-advection-dz 10.0 --v28-advection-dt 120.0 \
  --v28-advection-delta-scale 5.0 --v28-advection-gradient-scale 0.2 \
  --v28-advection-clip 20.0 --v28-advection-input-clip 8.0 \
  --keep-input-channel-names "${KEEP_CHANNELS}" \
  --texture-aware-sampler --texture-weight-power 1.05 \
  --texture-min-weight 0.45 --texture-max-weight 7.0 \
  --init-weights "${ROOT}/checkpoints/v38_initialization.pt" --init-allow-partial \
  --out-dir "${OUT_DIR}" --epochs 70 --batch-size 1 --num-workers 4 \
  --base-channels 32 --global-feature-channels 8 \
  --high-residual-scale 1.0 --min-high-gate 0.20 \
  --lr 3e-5 --weight-decay 1e-4 --lr-scheduler plateau \
  --lr-factor 0.5 --lr-patience 7 --min-lr 1e-6 \
  --early-stopping-patience 18 --early-stopping-min-delta 0.0 \
  --train-samples-per-epoch 512 --val-samples 128 --progress none \
  --delta-monitor-checkpoints --delta-monitor-active-threshold 0.75 \
  --delta-monitor-min-valid 256 --delta-monitor-hard-min-std 1.0 \
  --delta-monitor-hard-min-active-fraction 0.25 \
  --base-loss huber --huber-delta 2.0 --gradient-loss-weight 0.03 \
  --multiscale-loss-weight 0.0 --smoothness-loss-weight 0.0 \
  --variance-loss-weight 0.0 --residual-weight-alpha 0.5 \
  --residual-weight-scale 4.0 --residual-weight-max 4.0 \
  --target-gradient-weight-alpha 0.35 --target-gradient-weight-scale 1.0 \
  --target-gradient-weight-max 2.5 --low-layer-weight-alpha 0.0 \
  --normalized-loss-weight 0.0 --correlation-loss-weight 0.04 \
  --correlation-min-target-std 0.25 --correlation-min-valid-fraction 0.5 \
  --local-correlation-loss-weight 0.05 --local-correlation-pool 32 \
  --local-correlation-min-target-std 0.45 \
  --local-correlation-min-valid-fraction 0.40 --amplitude-loss-weight 0.05 \
  --amplitude-min-target-std 0.45 --active-delta-loss-weight 0.35 \
  --active-delta-threshold 0.75 --low-frequency-loss-weight 0.04 \
  --low-frequency-correlation-weight 0.02 --low-frequency-pool 16 \
  --low-frequency-min-valid-fraction 0.40 --high-frequency-loss-weight 0.04 \
  --high-frequency-huber-delta 0.85 --sign-loss-weight 0.015 \
  --sign-loss-min-abs 0.75 --sign-loss-scale 2.0 \
  --active-loss-weight 0.02 --active-loss-threshold 0.75 \
  --active-loss-pos-weight 2.0 --sign-class-loss-weight 0.02 \
  --sign-class-loss-min-abs 0.75 --sign-class-loss-pos-weight 1.0 \
  2>&1 | tee "${ROOT}/logs/reproduce_camden_v40_training.log"

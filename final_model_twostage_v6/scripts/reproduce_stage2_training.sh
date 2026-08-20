#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"
CACHE_ROOT="${ROOT}/generated/stage2_cache" \
PREV_SIDECAR_ROOT="${ROOT}/generated/previous_co2_sidecar" \
GLOBAL_SIDECAR_ROOT="${ROOT}/generated/full_domain_context_sidecar" \
INIT_CHECKPOINT="${ROOT}/checkpoints/v40_initialization.pt" \
OUT_DIR="${ROOT}/generated/stage2_run" \
LOG_PATH="${ROOT}/logs/reproduce_stage2_training.log" \
EPOCHS=110 \
bash scripts/train_twostage_v6_stage2.sh

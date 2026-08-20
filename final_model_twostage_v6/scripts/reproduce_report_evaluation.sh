#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"
CACHE_ROOT="${ROOT}/generated/stage2_cache" \
PREV_SIDECAR_ROOT="${ROOT}/generated/previous_co2_sidecar" \
GLOBAL_SIDECAR_ROOT="${ROOT}/generated/full_domain_context_sidecar" \
CHECKPOINT="${ROOT}/checkpoints/stage2_best_model.pt" \
OUT_DIR="${ROOT}/generated/report_evaluation" \
LOG_PATH="${ROOT}/logs/reproduce_report_evaluation.log" \
VISUAL_SAMPLES="${VISUAL_SAMPLES:-12}" \
bash scripts/evaluate_twostage_v6_stage2.sh

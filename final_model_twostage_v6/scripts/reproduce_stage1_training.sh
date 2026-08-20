#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
PYLIBS_ROOT="${PYLIBS_ROOT:-}"
export PYTHONPATH="${ROOT}/src${PYLIBS_ROOT:+:${PYLIBS_ROOT}}"

mkdir -p generated/stage1_run logs
"${PYTHON_BIN}" scripts/train_stage1_fno.py \
  --config configs/stage1_fno_camden.yaml \
  --cache-root generated/stage1_cache \
  --out-dir generated/stage1_run \
  --epochs 120 \
  --batch-size 1 \
  --lr 3e-4 \
  --weight-decay 1e-4 \
  --num-workers "${NUM_WORKERS:-2}" \
  --seed 42 \
  2>&1 | tee logs/reproduce_stage1_training.log

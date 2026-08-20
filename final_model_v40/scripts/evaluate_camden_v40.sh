#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
PYLIBS_ROOT="${PYLIBS_ROOT:-}"
cd "${ROOT}"

PYTHONPATH="${ROOT}/src${PYLIBS_ROOT:+:${PYLIBS_ROOT}}" \
  "${PYTHON_BIN}" -m final_model.evaluate "$@"

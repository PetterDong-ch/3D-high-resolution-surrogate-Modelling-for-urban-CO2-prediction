#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

PYTHONPATH="${ROOT}/src${PYLIBS_ROOT:+:${PYLIBS_ROOT}}" \
  "${PYTHON_BIN:-python3}" - <<'PY'
from pathlib import Path
import torch

from twostage_v6.model import build_final_model, count_trainable_parameters
from twostage_v6.stage1_fno import LocalFNOStage1, Stage1ModelConfig

root = Path.cwd()

stage2_ckpt = torch.load(root / "checkpoints/stage2_best_model.pt", map_location="cpu")
stage2_state = stage2_ckpt.get("model_state_dict", stage2_ckpt)
stage2 = build_final_model()
stage2.load_state_dict(stage2_state, strict=True)
print(f"Stage2 loaded: {count_trainable_parameters(stage2):,} trainable parameters")

stage1_ckpt = torch.load(root / "checkpoints/stage1_local_fno_best_model.pt", map_location="cpu")
stage1 = LocalFNOStage1(Stage1ModelConfig(**dict(stage1_ckpt["model_config"])))
stage1.load_state_dict(stage1_ckpt["model_state_dict"], strict=True)
params = sum(p.numel() for p in stage1.parameters() if p.requires_grad)
print(f"Stage1 loaded: {params:,} trainable parameters")
PY

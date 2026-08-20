#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

PYTHONPATH="${ROOT}/src${PYLIBS_ROOT:+:${PYLIBS_ROOT}}" \
  "${PYTHON_BIN:-python3}" - <<'PY'
from pathlib import Path
import torch

from task2_v3.model import CompactProfileNet

root = Path.cwd()
for name, path in {
    "Camden": root / "checkpoints/camden_background_enhancement_best_model.pt",
    "Richmond fine-tuned": root / "checkpoints/richmond_finetune_background_enhancement_best_model.pt",
}.items():
    ckpt = torch.load(path, map_location="cpu")
    cfg = ckpt.get("config", {})
    model = CompactProfileNet(
        int(cfg.get("local_feature_count", 30)),
        int(cfg.get("global_feature_count", 16)),
        int(cfg.get("base_channels", 64)),
    )
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"{name} checkpoint loaded: {params:,} trainable parameters")
PY

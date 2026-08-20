from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from .model import build_final_model, count_trainable_parameters


# Load final checkpoint from disk or cache.
def load_final_checkpoint(
    checkpoint_path: str | Path,
    map_location: str | torch.device = "cpu",
) -> tuple[torch.nn.Module, dict[str, Any]]:
    """Load a two-stage V6 Stage2 checkpoint into the clean model definition."""

    checkpoint_path = Path(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location=map_location)
    model_config = checkpoint.get("model_config", {})

    model = build_final_model(
        in_channels=int(model_config.get("in_channels", checkpoint.get("in_channels", 18))),
        base_channels=int(model_config.get("base_channels", checkpoint.get("base_channels", 32))),
        global_channels=int(model_config.get("global_channels", checkpoint.get("global_channels", 9))),
        global_feature_channels=int(
            model_config.get("global_feature_channels", checkpoint.get("global_feature_channels", 8))
        ),
        high_delta_scale=float(model_config.get("high_delta_scale", checkpoint.get("high_residual_scale", 1.0))),
        min_high_gate=float(model_config.get("min_high_gate", checkpoint.get("min_high_gate", 0.20))),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    metadata = {
        "checkpoint": str(checkpoint_path),
        "epoch": checkpoint.get("epoch"),
        "best_val_loss": checkpoint.get("best_val_loss", checkpoint.get("best_val")),
        "model_variant": model_config.get("variant", checkpoint.get("model_variant")),
        "target_mode": "autoregressive_delta_with_stage1_met",
        "in_channels": model_config.get("in_channels", checkpoint.get("in_channels", 18)),
        "global_channels": model_config.get("global_channels", checkpoint.get("global_channels", 9)),
        "parameters": count_trainable_parameters(model),
        "input_channels": model_config.get("input_channels"),
        "global_input_channels": model_config.get("global_input_channels"),
    }
    return model, metadata


# Print checkpoint metadata and parameter counts.
def print_checkpoint_summary(checkpoint_path: str | Path) -> None:
    _, metadata = load_final_checkpoint(checkpoint_path)
    for key, value in metadata.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Load and summarize the two-stage V6 Stage2 checkpoint.")
    parser.add_argument(
        "checkpoint",
        nargs="?",
        default=str(Path(__file__).resolve().parents[2] / "checkpoints" / "stage2_best_model.pt"),
    )
    args = parser.parse_args()
    print_checkpoint_summary(args.checkpoint)

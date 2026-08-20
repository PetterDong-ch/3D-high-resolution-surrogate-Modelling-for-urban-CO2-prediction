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
    """Load a V40 final checkpoint into the clean model definition."""

    checkpoint_path = Path(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location=map_location)

    model = build_final_model(
        in_channels=int(checkpoint.get("in_channels", 18)),
        base_channels=int(checkpoint.get("base_channels", 32)),
        global_channels=int(checkpoint.get("global_channels", 9)),
        global_feature_channels=int(checkpoint.get("global_feature_channels", 8)),
        high_delta_scale=float(checkpoint.get("high_residual_scale", 1.0)),
        min_high_gate=float(checkpoint.get("min_high_gate", 0.20)),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    metadata = {
        "checkpoint": str(checkpoint_path),
        "epoch": checkpoint.get("epoch"),
        "best_val": checkpoint.get("best_val"),
        # Older weight files used an experiment-era class label. The clean
        # package exposes the architecture only under its final V40 name.
        "model_variant": "v40_event_texture_context",
        "target_mode": checkpoint.get("target_mode"),
        "in_channels": checkpoint.get("in_channels", 18),
        "global_channels": checkpoint.get("global_channels", 9),
        "parameters": count_trainable_parameters(model),
        "kept_input_channels": checkpoint.get("kept_input_channels"),
    }
    return model, metadata


# Print checkpoint metadata and parameter counts.
def print_checkpoint_summary(checkpoint_path: str | Path) -> None:
    """Print human-readable checkpoint metadata for quick inspection."""

    _, metadata = load_final_checkpoint(checkpoint_path)
    for key, value in metadata.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Load and summarize the V40 final checkpoint.")
    parser.add_argument("checkpoint", nargs="?", default=str(Path(__file__).resolve().parents[2] / "checkpoints" / "best_model.pt"))
    args = parser.parse_args()
    print_checkpoint_summary(args.checkpoint)

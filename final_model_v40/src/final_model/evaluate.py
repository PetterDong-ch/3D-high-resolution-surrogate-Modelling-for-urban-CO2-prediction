from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .checkpoint import load_final_checkpoint
from .config import FinalV40Paths
from .train import fixed_subset, make_dataset
from torch.utils.data import DataLoader


# Accumulates streaming regression metrics without storing every prediction.
class StreamingMetrics:
    """Accumulate MAE/RMSE/R over valid cells across many 3-D batches."""

    # Store constructor arguments and initialize object state.
    def __init__(self) -> None:
        self.n = 0
        self.sum_abs = self.sum_sq = self.sum_p = self.sum_t = self.sum_p2 = self.sum_t2 = self.sum_pt = 0.0

    # Update running metric or statistic accumulators.
    def update(self, pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> None:
        valid = mask > 0
        p, t = pred[valid].double(), target[valid].double()
        if p.numel() == 0:
            return
        error = p - t
        self.n += p.numel()
        self.sum_abs += error.abs().sum().item()
        self.sum_sq += error.square().sum().item()
        self.sum_p += p.sum().item(); self.sum_t += t.sum().item()
        self.sum_p2 += p.square().sum().item(); self.sum_t2 += t.square().sum().item(); self.sum_pt += (p * t).sum().item()

    # Return the final accumulated metrics.
    def finish(self) -> dict[str, float]:
        n = max(self.n, 1)
        cov = self.sum_pt - self.sum_p * self.sum_t / n
        vp = self.sum_p2 - self.sum_p * self.sum_p / n
        vt = self.sum_t2 - self.sum_t * self.sum_t / n
        r = cov / np.sqrt(max(vp * vt, 1e-12))
        return {"valid_count": self.n, "MAE": self.sum_abs / n, "RMSE": np.sqrt(self.sum_sq / n), "R": float(r)}


# Build the command-line argument parser.
def parser() -> argparse.ArgumentParser:
    """Define evaluation arguments and repository-relative defaults."""

    paths = FinalV40Paths()
    p = argparse.ArgumentParser(description="Evaluate the standalone final V40 model on a prepared cache split.")
    p.add_argument("--cache-root", type=Path, default=paths.cache_root)
    p.add_argument("--context-sidecar-root", type=Path, default=paths.context_sidecar_root)
    p.add_argument("--previous-co2-sidecar-root", type=Path, default=paths.previous_co2_sidecar_root)
    p.add_argument("--normalization-root", type=Path, default=paths.normalization_root)
    p.add_argument("--texture-sidecar-root", type=Path, default=paths.texture_sidecar_root)
    p.add_argument("--checkpoint", type=Path, default=paths.checkpoint)
    p.add_argument("--out-dir", type=Path, default=paths.eval_dir)
    p.add_argument("--split", choices=("train", "val"), default="val")
    p.add_argument("--max-samples", type=int, default=128)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--layer-min", type=int, default=1)
    p.add_argument("--layer-max", type=int, default=10)
    p.add_argument("--min-layer-overlap", type=int, default=8)
    p.add_argument("--global-sample-size", type=int, default=64)
    p.add_argument("--device", default="auto")
    p.add_argument("--dry-run", action="store_true")
    return p


# Entry point for the command-line workflow.
def main() -> None:
    """Command-line entry point for V40 checkpoint evaluation."""

    args = parser().parse_args()
    if args.dry_run:
        print(json.dumps({k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}, indent=2))
        return
    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device))

    dataset = make_dataset(args, args.split)
    subset = fixed_subset(dataset, args.max_samples, 42)
    batches = DataLoader(subset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    model, metadata = load_final_checkpoint(args.checkpoint, map_location=device)
    model.to(device)
    delta = StreamingMetrics()
    with torch.no_grad():
        for local, context, grid, target, mask in batches:
            pred = model(local.to(device), context.to(device), grid.to(device))
            delta.update(pred.cpu(), target, mask)

    result = {"checkpoint": metadata, "split": args.split, "samples": len(subset), "delta": delta.finish()}
    (args.out_dir / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

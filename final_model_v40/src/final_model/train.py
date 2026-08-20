from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler, Subset, WeightedRandomSampler

from .config import LOCAL_CHANNELS, PROJECT_ROOT, FinalV40Paths
from .data import FinalV40Dataset, PatchCache
from .losses import V40LossWeights, v40_composite_loss
from .model import build_final_model


# Seed Python, NumPy, and Torch random number generators.
def seed_all(seed: int) -> None:
    """Seed Python, NumPy and PyTorch for best-effort reproducibility."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# Load checkpoint weights while allowing compatible key differences.
def flexible_load(model: torch.nn.Module, path: Path, device: torch.device) -> None:
    """Load matching tensors from an initialization checkpoint."""

    checkpoint = torch.load(path, map_location=device)
    source = checkpoint.get("model_state_dict", checkpoint)
    target = model.state_dict()
    compatible: dict[str, torch.Tensor] = {}
    for name, value in source.items():
        if name not in target:
            continue
        if value.shape == target[name].shape:
            compatible[name] = value
        elif value.ndim == target[name].ndim:
            adapted = target[name].clone()
            slices = tuple(slice(0, min(a, b)) for a, b in zip(value.shape, adapted.shape))
            adapted[slices] = value[slices].to(adapted.dtype)
            compatible[name] = adapted
    result = model.load_state_dict(compatible, strict=False)
    print(f"Initialized {len(compatible)} tensors from {path}; missing={len(result.missing_keys)}", flush=True)


# Create dataset for the workflow.
def make_dataset(args: argparse.Namespace, split: str) -> FinalV40Dataset:
    """Create the final V40 dataset for a train or validation split."""

    base = PatchCache(str(Path(args.cache_root) / split))
    return FinalV40Dataset(
        base,
        sidecar_root=str(args.context_sidecar_root),
        normalization_root=str(args.normalization_root),
        prev_sidecar_root=str(args.previous_co2_sidecar_root),
        split=split,
        global_sample_size=args.global_sample_size,
        layer_min=args.layer_min,
        layer_max=args.layer_max,
        min_layer_overlap=args.min_layer_overlap,
        use_global_context=True,
        surface_channel_indices=(7, 8, 9, 10, 11),
        height_gate_decay_levels=30.0,
        append_height_gate=False,
        exclude_months=(11, 12),
        advection_dx=5.0,
        advection_dy=5.0,
        advection_dz=10.0,
        advection_dt=120.0,
        advection_delta_scale=5.0,
        advection_gradient_scale=0.2,
        advection_clip=20.0,
        advection_input_clip=8.0,
        correction_target=False,
        keep_channels=tuple(LOCAL_CHANNELS),
    )


# Build a deterministic subset of dataset indices.
def fixed_subset(dataset: FinalV40Dataset, count: int, seed: int) -> Subset:
    """Pick a deterministic subset for validation/evaluation speed."""

    if count <= 0 or count >= len(dataset):
        return Subset(dataset, list(range(len(dataset))))
    rng = np.random.default_rng(seed)
    indices = np.sort(rng.choice(len(dataset), size=count, replace=False)).tolist()
    return Subset(dataset, indices)


# Compute texture-based loss weights.
def texture_weights(root: Path, dataset: FinalV40Dataset, power: float = 1.05) -> torch.Tensor:
    """Load deterministic sample weights from the texture sidecar."""

    data = np.load(root / "train_texture_stats.npz", allow_pickle=False)
    raw = data["sample_weight"] if "sample_weight" in data.files else data["texture_score"]
    raw = np.asarray(raw, dtype=np.float64)
    # Dataset indices refer back to source-cache rows after layer/month filtering.
    raw = raw[np.asarray(dataset.indices, dtype=np.int64)]
    raw = np.nan_to_num(raw, nan=1.0, posinf=7.0, neginf=0.45)
    return torch.as_tensor(np.clip(np.power(np.clip(raw, 0.45, 7.0), power), 0.45, 7.0), dtype=torch.double)


# Create a data loader for one split.
def loader(dataset: FinalV40Dataset, *, train: bool, samples: int, batch_size: int, workers: int, seed: int, texture_root: Path | None = None) -> DataLoader:
    """Build a DataLoader with train-time sampling or fixed validation subset."""

    if train:
        if texture_root is not None:
            weights = texture_weights(texture_root, dataset)
            count = samples if samples > 0 else len(dataset)
            sampler = WeightedRandomSampler(weights, count, replacement=True)
        else:
            sampler = RandomSampler(dataset, replacement=samples > 0, num_samples=samples if samples > 0 else None)
        source = dataset
    else:
        source = fixed_subset(dataset, samples, seed)
        sampler = None
    return DataLoader(
        source,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=False,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=workers > 0,
    )


# Run one training or validation epoch.
def run_epoch(model: torch.nn.Module, batches: DataLoader, device: torch.device, optimizer: torch.optim.Optimizer | None, weights: V40LossWeights) -> tuple[float, dict[str, float]]:
    """Run one train or validation epoch and average loss components."""

    training = optimizer is not None
    model.train(training)
    totals: dict[str, float] = {}
    count = 0
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for local, global_context, global_grid, target, mask in batches:
            local = local.to(device, non_blocking=True)
            global_context = global_context.to(device, non_blocking=True)
            global_grid = global_grid.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)
            if training:
                optimizer.zero_grad(set_to_none=True)
            output = model(local, global_context, global_grid, return_components=True)
            loss, parts = v40_composite_loss(output, target, mask, weights)
            if training:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
                optimizer.step()
            count += 1
            for name, value in parts.items():
                totals[name] = totals.get(name, 0.0) + float(value.detach().cpu())
    return totals.get("total", 0.0) / max(count, 1), {name: value / max(count, 1) for name, value in totals.items()}


# Write history to disk.
def write_history(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write training history rows as a CSV table."""

    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


# Build the command-line argument parser.
def parser() -> argparse.ArgumentParser:
    """Define command-line options for reproducing V40 training."""

    paths = FinalV40Paths()
    p = argparse.ArgumentParser(description="Train the standalone final V40 CO2 model.")
    p.add_argument("--cache-root", type=Path, default=paths.cache_root)
    p.add_argument("--context-sidecar-root", type=Path, default=paths.context_sidecar_root)
    p.add_argument("--previous-co2-sidecar-root", type=Path, default=paths.previous_co2_sidecar_root)
    p.add_argument("--normalization-root", type=Path, default=paths.normalization_root)
    p.add_argument("--texture-sidecar-root", type=Path, default=paths.texture_sidecar_root)
    p.add_argument("--out-dir", type=Path, default=paths.run_dir)
    p.add_argument("--init-weights", type=Path, default=PROJECT_ROOT / "checkpoints/best_model.pt")
    p.add_argument("--resume", type=Path)
    p.add_argument("--train-from-scratch", action="store_true")
    p.add_argument("--epochs", type=int, default=70)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--train-samples-per-epoch", type=int, default=512)
    p.add_argument("--val-samples", type=int, default=128)
    p.add_argument("--lr", type=float, default=3e-5)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--lr-factor", type=float, default=0.5)
    p.add_argument("--lr-patience", type=int, default=7)
    p.add_argument("--min-lr", type=float, default=1e-6)
    p.add_argument("--early-stopping-patience", type=int, default=18)
    p.add_argument("--layer-min", type=int, default=1)
    p.add_argument("--layer-max", type=int, default=10)
    p.add_argument("--min-layer-overlap", type=int, default=8)
    p.add_argument("--global-sample-size", type=int, default=64)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="auto")
    p.add_argument("--no-texture-sampler", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    return p


# Entry point for the command-line workflow.
def main() -> None:
    """Command-line entry point for standalone V40 training."""

    args = parser().parse_args()
    if args.dry_run:
        print(json.dumps({key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}, indent=2))
        return

    seed_all(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device))

    train_ds = make_dataset(args, "train")
    val_ds = make_dataset(args, "val")

    model = build_final_model().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=args.lr_factor, patience=args.lr_patience, min_lr=args.min_lr)
    start_epoch, best, stale, history = 1, float("inf"), 0, []
    if args.resume:
        state = torch.load(args.resume, map_location=device)
        model.load_state_dict(state["model_state_dict"])
        optimizer.load_state_dict(state["optimizer_state_dict"])
        start_epoch = int(state["epoch"]) + 1
        best = float(state.get("best_val", best))
        history = list(state.get("history", []))
    elif not args.train_from_scratch and args.init_weights:
        flexible_load(model, args.init_weights, device)

    train_batches = loader(train_ds, train=True, samples=args.train_samples_per_epoch, batch_size=args.batch_size, workers=args.num_workers, seed=args.seed, texture_root=None if args.no_texture_sampler else args.texture_sidecar_root)
    val_batches = loader(val_ds, train=False, samples=args.val_samples, batch_size=args.batch_size, workers=args.num_workers, seed=args.seed + 1)
    weights = V40LossWeights()
    print(f"V40 datasets: train={len(train_ds)} val={len(val_ds)} device={device}", flush=True)
    for epoch in range(start_epoch, args.epochs + 1):
        train_loss, _ = run_epoch(model, train_batches, device, optimizer, weights)
        val_loss, parts = run_epoch(model, val_batches, device, None, weights)
        scheduler.step(val_loss)

        improved = val_loss < best
        stale = 0 if improved else stale + 1
        best = min(best, val_loss)
        row = {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "best_val": best, "lr": optimizer.param_groups[0]["lr"], **{f"val_{k}": v for k, v in parts.items() if k != "total"}}
        history.append(row)
        write_history(args.out_dir / "history.csv", history)

        checkpoint = {"epoch": epoch, "model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(), "best_val": best, "history": history, "model_variant": "v40_event_texture_context", "target_mode": "autoregressive_delta", "in_channels": 18, "global_channels": 9}
        torch.save(checkpoint, args.out_dir / "last_model.pt")
        if improved:
            torch.save(checkpoint, args.out_dir / "best_model.pt")
        print(f"epoch={epoch:03d} train={train_loss:.6f} val={val_loss:.6f} best={best:.6f} stale={stale}", flush=True)
        if stale >= args.early_stopping_patience:
            break


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler, Subset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
V40_ROOT = PROJECT_ROOT / "runtime"
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(V40_ROOT))

from models.unet3d import V38EventTextureContextV7UNet3D  # noqa: E402
from scripts.train_3d_unet import (  # noqa: E402
    combined_masked_loss,
    current_lr,
    load_flexible_partial_state_dict,
    run_epoch,
)
from twostage_v6.stage2_datasets import (  # noqa: E402
    Stage2V40GlobalContextDataset,
    Stage2V40LocalDataset,
)
from twostage_v6.stage2_constants import (  # noqa: E402
    V40_STAGE1_GLOBAL_CONTEXT_CHANNELS,
    V40_STAGE1_MET_CHANNELS,
)


# Seed Python, NumPy, and Torch random number generators.
def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# Write csv to disk.
def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


# Build a deterministic subset of dataset indices.
def fixed_subset(dataset: Stage2V40LocalDataset, count: int, seed: int) -> Subset:
    if count <= 0 or count >= len(dataset):
        return Subset(dataset, list(range(len(dataset))))
    rng = np.random.default_rng(seed)
    indices = np.sort(rng.choice(len(dataset), size=count, replace=False)).astype(int).tolist()
    return Subset(dataset, indices)


# Train loader and update checkpoints.
def train_loader(dataset: Stage2V40LocalDataset, samples: int, batch_size: int, workers: int) -> DataLoader:
    sampler = RandomSampler(dataset, replacement=True, num_samples=samples) if samples > 0 else RandomSampler(dataset)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=workers > 0,
    )


# Evaluate eval loader and collect metrics.
def eval_loader(dataset: Stage2V40LocalDataset, samples: int, batch_size: int, workers: int, seed: int) -> DataLoader:
    subset = fixed_subset(dataset, samples, seed)
    return DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=workers > 0,
    )


# Compute the run loss smoke.
def run_loss_smoke(model: torch.nn.Module, dataset: Stage2V40LocalDataset, device: torch.device) -> None:
    batch = dataset[0]
    if len(batch) == 5:
        x, global_context, global_grid, y, m = batch
        global_context = global_context.unsqueeze(0).to(device)
        global_grid = global_grid.unsqueeze(0).to(device)
    else:
        x, y, m = batch
        global_context = None
        global_grid = None
    x = x.unsqueeze(0).to(device)
    y = y.unsqueeze(0).to(device)
    m = m.unsqueeze(0).to(device)
    with torch.no_grad():
        out = model(x, global_context=global_context, global_grid=global_grid, return_components=True)
        pred = out["final"] if isinstance(out, dict) else out
        loss, parts = combined_masked_loss(
            pred,
            y,
            m,
            input_x=x,
            components=out if isinstance(out, dict) else None,
            base_loss="huber",
            huber_delta=2.0,
            gradient_loss_weight=0.03,
            multiscale_loss_weight=0.0,
            multiscale_scales=(8, 16, 32, 64),
            multiscale_min_valid_fraction=0.5,
            smoothness_loss_weight=0.0,
            smoothness_kernel_size=9,
            height_gate_channel_idx=None,
            variance_loss_weight=0.0,
            variance_min_std=2.0,
            variance_eps=1.0e-6,
            residual_weight_alpha=0.5,
            residual_weight_scale=4.0,
            residual_weight_max=4.0,
            target_gradient_weight_alpha=0.35,
            target_gradient_weight_scale=1.0,
            target_gradient_weight_max=2.5,
            low_layer_weight_alpha=0.0,
            normalized_loss_weight=0.0,
            normalized_min_std=0.75,
            normalized_huber_delta=1.0,
            normalized_eps=1.0e-6,
            correlation_loss_weight=0.04,
            correlation_eps=1.0e-6,
            correlation_min_target_std=0.25,
            correlation_min_valid_fraction=0.5,
            low_frequency_loss_weight=0.04,
            low_frequency_pool=16,
            low_frequency_min_valid_fraction=0.40,
            low_frequency_correlation_weight=0.02,
            high_frequency_loss_weight=0.04,
            high_frequency_huber_delta=0.85,
            local_correlation_loss_weight=0.05,
            local_correlation_pool=32,
            local_correlation_min_target_std=0.45,
            local_correlation_min_valid_fraction=0.40,
            amplitude_loss_weight=0.05,
            amplitude_min_target_std=0.45,
            active_delta_loss_weight=0.35,
            active_delta_threshold=0.75,
            sign_loss_weight=0.015,
            sign_loss_min_abs=0.75,
            sign_loss_scale=2.0,
            active_loss_weight=0.02,
            active_loss_threshold=0.75,
            active_loss_pos_weight=2.0,
            sign_class_loss_weight=0.02,
            sign_class_loss_min_abs=0.75,
            sign_class_loss_pos_weight=1.0,
            pattern_height_decay=False,
        )
    print(
        f"Smoke sample: x={tuple(x.shape)} y={tuple(y.shape)} mask_valid={float(m.sum().item()):.0f} "
        f"loss={float(loss.item()):.6f} base={parts['base']:.6f}",
        flush=True,
    )


# Entry point for the command-line workflow.
def main() -> None:
    parser = argparse.ArgumentParser(description="Train V40-style local event-texture CO2 delta model using Stage1-predicted met fields.")
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--prev-sidecar-root", type=Path, required=True)
    parser.add_argument("--global-sidecar-root", type=Path, help="Optional full-domain Stage1-met context sidecar.")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--init-checkpoint", type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--epochs", type=int, default=70)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--train-samples-per-epoch", type=int, default=512)
    parser.add_argument("--val-samples", type=int, default=128)
    parser.add_argument("--base-channels", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3.0e-5)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--lr-factor", type=float, default=0.5)
    parser.add_argument("--lr-patience", type=int, default=7)
    parser.add_argument("--min-lr", type=float, default=1.0e-6)
    parser.add_argument("--early-stopping-patience", type=int, default=18)
    parser.add_argument("--early-stopping-min-delta", type=float, default=0.0)
    parser.add_argument("--layer-min", type=int, default=1)
    parser.add_argument("--layer-max", type=int, default=10)
    parser.add_argument("--min-layer-overlap", type=int, default=8)
    parser.add_argument("--dx", type=float, default=5.0)
    parser.add_argument("--dy", type=float, default=5.0)
    parser.add_argument("--dz", type=float, default=10.0)
    parser.add_argument("--high-residual-scale", type=float, default=1.0)
    parser.add_argument("--min-high-gate", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--progress", choices=["none", "summary", "tqdm"], default="none")
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--smoke-only", action="store_true")
    args = parser.parse_args()

    seed_all(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else args.device)
    if args.device == "auto" and not torch.cuda.is_available():
        device = torch.device("cpu")

    train_ds = Stage2V40GlobalContextDataset(
        args.cache_root,
        args.prev_sidecar_root,
        "train",
        layer_min=args.layer_min,
        layer_max=args.layer_max,
        min_layer_overlap=args.min_layer_overlap,
        dx=args.dx,
        dy=args.dy,
        dz=args.dz,
        global_sidecar_root=args.global_sidecar_root,
    )
    val_ds = Stage2V40GlobalContextDataset(
        args.cache_root,
        args.prev_sidecar_root,
        "val",
        layer_min=args.layer_min,
        layer_max=args.layer_max,
        min_layer_overlap=args.min_layer_overlap,
        dx=args.dx,
        dy=args.dy,
        dz=args.dz,
        global_sidecar_root=args.global_sidecar_root,
    )

    model = V38EventTextureContextV7UNet3D(
        in_channels=len(V40_STAGE1_MET_CHANNELS),
        out_channels=1,
        base_channels=args.base_channels,
        global_channels=len(V40_STAGE1_GLOBAL_CONTEXT_CHANNELS),
        global_feature_channels=8,
        context_correction_scale=args.high_residual_scale,
        high_delta_scale=args.high_residual_scale,
        min_high_gate=args.min_high_gate,
    ).to(device)

    if args.init_checkpoint is not None and args.resume is None:
        checkpoint = torch.load(args.init_checkpoint, map_location=device)
        source_state = checkpoint.get("model_state_dict", checkpoint)
        missing, unexpected, adapted = load_flexible_partial_state_dict(model, source_state)
        print(
            f"Initialized from {args.init_checkpoint}: missing={len(missing)} "
            f"unexpected_or_skipped={len(unexpected)} adapted={len(adapted)}",
            flush=True,
        )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=args.lr_factor,
        patience=args.lr_patience,
        min_lr=args.min_lr,
    )

    start_epoch = 1
    best_val = float("inf")
    best_delta_r = -float("inf")
    best_hard_delta_r = -float("inf")
    best_sign = -float("inf")
    best_pattern = -float("inf")
    no_improve = 0
    history: list[dict[str, Any]] = []
    if args.resume is not None:
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        if "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = int(checkpoint.get("epoch", 0)) + 1
        best_val = float(checkpoint.get("best_val_loss", best_val))
        best_delta_r = float(checkpoint.get("best_delta_r", best_delta_r))
        best_hard_delta_r = float(checkpoint.get("best_hard_delta_r", best_hard_delta_r))
        best_sign = float(checkpoint.get("best_sign_accuracy", best_sign))
        best_pattern = float(checkpoint.get("best_pattern_score", best_pattern))
        no_improve = int(checkpoint.get("no_improve", 0))
        history = list(checkpoint.get("history", []))
        print(f"Resumed from {args.resume} at epoch {start_epoch}", flush=True)

    print(f"Stage1-met cache: {args.cache_root}", flush=True)
    print(f"Prev-CO2 sidecar: {args.prev_sidecar_root}", flush=True)
    print(f"Dataset ready: train={len(train_ds)} val={len(val_ds)} channels={len(train_ds.channels)}", flush=True)
    print(f"Input channels: {list(train_ds.channels)}", flush=True)
    context_kind = "full-domain sidecar" if args.global_sidecar_root is not None else "patch-downsample fallback"
    print(
        f"Global context: enabled via {context_kind} channels={list(train_ds.global_channels)} "
        f"shape=[{len(train_ds.global_channels)},16,{train_ds.global_size},{train_ds.global_size}]",
        flush=True,
    )
    print(f"Using device: {device}", flush=True)
    print(f"Checkpoint path: {args.out_dir / 'best_model.pt'}", flush=True)

    if args.smoke_only:
        run_loss_smoke(model, train_ds, device)
        return

    tr_loader = train_loader(train_ds, args.train_samples_per_epoch, args.batch_size, args.num_workers)
    va_loader = eval_loader(val_ds, args.val_samples, args.batch_size, args.num_workers, args.seed + 100003)

    common_loss = {
        "huber_delta": 2.0,
        "base_loss": "huber",
        "gradient_loss_weight": 0.03,
        "multiscale_loss_weight": 0.0,
        "multiscale_scales": (8, 16, 32, 64),
        "multiscale_min_valid_fraction": 0.5,
        "smoothness_loss_weight": 0.0,
        "smoothness_kernel_size": 9,
        "height_gate_channel_idx": None,
        "variance_loss_weight": 0.0,
        "variance_min_std": 2.0,
        "variance_eps": 1.0e-6,
        "residual_weight_alpha": 0.5,
        "residual_weight_scale": 4.0,
        "residual_weight_max": 4.0,
        "target_gradient_weight_alpha": 0.35,
        "target_gradient_weight_scale": 1.0,
        "target_gradient_weight_max": 2.5,
        "low_layer_weight_alpha": 0.0,
        "normalized_loss_weight": 0.0,
        "normalized_min_std": 0.75,
        "normalized_huber_delta": 1.0,
        "normalized_eps": 1.0e-6,
        "correlation_loss_weight": 0.04,
        "correlation_eps": 1.0e-6,
        "correlation_min_target_std": 0.25,
        "correlation_min_valid_fraction": 0.5,
        "low_frequency_loss_weight": 0.04,
        "low_frequency_pool": 16,
        "low_frequency_min_valid_fraction": 0.40,
        "low_frequency_correlation_weight": 0.02,
        "high_frequency_loss_weight": 0.04,
        "high_frequency_huber_delta": 0.85,
        "local_correlation_loss_weight": 0.05,
        "local_correlation_pool": 32,
        "local_correlation_min_target_std": 0.45,
        "local_correlation_min_valid_fraction": 0.40,
        "amplitude_loss_weight": 0.05,
        "amplitude_min_target_std": 0.45,
        "active_delta_loss_weight": 0.35,
        "active_delta_threshold": 0.75,
        "sign_loss_weight": 0.015,
        "sign_loss_min_abs": 0.75,
        "sign_loss_scale": 2.0,
        "active_loss_weight": 0.02,
        "active_loss_threshold": 0.75,
        "active_loss_pos_weight": 2.0,
        "sign_class_loss_weight": 0.02,
        "sign_class_loss_min_abs": 0.75,
        "sign_class_loss_pos_weight": 1.0,
        "pattern_height_decay": False,
    }
    print(f"Loss config: V40 keep-prev monitor loss, Stage1 met replacement with global context", flush=True)

    for epoch in range(start_epoch, args.epochs + 1):
        train_loss = run_epoch(
            model,
            tr_loader,
            optimizer,
            device,
            desc=f"train {epoch}",
            epoch=epoch,
            log_every=args.log_every,
            progress=args.progress,
            collect_delta_monitor=False,
            **common_loss,
        )
        val_loss, monitor = run_epoch(
            model,
            va_loader,
            None,
            device,
            desc=f"val {epoch}",
            epoch=epoch,
            log_every=args.log_every,
            progress=args.progress,
            collect_delta_monitor=True,
            delta_monitor_active_threshold=0.75,
            delta_monitor_min_valid=256,
            delta_monitor_hard_min_std=1.0,
            delta_monitor_hard_min_active_fraction=0.25,
            **common_loss,
        )
        old_lr = current_lr(optimizer)
        scheduler.step(float(val_loss))
        new_lr = current_lr(optimizer)

        improved = float(val_loss) < best_val - args.early_stopping_min_delta
        if improved:
            best_val = float(val_loss)
            no_improve = 0
        else:
            no_improve += 1

        row = {
            "epoch": epoch,
            "train_loss": float(train_loss),
            "val_loss": float(val_loss),
            "best_val_loss": best_val,
            "lr": new_lr,
            "improved": int(improved),
            "no_improve": no_improve,
            **monitor,
        }
        history.append(row)
        write_csv(args.out_dir / "history.csv", history)

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "model_config": {
                "variant": "v40_local_event_texture_stage1_met",
                "in_channels": len(V40_STAGE1_MET_CHANNELS),
                "base_channels": args.base_channels,
                "global_channels": len(V40_STAGE1_GLOBAL_CONTEXT_CHANNELS),
                "global_feature_channels": 8,
                "input_channels": list(V40_STAGE1_MET_CHANNELS),
                "global_input_channels": list(V40_STAGE1_GLOBAL_CONTEXT_CHANNELS),
            },
            "dataset_config": {
                "cache_root": str(args.cache_root),
                "prev_sidecar_root": str(args.prev_sidecar_root),
                "global_sidecar_root": str(args.global_sidecar_root) if args.global_sidecar_root is not None else None,
                "layer_min": args.layer_min,
                "layer_max": args.layer_max,
                "dx": args.dx,
                "dy": args.dy,
                "dz": args.dz,
            },
            "loss_config": common_loss,
            "best_val_loss": best_val,
            "best_delta_r": best_delta_r,
            "best_hard_delta_r": best_hard_delta_r,
            "best_sign_accuracy": best_sign,
            "best_pattern_score": best_pattern,
            "no_improve": no_improve,
            "history": history,
        }
        torch.save(checkpoint, args.out_dir / "last_model.pt")
        if improved:
            torch.save(checkpoint, args.out_dir / "best_model.pt")

        metric_paths = [
            ("delta_r_mean", "best_delta_r", "best_delta_r_model.pt"),
            ("hard_delta_r_mean", "best_hard_delta_r", "best_hard_delta_r_model.pt"),
            ("sign_accuracy", "best_sign_accuracy", "best_sign_accuracy_model.pt"),
            ("pattern_score", "best_pattern_score", "best_pattern_score_model.pt"),
        ]
        for metric_key, best_key, filename in metric_paths:
            value = float(monitor.get(metric_key, float("nan")))
            if np.isfinite(value) and value > float(checkpoint[best_key]):
                if best_key == "best_delta_r":
                    best_delta_r = value
                elif best_key == "best_hard_delta_r":
                    best_hard_delta_r = value
                elif best_key == "best_sign_accuracy":
                    best_sign = value
                elif best_key == "best_pattern_score":
                    best_pattern = value
                checkpoint[best_key] = value
                torch.save(checkpoint, args.out_dir / filename)

        if new_lr < old_lr:
            print(f"LR reduced: {old_lr:.6g} -> {new_lr:.6g}", flush=True)
        print(
            f"epoch={epoch:03d}/{args.epochs:03d} train_loss={float(train_loss):.6f} "
            f"val_loss={float(val_loss):.6f} best_val={best_val:.6f} "
            f"delta_R={monitor.get('delta_r_mean', float('nan')):.6f} "
            f"hard_delta_R={monitor.get('hard_delta_r_mean', float('nan')):.6f} "
            f"sign_acc={monitor.get('sign_accuracy', float('nan')):.6f} "
            f"pattern={monitor.get('pattern_score', float('nan')):.6f} "
            f"lr={new_lr:.6g} improved={int(improved)} no_improve={no_improve}",
            flush=True,
        )
        if no_improve >= args.early_stopping_patience:
            print(f"Early stopping at epoch {epoch}", flush=True)
            break

    summary = {
        "out_dir": str(args.out_dir),
        "last_epoch": history[-1]["epoch"] if history else None,
        "best_val_loss": best_val,
        "best_delta_r": best_delta_r,
        "best_hard_delta_r": best_hard_delta_r,
        "best_sign_accuracy": best_sign,
        "best_pattern_score": best_pattern,
    }
    (args.out_dir / "training_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Saved outputs to {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()

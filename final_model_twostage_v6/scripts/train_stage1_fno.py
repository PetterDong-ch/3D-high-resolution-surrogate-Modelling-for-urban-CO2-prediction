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
import yaml
from torch.utils.data import DataLoader, RandomSampler, Sampler, SequentialSampler, Subset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_ROOT))

from two_stage_surrogate.data.stage1_cache import Stage1CacheDataset, denormalize_target  # noqa: E402
from two_stage_surrogate.models import LocalFNOStage1, Stage1ModelConfig  # noqa: E402
from two_stage_surrogate.training import Stage1Loss, Stage1LossConfig  # noqa: E402


# Accumulates train/validation metric totals.
class MetricAccumulator:
    # Store constructor arguments and initialize object state.
    def __init__(self) -> None:
        self.count = 0.0
        self.abs_sum = 0.0
        self.sq_sum = 0.0
        self.pred_sum = 0.0
        self.target_sum = 0.0
        self.pred_sq_sum = 0.0
        self.target_sq_sum = 0.0
        self.cross_sum = 0.0

    # Update running metric or statistic accumulators.
    def update(self, pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> None:
        mask = mask.to(dtype=torch.bool)
        while mask.ndim < pred.ndim:
            mask = mask.unsqueeze(1)
        mask = mask.expand_as(pred)
        pred_v = pred.detach()[mask].double().cpu()
        target_v = target.detach()[mask].double().cpu()
        if pred_v.numel() == 0:
            return
        diff = pred_v - target_v
        self.count += float(pred_v.numel())
        self.abs_sum += float(diff.abs().sum())
        self.sq_sum += float((diff * diff).sum())
        self.pred_sum += float(pred_v.sum())
        self.target_sum += float(target_v.sum())
        self.pred_sq_sum += float((pred_v * pred_v).sum())
        self.target_sq_sum += float((target_v * target_v).sum())
        self.cross_sum += float((pred_v * target_v).sum())

    # Return the final accumulated statistics.
    def finalize(self) -> dict[str, float]:
        if self.count <= 0:
            return {"MAE": float("nan"), "RMSE": float("nan"), "R": float("nan")}
        mae = self.abs_sum / self.count
        rmse = (self.sq_sum / self.count) ** 0.5
        cov = self.cross_sum - self.pred_sum * self.target_sum / self.count
        pred_var = self.pred_sq_sum - self.pred_sum * self.pred_sum / self.count
        target_var = self.target_sq_sum - self.target_sum * self.target_sum / self.count
        if pred_var <= 1e-12 or target_var <= 1e-12:
            corr = float("nan")
        else:
            corr = cov / ((pred_var * target_var) ** 0.5)
        return {"MAE": mae, "RMSE": rmse, "R": corr}


# Internal helper for seed all.
def _seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# Internal helper for to device.
def _to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            out[key] = value.to(device, non_blocking=True)
        else:
            out[key] = value
    return out


# Internal helper for write history.
def _write_history(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


# Internal helper for make loader.
def _make_loader(
    dataset: Stage1CacheDataset | Subset,
    *,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    samples_per_epoch: int | None = None,
    group_by_shard: bool = False,
    seed: int = 42,
) -> DataLoader:
    generator = torch.Generator().manual_seed(seed)
    if samples_per_epoch is not None:
        if group_by_shard and isinstance(dataset, Stage1CacheDataset):
            sampler = ShardGroupedRandomSampler(dataset, num_samples=samples_per_epoch, seed=seed)
            return DataLoader(dataset, batch_size=batch_size, sampler=sampler, num_workers=num_workers, pin_memory=torch.cuda.is_available())
        sampler = RandomSampler(dataset, replacement=True, num_samples=samples_per_epoch, generator=generator)
        return DataLoader(dataset, batch_size=batch_size, sampler=sampler, num_workers=num_workers, pin_memory=torch.cuda.is_available())
    sampler = RandomSampler(dataset, generator=generator) if shuffle else SequentialSampler(dataset)
    return DataLoader(dataset, batch_size=batch_size, sampler=sampler, num_workers=num_workers, pin_memory=torch.cuda.is_available())


# Samples rows while keeping shard-local access efficient.
class ShardGroupedRandomSampler(Sampler[int]):
    """Randomly samples indices, then yields them grouped by shard.

    This keeps the randomness of the epoch subset while avoiding pathological
    I/O where every sample reloads a different 100+ MB npz shard.
    """

    # Store constructor arguments and initialize object state.
    def __init__(self, dataset: Stage1CacheDataset, num_samples: int, seed: int) -> None:
        self.dataset = dataset
        self.num_samples = int(num_samples)
        self.seed = int(seed)
        self.epoch = 0

    # Internal helper for iter.
    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        self.epoch += 1
        selected = rng.integers(0, len(self.dataset), size=self.num_samples)
        groups: dict[int, list[int]] = {}
        for index in selected:
            shard_index, _ = self.dataset._local_index(int(index))
            groups.setdefault(shard_index, []).append(int(index))
        shard_order = list(groups)
        rng.shuffle(shard_order)
        for shard_index in shard_order:
            values = groups[shard_index]
            rng.shuffle(values)
            yield from values

    # Return the number of available samples.
    def __len__(self) -> int:
        return self.num_samples


# Internal helper for limited subset.
def _limited_subset(dataset: Stage1CacheDataset, limit: int | None) -> Stage1CacheDataset | Subset:
    if limit is None or limit >= len(dataset):
        return dataset
    return Subset(dataset, list(range(limit)))


# Internal helper for batch loss.
def _batch_loss(
    *,
    model: LocalFNOStage1,
    loss_fn: Stage1Loss,
    batch: dict[str, Any],
) -> tuple[torch.Tensor, dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    pred = model(
        geometry_3d=batch["geometry_3d"],
        surface_2d=batch["surface_2d"],
        profile=batch["profile"],
        scalar=batch["scalar"],
        theta_reference=None,
    )
    target = {
        "uv": batch["target_uv"],
        "theta_prime": batch["target_theta_prime"],
    }
    if model.config.predict_w and loss_fn.config.predict_w:
        target["w"] = batch["target_w"]
    loss, parts = loss_fn(pred, target, batch["mask"])
    return loss, parts, pred


# Internal helper for evaluate.
def _evaluate(
    *,
    model: LocalFNOStage1,
    loader: DataLoader,
    loss_fn: Stage1Loss,
    target_stats: dict[str, torch.Tensor],
    device: torch.device,
) -> tuple[float, dict[str, float], dict[str, dict[str, float]]]:
    model.eval()
    total_loss = 0.0
    steps = 0
    metrics = {
        "u": MetricAccumulator(),
        "v": MetricAccumulator(),
        "theta_prime": MetricAccumulator(),
        "theta_reconstructed": MetricAccumulator(),
    }
    if model.config.predict_w:
        metrics["w"] = MetricAccumulator()
    with torch.no_grad():
        for raw_batch in loader:
            batch = _to_device(raw_batch, device)
            loss, _, pred = _batch_loss(model=model, loss_fn=loss_fn, batch=batch)
            total_loss += float(loss.detach().cpu())
            steps += 1

            uv_pred = denormalize_target("target_uv", pred["uv"], target_stats)
            uv_target = denormalize_target("target_uv", batch["target_uv"], target_stats)
            theta_prime_pred = denormalize_target("target_theta_prime", pred["theta_prime"], target_stats)
            theta_prime_target = denormalize_target("target_theta_prime", batch["target_theta_prime"], target_stats)
            theta_pred = batch["theta_reference"] + theta_prime_pred

            metrics["u"].update(uv_pred[:, 0:1], uv_target[:, 0:1], batch["mask"])
            metrics["v"].update(uv_pred[:, 1:2], uv_target[:, 1:2], batch["mask"])
            if model.config.predict_w:
                w_pred = denormalize_target("target_w", pred["w"], target_stats)
                w_target = denormalize_target("target_w", batch["target_w"], target_stats)
                metrics["w"].update(w_pred, w_target, batch["mask"])
            metrics["theta_prime"].update(theta_prime_pred, theta_prime_target, batch["mask"])
            metrics["theta_reconstructed"].update(theta_pred, batch["target_theta"], batch["mask"])
    avg_loss = total_loss / max(1, steps)
    metric_dict = {name: acc.finalize() for name, acc in metrics.items()}
    flat = {f"{name}_{key}": value for name, values in metric_dict.items() for key, value in values.items()}
    return avg_loss, flat, metric_dict


# Entry point for the command-line workflow.
def main() -> None:
    parser = argparse.ArgumentParser(description="Train Stage 1 Local-FNO from a sharded cache.")
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs" / "stage1_fno_camden.yaml")
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=PROJECT_ROOT / "processed" / "stage1_fno_run")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--weight-decay", type=float)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--train-samples-per-epoch", type=int)
    parser.add_argument("--limit-train-samples", type=int)
    parser.add_argument("--limit-val-samples", type=int)
    parser.add_argument("--val-samples-per-epoch", type=int)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--disable-shard-grouping", action="store_true")
    parser.add_argument("--width", type=int)
    parser.add_argument("--depth", type=int)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    training_cfg = config.get("training", {})
    seed = int(args.seed if args.seed is not None else training_cfg.get("seed", 42))
    _seed_all(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    args.out_dir.mkdir(parents=True, exist_ok=True)

    train_ds = Stage1CacheDataset(args.cache_root, split="train")
    val_ds = Stage1CacheDataset(args.cache_root, split="val")
    train_data = _limited_subset(train_ds, args.limit_train_samples)
    val_data = _limited_subset(val_ds, args.limit_val_samples)
    print(f"Stage 1 cache: {args.cache_root}", flush=True)
    print(f"Dataset ready: train={len(train_data)} val={len(val_data)}", flush=True)

    example = train_ds[0]
    model_cfg = dict(config["model"])
    model_cfg["geometry_channels"] = int(example["geometry_3d"].shape[0])
    model_cfg["surface_channels"] = int(example["surface_2d"].shape[0])
    model_cfg["profile_channels"] = int(example["profile"].shape[0])
    model_cfg["scalar_channels"] = int(example["scalar"].shape[0])
    if args.width is not None:
        model_cfg["width"] = args.width
    if args.depth is not None:
        model_cfg["depth"] = args.depth

    model = LocalFNOStage1(Stage1ModelConfig(**model_cfg))
    loss_fn = Stage1Loss(Stage1LossConfig(**config["loss"]))
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    model.to(device)
    target_stats = train_ds.target_stat_tensors(device)

    epochs = int(args.epochs if args.epochs is not None else training_cfg.get("epochs", 120))
    batch_size = int(args.batch_size if args.batch_size is not None else training_cfg.get("batch_size", 1))
    lr = float(args.lr if args.lr is not None else training_cfg.get("lr", 3e-4))
    weight_decay = float(args.weight_decay if args.weight_decay is not None else training_cfg.get("weight_decay", 1e-4))

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler_cfg = training_cfg.get("scheduler", {})
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=float(scheduler_cfg.get("factor", 0.5)),
        patience=int(scheduler_cfg.get("patience", 8)),
        min_lr=float(scheduler_cfg.get("min_lr", 1e-6)),
    )
    early_cfg = training_cfg.get("early_stopping", {})
    early_patience = int(early_cfg.get("patience", 20))

    start_epoch = 1
    best_val = float("inf")
    no_improve = 0
    history: list[dict[str, Any]] = []
    if args.resume is not None:
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        if "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = int(checkpoint.get("epoch", 0)) + 1
        best_val = float(checkpoint.get("best_val_loss", best_val))
        history = list(checkpoint.get("history", []))
        print(f"Resumed from {args.resume} at epoch {start_epoch}", flush=True)

    train_loader = _make_loader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        samples_per_epoch=args.train_samples_per_epoch,
        group_by_shard=not args.disable_shard_grouping,
        seed=seed,
    )
    val_loader = _make_loader(
        val_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        samples_per_epoch=args.val_samples_per_epoch,
        group_by_shard=not args.disable_shard_grouping,
        seed=seed + 100003,
    )

    print(f"Using device: {device}", flush=True)
    print(f"Model config: {model_cfg}", flush=True)
    print(f"Loss config: {config['loss']}", flush=True)
    print(f"Checkpoint path: {args.out_dir / 'best_model.pt'}", flush=True)

    for epoch in range(start_epoch, epochs + 1):
        model.train()
        total_train = 0.0
        train_steps = 0
        for raw_batch in train_loader:
            batch = _to_device(raw_batch, device)
            loss, _, _ = _batch_loss(model=model, loss_fn=loss_fn, batch=batch)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total_train += float(loss.detach().cpu())
            train_steps += 1
            if args.log_every > 0 and (train_steps == 1 or train_steps % args.log_every == 0):
                print(
                    f"[train] epoch={epoch:03d} step={train_steps:04d} "
                    f"avg_loss={total_train / max(1, train_steps):.6f}",
                    flush=True,
                )
        train_loss = total_train / max(1, train_steps)
        val_loss, flat_metrics, metric_dict = _evaluate(
            model=model,
            loader=val_loader,
            loss_fn=loss_fn,
            target_stats=target_stats,
            device=device,
        )
        scheduler.step(val_loss)
        improved = val_loss < best_val
        if improved:
            best_val = val_loss
            no_improve = 0
        else:
            no_improve += 1

        row: dict[str, Any] = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "best_val_loss": best_val,
            "lr": optimizer.param_groups[0]["lr"],
            "improved": int(improved),
            "no_improve": no_improve,
        }
        row.update(flat_metrics)
        history.append(row)
        _write_history(args.out_dir / "history.csv", history)

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "model_config": model_cfg,
            "loss_config": config["loss"],
            "cache_root": str(args.cache_root),
            "cache_manifest": train_ds.manifest,
            "best_val_loss": best_val,
            "history": history,
            "metrics_physical_units": metric_dict,
        }
        torch.save(checkpoint, args.out_dir / "last_model.pt")
        if improved:
            torch.save(checkpoint, args.out_dir / "best_model.pt")

        metric_text = (
            f"u_R={flat_metrics['u_R']:.4f} "
            f"v_R={flat_metrics['v_R']:.4f} "
        )
        if model.config.predict_w:
            metric_text += f"w_R={flat_metrics['w_R']:.4f} "
        metric_text += f"theta_R={flat_metrics['theta_reconstructed_R']:.4f}"
        print(
            f"epoch={epoch:03d}/{epochs:03d} train_loss={train_loss:.6f} "
            f"val_loss={val_loss:.6f} {metric_text} best={best_val:.6f} "
            f"lr={optimizer.param_groups[0]['lr']:.2e} no_improve={no_improve}",
            flush=True,
        )

        if no_improve >= early_patience:
            print(f"Early stopping at epoch {epoch}", flush=True)
            break

    summary = {
        "config": str(args.config),
        "cache_root": str(args.cache_root),
        "out_dir": str(args.out_dir),
        "best_val_loss": best_val,
        "last_epoch": history[-1]["epoch"] if history else None,
        "model_config": model_cfg,
        "loss_config": config["loss"],
        "history_csv": str(args.out_dir / "history.csv"),
        "best_checkpoint": str(args.out_dir / "best_model.pt"),
        "last_checkpoint": str(args.out_dir / "last_model.pt"),
    }
    (args.out_dir / "training_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Saved training outputs to {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()

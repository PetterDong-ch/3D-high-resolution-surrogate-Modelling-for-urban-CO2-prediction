from __future__ import annotations

import argparse
import json
import os
import random
import time

import numpy as np
import torch

from .data import BackgroundEnhancementV3Dataset, compute_target_norm_stats, make_layer_weights
from .engine import evaluate, make_loader
from .io import load_history, save_history, write_json
from .losses import weighted_huber_loss
from .network import CompactProfileNet


# Entry point for the command-line workflow.
def main() -> None:
    parser = argparse.ArgumentParser(description="Task2_V3: normalized background-enhancement profile with low-layer weighted loss.")
    parser.add_argument("--profile-cache-root", required=True, help="Prepared profile-cache directory.")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--train-samples-per-epoch", type=int, default=4096)
    parser.add_argument("--val-samples", type=int, default=0)
    parser.add_argument("--base-channels", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3.0e-4)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--huber-delta", type=float, default=1.0)
    parser.add_argument("--profile-gradient-weight", type=float, default=0.05)
    parser.add_argument("--low-layer-alpha", type=float, default=1.5)
    parser.add_argument("--low-layer-tau", type=float, default=4.0)
    parser.add_argument("--valid-weight-power", type=float, default=0.25)
    parser.add_argument("--target-min-std", type=float, default=0.2)
    parser.add_argument("--drop-global-context", action="store_true")
    parser.add_argument("--early-stopping-patience", type=int, default=18)
    parser.add_argument("--lr-patience", type=int, default=8)
    parser.add_argument("--lr-factor", type=float, default=0.5)
    parser.add_argument("--min-lr", type=float, default=1.0e-6)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--init-checkpoint", default="", help="Load model weights only and start a fresh run. Useful for transfer learning.")
    parser.add_argument("--resume", default="")
    parser.add_argument("--reset-optimizer-on-resume", action="store_true")
    parser.add_argument("--override-lr-on-resume", action="store_true")
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--checkpoint", default="")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    os.makedirs(args.out_dir, exist_ok=True)
    if args.init_checkpoint and args.resume:
        raise ValueError("--init-checkpoint and --resume are mutually exclusive. Use init for transfer learning, resume for continuing a run.")

    target_mean, target_std, valid_fraction = compute_target_norm_stats(args.profile_cache_root, args.target_min_std)
    layer_weights = make_layer_weights(
        depth=len(target_mean),
        valid_fraction=valid_fraction,
        low_alpha=args.low_layer_alpha,
        low_tau=args.low_layer_tau,
        valid_power=args.valid_weight_power,
    )

    train_ds = BackgroundEnhancementV3Dataset(args.profile_cache_root, "train", target_mean, target_std, use_global_context=not args.drop_global_context)
    val_ds = BackgroundEnhancementV3Dataset(args.profile_cache_root, "val", target_mean, target_std, use_global_context=not args.drop_global_context)
    local_channels = int(train_ds.local.shape[1])
    global_channels = int(train_ds.global_context.shape[1])
    depth = int(train_ds.local.shape[2])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CompactProfileNet(local_channels, global_channels, base_channels=args.base_channels).to(device)
    model.initialize_output_bias(0.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=args.lr_factor,
        patience=args.lr_patience,
        min_lr=args.min_lr,
    )
    layer_weights_t = torch.from_numpy(layer_weights).to(device=device)

    print("Task2_V3 target=background-normalized enhancement", flush=True)
    print(f"Profile cache={os.path.abspath(args.profile_cache_root)}", flush=True)
    print(f"Train samples={len(train_ds)} val samples={len(val_ds)} depth={depth}", flush=True)
    print(f"Feature channels: local={local_channels}, global={global_channels}", flush=True)
    print(f"Global context enabled={not args.drop_global_context}", flush=True)
    print("Previous timestep CO2 is excluded completely. Target/reconstruction use current incoming background CO2.", flush=True)
    print(f"Using device={device}", flush=True)
    print(f"Enhancement mean per z={np.round(target_mean, 4).tolist()}", flush=True)
    print(f"Enhancement std per z={np.round(target_std, 4).tolist()}", flush=True)
    print(f"Valid fraction per z={np.round(valid_fraction, 4).tolist()}", flush=True)
    print(f"Layer weights={np.round(layer_weights, 4).tolist()}", flush=True)

    config = {
        "target": "normalized_background_enhancement",
        "profile_cache_root": os.path.abspath(args.profile_cache_root),
        "local_feature_count": local_channels,
        "global_feature_count": global_channels,
        "global_context_enabled": not args.drop_global_context,
        "depth": depth,
        "target_mean": target_mean.tolist(),
        "target_std": target_std.tolist(),
        "valid_fraction": valid_fraction.tolist(),
        "layer_weights": layer_weights.tolist(),
        "loss": {
            "base": "weighted_huber_on_normalized_background_enhancement",
            "huber_delta": args.huber_delta,
            "profile_gradient_weight": args.profile_gradient_weight,
            "low_layer_alpha": args.low_layer_alpha,
            "low_layer_tau": args.low_layer_tau,
            "valid_weight_power": args.valid_weight_power,
        },
        "note": "Final metrics are computed after denormalizing enhancement and reconstructing concentration as background_CO2 + predicted_enhancement.",
    }
    write_json(os.path.join(args.out_dir, "config.json"), config)

    start_epoch = 1
    best_val = float("inf")
    history_path = os.path.join(args.out_dir, "history.csv")
    history: list[dict[str, float]] = [] if args.eval_only else load_history(history_path)
    if args.init_checkpoint and not args.eval_only:
        checkpoint = torch.load(args.init_checkpoint, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"Initialized model weights from checkpoint: {args.init_checkpoint}", flush=True)
        print("Fresh optimizer/scheduler and epoch counter are used for this transfer run.", flush=True)
    ckpt_path = args.checkpoint or (os.path.join(args.out_dir, "best_model.pt") if args.eval_only else args.resume)
    if ckpt_path:
        checkpoint = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        if not args.eval_only:
            start_epoch = int(checkpoint.get("epoch", 0)) + 1
            best_val = float(checkpoint.get("best_val", best_val))
            if "optimizer_state_dict" in checkpoint and not args.reset_optimizer_on_resume:
                optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                scheduler.load_state_dict(checkpoint.get("scheduler_state_dict", scheduler.state_dict()))
            if args.override_lr_on_resume:
                for group in optimizer.param_groups:
                    group["lr"] = float(args.lr)
        print(f"Loaded checkpoint: {ckpt_path}", flush=True)
        if not args.eval_only:
            print(
                f"Resume start_epoch={start_epoch}, optimizer_reset={args.reset_optimizer_on_resume}, "
                f"lr={optimizer.param_groups[0]['lr']:g}",
                flush=True,
            )

    val_loader = make_loader(val_ds, args.val_samples, args.batch_size, args.num_workers, shuffle=False)
    if args.eval_only:
        metrics = evaluate(model, val_loader, device, target_mean, target_std, out_dir=args.out_dir)
        print(json.dumps(metrics["concentration"], indent=2), flush=True)
        return

    best_path = os.path.join(args.out_dir, "best_model.pt")
    last_path = os.path.join(args.out_dir, "last_model.pt")
    no_improve = 0
    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        train_loader = make_loader(train_ds, args.train_samples_per_epoch, args.batch_size, args.num_workers, shuffle=True)
        losses = []
        t0 = time.time()
        for batch in train_loader:
            local, global_context, target_norm, mask, *_ = batch
            local = local.to(device=device, non_blocking=True)
            global_context = global_context.to(device=device, non_blocking=True)
            target_norm = target_norm.to(device=device, non_blocking=True)
            mask = mask.to(device=device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            pred_norm = model(local, global_context)
            loss = weighted_huber_loss(
                pred_norm,
                target_norm,
                mask,
                layer_weights_t,
                huber_delta=args.huber_delta,
                gradient_weight=args.profile_gradient_weight,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu().item()))
        train_loss = float(np.mean(losses)) if losses else float("nan")
        eval_metrics = evaluate(model, val_loader, device, target_mean, target_std)
        val_metric = float(eval_metrics["concentration"]["RMSE"])
        val_target_rmse = float(eval_metrics["target_enhancement"]["RMSE"])
        val_conc_mae = float(eval_metrics["concentration"]["MAE"])
        val_conc_r = float(eval_metrics["concentration"]["R"])
        scheduler.step(val_metric)
        lr = float(optimizer.param_groups[0]["lr"])
        improved = val_metric < best_val
        if improved:
            best_val = val_metric
            no_improve = 0
        else:
            no_improve += 1
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_concentration_rmse": val_metric,
            "val_concentration_mae": val_conc_mae,
            "val_concentration_R": val_conc_r,
            "val_target_enhancement_rmse": val_target_rmse,
            "best_val_concentration_rmse": best_val,
            "lr": lr,
            "elapsed_s": time.time() - t0,
        }
        history.append(row)
        save_history(history_path, history)
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "best_val": best_val,
            "config": config,
        }
        torch.save(checkpoint, last_path)
        if improved:
            torch.save(checkpoint, best_path)
        print(
            f"epoch={epoch:03d} train_loss={train_loss:.6f} "
            f"val_conc_R={val_conc_r:.6f} val_conc_MAE={val_conc_mae:.6f} "
            f"val_conc_RMSE={val_metric:.6f} val_enhancement_RMSE={val_target_rmse:.6f} "
            f"best={best_val:.6f} lr={lr:g} improved={int(improved)} no_improve={no_improve} "
            f"elapsed={time.time() - t0:.1f}s",
            flush=True,
        )
        if no_improve >= args.early_stopping_patience:
            print(f"Early stopping at epoch {epoch}", flush=True)
            break

    print(f"Saved best checkpoint: {best_path}", flush=True)
    best_checkpoint = torch.load(best_path, map_location=device)
    model.load_state_dict(best_checkpoint["model_state_dict"])
    final_metrics = evaluate(model, val_loader, device, target_mean, target_std, out_dir=args.out_dir)
    print("Final validation concentration metrics:", flush=True)
    print(json.dumps(final_metrics["concentration"], indent=2), flush=True)


if __name__ == "__main__":
    main()

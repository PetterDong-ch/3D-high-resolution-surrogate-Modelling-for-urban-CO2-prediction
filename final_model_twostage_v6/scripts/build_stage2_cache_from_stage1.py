#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from netCDF4 import Dataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
V1_ROOT = PROJECT_ROOT
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(V1_ROOT / "src"))

from stage2_cache import BASE_CHANNELS, ChannelStats, assemble_stage2_input  # noqa: E402
from two_stage_surrogate.data.alignment import linear_time_match, values_to_seconds  # noqa: E402
from two_stage_surrogate.data.stage1_cache import denormalize_target, load_cache_manifest  # noqa: E402
from two_stage_surrogate.data.stage1_direct_reader import (  # noqa: E402
    _array,
    _interpolate_profile_to,
    _resize_nearest_2d,
    load_manifest_records,
    read_stage1_sample,
)
from two_stage_surrogate.models.stage1_fno import LocalFNOStage1, Stage1ModelConfig  # noqa: E402


ARRAY_FIELDS = (
    "met_pred",
    "emission_2d",
    "bg_profile",
    "surface_2d",
    "scalar",
    "target",
    "mask",
    "z0",
    "y0",
    "x0",
    "nz",
    "ny",
    "nx",
)


# Internal helper for safe prepare out root.
def _safe_prepare_out_root(out_root: Path, overwrite: bool) -> None:
    out_root = out_root.resolve()
    project = PROJECT_ROOT.resolve()
    if not str(out_root).startswith(str(project)):
        raise ValueError(f"Refusing to write outside project root: {out_root}")
    if out_root.exists():
        if not overwrite:
            raise FileExistsError(f"{out_root} already exists. Use --overwrite or choose another --out-root.")
        if "stage2_cache" not in out_root.name:
            raise ValueError(f"Refusing to overwrite path that does not look like a Stage 2 cache: {out_root}")
        shutil.rmtree(out_root)
    out_root.mkdir(parents=True, exist_ok=True)


# Internal helper for select records.
def _select_records(records: list[dict[str, Any]], count: int | None, selection: str, seed: int) -> list[dict[str, Any]]:
    if count is None or count >= len(records):
        return list(records)
    if selection == "first":
        return records[:count]
    if selection == "even":
        positions = np.linspace(0, len(records) - 1, count).round().astype(np.int64)
        return [records[int(pos)] for pos in positions]
    rng = np.random.default_rng(seed)
    indices = np.sort(rng.choice(len(records), size=count, replace=False))
    return [records[int(idx)] for idx in indices]


# Internal helper for norm array.
def _norm_array(arr: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return ((arr.astype(np.float32, copy=False) - mean) / np.maximum(std, 1.0e-6)).astype(np.float32, copy=False)


# Internal helper for load stage1 model.
def _load_stage1_model(checkpoint_path: Path, device: torch.device) -> tuple[LocalFNOStage1, dict[str, torch.Tensor]]:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model_cfg = dict(checkpoint["model_config"])
    model = LocalFNOStage1(Stage1ModelConfig(**model_cfg)).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    cache_manifest = checkpoint.get("cache_manifest")
    if cache_manifest is None:
        cache_manifest = load_cache_manifest(checkpoint["cache_root"])
    stats: dict[str, torch.Tensor] = {}
    for field in ("target_uv", "target_w", "target_theta_prime"):
        stats[f"{field}_mean"] = torch.tensor(cache_manifest["normalization"][f"{field}_mean"], device=device)
        stats[f"{field}_std"] = torch.tensor(cache_manifest["normalization"][f"{field}_std"], device=device)
    return model, stats


# Internal helper for stage1 predict.
def _stage1_predict(
    *,
    model: LocalFNOStage1,
    target_stats: dict[str, torch.Tensor],
    stage1_norm: dict[str, Any],
    sample: Any,
    device: torch.device,
    w_strategy: str,
) -> np.ndarray:
    inputs = {}
    for field in ("geometry_3d", "surface_2d", "profile", "scalar"):
        mean = np.asarray(stage1_norm[f"{field}_mean"], dtype=np.float32)
        std = np.asarray(stage1_norm[f"{field}_std"], dtype=np.float32)
        value = _norm_array(getattr(sample, field), mean, std)
        inputs[field] = torch.from_numpy(value[None]).to(device)

    with torch.no_grad():
        pred = model(
            geometry_3d=inputs["geometry_3d"],
            surface_2d=inputs["surface_2d"],
            profile=inputs["profile"],
            scalar=inputs["scalar"],
            theta_reference=None,
        )
        uv = denormalize_target("target_uv", pred["uv"], target_stats)[0].detach().cpu().numpy()
        theta_prime = denormalize_target("target_theta_prime", pred["theta_prime"], target_stats)[0].detach().cpu().numpy()
        w_pred = denormalize_target("target_w", pred["w"], target_stats)[0].detach().cpu().numpy()

    theta = sample.theta_reference + theta_prime
    if w_strategy == "predicted":
        w = w_pred
    elif w_strategy == "zero":
        w = np.zeros_like(w_pred, dtype=np.float32)
    elif w_strategy == "truth":
        w = sample.target_w.astype(np.float32, copy=False)
    elif w_strategy == "forcing_profile":
        # Stage 1 profile channel 2 is the dynamic right-boundary w profile.
        profile_w = np.asarray(sample.profile[2], dtype=np.float32)
        depth, height, width = w_pred.shape[-3:]
        take = np.interp(
            np.linspace(0, len(profile_w) - 1, depth, dtype=np.float32),
            np.arange(len(profile_w), dtype=np.float32),
            profile_w,
        ).astype(np.float32)
        w = np.broadcast_to(take[None, :, None, None], w_pred.shape).copy()
    else:
        raise ValueError(f"Unknown w strategy: {w_strategy}")

    return np.concatenate([uv, theta.astype(np.float32), w.astype(np.float32)], axis=0)


# Internal helper for locate av3d.
def _locate_av3d(job_files: dict[str, Any]) -> str | None:
    job_dir = Path(job_files["job_dir"])
    candidates = sorted((job_dir / "OUTPUT").glob("*_av_3d_N02*.nc")) + sorted((job_dir / "OUTPUT").glob("*_av_3d*.nc"))
    return str(candidates[0]) if candidates else None


# Internal helper for crop emission.
def _crop_emission(job_files: dict[str, Any], record: dict[str, Any], patch_h: int, patch_w: int) -> np.ndarray:
    y0 = int(record["y0"])
    x0 = int(record["x0"])
    idx = int(record.get("chemistry_nearest_index", record["output_time_index"]))
    with Dataset(job_files["chemistry"], "r") as ds:
        var = ds.variables["emission_values"]
        idx = min(idx, var.shape[0] - 1)
        arr = _array(var[idx])
    arr = np.squeeze(arr)
    if arr.ndim == 3:
        arr = arr[0]
    src_h, src_w = arr.shape[-2:]
    sy0 = int(round(y0 * src_h / 800.0))
    sx0 = int(round(x0 * src_w / 800.0))
    sy1 = int(round((y0 + patch_h) * src_h / 800.0))
    sx1 = int(round((x0 + patch_w) * src_w / 800.0))
    sy1 = max(sy0 + 1, min(src_h, sy1))
    sx1 = max(sx0 + 1, min(src_w, sx1))
    return _resize_nearest_2d(arr[sy0:sy1, sx0:sx1], patch_h, patch_w)


# Internal helper for read target and bg.
def _read_target_and_bg(
    job_files: dict[str, Any],
    record: dict[str, Any],
    *,
    patch_h: int,
    patch_w: int,
    target_source: str,
    target_mode: str,
) -> tuple[np.ndarray, np.ndarray, tuple[int, int, int]]:
    z0 = int(record["z0"])
    y0 = int(record["y0"])
    x0 = int(record["x0"])
    dz = int(record["dz"])
    t_idx = int(record["output_time_index"])
    target_path = _locate_av3d(job_files) if target_source == "av3d" else None
    if target_path is None:
        target_path = job_files["out3d"]

    with Dataset(target_path, "r") as ds:
        t_idx_use = min(t_idx, len(ds.dimensions["time"]) - 1)
        target = _array(ds.variables["kc_CO2"][t_idx_use, z0 : z0 + dz, y0 : y0 + patch_h, x0 : x0 + patch_w])
        zu = _array(ds.variables["zu_3d"][:])
        target_zu = zu[z0 : z0 + dz]
        nz = len(ds.dimensions["zu_3d"])
        ny = len(ds.dimensions["y"])
        nx = len(ds.dimensions["x"])

    with Dataset(job_files["dynamic"], "r") as dyn:
        dyn_seconds = values_to_seconds(_array(dyn.variables["time"][:]), getattr(dyn.variables["time"], "units", "seconds"))
        out_seconds = float(record.get("output_seconds", 0.0))
        match = linear_time_match(out_seconds, dyn_seconds)
        wr = float(record.get("dynamic_weight_right", match.weight_right))
        left = int(record.get("dynamic_left_index", match.left_index))
        right = int(record.get("dynamic_right_index", match.right_index))
        bg_left = _array(dyn.variables["ls_forcing_right_CO2"][left])
        bg_right = _array(dyn.variables["ls_forcing_right_CO2"][right])
        bg_values = (1.0 - wr) * bg_left + wr * bg_right
        dyn_z = _array(dyn.variables["z"][:])
    bg_profile = _interpolate_profile_to(bg_values, dyn_z, target_zu).astype(np.float32)
    bg_3d = bg_profile[:, None, None]

    if target_mode == "bg_residual":
        y = target[None].astype(np.float32) - bg_3d[None].astype(np.float32)
    elif target_mode == "absolute":
        y = target[None].astype(np.float32)
    else:
        raise ValueError(f"Unknown target mode: {target_mode}")
    return y, bg_profile, (nz, ny, nx)


# Internal helper for restrict mask to global layers.
def _restrict_mask_to_global_layers(mask: np.ndarray, z0: int, layer_min: int | None, layer_max: int | None) -> np.ndarray:
    if layer_min is None and layer_max is None:
        return mask
    depth = mask.shape[1]
    global_z = np.arange(depth, dtype=np.int32) + int(z0)
    keep = np.ones(depth, dtype=bool)
    if layer_min is not None:
        keep &= global_z >= int(layer_min)
    if layer_max is not None:
        keep &= global_z <= int(layer_max)
    return mask * keep[None, :, None, None].astype(mask.dtype)


# Internal helper for flush shard.
def _flush_shard(out_root: Path, split: str, shard_index: int, buffer: list[dict[str, Any]], dtype: np.dtype, compress: bool) -> dict[str, Any]:
    split_dir = out_root / split
    split_dir.mkdir(parents=True, exist_ok=True)
    filename = f"shard_{shard_index:05d}.npz"
    path = split_dir / filename
    arrays: dict[str, Any] = {}
    for field in ARRAY_FIELDS:
        if field == "mask":
            arrays[field] = np.stack([sample[field].astype(np.uint8) for sample in buffer], axis=0)
        elif field in {"z0", "y0", "x0", "nz", "ny", "nx"}:
            arrays[field] = np.asarray([sample[field] for sample in buffer], dtype=np.int32)
        else:
            arrays[field] = np.stack([np.asarray(sample[field], dtype=dtype) for sample in buffer], axis=0)
    arrays["sample_key"] = np.asarray([sample["sample_key"] for sample in buffer])
    arrays["metadata_json"] = np.asarray([json.dumps(sample["metadata"], sort_keys=True) for sample in buffer])
    if compress:
        np.savez_compressed(path, **arrays)
    else:
        np.savez(path, **arrays)
    return {"path": str(path.relative_to(out_root)), "count": len(buffer)}


# Entry point for the command-line workflow.
def main() -> None:
    parser = argparse.ArgumentParser(description="Build Stage 2 CO2 cache using frozen Stage 1 V1 microclimate predictions.")
    parser.add_argument("--stage1-cache-root", type=Path, default=PROJECT_ROOT / "generated" / "stage1_cache")
    parser.add_argument("--stage1-checkpoint", type=Path, default=PROJECT_ROOT / "checkpoints" / "stage1_local_fno_best_model.pt")
    parser.add_argument("--manifest", type=Path, default=PROJECT_ROOT / "generated" / "stage2_manifest" / "stage1_manifest.json")
    parser.add_argument("--out-root", type=Path, default=PROJECT_ROOT / "generated" / "stage2_cache")
    parser.add_argument("--splits", nargs="+", default=["train", "val"])
    parser.add_argument("--train-samples", type=int, default=1024)
    parser.add_argument("--val-samples", type=int, default=256)
    parser.add_argument("--dev-samples", type=int)
    parser.add_argument("--selection", choices=["first", "even", "random"], default="random")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--patch-h", type=int, default=256)
    parser.add_argument("--patch-w", type=int, default=256)
    parser.add_argument("--shard-size", type=int, default=4)
    parser.add_argument("--dtype", choices=["float16", "float32"], default="float16")
    parser.add_argument("--compress", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--target-mode", choices=["bg_residual", "absolute"], default="bg_residual")
    parser.add_argument("--target-source", choices=["av3d", "out3d"], default="av3d")
    parser.add_argument("--w-strategy", choices=["predicted", "zero", "forcing_profile", "truth"], default="predicted")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--stats-layer-min", type=int)
    parser.add_argument("--stats-layer-max", type=int)
    args = parser.parse_args()

    _safe_prepare_out_root(args.out_root, args.overwrite)
    dtype = np.dtype(args.dtype)
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else args.device)
    if args.device == "auto" and not torch.cuda.is_available():
        device = torch.device("cpu")

    stage1_manifest = load_cache_manifest(args.stage1_cache_root)
    stage1_norm = stage1_manifest["normalization"]
    model, target_stats = _load_stage1_model(args.stage1_checkpoint, device)
    print(f"Using Stage 1 checkpoint: {args.stage1_checkpoint}", flush=True)
    print(f"Using device: {device}", flush=True)
    print(f"Stage 2 target={args.target_mode} target_source={args.target_source} w_strategy={args.w_strategy}", flush=True)

    split_counts = {"train": args.train_samples, "val": args.val_samples, "dev": args.dev_samples}
    split_manifests: dict[str, dict[str, Any]] = {}
    x_stats = ChannelStats()
    target_stats_acc = ChannelStats()
    preview_rows: list[dict[str, Any]] = []

    for split in args.splits:
        records, jobs = load_manifest_records(args.manifest, split=split)
        selected = _select_records(records, split_counts.get(split), args.selection, args.seed + len(split))
        print(f"Building {split}: selected={len(selected)} from records={len(records)}", flush=True)
        shards: list[dict[str, Any]] = []
        buffer: list[dict[str, Any]] = []
        for idx, record in enumerate(selected, start=1):
            stage1_sample = read_stage1_sample(record, jobs[record["job"]], patch_h=args.patch_h, patch_w=args.patch_w)
            met_pred = _stage1_predict(
                model=model,
                target_stats=target_stats,
                stage1_norm=stage1_norm,
                sample=stage1_sample,
                device=device,
                w_strategy=args.w_strategy,
            )
            target, bg_profile, full_shape = _read_target_and_bg(
                jobs[record["job"]],
                record,
                patch_h=args.patch_h,
                patch_w=args.patch_w,
                target_source=args.target_source,
                target_mode=args.target_mode,
            )
            emission = _crop_emission(jobs[record["job"]], record, args.patch_h, args.patch_w)
            mask = stage1_sample.mask * np.isfinite(target).astype(np.float32)
            target = np.nan_to_num(target, nan=0.0, posinf=0.0, neginf=0.0)
            target = np.where(mask > 0, target, 0.0).astype(np.float32, copy=False)

            sample = {
                "sample_key": stage1_sample.sample_key,
                "met_pred": met_pred,
                "emission_2d": emission,
                "bg_profile": bg_profile,
                "surface_2d": stage1_sample.surface_2d,
                "scalar": stage1_sample.scalar,
                "target": target,
                "mask": mask,
                "z0": int(record["z0"]),
                "y0": int(record["y0"]),
                "x0": int(record["x0"]),
                "nz": int(full_shape[0]),
                "ny": int(full_shape[1]),
                "nx": int(full_shape[2]),
                "metadata": {
                    "job": record["job"],
                    "month": int(record["month"]),
                    "time_index": int(record["output_time_index"]),
                    "z0": int(record["z0"]),
                    "y0": int(record["y0"]),
                    "x0": int(record["x0"]),
                    "w_strategy": args.w_strategy,
                    "target_mode": args.target_mode,
                    "target_source": args.target_source,
                },
            }
            if split == "train":
                stats_mask = _restrict_mask_to_global_layers(
                    mask,
                    int(record["z0"]),
                    args.stats_layer_min,
                    args.stats_layer_max,
                )
                x_arr = assemble_stage2_input(sample, BASE_CHANNELS)
                x_stats.update(x_arr)
                target_stats_acc.update(target, stats_mask)
            buffer.append(sample)
            if len(preview_rows) < 300:
                preview_rows.append(
                    {
                        "split": split,
                        "sample_key": sample["sample_key"],
                        "job": record["job"],
                        "month": int(record["month"]),
                        "time_index": int(record["output_time_index"]),
                        "z0": int(record["z0"]),
                        "y0": int(record["y0"]),
                        "x0": int(record["x0"]),
                        "valid_fraction": float(mask.mean()),
                    }
                )
            if len(buffer) >= args.shard_size:
                shards.append(_flush_shard(args.out_root, split, len(shards), buffer, dtype, args.compress))
                buffer = []
            if idx == 1 or idx == len(selected) or idx % 25 == 0:
                print(f"cache {split}: {idx}/{len(selected)}", flush=True)
        if buffer:
            shards.append(_flush_shard(args.out_root, split, len(shards), buffer, dtype, args.compress))
        split_manifests[split] = {"count": len(selected), "shards": shards}

    x_mean, x_std = x_stats.finalize()
    y_mean, y_std = target_stats_acc.finalize()
    manifest = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_stage1_manifest": str(args.manifest),
        "stage1_cache_root": str(args.stage1_cache_root),
        "stage1_checkpoint": str(args.stage1_checkpoint),
        "target_mode": args.target_mode,
        "target_source": args.target_source,
        "w_strategy": args.w_strategy,
        "channels": list(BASE_CHANNELS),
        "patch_shape": [16, args.patch_h, args.patch_w],
        "normalization_layers": {"layer_min": args.stats_layer_min, "layer_max": args.stats_layer_max},
        "dtype": args.dtype,
        "compressed": bool(args.compress),
        "splits": split_manifests,
        "normalization": {
            "x_mean": x_mean.tolist(),
            "x_std": x_std.tolist(),
            "target_mean": y_mean.tolist(),
            "target_std": y_std.tolist(),
        },
    }
    (args.out_root / "cache_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    preview_path = args.out_root / "metadata_preview.csv"
    if preview_rows:
        with preview_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(preview_rows[0].keys()))
            writer.writeheader()
            writer.writerows(preview_rows)
    print(f"Saved Stage 2 cache manifest: {args.out_root / 'cache_manifest.json'}", flush=True)


if __name__ == "__main__":
    main()

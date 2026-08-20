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
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_ROOT))

from two_stage_surrogate.data.stage1_direct_reader import load_manifest_records, read_stage1_sample  # noqa: E402


ARRAY_FIELDS = (
    "geometry_3d",
    "surface_2d",
    "profile",
    "scalar",
    "theta_reference",
    "target_uv",
    "target_w",
    "target_theta_prime",
    "target_theta",
    "mask",
)
INPUT_STAT_FIELDS = ("geometry_3d", "surface_2d", "profile", "scalar")
TARGET_STAT_FIELDS = ("target_uv", "target_w", "target_theta_prime")


# Accumulates channel-wise mean and variance statistics.
class ChannelStats:
    # Store constructor arguments and initialize object state.
    def __init__(self) -> None:
        self.sum: np.ndarray | None = None
        self.sumsq: np.ndarray | None = None
        self.count: np.ndarray | None = None

    # Update running metric or statistic accumulators.
    def update(self, arr: np.ndarray, mask: np.ndarray | None = None) -> None:
        arr = np.asarray(arr, dtype=np.float64)
        channels = arr.shape[0]
        flat = arr.reshape(channels, -1)
        if mask is None:
            valid = np.ones_like(flat, dtype=bool)
        else:
            mask_flat = np.asarray(mask, dtype=bool).reshape(1, -1)
            valid = np.broadcast_to(mask_flat, flat.shape)
        values = np.where(valid, flat, 0.0)
        counts = valid.sum(axis=1).astype(np.float64)
        sums = values.sum(axis=1)
        sumsqs = (values * values).sum(axis=1)
        if self.sum is None:
            self.sum = np.zeros(channels, dtype=np.float64)
            self.sumsq = np.zeros(channels, dtype=np.float64)
            self.count = np.zeros(channels, dtype=np.float64)
        self.sum += sums
        self.sumsq += sumsqs
        self.count += counts

    # Return the final accumulated statistics.
    def finalize(self, shape: tuple[int, ...]) -> tuple[np.ndarray, np.ndarray]:
        if self.sum is None or self.sumsq is None or self.count is None:
            raise RuntimeError("Cannot finalize empty stats")
        count = np.maximum(self.count, 1.0)
        mean = self.sum / count
        var = np.maximum(self.sumsq / count - mean * mean, 1e-12)
        std = np.sqrt(var)
        return mean.astype(np.float32).reshape(shape), std.astype(np.float32).reshape(shape)


# Internal helper for safe prepare out root.
def _safe_prepare_out_root(out_root: Path, overwrite: bool) -> None:
    out_root = out_root.resolve()
    project = PROJECT_ROOT.resolve()
    if not str(out_root).startswith(str(project)):
        raise ValueError(f"Refusing to write Stage 1 cache outside project root: {out_root}")
    if out_root.exists():
        if not overwrite:
            raise FileExistsError(f"{out_root} already exists. Use --overwrite or choose another --out-root.")
        if "stage1_cache" not in out_root.name:
            raise ValueError(f"Refusing to overwrite a path that does not look like a Stage 1 cache: {out_root}")
        shutil.rmtree(out_root)
    out_root.mkdir(parents=True, exist_ok=True)


# Internal helper for select records.
def _select_records(records: list[dict[str, Any]], count: int | None, selection: str, seed: int) -> list[dict[str, Any]]:
    if count is None or count >= len(records):
        return list(records)
    if selection == "first":
        return records[:count]
    if selection == "even":
        positions = np.linspace(0, len(records) - 1, count).round().astype(int)
        return [records[int(pos)] for pos in positions]
    rng = np.random.default_rng(seed)
    indices = np.sort(rng.choice(len(records), size=count, replace=False))
    return [records[int(idx)] for idx in indices]


# Internal helper for cast for cache.
def _cast_for_cache(arr: np.ndarray, dtype: np.dtype) -> np.ndarray:
    return np.asarray(arr, dtype=dtype)


# Internal helper for flush shard.
def _flush_shard(
    *,
    out_root: Path,
    split: str,
    shard_index: int,
    buffer: list[dict[str, Any]],
    dtype: np.dtype,
    compress: bool,
) -> dict[str, Any]:
    split_dir = out_root / split
    split_dir.mkdir(parents=True, exist_ok=True)
    filename = f"shard_{shard_index:05d}.npz"
    path = split_dir / filename

    arrays: dict[str, Any] = {}
    for field in ARRAY_FIELDS:
        if field == "mask":
            arrays[field] = np.stack([sample[field].astype(np.uint8) for sample in buffer], axis=0)
        else:
            arrays[field] = np.stack([_cast_for_cache(sample[field], dtype) for sample in buffer], axis=0)
    arrays["sample_key"] = np.asarray([sample["sample_key"] for sample in buffer])
    arrays["metadata_json"] = np.asarray([json.dumps(sample["metadata"], sort_keys=True) for sample in buffer])

    if compress:
        np.savez_compressed(path, **arrays)
    else:
        np.savez(path, **arrays)
    return {"path": str(path.relative_to(out_root)), "count": len(buffer)}


# Internal helper for sample to dict.
def _sample_to_dict(sample: Any) -> dict[str, Any]:
    return {
        "sample_key": sample.sample_key,
        "geometry_3d": sample.geometry_3d,
        "surface_2d": sample.surface_2d,
        "profile": sample.profile,
        "scalar": sample.scalar,
        "theta_reference": sample.theta_reference,
        "target_uv": sample.target_uv,
        "target_w": sample.target_w,
        "target_theta_prime": sample.target_theta_prime,
        "target_theta": sample.target_theta,
        "mask": sample.mask,
        "metadata": sample.metadata,
    }


# Internal helper for write preview.
def _write_preview(out_root: Path, split: str, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    preview_path = out_root / "metadata" / f"{split}_metadata_preview.csv"
    preview_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["sample_key", "job", "month", "time_index", "z0", "y0", "x0", "valid_fraction"]
    with preview_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows[:200]:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


# Entry point for the command-line workflow.
def main() -> None:
    parser = argparse.ArgumentParser(description="Build a sharded Stage 1 microclimate cache.")
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs" / "stage1_fno_camden.yaml")
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--out-root", type=Path, default=PROJECT_ROOT / "processed" / "stage1_cache_camden")
    parser.add_argument("--splits", nargs="+", default=["train", "val"])
    parser.add_argument("--train-samples", type=int)
    parser.add_argument("--val-samples", type=int)
    parser.add_argument("--dev-samples", type=int)
    parser.add_argument("--selection", choices=["first", "even", "random"], default="random")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--patch-h", type=int)
    parser.add_argument("--patch-w", type=int)
    parser.add_argument("--shard-size", type=int, default=8)
    parser.add_argument("--dtype", choices=["float16", "float32"], default="float16")
    parser.add_argument("--compress", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    manifest_path = args.manifest or Path(config["data"]["manifest"])
    patch_h = args.patch_h or int(config["data"]["target_grid"]["y"])
    patch_w = args.patch_w or int(config["data"]["target_grid"]["x"])
    dtype = np.dtype(args.dtype)
    _safe_prepare_out_root(args.out_root, args.overwrite)

    stats = {field: ChannelStats() for field in INPUT_STAT_FIELDS + TARGET_STAT_FIELDS}
    split_manifests: dict[str, dict[str, Any]] = {}
    preview_rows: dict[str, list[dict[str, Any]]] = {}
    sample_counts = {"train": args.train_samples, "val": args.val_samples, "dev": args.dev_samples}

    for split in args.splits:
        records, jobs = load_manifest_records(manifest_path, split=split)
        selected = _select_records(records, sample_counts.get(split), args.selection, args.seed + len(split))
        print(f"Building {split} cache: selected={len(selected)} from records={len(records)}", flush=True)
        shards: list[dict[str, Any]] = []
        buffer: list[dict[str, Any]] = []
        preview_rows[split] = []
        for idx, record in enumerate(selected, start=1):
            sample = read_stage1_sample(record, jobs[record["job"]], patch_h=patch_h, patch_w=patch_w)
            sample_dict = _sample_to_dict(sample)
            if split == "train":
                for field in INPUT_STAT_FIELDS:
                    stats[field].update(sample_dict[field])
                mask = sample_dict["mask"]
                for field in TARGET_STAT_FIELDS:
                    stats[field].update(sample_dict[field], mask=mask)
            buffer.append(sample_dict)
            preview_rows[split].append(
                {
                    "sample_key": sample.sample_key,
                    "job": sample.metadata["job"],
                    "month": sample.metadata["month"],
                    "time_index": sample.metadata["time_index"],
                    "z0": sample.metadata["z0"],
                    "y0": sample.metadata["y0"],
                    "x0": sample.metadata["x0"],
                    "valid_fraction": sample.metadata["valid_fraction"],
                }
            )
            if len(buffer) >= args.shard_size:
                shards.append(
                    _flush_shard(
                        out_root=args.out_root,
                        split=split,
                        shard_index=len(shards),
                        buffer=buffer,
                        dtype=dtype,
                        compress=args.compress,
                    )
                )
                buffer = []
            if idx == 1 or idx == len(selected) or idx % 50 == 0:
                print(f"cache {split}: {idx}/{len(selected)} samples", flush=True)
        if buffer:
            shards.append(
                _flush_shard(
                    out_root=args.out_root,
                    split=split,
                    shard_index=len(shards),
                    buffer=buffer,
                    dtype=dtype,
                    compress=args.compress,
                )
            )
        split_manifests[split] = {"count": len(selected), "shards": shards}
        _write_preview(args.out_root, split, preview_rows[split])

    norm: dict[str, Any] = {}
    shape_map = {
        "geometry_3d": (-1, 1, 1, 1),
        "surface_2d": (-1, 1, 1),
        "profile": (-1, 1),
        "scalar": (-1,),
        "target_uv": (-1, 1, 1, 1),
        "target_w": (-1, 1, 1, 1),
        "target_theta_prime": (-1, 1, 1, 1),
    }
    for field, accumulator in stats.items():
        mean, std = accumulator.finalize(shape_map[field])
        norm[f"{field}_mean"] = mean.tolist()
        norm[f"{field}_std"] = std.tolist()

    cache_manifest = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_manifest": str(manifest_path),
        "config": str(args.config),
        "patch_h": patch_h,
        "patch_w": patch_w,
        "dtype": args.dtype,
        "compressed": bool(args.compress),
        "splits": split_manifests,
        "channels": config.get("channels", {}),
        "normalization": norm,
    }
    (args.out_root / "cache_manifest.json").write_text(json.dumps(cache_manifest, indent=2), encoding="utf-8")
    print(f"Saved Stage 1 cache manifest: {args.out_root / 'cache_manifest.json'}", flush=True)


if __name__ == "__main__":
    main()

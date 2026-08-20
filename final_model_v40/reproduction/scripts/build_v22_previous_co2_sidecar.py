#!/usr/bin/env python3
from __future__ import annotations

import argparse
import bisect
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np

os.environ.setdefault("PALM_KEEP_DATASETS_OPEN", "1")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from scripts.build_v13_sidecar import RunningStats, block_mean_2d  # noqa: E402
from scripts.train_3d_unet import (  # noqa: E402
    TARGET_VAR,
    align_3d_to_shape,
    locate_job_files,
    safe_open_dataset,
    to_numpy,
)


# Load cache manifest from disk or cache.
def load_cache_manifest(cache_root: Path, split: str) -> dict:
    manifest_path = cache_root / split / "manifest.json"
    with open(manifest_path, "r", encoding="utf-8") as f:
        return json.load(f)


# Load cache sample from disk or cache.
def load_cache_sample(cache_root: Path, split: str, manifest: dict, index: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    cumulative: list[int] = []
    total = 0
    for shard in manifest["shards"]:
        total += int(shard["count"])
        cumulative.append(total)
    shard_idx = bisect.bisect_right(cumulative, index)
    prev = 0 if shard_idx == 0 else cumulative[shard_idx - 1]
    local_idx = index - prev
    shard = manifest["shards"][shard_idx]
    split_dir = cache_root / split
    x = np.load(split_dir / shard["x"], mmap_mode="r")[local_idx]
    y = np.load(split_dir / shard["y"], mmap_mode="r")[local_idx]
    m = np.load(split_dir / shard["mask"], mmap_mode="r")[local_idx]
    return x, y, m


# Extract a previous-CO2 local patch.
def co2_patch(dsav, time_index: int, z0: int, z1: int, y0: int, y1: int, x0: int, x1: int) -> np.ndarray:
    target = dsav[TARGET_VAR].isel(time=int(time_index))
    z_name = "zu_3d" if "zu_3d" in target.dims else target.dims[0]
    patch = target.isel({z_name: slice(z0, z1), "y": slice(y0, y1), "x": slice(x0, x1)})
    arr = np.nan_to_num(to_numpy(patch), nan=0.0).astype(np.float32, copy=False)
    return align_3d_to_shape(arr, (z1 - z0, y1 - y0, x1 - x0))


# Build a low-resolution previous-CO2 context field.
def co2_global_lowres(dsav, time_index: int, z0: int, z1: int, ny: int, nx: int, out_size: int) -> np.ndarray:
    target = dsav[TARGET_VAR].isel(time=int(time_index))
    z_name = "zu_3d" if "zu_3d" in target.dims else target.dims[0]
    field = target.isel({z_name: slice(z0, z1), "y": slice(0, ny), "x": slice(0, nx)})
    arr = np.nan_to_num(to_numpy(field), nan=0.0).astype(np.float32, copy=False)
    arr = align_3d_to_shape(arr, (z1 - z0, ny, nx))
    return block_mean_2d(arr, out_size).astype(np.float32, copy=False)


# Save shard to disk.
def save_shard(
    split_dir: Path,
    shard_idx: int,
    prev_local: list[np.ndarray],
    prev_global: list[np.ndarray],
    has_prev: list[int],
) -> dict:
    prefix = f"shard_{shard_idx:05d}"
    local_arr = np.stack(prev_local, axis=0).astype(np.float16)
    global_arr = np.stack(prev_global, axis=0).astype(np.float16)
    valid_arr = np.asarray(has_prev, dtype=np.uint8)
    local_name = f"{prefix}_prev_co2.npy"
    global_name = f"{prefix}_prev_global_co2.npy"
    valid_name = f"{prefix}_has_prev.npy"
    np.save(split_dir / local_name, local_arr)
    np.save(split_dir / global_name, global_arr)
    np.save(split_dir / valid_name, valid_arr)
    return {
        "index": shard_idx,
        "count": int(local_arr.shape[0]),
        "prev_co2": local_name,
        "prev_global_co2": global_name,
        "has_prev": valid_name,
        "prev_co2_shape": list(local_arr.shape),
        "prev_global_co2_shape": list(global_arr.shape),
        "dtype": {
            "prev_co2": "float16",
            "prev_global_co2": "float16",
            "has_prev": "uint8",
        },
    }


# Generate one train or validation profile-cache split.
def build_split(args: argparse.Namespace, split: str, train_stats: bool) -> tuple[dict, RunningStats | None, RunningStats | None]:
    cache_root = Path(args.cache_root)
    meta_root = Path(args.metadata_root)
    out_dir = Path(args.out_dir)
    split_dir = out_dir / split
    split_dir.mkdir(parents=True, exist_ok=True)
    for old in split_dir.glob("shard_*.npy"):
        old.unlink()
    for old_name in ("manifest.json", "preview.csv"):
        old_path = split_dir / old_name
        if old_path.exists():
            old_path.unlink()

    cache_manifest = load_cache_manifest(cache_root, split)
    meta = np.load(meta_root / f"{split}_metadata.npz", allow_pickle=False)
    coord_manifest = json.load(open(meta_root / "metadata_manifest.json", "r", encoding="utf-8"))
    job_names = coord_manifest["splits"][0 if split == "train" else 1]["jobs"]
    total = min(int(cache_manifest["total_patches"]), len(meta["z0"]))
    if args.max_samples_per_split > 0:
        total = min(total, int(args.max_samples_per_split))
    patch_d, patch_h, patch_w = [int(v) for v in cache_manifest["patch_size"]]
    print(
        f"Building V22 previous-CO2 sidecar split={split}: samples={total} "
        f"patch=({patch_d},{patch_h},{patch_w}) global_size={args.global_size} shard_size={args.shard_size}",
        flush=True,
    )

    local_stats = RunningStats(1) if train_stats else None
    global_stats = RunningStats(1) if train_stats else None
    global_cache: dict[tuple[str, int, int], np.ndarray] = {}
    shards: list[dict] = []
    local_batch: list[np.ndarray] = []
    global_batch: list[np.ndarray] = []
    valid_batch: list[int] = []
    preview_rows: list[dict[str, object]] = []
    shard_idx = 0
    valid_total = 0

    for index in range(total):
        job = job_names[int(meta["job_idx"][index])]
        jf = locate_job_files(args.jobs_root, job)
        if jf is None:
            raise RuntimeError(f"Cannot locate job files for {job}")
        t_idx = int(meta["time_index"][index])
        z0 = int(meta["z0"][index])
        y0 = int(meta["y0"][index])
        x0 = int(meta["x0"][index])
        ny = int(meta["ny"][index])
        nx = int(meta["nx"][index])
        z1 = z0 + patch_d
        y1 = y0 + patch_h
        x1 = x0 + patch_w
        has_prev = 1 if t_idx > 0 else 0

        if has_prev:
            prev_t = t_idx - 1
            with safe_open_dataset(jf.av3d) as dsav:
                prev_patch = co2_patch(dsav, prev_t, z0, z1, y0, y1, x0, x1)
                global_key = (job, prev_t, z0)
                prev_global = global_cache.get(global_key)
                if prev_global is None:
                    prev_global = co2_global_lowres(dsav, prev_t, z0, z1, ny, nx, args.global_size)
                    global_cache[global_key] = prev_global
        else:
            prev_patch = np.zeros((patch_d, patch_h, patch_w), dtype=np.float32)
            prev_global = np.zeros((patch_d, args.global_size, args.global_size), dtype=np.float32)

        local_batch.append(prev_patch)
        global_batch.append(prev_global)
        valid_batch.append(has_prev)
        valid_total += int(has_prev)

        if train_stats and has_prev:
            assert local_stats is not None and global_stats is not None
            local_stats.update(prev_patch[None, :, :: args.stats_stride, :: args.stats_stride])
            global_stats.update(prev_global[None])

        if len(preview_rows) < args.preview_rows:
            current_mean = float("nan")
            delta_mae = float("nan")
            if has_prev:
                _, y_cache, m_cache = load_cache_sample(cache_root, split, cache_manifest, index)
                current = np.asarray(y_cache[0], dtype=np.float32)
                mask = np.asarray(m_cache[0], dtype=np.float32) > 0
                current_mean = float(np.nanmean(current[mask])) if np.any(mask) else float(np.nanmean(current))
                delta_mae = float(np.nanmean(np.abs((current - prev_patch)[mask]))) if np.any(mask) else float(np.nanmean(np.abs(current - prev_patch)))
            preview_rows.append(
                {
                    "index": index,
                    "job": job,
                    "time_index": t_idx,
                    "prev_time_index": t_idx - 1 if has_prev else "",
                    "has_prev": has_prev,
                    "z0": z0,
                    "y0": y0,
                    "x0": x0,
                    "prev_mean": float(np.nanmean(prev_patch)) if has_prev else "",
                    "current_mean": current_mean,
                    "persistence_delta_mae": delta_mae,
                }
            )

        if len(local_batch) >= args.shard_size or index + 1 == total:
            shards.append(save_shard(split_dir, shard_idx, local_batch, global_batch, valid_batch))
            local_batch, global_batch, valid_batch = [], [], []
            shard_idx += 1

        if (index + 1) % max(1, args.progress_every) == 0 or index + 1 == total:
            print(f"{split}: {index + 1}/{total} samples valid_prev={valid_total}", flush=True)

    if preview_rows:
        with open(split_dir / "preview.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(preview_rows[0].keys()))
            writer.writeheader()
            writer.writerows(preview_rows)

    manifest = {
        "split": split,
        "total": total,
        "valid_prev_total": valid_total,
        "patch_size": [patch_d, patch_h, patch_w],
        "global_size": args.global_size,
        "local_prev_channel": "prev_kc_CO2",
        "global_prev_channel": "prev_kc_CO2_global",
        "shards": shards,
        "source_cache_root": os.path.abspath(args.cache_root),
        "metadata_root": os.path.abspath(args.metadata_root),
    }
    with open(split_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    return manifest, local_stats, global_stats


# Entry point for the command-line workflow.
def main() -> None:
    parser = argparse.ArgumentParser(description="Build V22 previous-timestep CO2 sidecar for autoregressive residual training")
    parser.add_argument("--jobs-root", required=True)
    parser.add_argument("--cache-root", required=True)
    parser.add_argument("--metadata-root", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--splits", default="train,val")
    parser.add_argument("--global-size", type=int, default=80)
    parser.add_argument("--shard-size", type=int, default=8)
    parser.add_argument("--stats-stride", type=int, default=16)
    parser.add_argument("--preview-rows", type=int, default=50)
    parser.add_argument("--progress-every", type=int, default=50)
    parser.add_argument("--max-samples-per-split", type=int, default=0, help="Debug only: limit samples per split")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for old_name in ("manifest.json", "normalization.json"):
        old_path = out_dir / old_name
        if old_path.exists():
            old_path.unlink()

    manifests = []
    local_stats = None
    global_stats = None
    for split in [s.strip() for s in args.splits.split(",") if s.strip()]:
        manifest, l_stats, g_stats = build_split(args, split, train_stats=(split == "train"))
        manifests.append(manifest)
        if l_stats is not None:
            local_stats = l_stats
        if g_stats is not None:
            global_stats = g_stats

    if local_stats is not None and global_stats is not None:
        local_mean, local_std = local_stats.finalize()
        global_mean, global_std = global_stats.finalize()
        norm = {
            "local_prev_channel": "prev_kc_CO2",
            "local_prev_mean": float(local_mean[0]),
            "local_prev_std": float(local_std[0]),
            "global_prev_channel": "prev_kc_CO2_global",
            "global_prev_mean": float(global_mean[0]),
            "global_prev_std": float(global_std[0]),
        }
        with open(out_dir / "normalization.json", "w", encoding="utf-8") as f:
            json.dump(norm, f, indent=2)

    root_manifest = {
        "version": "v22_previous_co2_sidecar",
        "jobs_root": os.path.abspath(args.jobs_root),
        "cache_root": os.path.abspath(args.cache_root),
        "metadata_root": os.path.abspath(args.metadata_root),
        "global_size": args.global_size,
        "splits": manifests,
    }
    with open(out_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(root_manifest, f, indent=2)
    print(json.dumps(root_manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()

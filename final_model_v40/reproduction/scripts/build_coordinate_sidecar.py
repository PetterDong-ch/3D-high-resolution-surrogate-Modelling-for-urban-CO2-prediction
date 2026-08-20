#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import sys
from pathlib import Path

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from scripts.train_3d_unet import TARGET_VAR, discover_jobs, locate_job_files, safe_open_dataset


# Split available time indices into train/validation groups.
def split_time_indices(n_t: int, split: str, train_fraction: float, val_fraction: float) -> list[int]:
    indices = list(range(n_t))
    n_train = max(1, int(round(n_t * train_fraction)))
    n_val = max(1, int(round(n_t * val_fraction)))
    if n_train + n_val >= n_t:
        n_train = max(1, n_t - 2)
        n_val = 1 if n_t - n_train > 1 else 0
    if split == "train":
        return indices[:n_train]
    if split == "val":
        return indices[n_train : n_train + n_val]
    if split == "test":
        return indices[n_train + n_val :]
    raise ValueError(f"Unknown split: {split}")


# Choose a valid patch start index.
def choose_start(rng: random.Random, n: int, patch: int) -> int:
    if n <= patch:
        return 0
    return rng.randrange(0, n - patch + 1)


# Choose a valid vertical patch start index.
def choose_z_start(rng: random.Random, n: int, patch: int, z_min_start: int | None, z_max_start: int | None) -> int:
    if n <= patch:
        return 0
    max_allowed = n - patch
    lo = 0 if z_min_start is None else max(0, int(z_min_start))
    hi = max_allowed if z_max_start is None else min(max_allowed, int(z_max_start))
    if hi < lo:
        raise RuntimeError(
            f"Invalid z start range [{lo}, {hi}] for nz={n}, patch_d={patch}. "
            "Check cache manifest z_min_start/z_max_start."
        )
    return rng.randrange(lo, hi + 1)


# Load cache manifest from disk or cache.
def load_cache_manifest(cache_root: Path, split: str) -> dict:
    manifest_path = cache_root / split / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing cache manifest: {manifest_path}")
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    return manifest


# Count total cached patches from the manifest.
def cache_total_patches(manifest: dict) -> int:
    if "shards" in manifest:
        return int(sum(int(shard["count"]) for shard in manifest["shards"]))
    return int(manifest["total_patches"])


# Collect file paths and metadata for available jobs.
def collect_job_info(jobs_root: str, split: str, train_fraction: float, val_fraction: float) -> tuple[list, dict[str, dict]]:
    jobs = discover_jobs(jobs_root)
    files = []
    info: dict[str, dict] = {}
    for name in jobs:
        jf = locate_job_files(jobs_root, name)
        if jf is None:
            continue
        with safe_open_dataset(jf.out3d) as ds3d, safe_open_dataset(jf.av3d) as dsav:
            n_t = min(int(ds3d.sizes.get("time", 0)), int(dsav.sizes.get("time", 0)))
            times = split_time_indices(max(1, n_t), split, train_fraction, val_fraction)
            if not times:
                continue
            target = dsav[TARGET_VAR].isel(time=times[0])
            z_name = "zu_3d" if "zu_3d" in target.dims else target.dims[0]
            info[jf.name] = {
                "times": times,
                "nz": int(target.sizes[z_name]),
                "ny": int(target.sizes["y"]),
                "nx": int(target.sizes["x"]),
            }
        files.append(jf)
    if not files:
        raise RuntimeError(f"No valid job files found for split={split}")
    return files, info


# Build split sidecar for the workflow.
def build_split_sidecar(
    jobs_root: str,
    cache_root: Path,
    out_dir: Path,
    split: str,
    train_fraction: float,
    val_fraction: float,
) -> dict:
    manifest = load_cache_manifest(cache_root, split)
    total = cache_total_patches(manifest)
    patch_d, patch_h, patch_w = [int(x) for x in manifest["patch_size"]]
    seed = int(manifest["seed"])
    z_min_start = manifest.get("z_min_start", None)
    z_max_start = manifest.get("z_max_start", None)
    rng = random.Random(seed)

    all_job_names = discover_jobs(jobs_root)
    samples_per_job = math.ceil(total / len(all_job_names))
    files, info = collect_job_info(jobs_root, split, train_fraction, val_fraction)

    entries: list[tuple[object, int]] = []
    for jf in files:
        times = info[jf.name]["times"]
        for _ in range(samples_per_job):
            entries.append((jf, rng.choice(times)))
    if len(entries) < total:
        raise RuntimeError(f"Replayed only {len(entries)} entries but cache contains {total} patches")

    job_names = [jf.name for jf in files]
    job_to_idx = {name: idx for idx, name in enumerate(job_names)}
    arrays = {
        "job_idx": np.empty(total, dtype=np.int16),
        "month": np.empty(total, dtype=np.int16),
        "time_index": np.empty(total, dtype=np.int16),
        "z0": np.empty(total, dtype=np.int16),
        "y0": np.empty(total, dtype=np.int16),
        "x0": np.empty(total, dtype=np.int16),
        "nz": np.empty(total, dtype=np.int16),
        "ny": np.empty(total, dtype=np.int16),
        "nx": np.empty(total, dtype=np.int16),
    }

    for i in range(total):
        jf, t_idx = entries[i]
        dims = info[jf.name]
        nz = int(dims["nz"])
        ny = int(dims["ny"])
        nx = int(dims["nx"])
        arrays["job_idx"][i] = job_to_idx[jf.name]
        arrays["month"][i] = int(jf.month)
        arrays["time_index"][i] = int(t_idx)
        arrays["z0"][i] = choose_z_start(rng, nz, patch_d, z_min_start, z_max_start)
        arrays["y0"][i] = choose_start(rng, ny, patch_h)
        arrays["x0"][i] = choose_start(rng, nx, patch_w)
        arrays["nz"][i] = nz
        arrays["ny"][i] = ny
        arrays["nx"][i] = nx

    out_dir.mkdir(parents=True, exist_ok=True)
    npz_path = out_dir / f"{split}_metadata.npz"
    np.savez_compressed(npz_path, **arrays)

    preview_path = out_dir / f"{split}_metadata_preview.csv"
    with open(preview_path, "w", newline="", encoding="utf-8") as f:
        fieldnames = ["index", "job", "month", "time_index", "z0", "y0", "x0", "nz", "ny", "nx"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for i in range(min(50, total)):
            writer.writerow(
                {
                    "index": i,
                    "job": job_names[int(arrays["job_idx"][i])],
                    "month": int(arrays["month"][i]),
                    "time_index": int(arrays["time_index"][i]),
                    "z0": int(arrays["z0"][i]),
                    "y0": int(arrays["y0"][i]),
                    "x0": int(arrays["x0"][i]),
                    "nz": int(arrays["nz"][i]),
                    "ny": int(arrays["ny"][i]),
                    "nx": int(arrays["nx"][i]),
                }
            )

    return {
        "split": split,
        "metadata": str(npz_path),
        "preview": str(preview_path),
        "total_patches": total,
        "seed": seed,
        "patch_size": [patch_d, patch_h, patch_w],
        "z_min_start": z_min_start,
        "z_max_start": z_max_start,
        "samples_per_job": samples_per_job,
        "jobs": job_names,
    }


# Entry point for the command-line workflow.
def main() -> None:
    parser = argparse.ArgumentParser(description="Build small coordinate metadata sidecars for an existing V7/V8 patch cache")
    parser.add_argument("--jobs-root", required=True)
    parser.add_argument("--cache-root", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--train-fraction", type=float, default=0.70)
    parser.add_argument("--val-fraction", type=float, default=0.15)
    args = parser.parse_args()

    cache_root = Path(args.cache_root)
    out_dir = Path(args.out_dir)
    results = [
        build_split_sidecar(
            jobs_root=args.jobs_root,
            cache_root=cache_root,
            out_dir=out_dir,
            split="train",
            train_fraction=args.train_fraction,
            val_fraction=args.val_fraction,
        ),
        build_split_sidecar(
            jobs_root=args.jobs_root,
            cache_root=cache_root,
            out_dir=out_dir,
            split="val",
            train_fraction=args.train_fraction,
            val_fraction=args.val_fraction,
        ),
    ]

    manifest = {
        "cache_root": str(cache_root.resolve()),
        "jobs_root": os.path.abspath(args.jobs_root),
        "train_fraction": args.train_fraction,
        "val_fraction": args.val_fraction,
        "coordinate_channels": ["x_norm", "y_norm", "z_norm"],
        "splits": results,
    }
    manifest_path = out_dir / "metadata_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(json.dumps(manifest, indent=2), flush=True)
    print(f"Saved coordinate sidecar manifest: {manifest_path}", flush=True)


if __name__ == "__main__":
    main()

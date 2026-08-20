#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from netCDF4 import Dataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
import sys

sys.path.insert(0, str(PROJECT_ROOT / "src"))
from stage2_v40_cache import Stage2ShardReader, finite_difference_3d_np  # noqa: E402


# Accumulates running mean and variance statistics.
class RunningStats:
    # Store constructor arguments and initialize object state.
    def __init__(self, channels: int) -> None:
        self.sum = np.zeros(channels, dtype=np.float64)
        self.sumsq = np.zeros(channels, dtype=np.float64)
        self.count = np.zeros(channels, dtype=np.float64)

    # Update running metric or statistic accumulators.
    def update(self, arr: np.ndarray, mask: np.ndarray | None = None) -> None:
        values = np.asarray(arr, dtype=np.float64).reshape(arr.shape[0], -1)
        if mask is None:
            valid = np.isfinite(values)
        else:
            valid_mask = np.asarray(mask, dtype=bool).reshape(1, -1)
            valid = np.broadcast_to(valid_mask, values.shape) & np.isfinite(values)
        clean = np.where(valid, values, 0.0)
        self.sum += clean.sum(axis=1)
        self.sumsq += (clean * clean).sum(axis=1)
        self.count += valid.sum(axis=1)

    # Return the final accumulated statistics.
    def finalize(self) -> tuple[np.ndarray, np.ndarray]:
        count = np.maximum(self.count, 1.0)
        mean = self.sum / count
        var = np.maximum(self.sumsq / count - mean * mean, 1.0e-12)
        return mean.astype(np.float32), np.sqrt(var).astype(np.float32)


# Prepare a safe output directory.
def safe_out_root(path: Path, overwrite: bool) -> None:
    path = path.resolve()
    project = PROJECT_ROOT.resolve()
    if not str(path).startswith(str(project)):
        raise ValueError(f"Refusing to write outside project root: {path}")
    if path.exists():
        if not overwrite:
            raise FileExistsError(f"{path} exists; pass --overwrite or choose another output root")
        if "prev" not in path.name and "sidecar" not in path.name:
            raise ValueError(f"Refusing to overwrite non-sidecar path: {path}")
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


# Locate the CO2 NetCDF file for one job.
def locate_co2_file(jobs_root: Path, job: str, target_source: str) -> Path:
    output = jobs_root / job / "OUTPUT"
    patterns = (
        ("*_av_3d_N02*.nc", "*_av_3d*.nc", "*_3d_N02*.nc", "*_3d*.nc")
        if target_source == "av3d"
        else ("*_3d_N02*.nc", "*_3d*.nc", "*_av_3d_N02*.nc", "*_av_3d*.nc")
    )
    for pattern in patterns:
        matches = sorted(output.glob(pattern))
        if matches:
            return matches[0]
    raise FileNotFoundError(f"No CO2 file found for {job} in {output}")


# Convert input data to a float NumPy array.
def as_float_array(value: Any) -> np.ndarray:
    arr = np.asanyarray(value)
    if np.ma.isMaskedArray(arr):
        arr = arr.astype(np.float32).filled(np.nan)
    return np.asarray(arr, dtype=np.float32)


# Read prev co2 from disk.
def read_prev_co2(ds: Dataset, t_idx: int, z0: int, y0: int, x0: int, depth: int, height: int, width: int) -> np.ndarray:
    if t_idx <= 0:
        return np.zeros((depth, height, width), dtype=np.float32)
    var = ds.variables["kc_CO2"]
    prev_t = min(t_idx - 1, var.shape[0] - 1)
    arr = as_float_array(var[prev_t, z0 : z0 + depth, y0 : y0 + height, x0 : x0 + width])
    if arr.shape != (depth, height, width):
        out = np.zeros((depth, height, width), dtype=np.float32)
        dz = min(depth, arr.shape[-3])
        dy = min(height, arr.shape[-2])
        dx = min(width, arr.shape[-1])
        out[:dz, :dy, :dx] = arr[:dz, :dy, :dx]
        arr = out
    return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)


# Write split to disk.
def write_split(
    *,
    cache_root: Path,
    jobs_root: Path,
    out_root: Path,
    split: str,
    shard_size: int,
    stats: RunningStats | None,
    dx: float,
    dy: float,
    dz: float,
    progress_every: int,
    max_samples: int,
) -> dict[str, Any]:
    reader = Stage2ShardReader(cache_root, split)
    split_dir = out_root / split
    split_dir.mkdir(parents=True, exist_ok=True)
    total = len(reader) if max_samples <= 0 else min(len(reader), max_samples)
    print(f"Building prev-CO2 sidecar split={split}: samples={total} shard_size={shard_size}", flush=True)

    datasets: dict[tuple[str, str], Dataset] = {}
    shards: list[dict[str, Any]] = []
    prev_buffer: list[np.ndarray] = []
    valid_buffer: list[int] = []
    key_buffer: list[str] = []
    valid_total = 0

    def flush(shard_idx: int) -> None:
        if not prev_buffer:
            return
        path = split_dir / f"shard_{shard_idx:05d}.npz"
        np.savez_compressed(
            path,
            prev_co2=np.stack(prev_buffer, axis=0).astype(np.float16),
            has_prev=np.asarray(valid_buffer, dtype=np.uint8),
            sample_key=np.asarray(key_buffer),
        )
        shards.append(
            {
                "path": str(path.relative_to(out_root)),
                "count": len(prev_buffer),
                "prev_co2_shape": list(np.stack(prev_buffer, axis=0).shape),
                "dtype": {"prev_co2": "float16", "has_prev": "uint8"},
            }
        )
        prev_buffer.clear()
        valid_buffer.clear()
        key_buffer.clear()

    shard_idx = 0
    for index in range(total):
        sample = reader.sample(index)
        metadata = json.loads(str(sample["metadata_json"]))
        job = str(metadata["job"])
        target_source = str(metadata.get("target_source", "av3d"))
        t_idx = int(metadata["time_index"])
        z0 = int(metadata["z0"])
        y0 = int(metadata["y0"])
        x0 = int(metadata["x0"])
        depth = int(np.asarray(sample["target"]).shape[1])
        height = int(np.asarray(sample["target"]).shape[2])
        width = int(np.asarray(sample["target"]).shape[3])
        key = (job, target_source)
        ds = datasets.get(key)
        if ds is None:
            ds = Dataset(locate_co2_file(jobs_root, job, target_source), "r")
            datasets[key] = ds
        has_prev = 1 if t_idx > 0 else 0
        prev = read_prev_co2(ds, t_idx, z0, y0, x0, depth, height, width) if has_prev else np.zeros((depth, height, width), dtype=np.float32)

        prev_buffer.append(prev)
        valid_buffer.append(has_prev)
        key_buffer.append(str(sample["sample_key"]))
        valid_total += int(has_prev)
        if stats is not None and has_prev:
            dc_dx = finite_difference_3d_np(prev, dx, axis=2)
            dc_dy = finite_difference_3d_np(prev, dy, axis=1)
            dc_dz = finite_difference_3d_np(prev, dz, axis=0)
            mask = np.asarray(sample["mask"], dtype=np.float32)[0] > 0
            stats.update(np.stack((prev, dc_dx, dc_dy, dc_dz), axis=0), mask=mask)

        if len(prev_buffer) >= shard_size:
            flush(shard_idx)
            shard_idx += 1
        if (index + 1) % max(1, progress_every) == 0 or index + 1 == total:
            print(f"{split}: {index + 1}/{total} valid_prev={valid_total}", flush=True)

    flush(shard_idx)
    for ds in datasets.values():
        ds.close()
    return {
        "count": total,
        "valid_prev_total": valid_total,
        "source_cache_root": str(cache_root),
        "shards": shards,
    }


# Entry point for the command-line workflow.
def main() -> None:
    parser = argparse.ArgumentParser(description="Build previous-CO2 sidecar aligned to Stage2 Stage1-met cache.")
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--jobs-root", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--splits", default="train,val")
    parser.add_argument("--shard-size", type=int, default=8)
    parser.add_argument("--dx", type=float, default=10.0)
    parser.add_argument("--dy", type=float, default=10.0)
    parser.add_argument("--dz", type=float, default=10.0)
    parser.add_argument("--progress-every", type=int, default=100)
    parser.add_argument("--max-samples-per-split", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    safe_out_root(args.out_root, args.overwrite)
    split_names = [s.strip() for s in args.splits.split(",") if s.strip()]
    split_manifests: dict[str, Any] = {}
    train_stats = RunningStats(4)
    for split in split_names:
        split_manifests[split] = write_split(
            cache_root=args.cache_root,
            jobs_root=args.jobs_root,
            out_root=args.out_root,
            split=split,
            shard_size=args.shard_size,
            stats=train_stats if split == "train" else None,
            dx=args.dx,
            dy=args.dy,
            dz=args.dz,
            progress_every=args.progress_every,
            max_samples=args.max_samples_per_split,
        )

    mean, std = train_stats.finalize()
    norm = {
        "channels": ["prev_kc_CO2", "prev_dCdx", "prev_dCdy", "prev_dCdz"],
        "prev_kc_CO2_mean": float(mean[0]),
        "prev_kc_CO2_std": float(std[0]),
        "prev_dCdx_mean": float(mean[1]),
        "prev_dCdx_std": float(std[1]),
        "prev_dCdy_mean": float(mean[2]),
        "prev_dCdy_std": float(std[2]),
        "prev_dCdz_mean": float(mean[3]),
        "prev_dCdz_std": float(std[3]),
        "spacing": {"dx": args.dx, "dy": args.dy, "dz": args.dz},
    }
    (args.out_root / "normalization.json").write_text(json.dumps(norm, indent=2), encoding="utf-8")
    manifest = {
        "schema_version": "stage2_v40_prev_co2_sidecar_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "cache_root": str(args.cache_root),
        "jobs_root": str(args.jobs_root),
        "splits": split_manifests,
    }
    (args.out_root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()

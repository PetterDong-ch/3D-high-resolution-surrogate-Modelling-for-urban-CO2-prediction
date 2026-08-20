#!/usr/bin/env python3
from __future__ import annotations

import argparse
import bisect
import csv
import json
import math
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

os.environ.setdefault("PALM_KEEP_DATASETS_OPEN", "1")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from scripts.train_3d_unet import (  # noqa: E402
    STATIC_VARS,
    TARGET_VAR,
    align_3d_to_shape,
    discover_jobs,
    interp_w_patch_from_zw_to_zu,
    locate_job_files,
    safe_open_dataset,
    to_numpy,
)


LOCAL_BASE_CHANNELS = [
    "emission_values",
    "ls_forcing_right_CO2",
    "u",
    "v",
    "p",
    "theta",
    "w",
    *STATIC_VARS,
    "month_sin",
    "month_cos",
    "tod_sin",
    "tod_cos",
    "x_norm",
    "y_norm",
    "z_norm",
    "fluid_mask",
]

GLOBAL_CHANNELS = [
    "emission_values",
    "ls_forcing_right_CO2",
    "u",
    "v",
    "w",
    "p",
    "theta",
    "fluid_mask",
]


# Accumulates running mean and variance statistics.
class RunningStats:
    # Store constructor arguments and initialize object state.
    def __init__(self, n_channels: int) -> None:
        self.count = np.zeros(n_channels, dtype=np.float64)
        self.sum = np.zeros(n_channels, dtype=np.float64)
        self.sumsq = np.zeros(n_channels, dtype=np.float64)

    # Update running metric or statistic accumulators.
    def update(self, arr: np.ndarray) -> None:
        # arr shape: (C, ...)
        flat = arr.reshape(arr.shape[0], -1).astype(np.float64, copy=False)
        finite = np.isfinite(flat)
        values = np.where(finite, flat, 0.0)
        self.count += finite.sum(axis=1)
        self.sum += values.sum(axis=1)
        self.sumsq += (values * values).sum(axis=1)

    # Return the final accumulated statistics.
    def finalize(self) -> tuple[np.ndarray, np.ndarray]:
        count = np.maximum(self.count, 1.0)
        mean = self.sum / count
        var = np.maximum(self.sumsq / count - mean * mean, 1.0e-12)
        std = np.sqrt(var)
        std = np.where(std < 1.0e-6, 1.0, std)
        return mean.astype(np.float32), std.astype(np.float32)


# Parse timestamp bytes from PALM metadata.
def parse_timestamp_bytes(value: object) -> datetime | None:
    try:
        if isinstance(value, bytes):
            text = value.decode("utf-8")
        else:
            text = str(value)
        text = text.replace("+0000", "+00:00")
        return datetime.fromisoformat(text)
    except Exception:
        return None


# Read chemistry timestamps in seconds.
def chemistry_seconds(dsch) -> np.ndarray:
    if "timestamp" in dsch:
        timestamps = np.asarray(dsch["timestamp"].values)
        parsed = [parse_timestamp_bytes(v) for v in timestamps]
        if parsed and parsed[0] is not None and all(v is not None for v in parsed):
            origin = parsed[0]
            return np.asarray([(v - origin).total_seconds() for v in parsed], dtype=np.float64)
    if "time" in dsch:
        try:
            values = np.asarray(dsch["time"].values, dtype=np.float64)
            if values.size > 0:
                values = values - values[0]
                if float(np.nanmax(np.abs(values))) <= 10000.0:
                    values = values * 3600.0
                return values.astype(np.float64)
        except Exception:
            pass
    n = int(dsch.sizes.get("time", 0))
    return np.arange(n, dtype=np.float64) * 3600.0


# Load emission 2d from disk or cache.
def load_emission_2d(dsch, index: int) -> np.ndarray:
    index = int(np.clip(index, 0, int(dsch.sizes.get("time", 1)) - 1))
    emis = dsch["emission_values"].isel(time=index)
    for dim in ("nspecies", "z"):
        if dim in emis.dims and emis.sizes[dim] > 1:
            emis = emis.isel({dim: 0})
        elif dim in emis.dims:
            emis = emis.squeeze(dim)
    return np.nan_to_num(to_numpy(emis), nan=0.0).astype(np.float32, copy=False)


# Build the corrected 2D emission field.
def corrected_emission_2d(dsch, target_seconds: float) -> np.ndarray:
    seconds = chemistry_seconds(dsch)
    if len(seconds) <= 1:
        return load_emission_2d(dsch, 0)
    if target_seconds <= seconds[0]:
        return load_emission_2d(dsch, 0)
    if target_seconds >= seconds[-1]:
        return load_emission_2d(dsch, len(seconds) - 1)
    right = int(np.searchsorted(seconds, target_seconds, side="right"))
    left = right - 1
    denom = max(float(seconds[right] - seconds[left]), 1.0e-6)
    weight = float((target_seconds - seconds[left]) / denom)
    left_map = load_emission_2d(dsch, left)
    right_map = load_emission_2d(dsch, right)
    return ((1.0 - weight) * left_map + weight * right_map).astype(np.float32, copy=False)


# Build the corrected background CO2 profile.
def corrected_bg_profile(dsdyn, target_seconds: float, target_heights: np.ndarray) -> np.ndarray:
    bg = dsdyn["ls_forcing_right_CO2"]
    values = to_numpy(bg)
    if values.ndim == 1:
        profile = values
        z_name = "z" if "z" in bg.coords else bg.dims[0]
    else:
        if "time" not in bg.dims:
            raise RuntimeError(f"Unexpected background CO2 dims: {bg.dims}")
        time_values = np.asarray(dsdyn["time"].values, dtype=np.float64)
        time_axis = bg.dims.index("time")
        z_name = "z" if "z" in bg.dims else [d for d in bg.dims if d != "time"][0]
        z_axis = bg.dims.index(z_name)
        if time_axis != 0 or z_axis != 1:
            values = np.moveaxis(values, (time_axis, z_axis), (0, 1))
        profile = np.asarray(
            [np.interp(target_seconds, time_values, values[:, zi]) for zi in range(values.shape[1])],
            dtype=np.float32,
        )
    if z_name in bg.coords:
        bg_heights = np.asarray(bg.coords[z_name].values, dtype=np.float32)
    else:
        bg_heights = 25.0 + 50.0 * np.arange(profile.shape[0], dtype=np.float32)
    n_bg = min(20, len(profile), len(bg_heights))
    if n_bg <= 0:
        return np.zeros_like(target_heights, dtype=np.float32)
    return np.interp(
        target_heights.astype(np.float32),
        bg_heights[:n_bg],
        profile[:n_bg],
        left=profile[0],
        right=profile[n_bg - 1],
    ).astype(np.float32)


# Average a 2D field into coarse blocks.
def block_mean_2d(arr: np.ndarray, out_size: int) -> np.ndarray:
    h, w = arr.shape[-2:]
    if h % out_size == 0 and w % out_size == 0:
        bh = h // out_size
        bw = w // out_size
        reshaped = arr.reshape(*arr.shape[:-2], out_size, bh, out_size, bw)
        return np.nanmean(reshaped, axis=(-3, -1)).astype(np.float32)

    y_edges = np.linspace(0, h, out_size + 1).round().astype(np.int64)
    x_edges = np.linspace(0, w, out_size + 1).round().astype(np.int64)
    out_shape = (*arr.shape[:-2], out_size, out_size)
    out = np.empty(out_shape, dtype=np.float32)
    for yi in range(out_size):
        for xi in range(out_size):
            patch = arr[..., y_edges[yi] : y_edges[yi + 1], x_edges[xi] : x_edges[xi + 1]]
            out[..., yi, xi] = np.nanmean(patch, axis=(-2, -1))
    return out


# Build topography and fluid-mask context channels.
def topo_fluid_context(topo_values: np.ndarray | None, z0: int, d: int, out_size: int, ny: int, nx: int) -> np.ndarray:
    if topo_values is None or z0 >= topo_values.shape[0]:
        return np.ones((d, out_size, out_size), dtype=np.float32)
    fluid = np.ones((d, ny, nx), dtype=np.float32)
    z1 = min(z0 + d, topo_values.shape[0])
    y1 = min(ny, topo_values.shape[1])
    x1 = min(nx, topo_values.shape[2])
    if z1 > z0 and y1 > 0 and x1 > 0:
        fluid[: z1 - z0, :y1, :x1] = (topo_values[z0:z1, :y1, :x1] == 0).astype(np.float32)
    return block_mean_2d(fluid, out_size)


# Load a full-domain PALM variable.
def full_domain_variable(ds3d, var: str, t_idx: int, z0: int, z1: int, target_heights, target_shape: tuple[int, int, int]) -> np.ndarray:
    da = ds3d[var].isel(time=min(t_idx, int(ds3d.sizes["time"]) - 1))
    if var == "w" and "zw_3d" in da.dims:
        zw1 = min(int(da.sizes["zw_3d"]), z1 + 1)
        patch = da.isel({"zw_3d": slice(z0, zw1), "y": slice(0, target_shape[1]), "x": slice(0, target_shape[2])})
        return align_3d_to_shape(interp_w_patch_from_zw_to_zu(patch, target_heights, target_shape), target_shape)
    if "xu" in da.dims:
        da = da.rename({"xu": "x"})
    if "yv" in da.dims:
        da = da.rename({"yv": "y"})
    da = da.isel({"zu_3d": slice(z0, z1), "y": slice(0, target_shape[1]), "x": slice(0, target_shape[2])})
    return align_3d_to_shape(to_numpy(da), target_shape)


# Build global context for the workflow.
def build_global_context(
    corrected_emission: np.ndarray,
    corrected_bg: np.ndarray,
    ds3d,
    topo_values: np.ndarray | None,
    t_idx: int,
    z0: int,
    z1: int,
    target_heights,
    ny: int,
    nx: int,
    out_size: int,
) -> np.ndarray:
    d = z1 - z0
    target_shape = (d, ny, nx)
    emission_low = block_mean_2d(corrected_emission, out_size)
    emission_3d = np.broadcast_to(emission_low[None, :, :], (d, out_size, out_size)).astype(np.float32)
    bg_3d = np.broadcast_to(corrected_bg[:, None, None], (d, out_size, out_size)).astype(np.float32)
    fields = [emission_3d, bg_3d]
    for var in ("u", "v", "w", "p", "theta"):
        arr = full_domain_variable(ds3d, var, t_idx, z0, z1, target_heights, target_shape)
        fields.append(block_mean_2d(np.nan_to_num(arr, nan=0.0), out_size))
    fields.append(topo_fluid_context(topo_values, z0, d, out_size, ny, nx))
    return np.stack(fields, axis=0).astype(np.float32)


# Load cache sample from disk or cache.
def load_cache_sample(cache_root: Path, split: str, manifest: dict, index: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    cumulative = []
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


# Transform local channels before accumulating statistics.
def local_transformed_for_stats(
    x_cache: np.ndarray,
    mask: np.ndarray,
    corrected_emission_patch: np.ndarray,
    corrected_bg: np.ndarray,
    z0: int,
    y0: int,
    x0: int,
    nz: int,
    ny: int,
    nx: int,
    height_gate_decay_levels: float,
) -> np.ndarray:
    x = np.array(x_cache, dtype=np.float32, copy=True)
    d, h, w = x.shape[-3:]
    x[0] = np.broadcast_to(corrected_emission_patch[None, :, :], (d, h, w))
    x[1] = np.broadcast_to(corrected_bg[:, None, None], (d, h, w))
    z_values = ((np.arange(d, dtype=np.float32) + float(z0)) / float(max(nz - 1, 1))) * 2.0 - 1.0
    y_values = ((np.arange(h, dtype=np.float32) + float(y0)) / float(max(ny - 1, 1))) * 2.0 - 1.0
    x_values = ((np.arange(w, dtype=np.float32) + float(x0)) / float(max(nx - 1, 1))) * 2.0 - 1.0
    x_coord = np.broadcast_to(x_values[None, None, :], (d, h, w))
    y_coord = np.broadcast_to(y_values[None, :, None], (d, h, w))
    z_coord = np.broadcast_to(z_values[:, None, None], (d, h, w))
    extras = np.stack((x_coord, y_coord, z_coord, mask[0].astype(np.float32)), axis=0)
    return np.concatenate((x, extras), axis=0)


# Save shard to disk.
def save_shard(
    split_dir: Path,
    shard_idx: int,
    emissions: list[np.ndarray],
    bgs: list[np.ndarray],
    globals_: list[np.ndarray] | None,
) -> dict:
    prefix = f"shard_{shard_idx:05d}"
    emission = np.stack(emissions, axis=0).astype(np.float16)
    bg = np.stack(bgs, axis=0).astype(np.float32)
    emission_name = f"{prefix}_corrected_emission.npy"
    bg_name = f"{prefix}_corrected_bg.npy"
    np.save(split_dir / emission_name, emission)
    np.save(split_dir / bg_name, bg)

    shard = {
        "index": shard_idx,
        "count": int(emission.shape[0]),
        "corrected_emission": emission_name,
        "corrected_bg": bg_name,
        "corrected_emission_shape": list(emission.shape),
        "corrected_bg_shape": list(bg.shape),
        "dtype": {"corrected_emission": "float16", "corrected_bg": "float32"},
    }
    if globals_ is not None:
        global_context = np.stack(globals_, axis=0).astype(np.float16)
        global_name = f"{prefix}_global_context.npy"
        np.save(split_dir / global_name, global_context)
        shard["global_context"] = global_name
        shard["global_context_shape"] = list(global_context.shape)
        shard["dtype"]["global_context"] = "float16"
    return shard


# Generate one train or validation profile-cache split.
def build_split(args: argparse.Namespace, split: str, topo_values: np.ndarray | None, train_stats: bool) -> tuple[dict, RunningStats | None, RunningStats | None]:
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
    manifest_path = cache_root / split / "manifest.json"
    with open(manifest_path, "r", encoding="utf-8") as f:
        cache_manifest = json.load(f)
    meta = np.load(meta_root / f"{split}_metadata.npz", allow_pickle=False)
    coord_manifest = json.load(open(meta_root / "metadata_manifest.json", "r", encoding="utf-8"))
    job_names = coord_manifest["splits"][0 if split == "train" else 1]["jobs"]
    total = min(int(cache_manifest["total_patches"]), len(meta["z0"]))
    if args.max_samples_per_split > 0:
        total = min(total, int(args.max_samples_per_split))
    patch_d, patch_h, patch_w = [int(v) for v in cache_manifest["patch_size"]]
    print(
        f"Building V13 sidecar split={split}: samples={total} patch=({patch_d},{patch_h},{patch_w}) "
        f"global_size={args.global_size} shard_size={args.shard_size}",
        flush=True,
    )

    local_stats = RunningStats(len(LOCAL_BASE_CHANNELS)) if train_stats else None
    use_global_context = not args.no_global_context
    global_stats = RunningStats(len(GLOBAL_CHANNELS)) if (train_stats and use_global_context) else None
    shards: list[dict] = []
    emissions: list[np.ndarray] = []
    bgs: list[np.ndarray] = []
    globals_: list[np.ndarray] | None = [] if use_global_context else None
    preview_rows: list[dict[str, object]] = []
    shard_idx = 0

    for index in range(total):
        job = job_names[int(meta["job_idx"][index])]
        jf = locate_job_files(args.jobs_root, job)
        t_idx = int(meta["time_index"][index])
        z0 = int(meta["z0"][index])
        y0 = int(meta["y0"][index])
        x0 = int(meta["x0"][index])
        nz = int(meta["nz"][index])
        ny = int(meta["ny"][index])
        nx = int(meta["nx"][index])
        z1 = z0 + patch_d
        y1 = y0 + patch_h
        x1 = x0 + patch_w

        with safe_open_dataset(jf.chemistry) as dsch, safe_open_dataset(jf.dynamic) as dsdyn, safe_open_dataset(jf.out3d) as ds3d, safe_open_dataset(jf.av3d) as dsav:
            target = dsav[TARGET_VAR].isel(time=t_idx)
            z_name = "zu_3d" if "zu_3d" in target.dims else target.dims[0]
            target_z_coords = target.coords[z_name].isel({z_name: slice(z0, z1)}) if z_name in target.coords else None
            if target_z_coords is not None:
                target_heights = np.asarray(target_z_coords.values, dtype=np.float32)
            else:
                target_heights = np.arange(z0, z1, dtype=np.float32)
            target_seconds = float(target["time"].values) if "time" in target.coords else float(t_idx) * 1800.0
            emission_full = corrected_emission_2d(dsch, target_seconds)
            bg_profile = corrected_bg_profile(dsdyn, target_seconds, target_heights)
            if use_global_context:
                global_context = build_global_context(
                    corrected_emission=emission_full,
                    corrected_bg=bg_profile,
                    ds3d=ds3d,
                    topo_values=topo_values,
                    t_idx=t_idx,
                    z0=z0,
                    z1=z1,
                    target_heights=target_z_coords,
                    ny=ny,
                    nx=nx,
                    out_size=args.global_size,
                )
            else:
                global_context = None
        emission_patch = emission_full[y0:y1, x0:x1].astype(np.float32, copy=False)
        emissions.append(emission_patch)
        bgs.append(bg_profile)
        if globals_ is not None and global_context is not None:
            globals_.append(global_context)

        if train_stats:
            x_cache, _, mask = load_cache_sample(cache_root, split, cache_manifest, index)
            local_x = local_transformed_for_stats(
                x_cache=x_cache,
                mask=mask,
                corrected_emission_patch=emission_patch,
                corrected_bg=bg_profile,
                z0=z0,
                y0=y0,
                x0=x0,
                nz=nz,
                ny=ny,
                nx=nx,
                height_gate_decay_levels=args.height_gate_decay_levels,
            )
            local_stats.update(local_x[:, :, :: args.stats_stride, :: args.stats_stride])
            if global_stats is not None and global_context is not None:
                global_stats.update(global_context)

        if len(preview_rows) < args.preview_rows:
            preview_rows.append(
                {
                    "index": index,
                    "job": job,
                    "time_index": t_idx,
                    "target_seconds": target_seconds,
                    "z0": z0,
                    "y0": y0,
                    "x0": x0,
                    "emission_mean": float(np.nanmean(emission_patch)),
                    "bg_mean": float(np.nanmean(bg_profile)),
                }
            )

        if len(emissions) >= args.shard_size or index + 1 == total:
            shards.append(save_shard(split_dir, shard_idx, emissions, bgs, globals_))
            emissions, bgs = [], []
            globals_ = [] if use_global_context else None
            shard_idx += 1

        if (index + 1) % max(1, args.progress_every) == 0 or index + 1 == total:
            print(f"{split}: {index + 1}/{total} samples", flush=True)

    with open(split_dir / "preview.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(preview_rows[0].keys()))
        writer.writeheader()
        writer.writerows(preview_rows)

    manifest = {
        "split": split,
        "total": total,
        "patch_size": [patch_d, patch_h, patch_w],
        "global_size": args.global_size,
        "local_channels": LOCAL_BASE_CHANNELS,
        "global_channels": GLOBAL_CHANNELS if use_global_context else [],
        "shards": shards,
        "source_cache_root": os.path.abspath(args.cache_root),
        "metadata_root": os.path.abspath(args.metadata_root),
    }
    with open(split_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    return manifest, local_stats, global_stats


# Entry point for the command-line workflow.
def main() -> None:
    parser = argparse.ArgumentParser(description="Build V13 corrected-time sidecar for V7 256x256 patch cache")
    parser.add_argument("--jobs-root", required=True)
    parser.add_argument("--cache-root", required=True)
    parser.add_argument("--metadata-root", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--topography-path", default="/data/linfeng/palm/london_camden_2019_new/JOBS/cam07_175vm_topo_surf_N02.000.nc")
    parser.add_argument("--splits", default="train,val")
    parser.add_argument("--global-size", type=int, default=80)
    parser.add_argument("--no-global-context", action="store_true", help="Do not build full-domain low-resolution global context")
    parser.add_argument("--shard-size", type=int, default=8)
    parser.add_argument("--height-gate-decay-levels", type=float, default=40.0)
    parser.add_argument("--stats-stride", type=int, default=16)
    parser.add_argument("--preview-rows", type=int, default=50)
    parser.add_argument("--progress-every", type=int, default=100)
    parser.add_argument("--max-samples-per-split", type=int, default=0, help="Debug only: limit samples per split")
    args = parser.parse_args()

    with safe_open_dataset(args.topography_path) as dstopo:
        topo_values = to_numpy(dstopo["topo_all"]) if "topo_all" in dstopo else None

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
        manifest, l_stats, g_stats = build_split(args, split, topo_values, train_stats=(split == "train"))
        manifests.append(manifest)
        if l_stats is not None:
            local_stats = l_stats
        if g_stats is not None:
            global_stats = g_stats

    if local_stats is not None:
        local_mean, local_std = local_stats.finalize()
        if global_stats is not None:
            global_mean, global_std = global_stats.finalize()
            global_channels = GLOBAL_CHANNELS
        else:
            global_mean = np.asarray([], dtype=np.float32)
            global_std = np.asarray([], dtype=np.float32)
            global_channels = []
        norm = {
            "local_channels": LOCAL_BASE_CHANNELS,
            "local_mean": local_mean.tolist(),
            "local_std": local_std.tolist(),
            "global_channels": global_channels,
            "global_mean": global_mean.tolist(),
            "global_std": global_std.tolist(),
        }
        with open(out_dir / "normalization.json", "w", encoding="utf-8") as f:
            json.dump(norm, f, indent=2)
    root_manifest = {
        "version": "v13_corrected_time_sidecar",
        "jobs_root": os.path.abspath(args.jobs_root),
        "cache_root": os.path.abspath(args.cache_root),
        "metadata_root": os.path.abspath(args.metadata_root),
        "global_size": args.global_size,
        "global_context": not args.no_global_context,
        "splits": manifests,
    }
    with open(out_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(root_manifest, f, indent=2)
    print(json.dumps(root_manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()

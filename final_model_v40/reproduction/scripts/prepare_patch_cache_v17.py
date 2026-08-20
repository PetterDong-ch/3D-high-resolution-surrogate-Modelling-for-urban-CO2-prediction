#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import sys
from collections import defaultdict
from dataclasses import dataclass
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
    month_features,
    safe_open_dataset,
    time_of_day_features,
    to_numpy,
)


# Stores one candidate patch before balanced sampling.
@dataclass(frozen=True)
class Candidate:
    job_name: str
    job_idx: int
    month: int
    time_index: int
    time_bin: int
    z0: int
    y0: int
    x0: int
    nz: int
    ny: int
    nx: int
    score: float = 0.0
    amplitude_bin: int = 1


# Compute a stable modulo value for deterministic sampling.
def stable_mod(parts: tuple[object, ...], modulo: int) -> int:
    text = "|".join(str(p) for p in parts).encode("utf-8")
    digest = hashlib.blake2b(text, digest_size=8).digest()
    return int.from_bytes(digest, "little") % int(modulo)


# Parse a set of integer command-line values.
def parse_int_set(text: str) -> set[int]:
    values: set[int] = set()
    for item in str(text).split(","):
        item = item.strip()
        if item:
            values.add(int(item))
    return values


# Generate valid grid start positions for patch extraction.
def grid_starts(n: int, patch: int, stride: int) -> list[int]:
    if n <= patch:
        return [0]
    last = n - patch
    starts = list(range(0, last + 1, stride))
    if starts[-1] != last:
        starts.append(last)
    return starts


# Create time bins for balanced sampling.
def time_bins(n_t: int, bin_size: int) -> list[list[int]]:
    return [list(range(start, min(start + bin_size, n_t))) for start in range(0, n_t, bin_size)]


# Create an empty output directory.
def ensure_empty_or_create(path: Path, overwrite: bool) -> None:
    path.mkdir(parents=True, exist_ok=True)
    existing = list(path.glob("shard_*.npy"))
    manifest = path / "manifest.json"
    if (existing or manifest.exists()) and not overwrite:
        raise RuntimeError(f"Cache files already exist: {path}. Pass --overwrite to replace them.")
    if overwrite:
        for child in existing:
            child.unlink()
        if manifest.exists():
            manifest.unlink()


# Build topography mask for the workflow.
def build_topography_mask(
    dstopo,
    z0: int,
    z1: int,
    y0: int,
    y1: int,
    x0: int,
    x1: int,
    target_shape: tuple[int, int, int],
) -> np.ndarray:
    mask = np.ones(target_shape, dtype=np.float32)
    if "topo_all" not in dstopo:
        return mask
    topo = dstopo["topo_all"]
    z_dim = "z" if "z" in topo.dims else topo.dims[0]
    topo_z = int(topo.sizes.get(z_dim, 0))
    topo_y = int(topo.sizes.get("y", 0))
    topo_x = int(topo.sizes.get("x", 0))
    if topo_z <= z0 or topo_y <= y0 or topo_x <= x0:
        return mask
    z_read1 = min(z1, topo_z)
    y_read1 = min(y1, topo_y)
    x_read1 = min(x1, topo_x)
    topo_patch = topo.isel({z_dim: slice(z0, z_read1), "y": slice(y0, y_read1), "x": slice(x0, x_read1)})
    fluid = (to_numpy(topo_patch) == 0).astype(np.float32)
    mask[: fluid.shape[0], : fluid.shape[1], : fluid.shape[2]] = fluid
    return mask


# Extract the background CO2 patch for a sample.
def background_patch(dsdyn, target_patch, t3d: int, z0: int, z1: int, target_shape: tuple[int, int, int]) -> np.ndarray:
    z_name = "zu_3d" if "zu_3d" in target_patch.dims else target_patch.dims[0]
    bg_co2 = dsdyn["ls_forcing_right_CO2"]
    if "time" in bg_co2.dims:
        bg_co2 = bg_co2.isel(time=min(t3d, int(bg_co2.sizes.get("time", 1)) - 1))

    if "z" in bg_co2.dims:
        bg_z_name = "z"
    elif z_name in bg_co2.dims:
        bg_z_name = z_name
    else:
        bg_z_name = None

    if bg_z_name is not None and set(bg_co2.dims) == {bg_z_name}:
        bg_values = to_numpy(bg_co2)
        if bg_z_name in bg_co2.coords:
            bg_heights = np.asarray(bg_co2.coords[bg_z_name].values, dtype=np.float32)
        else:
            bg_heights = 25.0 + 50.0 * np.arange(bg_values.shape[0], dtype=np.float32)
        n_bg = min(20, bg_values.shape[0], bg_heights.shape[0])
        if n_bg == 0:
            bg_z_values = np.zeros(target_shape[0], dtype=np.float32)
        else:
            if z_name in target_patch.coords:
                target_heights = np.asarray(target_patch.coords[z_name].values, dtype=np.float32)
            else:
                target_heights = np.arange(z0, z1, dtype=np.float32)
            bg_z_values = np.interp(
                target_heights,
                bg_heights[:n_bg],
                bg_values[:n_bg],
                left=bg_values[0],
                right=bg_values[n_bg - 1],
            ).astype(np.float32)
        return np.broadcast_to(bg_z_values[:, None, None], target_shape).copy()

    if "z" in bg_co2.dims and z_name not in bg_co2.dims:
        bg_co2 = bg_co2.rename({"z": z_name})
    if z_name in bg_co2.dims:
        bg_co2 = bg_co2.isel({z_name: slice(z0, z1)})
    else:
        bg_co2 = bg_co2.expand_dims({z_name: target_patch.sizes[z_name]})
    if "y" in bg_co2.dims:
        bg_co2 = bg_co2.isel({"y": slice(0, target_shape[1])})
    else:
        bg_co2 = bg_co2.expand_dims({"y": target_shape[1]})
    if "x" in bg_co2.dims:
        bg_co2 = bg_co2.isel({"x": slice(0, target_shape[2])})
    else:
        bg_co2 = bg_co2.expand_dims({"x": target_shape[2]})
    return align_3d_to_shape(to_numpy(bg_co2.transpose(z_name, "y", "x")), target_shape)


# Load fixed patch from disk or cache.
def load_fixed_patch(
    jf,
    cand: Candidate,
    patch_size: tuple[int, int, int],
    topography_path: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    patch_d, patch_h, patch_w = patch_size
    z0, y0, x0 = cand.z0, cand.y0, cand.x0
    z1, y1, x1 = z0 + patch_d, y0 + patch_h, x0 + patch_w

    with safe_open_dataset(jf.chemistry) as dsch, safe_open_dataset(jf.dynamic) as dsdyn, safe_open_dataset(
        jf.static
    ) as dsst, safe_open_dataset(jf.out3d) as ds3d, safe_open_dataset(jf.av3d) as dsav, safe_open_dataset(
        topography_path
    ) as dstopo:
        t3d = min(cand.time_index, int(ds3d.sizes["time"]) - 1)
        tav = min(cand.time_index, int(dsav.sizes["time"]) - 1)
        target = dsav[TARGET_VAR].isel(time=tav)
        z_name = "zu_3d" if "zu_3d" in target.dims else target.dims[0]
        target_patch = target.isel({z_name: slice(z0, z1), "y": slice(y0, y1), "x": slice(x0, x1)})
        target_shape = (
            int(target_patch.sizes[z_name]),
            int(target_patch.sizes["y"]),
            int(target_patch.sizes["x"]),
        )

        dyn_arrays: list[np.ndarray] = []
        emis = dsch["emission_values"].isel(time=min(t3d, int(dsch.sizes.get("time", 1)) - 1))
        for dim in ("nspecies", "z"):
            if dim in emis.dims and emis.sizes[dim] > 1:
                emis = emis.isel({dim: 0})
            elif dim in emis.dims:
                emis = emis.squeeze(dim)
        emis = emis.isel({"y": slice(y0, y1), "x": slice(x0, x1)})
        emis3d = emis.expand_dims({z_name: target_patch.sizes[z_name]}).transpose(z_name, "y", "x")
        dyn_arrays.append(align_3d_to_shape(to_numpy(emis3d), target_shape))

        dyn_arrays.append(background_patch(dsdyn, target_patch, t3d, z0, z1, target_shape))

        for var in ("u", "v", "p", "theta"):
            da = ds3d[var].isel(time=t3d)
            if "xu" in da.dims:
                da = da.rename({"xu": "x"})
            if "yv" in da.dims:
                da = da.rename({"yv": "y"})
            da = da.isel({"zu_3d": slice(z0, z1), "y": slice(y0, y1), "x": slice(x0, x1)})
            dyn_arrays.append(align_3d_to_shape(to_numpy(da), target_shape))

        w_zw = ds3d["w"].isel(time=t3d)
        if "zw_3d" in w_zw.dims:
            zw1 = min(int(w_zw.sizes["zw_3d"]), z1 + 1)
            w_patch = w_zw.isel({"zw_3d": slice(z0, zw1), "y": slice(y0, y1), "x": slice(x0, x1)})
            w_np = interp_w_patch_from_zw_to_zu(w_patch, target_patch.coords.get(z_name), target_shape)
        else:
            w_patch = w_zw.isel({"zu_3d": slice(z0, z1), "y": slice(y0, y1), "x": slice(x0, x1)})
            w_np = to_numpy(w_patch)
        dyn_arrays.append(align_3d_to_shape(w_np, target_shape))

        static_arrays: list[np.ndarray] = []
        for var in STATIC_VARS:
            s2d = dsst[var].isel({"y": slice(y0, y1), "x": slice(x0, x1)})
            s2d_np = to_numpy(s2d)
            static_arrays.append(align_3d_to_shape(np.repeat(s2d_np[None, :, :], target_shape[0], axis=0), target_shape))

        msin, mcos = month_features(jf.month)
        month_sin = np.full_like(static_arrays[0], msin)
        month_cos = np.full_like(static_arrays[0], mcos)
        t_value = ds3d["time"].values[t3d] if "time" in ds3d else float(t3d)
        tsin, tcos = time_of_day_features(t_value)
        tod_sin = np.full_like(static_arrays[0], tsin)
        tod_cos = np.full_like(static_arrays[0], tcos)

        x = np.stack(dyn_arrays + static_arrays + [month_sin, month_cos, tod_sin, tod_cos], axis=0)
        y = to_numpy(target_patch)[None, ...]
        topo_mask = build_topography_mask(dstopo, z0, z1, y0, y1, x0, x1, target_shape)[None, ...]

    mask = np.isfinite(y).astype(np.float32) * topo_mask
    return np.nan_to_num(x, nan=0.0).astype(np.float32), np.nan_to_num(y, nan=0.0).astype(np.float32), mask.astype(np.float32)


# Score a candidate target patch for sampling.
def score_target_patch(jf, cand: Candidate, patch_size: tuple[int, int, int]) -> float:
    patch_d, patch_h, patch_w = patch_size
    z0, y0, x0 = cand.z0, cand.y0, cand.x0
    with safe_open_dataset(jf.av3d) as dsav:
        target = dsav[TARGET_VAR].isel(time=min(cand.time_index, int(dsav.sizes["time"]) - 1))
        z_name = "zu_3d" if "zu_3d" in target.dims else target.dims[0]
        patch = target.isel(
            {
                z_name: slice(z0, z0 + patch_d),
                "y": slice(y0, y0 + patch_h),
                "x": slice(x0, x0 + patch_w),
            }
        )
        arr = to_numpy(patch).astype(np.float32, copy=False)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return 0.0
    center = float(np.nanmedian(finite))
    return float(np.nanpercentile(np.abs(finite - center), 90.0))


# Collect file paths and metadata for available jobs.
def collect_job_info(jobs_root: str) -> tuple[list, dict[str, dict]]:
    jobs = discover_jobs(jobs_root)
    files = []
    info: dict[str, dict] = {}
    for job_idx, name in enumerate(jobs):
        jf = locate_job_files(jobs_root, name)
        if jf is None:
            continue
        with safe_open_dataset(jf.out3d) as ds3d, safe_open_dataset(jf.av3d) as dsav:
            n_t = min(int(ds3d.sizes.get("time", 0)), int(dsav.sizes.get("time", 0)))
            target = dsav[TARGET_VAR].isel(time=0)
            z_name = "zu_3d" if "zu_3d" in target.dims else target.dims[0]
            info[jf.name] = {
                "job_idx": job_idx,
                "month": int(jf.month),
                "n_t": int(n_t),
                "nz": int(target.sizes[z_name]),
                "ny": int(target.sizes["y"]),
                "nx": int(target.sizes["x"]),
            }
        files.append(jf)
    if not files:
        raise RuntimeError(f"No valid PALM jobs found under {jobs_root}")
    return files, info


# Sample candidate patches before balanced selection.
def sample_scoring_candidates(
    files: list,
    info: dict[str, dict],
    split: str,
    patch_size: tuple[int, int, int],
    xy_stride: int,
    time_bin_size: int,
    z_min_start: int,
    z_max_start: int,
    split_modulo: int,
    val_mod_value: int,
    candidates_per_stratum: int,
    seed: int,
) -> list[Candidate]:
    rng = random.Random(seed)
    patch_d, patch_h, patch_w = patch_size
    out: list[Candidate] = []
    for jf in files:
        meta = info[jf.name]
        y_starts = grid_starts(meta["ny"], patch_h, xy_stride)
        x_starts = grid_starts(meta["nx"], patch_w, xy_stride)
        z_hi = min(z_max_start, meta["nz"] - patch_d)
        z_lo = max(0, z_min_start)
        if z_hi < z_lo:
            continue
        for time_bin, times in enumerate(time_bins(meta["n_t"], time_bin_size)):
            for z0 in range(z_lo, z_hi + 1):
                candidates: list[Candidate] = []
                for t in times:
                    for y0 in y_starts:
                        for x0 in x_starts:
                            mod = stable_mod((jf.name, t, z0, y0, x0), split_modulo)
                            if split == "val" and mod != val_mod_value:
                                continue
                            if split == "train" and mod == val_mod_value:
                                continue
                            candidates.append(
                                Candidate(
                                    job_name=jf.name,
                                    job_idx=int(meta["job_idx"]),
                                    month=int(meta["month"]),
                                    time_index=int(t),
                                    time_bin=int(time_bin),
                                    z0=int(z0),
                                    y0=int(y0),
                                    x0=int(x0),
                                    nz=int(meta["nz"]),
                                    ny=int(meta["ny"]),
                                    nx=int(meta["nx"]),
                                )
                            )
                if len(candidates) <= candidates_per_stratum:
                    out.extend(candidates)
                else:
                    out.extend(rng.sample(candidates, candidates_per_stratum))
    rng.shuffle(out)
    return out


# Score candidate patches for balanced sampling.
def score_candidates(files: list, candidates: list[Candidate], patch_size: tuple[int, int, int], progress_every: int) -> list[Candidate]:
    by_name = {jf.name: jf for jf in files}
    scored: list[Candidate] = []
    for i, cand in enumerate(candidates, start=1):
        score = score_target_patch(by_name[cand.job_name], cand, patch_size)
        scored.append(
            Candidate(
                **{**cand.__dict__, "score": float(score), "amplitude_bin": 1}
            )
        )
        if i % max(1, progress_every) == 0 or i == len(candidates):
            print(f"scored candidates: {i}/{len(candidates)}", flush=True)
    return scored


# Assign candidates to amplitude bins.
def assign_amplitude_bins(scored: list[Candidate]) -> list[Candidate]:
    by_base: dict[tuple[int, int, int], list[Candidate]] = defaultdict(list)
    for cand in scored:
        by_base[(cand.month, cand.time_bin, cand.z0)].append(cand)
    out: list[Candidate] = []
    for group in by_base.values():
        scores = np.asarray([c.score for c in group], dtype=np.float32)
        if len(group) < 3 or float(scores.max() - scores.min()) <= 1.0e-6:
            cuts = (float("inf"), float("inf"))
        else:
            cuts = tuple(float(v) for v in np.quantile(scores, [1.0 / 3.0, 2.0 / 3.0]))
        for cand in group:
            amp_bin = 0 if cand.score <= cuts[0] else (1 if cand.score <= cuts[1] else 2)
            out.append(Candidate(**{**cand.__dict__, "amplitude_bin": amp_bin}))
    return out


# Select a balanced set of training patches.
def select_balanced(scored: list[Candidate], total: int, seed: int) -> list[Candidate]:
    rng = random.Random(seed)
    buckets: dict[tuple[int, int, int, int], list[Candidate]] = defaultdict(list)
    for cand in scored:
        buckets[(cand.month, cand.time_bin, cand.z0, cand.amplitude_bin)].append(cand)
    for bucket in buckets.values():
        rng.shuffle(bucket)
    keys = list(buckets)
    rng.shuffle(keys)
    selected: list[Candidate] = []
    while len(selected) < total and keys:
        next_keys: list[tuple[int, int, int, int]] = []
        for key in keys:
            bucket = buckets[key]
            if bucket:
                selected.append(bucket.pop())
                if len(selected) >= total:
                    break
            if bucket:
                next_keys.append(key)
        keys = next_keys
        rng.shuffle(keys)
    if len(selected) < total:
        raise RuntimeError(f"Only selected {len(selected)} candidates, requested {total}. Increase --score-candidates-per-stratum.")
    rng.shuffle(selected)
    return selected


# Save shard to disk.
def save_shard(split_dir: Path, shard_idx: int, xs: list[np.ndarray], ys: list[np.ndarray], masks: list[np.ndarray]) -> dict:
    x = np.stack(xs, axis=0).astype(np.float32, copy=False)
    y = np.stack(ys, axis=0).astype(np.float32, copy=False)
    mask = np.stack(masks, axis=0).astype(np.uint8, copy=False)
    prefix = f"shard_{shard_idx:05d}"
    x_name = f"{prefix}_x.npy"
    y_name = f"{prefix}_y.npy"
    mask_name = f"{prefix}_mask.npy"
    np.save(split_dir / x_name, x)
    np.save(split_dir / y_name, y)
    np.save(split_dir / mask_name, mask)
    return {
        "index": shard_idx,
        "count": int(x.shape[0]),
        "x": x_name,
        "y": y_name,
        "mask": mask_name,
        "x_shape": list(x.shape),
        "y_shape": list(y.shape),
        "mask_shape": list(mask.shape),
        "dtype": {"x": "float32", "y": "float32", "mask": "uint8"},
    }


# Write metadata to disk.
def write_metadata(
    metadata_root: Path,
    split: str,
    selected: list[Candidate],
    patch_size: tuple[int, int, int],
    jobs_by_idx: list[str],
) -> dict:
    metadata_root.mkdir(parents=True, exist_ok=True)
    arrays = {
        "job_idx": np.asarray([c.job_idx for c in selected], dtype=np.int16),
        "month": np.asarray([c.month for c in selected], dtype=np.int16),
        "time_index": np.asarray([c.time_index for c in selected], dtype=np.int16),
        "time_bin": np.asarray([c.time_bin for c in selected], dtype=np.int16),
        "z0": np.asarray([c.z0 for c in selected], dtype=np.int16),
        "y0": np.asarray([c.y0 for c in selected], dtype=np.int16),
        "x0": np.asarray([c.x0 for c in selected], dtype=np.int16),
        "nz": np.asarray([c.nz for c in selected], dtype=np.int16),
        "ny": np.asarray([c.ny for c in selected], dtype=np.int16),
        "nx": np.asarray([c.nx for c in selected], dtype=np.int16),
        "residual_amplitude_score": np.asarray([c.score for c in selected], dtype=np.float32),
        "amplitude_bin": np.asarray([c.amplitude_bin for c in selected], dtype=np.int16),
    }
    npz_path = metadata_root / f"{split}_metadata.npz"
    np.savez_compressed(npz_path, **arrays)
    preview_path = metadata_root / f"{split}_metadata_preview.csv"
    with open(preview_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "index",
                "job",
                "month",
                "time_index",
                "time_bin",
                "z0",
                "y0",
                "x0",
                "score",
                "amplitude_bin",
            ],
        )
        writer.writeheader()
        for i, cand in enumerate(selected[:100]):
            writer.writerow(
                {
                    "index": i,
                    "job": cand.job_name,
                    "month": cand.month,
                    "time_index": cand.time_index,
                    "time_bin": cand.time_bin,
                    "z0": cand.z0,
                    "y0": cand.y0,
                    "x0": cand.x0,
                    "score": f"{cand.score:.6g}",
                    "amplitude_bin": cand.amplitude_bin,
                }
            )
    jobs = list(jobs_by_idx)
    return {
        "split": split,
        "metadata": str(npz_path),
        "preview": str(preview_path),
        "total_patches": len(selected),
        "patch_size": list(patch_size),
        "jobs": jobs,
    }


# Generate one train or validation profile-cache split.
def build_split(args: argparse.Namespace, split: str, total_patches: int, seed: int, files: list, info: dict[str, dict]) -> dict:
    patch_size = (args.patch_d, args.patch_h, args.patch_w)
    split_dir = Path(args.out_dir) / split
    ensure_empty_or_create(split_dir, overwrite=args.overwrite)
    print(
        f"Building V17 {split} cache: target={total_patches} stride={args.xy_stride} "
        f"time_bin={args.time_bin_size} z_start=[{args.z_min_start},{args.z_max_start}]",
        flush=True,
    )
    scoring = sample_scoring_candidates(
        files=files,
        info=info,
        split=split,
        patch_size=patch_size,
        xy_stride=args.xy_stride,
        time_bin_size=args.time_bin_size,
        z_min_start=args.z_min_start,
        z_max_start=args.z_max_start,
        split_modulo=args.split_modulo,
        val_mod_value=args.val_mod_value,
        candidates_per_stratum=args.score_candidates_per_stratum,
        seed=seed,
    )
    print(f"{split}: scoring candidate count={len(scoring)}", flush=True)
    scored = assign_amplitude_bins(score_candidates(files, scoring, patch_size, args.progress_every))
    selected = select_balanced(scored, total_patches, seed + 19)
    max_job_idx = max(int(meta["job_idx"]) for meta in info.values())
    jobs_by_idx = [""] * (max_job_idx + 1)
    for name, meta in info.items():
        jobs_by_idx[int(meta["job_idx"])] = name
    meta_result = write_metadata(Path(args.metadata_out_dir), split, selected, patch_size, jobs_by_idx)

    by_name = {jf.name: jf for jf in files}
    shards: list[dict] = []
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    shard_idx = 0
    for i, cand in enumerate(selected):
        x, y, mask = load_fixed_patch(by_name[cand.job_name], cand, patch_size, args.topography_path)
        xs.append(x)
        ys.append(y)
        masks.append(mask)
        if len(xs) >= args.shard_size or i + 1 == len(selected):
            shards.append(save_shard(split_dir, shard_idx, xs, ys, masks))
            xs, ys, masks = [], [], []
            shard_idx += 1
        if (i + 1) % max(1, args.progress_every) == 0 or i + 1 == len(selected):
            print(f"cache {split}: {i + 1}/{len(selected)} patches", flush=True)

    manifest = {
        "split": split,
        "total_patches": len(selected),
        "patch_size": list(patch_size),
        "seed": seed,
        "shard_size": args.shard_size,
        "z_min_start": args.z_min_start,
        "z_max_start": args.z_max_start,
        "xy_stride": args.xy_stride,
        "time_bin_size": args.time_bin_size,
        "split_modulo": args.split_modulo,
        "val_mod_value": args.val_mod_value,
        "strategy": "v17_time_balanced_month_timebin_z0_residual_amplitude_stratified",
        "format": "npy_mmap_shards_v1",
        "metadata": meta_result,
        "shards": shards,
    }
    with open(split_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"Saved {split} manifest: {split_dir / 'manifest.json'}", flush=True)
    return manifest


# Entry point for the command-line workflow.
def main() -> None:
    parser = argparse.ArgumentParser(description="Build V17 time-balanced stratified 16x256x256 patch cache")
    parser.add_argument("--jobs-root", required=True)
    parser.add_argument("--out-dir", default=os.path.join(PROJECT_ROOT, "processed", "patch_cache_v17_time_balanced"))
    parser.add_argument("--metadata-out-dir", default=os.path.join(PROJECT_ROOT, "processed", "coordinate_sidecar_v17_time_balanced"))
    parser.add_argument("--topography-path", default="/data/linfeng/palm/london_camden_2019_new/JOBS/cam07_175vm_topo_surf_N02.000.nc")
    parser.add_argument("--train-patches", type=int, default=10240)
    parser.add_argument("--val-patches", type=int, default=1024)
    parser.add_argument("--shard-size", type=int, default=4)
    parser.add_argument("--patch-d", type=int, default=16)
    parser.add_argument("--patch-h", type=int, default=256)
    parser.add_argument("--patch-w", type=int, default=256)
    parser.add_argument("--z-min-start", type=int, default=0)
    parser.add_argument("--z-max-start", type=int, default=8)
    parser.add_argument("--xy-stride", type=int, default=64)
    parser.add_argument("--time-bin-size", type=int, default=8)
    parser.add_argument("--score-candidates-per-stratum", type=int, default=24)
    parser.add_argument("--split-modulo", type=int, default=10)
    parser.add_argument("--val-mod-value", type=int, default=0)
    parser.add_argument("--exclude-months", default="", help="Comma-separated months to exclude, e.g. 11,12")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--progress-every", type=int, default=50)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    meta_dir = Path(args.metadata_out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    meta_dir.mkdir(parents=True, exist_ok=True)
    files, info = collect_job_info(args.jobs_root)
    exclude_months = parse_int_set(args.exclude_months)
    if exclude_months:
        before = len(files)
        files = [jf for jf in files if int(jf.month) not in exclude_months]
        info = {name: meta for name, meta in info.items() if int(meta["month"]) not in exclude_months}
        print(f"Excluded months from cache candidates: {sorted(exclude_months)} jobs {before}->{len(files)}", flush=True)
    if not files:
        raise RuntimeError("No valid jobs remain after month filtering")
    train_manifest = build_split(args, "train", args.train_patches, args.seed, files, info)
    val_manifest = build_split(args, "val", args.val_patches, args.seed + 1, files, info)
    metadata_manifest = {
        "version": "v17_time_balanced_coordinate_metadata",
        "jobs_root": os.path.abspath(args.jobs_root),
        "cache_root": str(out_dir.resolve()),
        "coordinate_channels": ["x_norm", "y_norm", "z_norm"],
        "strategy": train_manifest["strategy"],
        "splits": [train_manifest["metadata"], val_manifest["metadata"]],
    }
    with open(meta_dir / "metadata_manifest.json", "w", encoding="utf-8") as f:
        json.dump(metadata_manifest, f, indent=2)
    root_manifest = {
        "version": "v17_time_balanced_patch_cache",
        "jobs_root": os.path.abspath(args.jobs_root),
        "metadata_root": str(meta_dir.resolve()),
        "train": {"total_patches": args.train_patches, "manifest": str((out_dir / "train" / "manifest.json").resolve())},
        "val": {"total_patches": args.val_patches, "manifest": str((out_dir / "val" / "manifest.json").resolve())},
        "stratification": {
            "month": True,
            "excluded_months": sorted(exclude_months),
            "time_bin_size": args.time_bin_size,
            "z0": [args.z_min_start, args.z_max_start],
            "residual_amplitude_score": "p90(abs(kc_CO2 - median(kc_CO2))) on scored candidates",
            "xy_stride": args.xy_stride,
        },
    }
    with open(out_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(root_manifest, f, indent=2)
    print(json.dumps(root_manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()

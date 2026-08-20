#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from netCDF4 import Dataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from twostage_v6.stage2_readers import Stage2ShardReader  # noqa: E402
from twostage_v6.stage2_constants import V40_STAGE1_GLOBAL_CONTEXT_CHANNELS  # noqa: E402
from two_stage_surrogate.data.alignment import linear_time_match, values_to_seconds  # noqa: E402
from two_stage_surrogate.data.stage1_cache import denormalize_target, load_cache_manifest  # noqa: E402
from two_stage_surrogate.data.stage1_direct_reader import (  # noqa: E402
    TOPOGRAPHY_PATH,
    _array,
    _interpolate_profile_to,
    _interpolate_w_to_zu,
    _month_encoding,
    _norm_coord,
    _resize_nearest_2d,
    _static_patch_600_to_output,
    _tod_encoding,
    _topography_3d_patch,
    _topography_surface_patch,
)
from two_stage_surrogate.models.stage1_fno import LocalFNOStage1, Stage1ModelConfig  # noqa: E402


# Groups full-domain Stage 1 input arrays and metadata.
@dataclass(frozen=True)
class FullStage1Input:
    geometry_3d: np.ndarray
    surface_2d: np.ndarray
    profile: np.ndarray
    scalar: np.ndarray
    theta_reference: np.ndarray
    bg_profile: np.ndarray
    emission_2d: np.ndarray
    fluid_mask: np.ndarray
    metadata: dict[str, Any]


# Accumulates running mean and variance statistics.
class RunningStats:
    # Store constructor arguments and initialize object state.
    def __init__(self, channels: int) -> None:
        self.sum = np.zeros(channels, dtype=np.float64)
        self.sumsq = np.zeros(channels, dtype=np.float64)
        self.count = np.zeros(channels, dtype=np.float64)

    # Update running metric or statistic accumulators.
    def update(self, arr: np.ndarray) -> None:
        values = np.asarray(arr, dtype=np.float64).reshape(arr.shape[0], -1)
        valid = np.isfinite(values)
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
def safe_prepare_out_root(out_root: Path, overwrite: bool) -> None:
    out_root = out_root.resolve()
    project = PROJECT_ROOT.resolve()
    if not str(out_root).startswith(str(project)):
        raise ValueError(f"Refusing to write outside V6 project root: {out_root}")
    if out_root.exists():
        if not overwrite:
            raise FileExistsError(f"{out_root} already exists; use --overwrite or choose another --out-root")
        if "sidecar" not in out_root.name and "context" not in out_root.name:
            raise ValueError(f"Refusing to overwrite path that does not look like a sidecar: {out_root}")
        shutil.rmtree(out_root)
    out_root.mkdir(parents=True, exist_ok=True)


# Load stage1 manifest from disk or cache.
def load_stage1_manifest(path: Path) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    jobs = {job["job"]: job for job in manifest["jobs"]}
    return list(manifest["records"]), jobs


# Load stage1 model from disk or cache.
def load_stage1_model(checkpoint_path: Path, device: torch.device) -> tuple[LocalFNOStage1, dict[str, torch.Tensor], dict[str, Any]]:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = LocalFNOStage1(Stage1ModelConfig(**dict(checkpoint["model_config"]))).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    cache_manifest = checkpoint.get("cache_manifest")
    if cache_manifest is None:
        cache_manifest = load_cache_manifest(checkpoint["cache_root"])
    target_stats: dict[str, torch.Tensor] = {}
    for field in ("target_uv", "target_w", "target_theta_prime"):
        target_stats[f"{field}_mean"] = torch.tensor(cache_manifest["normalization"][f"{field}_mean"], device=device)
        target_stats[f"{field}_std"] = torch.tensor(cache_manifest["normalization"][f"{field}_std"], device=device)
    return model, target_stats, cache_manifest["normalization"]


# Normalize model input channels.
def normalize_input(arr: np.ndarray, mean: Any, std: Any) -> np.ndarray:
    mean_arr = np.asarray(mean, dtype=np.float32)
    std_arr = np.maximum(np.asarray(std, dtype=np.float32), 1.0e-6)
    return ((arr.astype(np.float32, copy=False) - mean_arr) / std_arr).astype(np.float32, copy=False)


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


# Read prev co2 full from disk.
def read_prev_co2_full(jobs_root: Path, job: str, target_source: str, t_idx: int, z0: int, depth: int, height: int, width: int) -> np.ndarray:
    if t_idx <= 0:
        return np.zeros((depth, height, width), dtype=np.float32)
    with Dataset(locate_co2_file(jobs_root, job, target_source), "r") as ds:
        var = ds.variables["kc_CO2"]
        prev_t = min(t_idx - 1, var.shape[0] - 1)
        arr = _array(var[prev_t, z0 : z0 + depth, 0:height, 0:width])
    out = np.zeros((depth, height, width), dtype=np.float32)
    dz = min(depth, arr.shape[-3])
    dy = min(height, arr.shape[-2])
    dx = min(width, arr.shape[-1])
    out[:dz, :dy, :dx] = arr[:dz, :dy, :dx]
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


# Read emission full from disk.
def read_emission_full(job_files: dict[str, Any], record: dict[str, Any], height: int, width: int) -> np.ndarray:
    idx = int(record.get("chemistry_nearest_index", record.get("output_time_index", 0)))
    with Dataset(job_files["chemistry"], "r") as ds:
        var = ds.variables["emission_values"]
        idx = min(idx, var.shape[0] - 1)
        arr = np.squeeze(_array(var[idx]))
    if arr.ndim == 3:
        arr = arr[0]
    return _resize_nearest_2d(arr, height, width)


# Read full stage1 input from disk.
def read_full_stage1_input(
    record: dict[str, Any],
    job_files: dict[str, Any],
    *,
    full_h: int,
    full_w: int,
    topo_path: str,
) -> FullStage1Input:
    z0 = int(record["z0"])
    depth = int(record.get("dz", 16))
    t_idx = int(record["output_time_index"])

    with Dataset(job_files["out3d"], "r") as out_ds:
        output_seconds = values_to_seconds(_array(out_ds.variables["time"][:]), getattr(out_ds.variables["time"], "units", "seconds"))
        zu = _array(out_ds.variables["zu_3d"][:])
        zw = _array(out_ds.variables["zw_3d"][:])
        target_zu = zu[z0 : z0 + depth]
        output_nz = len(out_ds.dimensions["zu_3d"])
        output_ny = len(out_ds.dimensions["y"])
        output_nx = len(out_ds.dimensions["x"])
        out_sec = float(output_seconds[min(t_idx, len(output_seconds) - 1)])

    with Dataset(job_files["dynamic"], "r") as dyn_ds:
        dyn_seconds = values_to_seconds(_array(dyn_ds.variables["time"][:]), getattr(dyn_ds.variables["time"], "units", "seconds"))
        match = linear_time_match(out_sec, dyn_seconds)
        dyn_z = _array(dyn_ds.variables["z"][:])
        dyn_zw = _array(dyn_ds.variables["zw"][:])
        wr = match.weight_right

        def profile(name: str) -> np.ndarray:
            left = _array(dyn_ds.variables[name][match.left_index])
            right = _array(dyn_ds.variables[name][match.right_index])
            return (1.0 - wr) * left + wr * right

        profile_u = profile("ls_forcing_right_u")
        profile_v = profile("ls_forcing_right_v")
        profile_w = _interpolate_profile_to(profile("ls_forcing_right_w"), dyn_zw, dyn_z)
        profile_pt = profile("ls_forcing_right_pt")
        profile_qv = profile("ls_forcing_right_qv")
        profile_co2 = profile("ls_forcing_right_CO2")
        theta_reference_1d = _interpolate_profile_to(profile_pt, dyn_z, target_zu)
        bg_profile = _interpolate_profile_to(profile_co2, dyn_z, target_zu)

    profile_stack = np.stack([profile_u, profile_v, profile_w, profile_pt, profile_qv, profile_co2], axis=0)
    theta_reference = theta_reference_1d[None, :, None, None].astype(np.float32)

    topo_3d = _topography_3d_patch(topo_path, z0, 0, 0, depth, full_h, full_w)
    fluid_mask = (topo_3d == 0).astype(np.float32)
    building_voxel_mask = (topo_3d != 0).astype(np.float32)

    z_grid = _norm_coord(z0, depth, output_nz)[:, None, None]
    y_grid = _norm_coord(0, full_h, output_ny)[None, :, None]
    x_grid = _norm_coord(0, full_w, output_nx)[None, None, :]
    geometry = np.stack(
        [
            fluid_mask,
            np.broadcast_to(x_grid, (depth, full_h, full_w)),
            np.broadcast_to(y_grid, (depth, full_h, full_w)),
            np.broadcast_to(z_grid, (depth, full_h, full_w)),
            building_voxel_mask,
        ],
        axis=0,
    ).astype(np.float32)

    topo_2d, building_height = _topography_surface_patch(topo_path, 0, 0, full_h, full_w)
    with Dataset(job_files["static"], "r") as static_ds:
        surface = [
            topo_2d,
            building_height,
            _static_patch_600_to_output(static_ds.variables["vegetation_type"], 0, 0, full_h, full_w),
            _static_patch_600_to_output(static_ds.variables["pavement_type"], 0, 0, full_h, full_w),
            _static_patch_600_to_output(static_ds.variables["water_type"], 0, 0, full_h, full_w),
            _static_patch_600_to_output(static_ds.variables["albedo_type"], 0, 0, full_h, full_w),
            _static_patch_600_to_output(static_ds.variables["evi_pft"], 0, 0, full_h, full_w),
            _static_patch_600_to_output(static_ds.variables["lswi_pft"], 0, 0, full_h, full_w),
        ]
    month_sin, month_cos = _month_encoding(int(record["month"]))
    tod_sin, tod_cos = _tod_encoding(out_sec)
    metadata = {
        "job": record["job"],
        "month": int(record["month"]),
        "time_index": t_idx,
        "output_seconds": out_sec,
        "z0": z0,
        "dz": depth,
        "full_h": full_h,
        "full_w": full_w,
        "dynamic_left_index": match.left_index,
        "dynamic_right_index": match.right_index,
        "dynamic_weight_right": match.weight_right,
    }
    return FullStage1Input(
        geometry_3d=np.nan_to_num(geometry, nan=0.0, posinf=0.0, neginf=0.0),
        surface_2d=np.nan_to_num(np.stack(surface, axis=0).astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0),
        profile=np.nan_to_num(profile_stack.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0),
        scalar=np.asarray([month_sin, month_cos, tod_sin, tod_cos], dtype=np.float32),
        theta_reference=np.nan_to_num(theta_reference, nan=0.0, posinf=0.0, neginf=0.0),
        bg_profile=np.nan_to_num(bg_profile.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0),
        emission_2d=np.nan_to_num(read_emission_full(job_files, record, full_h, full_w), nan=0.0, posinf=0.0, neginf=0.0),
        fluid_mask=fluid_mask,
        metadata=metadata,
    )


# Generate tile start positions for full-domain inference.
def tile_starts(size: int, tile_size: int, stride: int) -> list[int]:
    if tile_size >= size:
        return [0]
    starts = list(range(0, max(size - tile_size + 1, 1), max(stride, 1)))
    last = size - tile_size
    if starts[-1] != last:
        starts.append(last)
    return starts


# Predict full-domain Stage 1 meteorological fields.
def predict_stage1_met_full(
    *,
    model: LocalFNOStage1,
    target_stats: dict[str, torch.Tensor],
    stage1_norm: dict[str, Any],
    sample: FullStage1Input,
    device: torch.device,
    tile_size: int,
    tile_stride: int,
    w_strategy: str,
) -> np.ndarray:
    _, depth, height, width = sample.geometry_3d.shape
    starts_y = tile_starts(height, tile_size, tile_stride)
    starts_x = tile_starts(width, tile_size, tile_stride)
    acc = np.zeros((4, depth, height, width), dtype=np.float32)
    weight = np.zeros((depth, height, width), dtype=np.float32)

    profile = normalize_input(sample.profile, stage1_norm["profile_mean"], stage1_norm["profile_std"])
    scalar = normalize_input(sample.scalar, stage1_norm["scalar_mean"], stage1_norm["scalar_std"])
    profile_t = torch.from_numpy(profile[None]).to(device)
    scalar_t = torch.from_numpy(scalar[None]).to(device)

    with torch.no_grad():
        for y0 in starts_y:
            y1 = y0 + min(tile_size, height)
            for x0 in starts_x:
                x1 = x0 + min(tile_size, width)
                geom = normalize_input(
                    sample.geometry_3d[:, :, y0:y1, x0:x1],
                    stage1_norm["geometry_3d_mean"],
                    stage1_norm["geometry_3d_std"],
                )
                surf = normalize_input(
                    sample.surface_2d[:, y0:y1, x0:x1],
                    stage1_norm["surface_2d_mean"],
                    stage1_norm["surface_2d_std"],
                )
                pred = model(
                    geometry_3d=torch.from_numpy(geom[None]).to(device),
                    surface_2d=torch.from_numpy(surf[None]).to(device),
                    profile=profile_t,
                    scalar=scalar_t,
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
                elif w_strategy == "forcing_profile":
                    profile_w = np.asarray(sample.profile[2], dtype=np.float32)
                    interp = np.interp(
                        np.linspace(0, len(profile_w) - 1, depth, dtype=np.float32),
                        np.arange(len(profile_w), dtype=np.float32),
                        profile_w,
                    ).astype(np.float32)
                    w = np.broadcast_to(interp[None, :, None, None], w_pred.shape).copy()
                else:
                    raise ValueError(f"Unknown w strategy: {w_strategy}")
                met = np.concatenate([uv, theta.astype(np.float32), w.astype(np.float32)], axis=0)
                acc[:, :, y0:y1, x0:x1] += met
                weight[:, y0:y1, x0:x1] += 1.0
    return acc / np.maximum(weight[None], 1.0)


# Downsample a stack of full-domain fields.
def downsample_stack(stack: np.ndarray, out_hw: int) -> np.ndarray:
    c, d, h, w = stack.shape
    out = np.empty((c, d, out_hw, out_hw), dtype=np.float32)
    for channel in range(c):
        tensor = torch.from_numpy(np.ascontiguousarray(stack[channel])).view(d, 1, h, w)
        resized = F.interpolate(tensor, size=(out_hw, out_hw), mode="bilinear", align_corners=False)
        out[channel] = resized.view(d, out_hw, out_hw).numpy()
    return out


# Assemble low-resolution context channels.
def assemble_context(sample: FullStage1Input, met: np.ndarray, prev_co2: np.ndarray, channels: tuple[str, ...]) -> np.ndarray:
    depth, height, width = sample.fluid_mask.shape
    emission = np.broadcast_to(sample.emission_2d[None], (depth, height, width)).astype(np.float32, copy=False)
    bg = np.broadcast_to(sample.bg_profile[:, None, None], (depth, height, width)).astype(np.float32, copy=False)
    values = {
        "emission_values": emission,
        "ls_forcing_right_CO2": bg,
        "u": met[0],
        "v": met[1],
        "w": met[3],
        "p": np.zeros((depth, height, width), dtype=np.float32),
        "theta": met[2],
        "fluid_mask": sample.fluid_mask.astype(np.float32, copy=False),
        "prev_kc_CO2": prev_co2.astype(np.float32, copy=False),
    }
    return np.stack([values[name] for name in channels], axis=0).astype(np.float32, copy=False)


# Collect context shards for one split.
def collect_split_contexts(
    cache_root: Path,
    split: str,
    *,
    max_source_samples: int,
    max_contexts: int,
) -> tuple[list[dict[str, Any]], np.ndarray]:
    manifest = json.loads((cache_root / "cache_manifest.json").read_text(encoding="utf-8"))
    split_info = manifest["splits"][split]
    key_to_index: dict[tuple[str, int, int], int] = {}
    records: list[dict[str, Any]] = []
    total = int(split_info["count"])
    if max_source_samples > 0:
        total = min(total, max_source_samples)
    sample_context_index = np.full(total, -1, dtype=np.int32)
    sample_index = 0
    stop = False
    for shard in split_info["shards"]:
        with np.load(cache_root / shard["path"], allow_pickle=False) as data:
            for meta_raw in data["metadata_json"]:
                if sample_index >= total:
                    stop = True
                    break
                meta = json.loads(str(meta_raw))
                key = (str(meta["job"]), int(meta["time_index"]), int(meta["z0"]))
                if key not in key_to_index:
                    if max_contexts > 0 and len(records) >= max_contexts:
                        sample_context_index[sample_index] = -1
                        sample_index += 1
                        continue
                    key_to_index[key] = len(records)
                    records.append(
                        {
                            "job": key[0],
                            "time_index": key[1],
                            "z0": key[2],
                            "month": int(meta["month"]),
                            "target_source": str(meta.get("target_source", "av3d")),
                        }
                    )
                sample_context_index[sample_index] = key_to_index[key]
                sample_index += 1
        if stop:
            break
    return records, sample_context_index


# Build record lookup for the workflow.
def build_record_lookup(stage1_records: list[dict[str, Any]]) -> dict[tuple[str, int], dict[str, Any]]:
    lookup: dict[tuple[str, int], dict[str, Any]] = {}
    for record in stage1_records:
        key = (str(record["job"]), int(record["output_time_index"]))
        lookup.setdefault(key, record)
    return lookup


# Write split to disk.
def write_split(
    *,
    cache_root: Path,
    jobs_root: Path,
    stage1_records: list[dict[str, Any]],
    jobs: dict[str, dict[str, Any]],
    out_root: Path,
    split: str,
    model: LocalFNOStage1,
    target_stats: dict[str, torch.Tensor],
    stage1_norm: dict[str, Any],
    device: torch.device,
    shard_size: int,
    context_size: int,
    full_h: int,
    full_w: int,
    tile_size: int,
    tile_stride: int,
    w_strategy: str,
    channels: tuple[str, ...],
    target_source: str,
    topo_path: str,
    progress_every: int,
    max_source_samples: int,
    max_contexts_per_split: int,
    stats: RunningStats | None,
) -> dict[str, Any]:
    records, sample_context_index = collect_split_contexts(
        cache_root,
        split,
        max_source_samples=max_source_samples,
        max_contexts=max_contexts_per_split,
    )
    split_dir = out_root / split
    split_dir.mkdir(parents=True, exist_ok=True)
    np.save(split_dir / "sample_context_index.npy", sample_context_index)

    record_lookup = build_record_lookup(stage1_records)
    shards: list[dict[str, Any]] = []
    key_records: list[dict[str, Any]] = []
    buffer: list[np.ndarray] = []
    start_index = 0

    def flush(shard_idx: int) -> None:
        nonlocal start_index
        if not buffer:
            return
        path = split_dir / f"context_shard_{shard_idx:05d}.npy"
        arr = np.stack(buffer, axis=0).astype(np.float16)
        np.save(path, arr, allow_pickle=False)
        shards.append(
            {
                "path": str(path.relative_to(out_root)),
                "start": start_index,
                "count": int(arr.shape[0]),
                "shape": list(arr.shape),
                "dtype": "float16",
            }
        )
        start_index += int(arr.shape[0])
        buffer.clear()

    print(
        f"Building full-domain Stage1 context split={split}: source_samples={sample_context_index.shape[0]} "
        f"unique_contexts={len(records)} tile={tile_size} stride={tile_stride}",
        flush=True,
    )
    for idx, meta in enumerate(records):
        base = dict(record_lookup.get((meta["job"], meta["time_index"]), {}))
        base.update(
            {
                "sample_key": f"full_domain_context/{meta['job']}/t{meta['time_index']:04d}/z{meta['z0']:03d}",
                "job": meta["job"],
                "month": int(meta["month"]),
                "output_time_index": int(meta["time_index"]),
                "z0": int(meta["z0"]),
                "y0": 0,
                "x0": 0,
                "dz": 16,
                "dy": full_h,
                "dx": full_w,
            }
        )
        job_files = jobs[meta["job"]]
        full_input = read_full_stage1_input(base, job_files, full_h=full_h, full_w=full_w, topo_path=topo_path)
        met = predict_stage1_met_full(
            model=model,
            target_stats=target_stats,
            stage1_norm=stage1_norm,
            sample=full_input,
            device=device,
            tile_size=tile_size,
            tile_stride=tile_stride,
            w_strategy=w_strategy,
        )
        prev_co2 = read_prev_co2_full(
            jobs_root,
            meta["job"],
            target_source=str(meta.get("target_source", target_source)),
            t_idx=int(meta["time_index"]),
            z0=int(meta["z0"]),
            depth=16,
            height=full_h,
            width=full_w,
        )
        context_full = assemble_context(full_input, met, prev_co2, channels)
        context_low = downsample_stack(context_full, context_size)
        if stats is not None and int(meta["time_index"]) > 0:
            stats.update(context_low)
        buffer.append(context_low)
        key_record = dict(full_input.metadata)
        key_record["context_index"] = idx
        key_record["sample_key"] = base["sample_key"]
        key_record["target_source"] = str(meta.get("target_source", target_source))
        key_records.append(key_record)
        if len(buffer) >= shard_size:
            flush(len(shards))
        if (idx + 1) % max(1, progress_every) == 0 or idx + 1 == len(records):
            print(f"{split}: {idx + 1}/{len(records)} contexts", flush=True)

    flush(len(shards))
    (split_dir / "context_keys.json").write_text(json.dumps(key_records, indent=2), encoding="utf-8")
    return {
        "source_samples": int(sample_context_index.shape[0]),
        "unique_contexts": len(records),
        "sample_context_index": str((split_dir / "sample_context_index.npy").relative_to(out_root)),
        "keys_path": str((split_dir / "context_keys.json").relative_to(out_root)),
        "shards": shards,
    }


# Entry point for the command-line workflow.
def main() -> None:
    parser = argparse.ArgumentParser(description="Build full-domain Stage1-met context sidecar aligned with the Stage2 cache.")
    parser.add_argument("--cache-root", type=Path, default=PROJECT_ROOT / "generated" / "stage2_cache")
    parser.add_argument("--stage1-manifest", type=Path, default=PROJECT_ROOT / "generated" / "stage1_manifest" / "stage1_manifest.json")
    parser.add_argument("--stage1-checkpoint", type=Path, default=PROJECT_ROOT / "checkpoints" / "stage1_local_fno_best_model.pt")
    parser.add_argument("--jobs-root", type=Path, default=PROJECT_ROOT / "external_data" / "camden" / "JOBS")
    parser.add_argument("--out-root", type=Path, default=PROJECT_ROOT / "generated" / "full_domain_context_sidecar")
    parser.add_argument("--splits", default="train,val")
    parser.add_argument("--target-source", choices=["av3d", "out3d"], default="av3d")
    parser.add_argument("--context-size", type=int, default=80)
    parser.add_argument("--full-h", type=int, default=800)
    parser.add_argument("--full-w", type=int, default=800)
    parser.add_argument("--tile-size", type=int, default=256)
    parser.add_argument("--tile-stride", type=int, default=256)
    parser.add_argument("--shard-size", type=int, default=16)
    parser.add_argument("--w-strategy", choices=["predicted", "zero", "forcing_profile"], default="predicted")
    parser.add_argument("--topography-path", default=TOPOGRAPHY_PATH)
    parser.add_argument("--progress-every", type=int, default=20)
    parser.add_argument("--max-source-samples-per-split", type=int, default=0)
    parser.add_argument("--max-contexts-per-split", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    safe_prepare_out_root(args.out_root, args.overwrite)
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else args.device)
    if args.device == "auto" and not torch.cuda.is_available():
        device = torch.device("cpu")

    stage1_records, jobs = load_stage1_manifest(args.stage1_manifest)
    model, target_stats, stage1_norm = load_stage1_model(args.stage1_checkpoint, device)
    channels = tuple(V40_STAGE1_GLOBAL_CONTEXT_CHANNELS)
    train_stats = RunningStats(len(channels))
    split_manifests: dict[str, Any] = {}
    for split in [s.strip() for s in args.splits.split(",") if s.strip()]:
        split_manifests[split] = write_split(
            cache_root=args.cache_root,
            jobs_root=args.jobs_root,
            stage1_records=stage1_records,
            jobs=jobs,
            out_root=args.out_root,
            split=split,
            model=model,
            target_stats=target_stats,
            stage1_norm=stage1_norm,
            device=device,
            shard_size=args.shard_size,
            context_size=args.context_size,
            full_h=args.full_h,
            full_w=args.full_w,
            tile_size=args.tile_size,
            tile_stride=args.tile_stride,
            w_strategy=args.w_strategy,
            channels=channels,
            target_source=args.target_source,
            topo_path=args.topography_path,
            progress_every=args.progress_every,
            max_source_samples=args.max_source_samples_per_split,
            max_contexts_per_split=args.max_contexts_per_split,
            stats=train_stats if split == "train" else None,
        )

    mean, std = train_stats.finalize()
    norm = {
        "channels": list(channels),
        "mean": [float(v) for v in mean],
        "std": [float(max(v, 1.0e-6)) for v in std],
        "computed_from": "train contexts with time_index > 0",
    }
    (args.out_root / "normalization.json").write_text(json.dumps(norm, indent=2), encoding="utf-8")
    manifest = {
        "schema_version": "two_stage_surrogate_v6_full_domain_stage1_context_sidecar_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "cache_root": str(args.cache_root),
        "stage1_manifest": str(args.stage1_manifest),
        "stage1_checkpoint": str(args.stage1_checkpoint),
        "jobs_root": str(args.jobs_root),
        "target_source": args.target_source,
        "w_strategy": args.w_strategy,
        "global_channels": list(channels),
        "global_shape": [len(channels), 16, args.context_size, args.context_size],
        "full_domain_shape": [16, args.full_h, args.full_w],
        "tile_size": args.tile_size,
        "tile_stride": args.tile_stride,
        "splits": split_manifests,
    }
    (args.out_root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()

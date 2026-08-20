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

# An optional external site-packages path is supported for the original HPC
# layout. A clean virtual environment should leave PYLIBS_ROOT unset.
PYLIBS = os.environ.get("PYLIBS_ROOT", "").strip()
if PYLIBS:
    sys.path = [p for p in sys.path if os.path.abspath(p) != os.path.abspath(PYLIBS)]

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm, ListedColormap

if PYLIBS:
    sys.path.append(PYLIBS)
import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from models.unet3d import (
    UNet3D,
    V7StyleUNet3D,
    V20ContextV7UNet3D,
    V35MultiTaskContextV7UNet3D,
    V37HardPatternContextV7UNet3D,
    V38EventTextureContextV7UNet3D,
)
from scripts.train_3d_unet import (
    JobFiles,
    STATIC_VARS,
    TARGET_VAR,
    align_3d_to_shape,
    discover_jobs,
    find_one,
    interp_w_patch_from_zw_to_zu,
    locate_job_files,
    month_features,
    parse_month_from_job,
    safe_open_dataset,
    time_of_day_features,
    to_numpy,
)
from scripts.build_v13_sidecar import (
    LOCAL_BASE_CHANNELS,
    block_mean_2d,
    corrected_bg_profile,
    corrected_emission_2d,
    build_global_context,
)


# Return a safe matplotlib colormap.
def safe_cmap(name: str, fallback: str):
    try:
        return plt.get_cmap(name).copy()
    except ValueError:
        return plt.get_cmap(fallback).copy()


# Parse a list of integer command-line values.
def parse_int_list(text: str) -> list[int]:
    values: list[int] = []
    for item in text.split(","):
        item = item.strip()
        if item:
            values.append(int(item))
    return values


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


# Locate direct job root.
def locate_direct_job_root(job_root: str, job_name: str | None = None, month: int | None = None) -> JobFiles | None:
    """Locate one PALM job when INPUT/OUTPUT are directly under job_root.

    Camden jobs are arranged as JOBS/<job_name>/INPUT and are handled by
    locate_job_files(). Richmond was copied as one standalone job directory, so
    discover_jobs() does not see it. This keeps the generic evaluator usable for
    a domain-translation test without changing Linfeng's source directory.
    """
    root = os.path.abspath(job_root)
    if not (os.path.isdir(os.path.join(root, "INPUT")) and os.path.isdir(os.path.join(root, "OUTPUT"))):
        return None

    chemistry = find_one(
        [
            os.path.join(root, "INPUT", "*_chemistry_N02"),
            os.path.join(root, "INPUT", "*_chemistry_N02.nc"),
            os.path.join(root, "INPUT", "*_chemistry_N02*"),
        ]
    )
    dynamic = find_one(
        [
            os.path.join(root, "INPUT", "*_dynamic_N02"),
            os.path.join(root, "INPUT", "*_dynamic_N02.nc"),
            os.path.join(root, "INPUT", "*_dynamic"),
            os.path.join(root, "INPUT", "*_dynamic.nc"),
            os.path.join(root, "INPUT", "*_dynamic*"),
        ]
    )
    static = find_one(
        [
            os.path.join(root, "INPUT", "*_static_N02"),
            os.path.join(root, "INPUT", "*_static_N02.nc"),
            os.path.join(root, "INPUT", "*_static_N02*"),
        ]
    )
    out3d = find_one(
        [
            os.path.join(root, "OUTPUT", "*_3d_N02.nc"),
            os.path.join(root, "OUTPUT", "*_3d_N02.*.nc"),
            os.path.join(root, "OUTPUT", "*_3d_N02*"),
        ]
    )
    av3d = find_one(
        [
            os.path.join(root, "OUTPUT", "*_av_3d_N02.nc"),
            os.path.join(root, "OUTPUT", "*_av_3d_N02.*.nc"),
            os.path.join(root, "OUTPUT", "*_av_3d_N02*"),
        ]
    )
    if not all([chemistry, dynamic, static, out3d, av3d]):
        return None

    resolved_name = job_name or os.path.basename(root.rstrip(os.sep))
    resolved_month = int(month) if month is not None else parse_month_from_job(resolved_name)
    return JobFiles(
        name=resolved_name,
        month=resolved_month,
        chemistry=chemistry,
        dynamic=dynamic,
        static=static,
        out3d=out3d,
        av3d=av3d,
    )


# Choose a valid patch start index.
def choose_start(rng: random.Random, n: int, patch: int) -> int:
    if n <= patch:
        return 0
    return rng.randrange(0, n - patch + 1)


# Count how many selected layers overlap a patch.
def layer_overlap_count(z0: int, depth: int, layer_min: int, layer_max: int) -> int:
    z1 = z0 + depth - 1
    return max(0, min(z1, layer_max) - max(z0, layer_min) + 1)


# Choose a vertical patch start focused on selected layers.
def choose_layer_focused_z_start(
    rng: random.Random,
    n: int,
    patch: int,
    layer_min: int,
    layer_max: int,
    min_overlap: int,
    require_full_range: bool,
) -> int | None:
    if n <= 0:
        return None
    max_start = max(0, n - patch)
    if require_full_range:
        lo = max(0, int(layer_min))
        hi = min(max_start, int(layer_max) - patch + 1)
        if lo <= hi:
            return rng.randrange(lo, hi + 1)

    min_overlap = max(1, int(min_overlap))
    valid = [
        z0
        for z0 in range(max_start + 1)
        if layer_overlap_count(z0, patch, int(layer_min), int(layer_max)) >= min_overlap
    ]
    if not valid:
        return None
    return rng.choice(valid)


# Restrict a mask to the selected vertical layer range.
def restrict_mask_to_layer_range(mask: np.ndarray, z0: int, layer_min: int | None, layer_max: int | None) -> np.ndarray:
    if layer_min is None or layer_max is None:
        return mask
    d = mask.shape[0]
    global_z = np.arange(d, dtype=np.int64) + int(z0)
    z_mask = ((global_z >= int(layer_min)) & (global_z <= int(layer_max))).astype(np.float32)
    return mask * z_mask[:, None, None]


# Build topography mask for the workflow.
def build_topography_mask(
    topo_values: np.ndarray | None,
    z0: int,
    z1: int,
    y0: int,
    y1: int,
    x0: int,
    x1: int,
    target_shape: tuple[int, int, int],
) -> np.ndarray:
    mask = np.ones(target_shape, dtype=np.float32)
    if topo_values is None:
        return mask
    topo_z, topo_y, topo_x = topo_values.shape
    if topo_z <= z0 or topo_y <= y0 or topo_x <= x0:
        return mask
    z_read1 = min(z1, topo_z)
    y_read1 = min(y1, topo_y)
    x_read1 = min(x1, topo_x)
    fluid = (topo_values[z0:z_read1, y0:y_read1, x0:x_read1] == 0).astype(np.float32)
    mask[: fluid.shape[0], : fluid.shape[1], : fluid.shape[2]] = fluid
    return mask


# Build input patch for the workflow.
def build_input_patch(
    dsch,
    dsdyn,
    dsst,
    ds3d,
    dsav,
    jf,
    t_idx: int,
    z0: int,
    y0: int,
    x0: int,
    patch_size: tuple[int, int, int],
) -> tuple[np.ndarray, np.ndarray]:
    patch_d, patch_h, patch_w = patch_size
    t3d = min(t_idx, int(ds3d.sizes["time"]) - 1)
    tav = min(t_idx, int(dsav.sizes["time"]) - 1)

    target = dsav[TARGET_VAR].isel(time=tav)
    z_name = "zu_3d" if "zu_3d" in target.dims else target.dims[0]
    y_name = "y"
    x_name = "x"
    z1 = min(int(target.sizes[z_name]), z0 + patch_d)
    y1 = min(int(target.sizes[y_name]), y0 + patch_h)
    x1 = min(int(target.sizes[x_name]), x0 + patch_w)

    target_patch = target.isel({z_name: slice(z0, z1), y_name: slice(y0, y1), x_name: slice(x0, x1)})
    target_shape = (
        int(target_patch.sizes[z_name]),
        int(target_patch.sizes[y_name]),
        int(target_patch.sizes[x_name]),
    )

    dyn_arrays: list[np.ndarray] = []

    emis = dsch["emission_values"].isel(time=min(t3d, int(dsch.sizes.get("time", 1)) - 1))
    for dim in ["nspecies", "z"]:
        if dim in emis.dims and emis.sizes[dim] > 1:
            emis = emis.isel({dim: 0})
        elif dim in emis.dims:
            emis = emis.squeeze(dim)
    emis = emis.isel({y_name: slice(y0, y1), x_name: slice(x0, x1)})
    emis3d = emis.expand_dims({z_name: target_patch.sizes[z_name]}).transpose(z_name, y_name, x_name)
    dyn_arrays.append(align_3d_to_shape(to_numpy(emis3d), target_shape))

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
        bg_np = np.broadcast_to(bg_z_values[:, None, None], target_shape).copy()
    else:
        if "z" in bg_co2.dims and z_name not in bg_co2.dims:
            bg_co2 = bg_co2.rename({"z": z_name})
        if z_name in bg_co2.dims:
            bg_co2 = bg_co2.isel({z_name: slice(z0, z1)})
        else:
            bg_co2 = bg_co2.expand_dims({z_name: target_patch.sizes[z_name]})
        if y_name in bg_co2.dims:
            bg_co2 = bg_co2.isel({y_name: slice(y0, y1)})
        else:
            bg_co2 = bg_co2.expand_dims({y_name: target_patch.sizes[y_name]})
        if x_name in bg_co2.dims:
            bg_co2 = bg_co2.isel({x_name: slice(x0, x1)})
        else:
            bg_co2 = bg_co2.expand_dims({x_name: target_patch.sizes[x_name]})
        bg_co2 = bg_co2.transpose(z_name, y_name, x_name)
        bg_np = align_3d_to_shape(to_numpy(bg_co2), target_shape)
    dyn_arrays.append(bg_np)

    for var in ["u", "v", "p", "theta"]:
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
    elif "zu_3d" in w_zw.dims:
        w_patch = w_zw.isel({"zu_3d": slice(z0, z1), "y": slice(y0, y1), "x": slice(x0, x1)})
        w_np = to_numpy(w_patch)
    else:
        w_patch = w_zw.isel({z_name: slice(z0, z1), "y": slice(y0, y1), "x": slice(x0, x1)})
        w_np = to_numpy(w_patch)
    dyn_arrays.append(align_3d_to_shape(w_np, target_shape))

    static_arrays: list[np.ndarray] = []
    for var in STATIC_VARS:
        s2d = dsst[var].isel({"y": slice(y0, y1), "x": slice(x0, x1)})
        s3d = np.repeat(to_numpy(s2d)[None, :, :], target_shape[0], axis=0)
        static_arrays.append(align_3d_to_shape(s3d, target_shape))

    msin, mcos = month_features(jf.month)
    month_sin = np.full_like(static_arrays[0], msin)
    month_cos = np.full_like(static_arrays[0], mcos)

    if "time" in ds3d:
        t_value = ds3d["time"].values[t3d]
    else:
        t_value = float(t3d)
    tsin, tcos = time_of_day_features(t_value)
    tod_sin = np.full_like(static_arrays[0], tsin)
    tod_cos = np.full_like(static_arrays[0], tcos)

    x = np.stack(dyn_arrays + static_arrays + [month_sin, month_cos, tod_sin, tod_cos], axis=0)
    y = to_numpy(target_patch)
    return np.nan_to_num(x, nan=0.0).astype(np.float32), np.nan_to_num(y, nan=0.0).astype(np.float32)


# Append normalized coordinate channels to the input tensor.
def append_coord_channels(
    x: np.ndarray,
    z0: int,
    y0: int,
    x0: int,
    nz: int,
    ny: int,
    nx: int,
) -> np.ndarray:
    d, h, w = x.shape[-3:]
    z_values = ((np.arange(d, dtype=np.float32) + float(z0)) / float(max(nz - 1, 1))) * 2.0 - 1.0
    y_values = ((np.arange(h, dtype=np.float32) + float(y0)) / float(max(ny - 1, 1))) * 2.0 - 1.0
    x_values = ((np.arange(w, dtype=np.float32) + float(x0)) / float(max(nx - 1, 1))) * 2.0 - 1.0
    x_coord = np.broadcast_to(x_values[None, None, :], (d, h, w))
    y_coord = np.broadcast_to(y_values[None, :, None], (d, h, w))
    z_coord = np.broadcast_to(z_values[:, None, None], (d, h, w))
    coords = np.stack((x_coord, y_coord, z_coord), axis=0).astype(np.float32)
    return np.concatenate((x, coords), axis=0).astype(np.float32, copy=False)


# Drop input channels requested by the experiment config.
def drop_input_channels(x: np.ndarray, dropped_channels: list[int]) -> np.ndarray:
    if not dropped_channels:
        return x
    drop = set(int(i) for i in dropped_channels)
    keep = [i for i in range(x.shape[0]) if i not in drop]
    if len(keep) == x.shape[0]:
        return x
    if not keep:
        raise RuntimeError("Cannot drop every input channel")
    return x[keep].astype(np.float32, copy=False)


# Keep only the input channels expected by a checkpoint.
def keep_input_channels_for_checkpoint(x: np.ndarray, checkpoint: dict[str, object]) -> np.ndarray:
    indices = checkpoint.get("kept_input_channel_indices", [])
    if not indices:
        return x
    keep = [int(i) for i in indices]
    missing = [i for i in keep if i < 0 or i >= x.shape[0]]
    if missing:
        raise RuntimeError(
            f"Checkpoint expects kept input channel indices {keep}, "
            f"but transformed input has only {x.shape[0]} channels"
        )
    return x[keep].astype(np.float32, copy=False)


# Apply the V12 input-channel transform.
def apply_v12_input_transform(
    x: np.ndarray,
    mask: np.ndarray,
    z0: int,
    y0: int,
    x0: int,
    nz: int,
    ny: int,
    nx: int,
    checkpoint: dict,
) -> np.ndarray:
    appended_channels = list(checkpoint.get("appended_channels", checkpoint.get("coord_channels", [])))
    surface_channels = [int(v) for v in checkpoint.get("surface_gated_channels", [])]
    decay = float(checkpoint.get("height_gate_decay_levels", 20.0))
    if not appended_channels and not surface_channels:
        return x

    d, h, w = x.shape[-3:]
    z_index = np.arange(d, dtype=np.float32) + float(z0)
    gate_z = np.exp(-z_index / max(decay, 1.0)).clip(0.0, 1.0)
    gate = np.broadcast_to(gate_z[:, None, None], (d, h, w)).astype(np.float32)

    if surface_channels:
        x = x.copy()
        for channel_idx in surface_channels:
            if 0 <= channel_idx < x.shape[0]:
                x[channel_idx] = x[channel_idx] * gate

    extras: list[np.ndarray] = []
    if "x_norm" in appended_channels or "y_norm" in appended_channels or "z_norm" in appended_channels:
        z_values = ((np.arange(d, dtype=np.float32) + float(z0)) / float(max(nz - 1, 1))) * 2.0 - 1.0
        y_values = ((np.arange(h, dtype=np.float32) + float(y0)) / float(max(ny - 1, 1))) * 2.0 - 1.0
        x_values = ((np.arange(w, dtype=np.float32) + float(x0)) / float(max(nx - 1, 1))) * 2.0 - 1.0
        x_coord = np.broadcast_to(x_values[None, None, :], (d, h, w))
        y_coord = np.broadcast_to(y_values[None, :, None], (d, h, w))
        z_coord = np.broadcast_to(z_values[:, None, None], (d, h, w))
        extras.extend([x_coord, y_coord, z_coord])
    if "fluid_mask" in appended_channels:
        extras.append(mask.astype(np.float32, copy=False))
    if "height_gate" in appended_channels:
        extras.append(gate)
    if extras:
        x = np.concatenate((x, np.stack(extras, axis=0).astype(np.float32)), axis=0)
    return x.astype(np.float32, copy=False)


# Load v13 normalization from disk or cache.
def load_v13_normalization(sidecar_root: str) -> dict[str, object]:
    norm_path = os.path.join(sidecar_root, "normalization.json")
    if not os.path.exists(norm_path):
        raise FileNotFoundError(f"Missing V13 normalization file: {norm_path}")
    with open(norm_path, "r", encoding="utf-8") as f:
        return json.load(f)


# Load optional target-normalization metadata from disk.
def load_target_normalization(path: str | None) -> dict[str, object] | None:
    if not path:
        return None
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing target-normalization stats: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# Choose the month or background-bin normalization group.
def target_norm_group_key(
    stats: dict[str, object],
    mode: str,
    month: int,
    bg_values: np.ndarray | None = None,
) -> str:
    if mode == "month":
        return f"{int(month):02d}"
    if mode == "background_bin":
        if bg_values is None:
            raise RuntimeError("background_bin target normalization requires corrected background values")
        value = float(np.asarray(bg_values, dtype=np.float32).mean())
        bins = stats.get("background_bins")
        if not isinstance(bins, list) or len(bins) < 2:
            raise RuntimeError("background_bin target normalization requires background_bins in stats")
        for idx in range(len(bins) - 1):
            lo = float(bins[idx])
            hi = float(bins[idx + 1])
            if (value >= lo and value < hi) or (idx == len(bins) - 2 and value <= hi):
                return f"bin_{idx:02d}"
        return "global"
    raise RuntimeError(f"Unknown target normalization mode: {mode}")


# Return safe mean and standard deviation values for a normalization group.
def target_norm_mean_std(stats: dict[str, object], key: str) -> tuple[float, float]:
    groups = stats.get("groups", {})
    group = groups.get(key) if isinstance(groups, dict) else None
    if group is None:
        group = stats.get("global", {})
    mean = float(group.get("mean", 0.0))
    std = max(float(group.get("std", 1.0)), 1.0e-6)
    return mean, std


# Build the V13 global-context sampling grid.
def v13_global_grid(x_transformed: np.ndarray, local_channels: list[str], sample_size: int) -> np.ndarray:
    d, h, w = x_transformed.shape[-3:]
    x_idx = local_channels.index("x_norm")
    y_idx = local_channels.index("y_norm")
    xs = np.rint(np.linspace(0, w - 1, sample_size)).astype(np.int64).clip(0, w - 1)
    ys = np.rint(np.linspace(0, h - 1, sample_size)).astype(np.int64).clip(0, h - 1)
    xx = x_transformed[x_idx][:, ys][:, :, xs]
    yy = x_transformed[y_idx][:, ys][:, :, xs]
    zz = np.broadcast_to(np.linspace(-1.0, 1.0, d, dtype=np.float32)[:, None, None], (d, sample_size, sample_size))
    return np.stack((xx, yy, zz), axis=-1).astype(np.float32)


# Apply the V13 corrected-context transform.
def apply_v13_transform(
    x: np.ndarray,
    mask: np.ndarray,
    corrected_emission_patch: np.ndarray,
    corrected_bg: np.ndarray,
    z0: int,
    y0: int,
    x0: int,
    nz: int,
    ny: int,
    nx: int,
    checkpoint: dict,
    normalization: dict[str, object],
    force_fluid_mask_ones: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    d, h, w = x.shape[-3:]
    local_channels = list(normalization["local_channels"])
    x = x.astype(np.float32, copy=True)
    bg_3d = np.broadcast_to(corrected_bg[:, None, None], (d, h, w)).astype(np.float32)
    x[0] = np.broadcast_to(corrected_emission_patch[None, :, :], (d, h, w)).astype(np.float32)
    x[1] = bg_3d
    if "month_sin" not in local_channels or "month_cos" not in local_channels:
        x = drop_input_channels(x, [14, 15])

    decay = float(checkpoint.get("height_gate_decay_levels", 40.0))
    gate_z = np.exp(-((np.arange(d, dtype=np.float32) + float(z0)) / max(decay, 1.0))).clip(0.0, 1.0)
    gate = np.broadcast_to(gate_z[:, None, None], (d, h, w)).astype(np.float32)
    if "height_gate" in local_channels:
        for channel_idx in (0, 7, 8, 9, 10, 11, 12, 13):
            if 0 <= channel_idx < x.shape[0]:
                x[channel_idx] = x[channel_idx] * gate

    z_values = ((np.arange(d, dtype=np.float32) + float(z0)) / float(max(nz - 1, 1))) * 2.0 - 1.0
    y_values = ((np.arange(h, dtype=np.float32) + float(y0)) / float(max(ny - 1, 1))) * 2.0 - 1.0
    x_values = ((np.arange(w, dtype=np.float32) + float(x0)) / float(max(nx - 1, 1))) * 2.0 - 1.0
    x_coord = np.broadcast_to(x_values[None, None, :], (d, h, w))
    y_coord = np.broadcast_to(y_values[None, :, None], (d, h, w))
    z_coord = np.broadcast_to(z_values[:, None, None], (d, h, w))
    extras = {
        "x_norm": x_coord,
        "y_norm": y_coord,
        "z_norm": z_coord,
        "fluid_mask": np.ones_like(mask, dtype=np.float32) if force_fluid_mask_ones else mask.astype(np.float32),
        "height_gate": gate,
    }
    extra_tensors = [extras[name] for name in local_channels[x.shape[0] :]]
    if extra_tensors:
        x = np.concatenate((x, np.stack(extra_tensors, axis=0)), axis=0).astype(np.float32)

    mean = np.asarray(normalization["local_mean"], dtype=np.float32)[:, None, None, None]
    std = np.maximum(np.asarray(normalization["local_std"], dtype=np.float32), 1.0e-6)[:, None, None, None]
    raw_extra = {"x_norm", "y_norm", "z_norm", "fluid_mask", "height_gate"}
    for i, name in enumerate(local_channels):
        if name in raw_extra:
            mean[i] = 0.0
            std[i] = 1.0
    if x.shape[0] != len(local_channels):
        raise RuntimeError(f"V13 eval channel count {x.shape[0]} does not match normalization {len(local_channels)}")
    x = (x - mean) / std

    surface_channels = [int(v) for v in checkpoint.get("surface_gated_channels", [])]
    if surface_channels:
        x = x.copy()
        for channel_idx in surface_channels:
            if 0 <= channel_idx < x.shape[0]:
                x[channel_idx] = x[channel_idx] * gate

    appended_channels = list(checkpoint.get("appended_channels", checkpoint.get("coord_channels", [])))
    if "height_gate" in appended_channels and "height_gate" not in local_channels:
        x = np.concatenate((x, gate[None].astype(np.float32)), axis=0)
    return x.astype(np.float32, copy=False), bg_3d


# Normalize the V13 global-context channels.
def normalize_v13_global_context(global_context: np.ndarray, normalization: dict[str, object]) -> np.ndarray:
    mean = np.asarray(normalization["global_mean"], dtype=np.float32)[:, None, None, None]
    std = np.maximum(np.asarray(normalization["global_std"], dtype=np.float32), 1.0e-6)[:, None, None, None]
    return ((global_context.astype(np.float32) - mean) / std).astype(np.float32)


# Load v22 prev normalization from disk or cache.
def load_v22_prev_normalization(prev_sidecar_root: str) -> dict[str, float | str]:
    norm_path = os.path.join(prev_sidecar_root, "normalization.json")
    if not os.path.exists(norm_path):
        raise FileNotFoundError(f"Missing V22 previous-CO2 normalization stats: {norm_path}")
    with open(norm_path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    return {
        "local_prev_channel": str(raw.get("local_prev_channel", "prev_kc_CO2")),
        "local_prev_mean": float(raw.get("local_prev_mean", 0.0)),
        "local_prev_std": max(float(raw.get("local_prev_std", 1.0)), 1.0e-6),
        "global_prev_channel": str(raw.get("global_prev_channel", "prev_kc_CO2_global")),
        "global_prev_mean": float(raw.get("global_prev_mean", 0.0)),
        "global_prev_std": max(float(raw.get("global_prev_std", 1.0)), 1.0e-6),
    }


# Load co2 patch from disk or cache.
def load_co2_patch(dsav, time_index: int, z0: int, z1: int, y0: int, y1: int, x0: int, x1: int) -> np.ndarray:
    target = dsav[TARGET_VAR].isel(time=int(time_index))
    z_name = "zu_3d" if "zu_3d" in target.dims else target.dims[0]
    patch = target.isel({z_name: slice(z0, z1), "y": slice(y0, y1), "x": slice(x0, x1)})
    arr = np.nan_to_num(to_numpy(patch), nan=0.0).astype(np.float32, copy=False)
    return align_3d_to_shape(arr, (z1 - z0, y1 - y0, x1 - x0))


# Load co2 global lowres from disk or cache.
def load_co2_global_lowres(dsav, time_index: int, z0: int, z1: int, ny: int, nx: int, out_size: int) -> np.ndarray:
    target = dsav[TARGET_VAR].isel(time=int(time_index))
    z_name = "zu_3d" if "zu_3d" in target.dims else target.dims[0]
    field = target.isel({z_name: slice(z0, z1), "y": slice(0, ny), "x": slice(0, nx)})
    arr = np.nan_to_num(to_numpy(field), nan=0.0).astype(np.float32, copy=False)
    arr = align_3d_to_shape(arr, (z1 - z0, ny, nx))
    return block_mean_2d(arr, out_size).astype(np.float32, copy=False)


# Append the V22 previous-CO2 local channel.
def append_v22_prev_channel(x: np.ndarray, prev_co2: np.ndarray, prev_norm: dict[str, float | str]) -> np.ndarray:
    mean = float(prev_norm["local_prev_mean"])
    std = float(prev_norm["local_prev_std"])
    prev_normalized = ((prev_co2.astype(np.float32) - mean) / std).astype(np.float32)
    return np.concatenate((x, prev_normalized[None]), axis=0).astype(np.float32, copy=False)


# Append the V22 previous-CO2 global context.
def append_v22_prev_global(global_context: np.ndarray, prev_global: np.ndarray, prev_norm: dict[str, float | str]) -> np.ndarray:
    mean = float(prev_norm["global_prev_mean"])
    std = float(prev_norm["global_prev_std"])
    prev_normalized = ((prev_global.astype(np.float32) - mean) / std).astype(np.float32)
    return np.concatenate((global_context, prev_normalized[None]), axis=0).astype(np.float32, copy=False)


V28_ADVECTION_CHANNELS = [
    "prev_dCdx",
    "prev_dCdy",
    "prev_dCdz",
    "adv_x_delta",
    "adv_y_delta",
    "adv_z_delta",
    "adv_total_delta",
]


# Compute NumPy finite differences for cached arrays.
def finite_difference_np(field: np.ndarray, spacing: float, axis: int) -> np.ndarray:
    spacing = max(float(spacing), 1.0e-6)
    field = field.astype(np.float32, copy=False)
    out = np.zeros_like(field, dtype=np.float32)
    n = int(field.shape[axis])
    if n <= 1:
        return out
    first = [slice(None)] * field.ndim
    first[axis] = 0
    second = [slice(None)] * field.ndim
    second[axis] = 1
    out[tuple(first)] = (field[tuple(second)] - field[tuple(first)]) / spacing
    last = [slice(None)] * field.ndim
    last[axis] = n - 1
    before_last = [slice(None)] * field.ndim
    before_last[axis] = n - 2
    out[tuple(last)] = (field[tuple(last)] - field[tuple(before_last)]) / spacing
    if n > 2:
        middle = [slice(None)] * field.ndim
        middle[axis] = slice(1, n - 1)
        plus = [slice(None)] * field.ndim
        plus[axis] = slice(2, n)
        minus = [slice(None)] * field.ndim
        minus[axis] = slice(0, n - 2)
        out[tuple(middle)] = (field[tuple(plus)] - field[tuple(minus)]) / (2.0 * spacing)
    return out


# Recover a denormalized V13 input channel.
def denormalized_v13_channel(x: np.ndarray, normalization: dict[str, object], name: str) -> np.ndarray:
    channels = list(normalization["local_channels"])
    if name not in channels:
        raise RuntimeError(f"Missing V28 advection channel source: {name}")
    idx = channels.index(name)
    mean = np.asarray(normalization["local_mean"], dtype=np.float32)
    std = np.maximum(np.asarray(normalization["local_std"], dtype=np.float32), 1.0e-6)
    return x[idx].astype(np.float32, copy=False) * std[idx] + mean[idx]


# Compute v28 advection delta and features for the workflow.
def compute_v28_advection_delta_and_features(
    x: np.ndarray,
    prev_co2: np.ndarray,
    normalization: dict[str, object],
    checkpoint: dict,
) -> tuple[np.ndarray, np.ndarray]:
    dx = float(checkpoint.get("v28_advection_dx", 5.0))
    dy = float(checkpoint.get("v28_advection_dy", 5.0))
    dz = float(checkpoint.get("v28_advection_dz", 10.0))
    dt = float(checkpoint.get("v28_advection_dt", 300.0))
    delta_scale = max(float(checkpoint.get("v28_advection_delta_scale", 5.0)), 1.0e-6)
    grad_scale = max(float(checkpoint.get("v28_advection_gradient_scale", 0.2)), 1.0e-6)
    adv_clip = float(checkpoint.get("v28_advection_clip", 20.0))
    input_clip = float(checkpoint.get("v28_advection_input_clip", 8.0))

    u = denormalized_v13_channel(x, normalization, "u")
    v = denormalized_v13_channel(x, normalization, "v")
    w = denormalized_v13_channel(x, normalization, "w")
    dc_dx = finite_difference_np(prev_co2, dx, axis=-1)
    dc_dy = finite_difference_np(prev_co2, dy, axis=-2)
    dc_dz = finite_difference_np(prev_co2, dz, axis=-3)
    adv_x = -u * dc_dx * dt
    adv_y = -v * dc_dy * dt
    adv_z = -w * dc_dz * dt
    adv_total = adv_x + adv_y + adv_z
    if adv_clip > 0:
        adv_x = np.clip(adv_x, -adv_clip, adv_clip)
        adv_y = np.clip(adv_y, -adv_clip, adv_clip)
        adv_z = np.clip(adv_z, -adv_clip, adv_clip)
        adv_total = np.clip(adv_total, -adv_clip, adv_clip)
    features = np.stack(
        (
            dc_dx / grad_scale,
            dc_dy / grad_scale,
            dc_dz / grad_scale,
            adv_x / delta_scale,
            adv_y / delta_scale,
            adv_z / delta_scale,
            adv_total / delta_scale,
        ),
        axis=0,
    ).astype(np.float32, copy=False)
    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
    if input_clip > 0:
        features = np.clip(features, -input_clip, input_clip)
    return adv_total.astype(np.float32, copy=False), features.astype(np.float32, copy=False)


# Append V28 advection-derived input features.
def append_v28_advection_features(
    x: np.ndarray,
    prev_co2: np.ndarray,
    normalization: dict[str, object],
    checkpoint: dict,
) -> tuple[np.ndarray, np.ndarray]:
    physics_delta, features = compute_v28_advection_delta_and_features(x, prev_co2, normalization, checkpoint)
    x = np.concatenate((x, features), axis=0).astype(np.float32, copy=False)
    return x, physics_delta


# Create an empty statistics accumulator.
def new_stats() -> dict[str, float]:
    return {
        "n": 0.0,
        "sum_y": 0.0,
        "sum_y2": 0.0,
        "sum_p": 0.0,
        "sum_p2": 0.0,
        "sum_py": 0.0,
        "abs_err": 0.0,
        "sq_err": 0.0,
    }


# Update statistics from one batch of values.
def update_stats(stats: dict[str, float], pred: np.ndarray, truth: np.ndarray, mask: np.ndarray) -> None:
    valid = np.isfinite(pred) & np.isfinite(truth) & (mask > 0)
    if not np.any(valid):
        return
    p = pred[valid].astype(np.float64, copy=False)
    y = truth[valid].astype(np.float64, copy=False)
    diff = p - y
    stats["n"] += int(y.size)
    stats["sum_y"] += float(y.sum())
    stats["sum_y2"] += float((y * y).sum())
    stats["sum_p"] += float(p.sum())
    stats["sum_p2"] += float((p * p).sum())
    stats["sum_py"] += float((p * y).sum())
    stats["abs_err"] += float(np.abs(diff).sum())
    stats["sq_err"] += float((diff * diff).sum())


# Finalize accumulated statistics.
def finalize_stats(stats: dict[str, float]) -> dict[str, float | int]:
    n = int(stats["n"])
    if n <= 0:
        return {"valid_count": 0, "R": float("nan"), "R2": float("nan"), "MAE": float("nan"), "RMSE": float("nan")}
    mae = stats["abs_err"] / n
    rmse = math.sqrt(stats["sq_err"] / n)
    y_mean = stats["sum_y"] / n
    p_mean = stats["sum_p"] / n
    ss_tot = stats["sum_y2"] - n * y_mean * y_mean
    ss_pred = stats["sum_p2"] - n * p_mean * p_mean
    cov = stats["sum_py"] - n * p_mean * y_mean
    r = cov / math.sqrt(ss_pred * ss_tot) if ss_pred > 0 and ss_tot > 0 else float("nan")
    r2 = 1.0 - stats["sq_err"] / ss_tot if ss_tot > 0 else float("nan")
    return {"valid_count": n, "R": r, "R2": r2, "MAE": mae, "RMSE": rmse}


# Internal helper for safe div.
def _safe_div(num: float, den: float) -> float:
    return float(num / den) if den > 0 else float("nan")


# Internal helper for binary scores.
def _binary_scores(pred_positive: np.ndarray, truth_positive: np.ndarray) -> tuple[float, float, float]:
    tp = float(np.logical_and(pred_positive, truth_positive).sum())
    fp = float(np.logical_and(pred_positive, ~truth_positive).sum())
    fn = float(np.logical_and(~pred_positive, truth_positive).sum())
    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    f1 = _safe_div(2.0 * precision * recall, precision + recall) if np.isfinite(precision) and np.isfinite(recall) else float("nan")
    return precision, recall, f1


# Compute diagnostic metrics for concentration deltas.
def delta_diagnostic_metrics(
    pred_delta: np.ndarray,
    truth_delta: np.ndarray,
    mask: np.ndarray,
    active_threshold: float,
) -> dict[str, float | int]:
    valid = np.isfinite(pred_delta) & np.isfinite(truth_delta) & (mask > 0)
    valid_count = int(valid.sum())
    if valid_count <= 0:
        return {
            "active_valid_count": 0,
            "active_fraction": float("nan"),
            "active_R": float("nan"),
            "active_R2": float("nan"),
            "active_MAE": float("nan"),
            "active_RMSE": float("nan"),
            "sign_accuracy": float("nan"),
            "pos_precision": float("nan"),
            "pos_recall": float("nan"),
            "pos_f1": float("nan"),
            "neg_precision": float("nan"),
            "neg_recall": float("nan"),
            "neg_f1": float("nan"),
            "amplitude_ratio": float("nan"),
            "pred_delta_std": float("nan"),
            "truth_delta_std": float("nan"),
            "active_pred_std": float("nan"),
            "active_truth_std": float("nan"),
            "pred_abs_mean": float("nan"),
            "truth_abs_mean": float("nan"),
        }

    p_all = pred_delta[valid].astype(np.float64, copy=False)
    y_all = truth_delta[valid].astype(np.float64, copy=False)
    pred_std = float(np.std(p_all))
    truth_std = float(np.std(y_all))
    active = valid & (np.abs(truth_delta) >= float(active_threshold))
    active_count = int(active.sum())
    active_fraction = active_count / valid_count
    active_stats = new_stats()
    update_stats(active_stats, pred_delta, truth_delta, active.astype(np.float32))
    active_metrics = finalize_stats(active_stats)

    if active_count > 0:
        p = pred_delta[active].astype(np.float64, copy=False)
        y = truth_delta[active].astype(np.float64, copy=False)
        sign_accuracy = float((np.sign(p) == np.sign(y)).mean())
        pos_precision, pos_recall, pos_f1 = _binary_scores(p > 0.0, y > 0.0)
        neg_precision, neg_recall, neg_f1 = _binary_scores(p < 0.0, y < 0.0)
        active_pred_std = float(np.std(p))
        active_truth_std = float(np.std(y))
        amplitude_ratio = _safe_div(active_pred_std, active_truth_std)
        pred_abs_mean = float(np.mean(np.abs(p)))
        truth_abs_mean = float(np.mean(np.abs(y)))
    else:
        sign_accuracy = float("nan")
        pos_precision = pos_recall = pos_f1 = float("nan")
        neg_precision = neg_recall = neg_f1 = float("nan")
        active_pred_std = active_truth_std = amplitude_ratio = float("nan")
        pred_abs_mean = truth_abs_mean = float("nan")

    return {
        "active_valid_count": active_count,
        "active_fraction": active_fraction,
        "active_R": active_metrics["R"],
        "active_R2": active_metrics["R2"],
        "active_MAE": active_metrics["MAE"],
        "active_RMSE": active_metrics["RMSE"],
        "sign_accuracy": sign_accuracy,
        "pos_precision": pos_precision,
        "pos_recall": pos_recall,
        "pos_f1": pos_f1,
        "neg_precision": neg_precision,
        "neg_recall": neg_recall,
        "neg_f1": neg_f1,
        "amplitude_ratio": amplitude_ratio,
        "pred_delta_std": pred_std,
        "truth_delta_std": truth_std,
        "active_pred_std": active_pred_std,
        "active_truth_std": active_truth_std,
        "pred_abs_mean": pred_abs_mean,
        "truth_abs_mean": truth_abs_mean,
    }


# Return plotting limits for CO2 concentration.
def co2_range(pred: np.ndarray, truth: np.ndarray, mask: np.ndarray) -> tuple[float, float]:
    pred_plot = pred.copy()
    truth_plot = truth.copy()
    pred_plot[mask <= 0] = np.nan
    truth_plot[mask <= 0] = np.nan
    vals = np.concatenate([pred_plot[np.isfinite(pred_plot)], truth_plot[np.isfinite(truth_plot)]])
    if vals.size == 0:
        return 0.0, 1.0
    vmin = float(np.nanpercentile(vals, 1))
    vmax = float(np.nanpercentile(vals, 99))
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
        vmin = float(np.nanmin(vals))
        vmax = float(np.nanmax(vals))
    if vmax <= vmin:
        vmax = vmin + 1.0
    return vmin, vmax


# Save visual to disk.
def save_visual(
    path: Path,
    topo_layer: np.ndarray,
    pred: np.ndarray,
    truth: np.ndarray,
    mask: np.ndarray,
    metadata: dict[str, object],
    layer_metrics: dict[str, float | int],
) -> tuple[float, float]:
    pred_plot = pred.copy()
    truth_plot = truth.copy()
    err_plot = np.abs(pred - truth)
    pred_plot[mask <= 0] = np.nan
    truth_plot[mask <= 0] = np.nan
    err_plot[mask <= 0] = np.nan

    vmin, vmax = co2_range(pred, truth, mask)
    err_vals = err_plot[np.isfinite(err_plot)]
    err_vmax = float(np.nanpercentile(err_vals, 99)) if err_vals.size else 1.0
    if err_vmax <= 0:
        err_vmax = 1.0

    topo_cmap = ListedColormap(["#f8fafc", "#8b5e34", "#111827", "#7c3aed"])
    topo_norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], topo_cmap.N)
    co2_cmap = safe_cmap("turbo", "jet")
    co2_cmap.set_bad("#ffffff")
    err_cmap = plt.get_cmap("magma").copy()
    err_cmap.set_bad("#ffffff")

    fig, axes = plt.subplots(1, 4, figsize=(16, 4.4), dpi=180, constrained_layout=True)
    height_value = metadata.get("height_m", None)
    if height_value is None:
        height_text = "height=unknown"
    else:
        height_m = float(height_value)
        if np.isfinite(height_m) and abs(height_m - round(height_m)) < 1e-3:
            height_text = f"height={int(round(height_m))} m"
        elif np.isfinite(height_m):
            height_text = f"height={height_m:.1f} m"
        else:
            height_text = "height=unknown"

    fig.suptitle(
        f"{metadata['job']} | month {metadata['month']} | time {metadata['time_index']} | "
        f"{height_text} | {metadata['inference_label']} | "
        f"R={layer_metrics['R']:.3f}, R2={layer_metrics['R2']:.3f}, "
        f"MAE={layer_metrics['MAE']:.3f}, RMSE={layer_metrics['RMSE']:.3f}",
        fontsize=11,
        fontweight="bold",
    )
    im0 = axes[0].imshow(topo_layer, cmap=topo_cmap, norm=topo_norm, interpolation="nearest")
    axes[0].set_title("Topography")
    cbar0 = fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04, ticks=[0, 1, 2, 3])
    cbar0.ax.set_yticklabels(["0 non-topo", "1 terrain", "2 building", "3 other"])
    im1 = axes[1].imshow(pred_plot, cmap=co2_cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
    axes[1].set_title("CO2 prediction")
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
    im2 = axes[2].imshow(truth_plot, cmap=co2_cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
    axes[2].set_title("CO2 ground truth")
    fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)
    im3 = axes[3].imshow(err_plot, cmap=err_cmap, vmin=0.0, vmax=err_vmax, interpolation="nearest")
    axes[3].set_title("|prediction - truth|")
    fig.colorbar(im3, ax=axes[3], fraction=0.046, pad=0.04)
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
    fig.text(
        0.5,
        0.01,
        f"CO2 prediction/truth share range {vmin:.2f}-{vmax:.2f}; topography-masked cells are hidden.",
        ha="center",
        fontsize=8.5,
        color="#475569",
    )
    fig.savefig(path)
    plt.close(fig)
    return vmin, vmax


# Return plotting limits for CO2 change.
def delta_range(pred_delta: np.ndarray, truth_delta: np.ndarray, mask: np.ndarray) -> tuple[float, float]:
    pred_plot = pred_delta.copy()
    truth_plot = truth_delta.copy()
    pred_plot[mask <= 0] = np.nan
    truth_plot[mask <= 0] = np.nan
    vals = np.concatenate([pred_plot[np.isfinite(pred_plot)], truth_plot[np.isfinite(truth_plot)]])
    if vals.size == 0:
        return -1.0, 1.0
    abs_lim = float(np.nanpercentile(np.abs(vals), 99))
    if not np.isfinite(abs_lim) or abs_lim <= 0:
        abs_lim = float(np.nanmax(np.abs(vals))) if vals.size else 1.0
    if not np.isfinite(abs_lim) or abs_lim <= 0:
        abs_lim = 1.0
    return -abs_lim, abs_lim


# Save delta visual to disk.
def save_delta_visual(
    path: Path,
    topo_layer: np.ndarray,
    prev_co2: np.ndarray,
    pred_delta: np.ndarray,
    truth_delta: np.ndarray,
    mask: np.ndarray,
    metadata: dict[str, object],
    layer_metrics: dict[str, float | int],
) -> tuple[float, float]:
    prev_plot = prev_co2.copy()
    pred_plot = pred_delta.copy()
    truth_plot = truth_delta.copy()
    err_plot = np.abs(pred_delta - truth_delta)
    prev_plot[mask <= 0] = np.nan
    pred_plot[mask <= 0] = np.nan
    truth_plot[mask <= 0] = np.nan
    err_plot[mask <= 0] = np.nan

    vmin, vmax = delta_range(pred_delta, truth_delta, mask)
    prev_vals = prev_plot[np.isfinite(prev_plot)]
    if prev_vals.size:
        prev_vmin = float(np.nanpercentile(prev_vals, 1))
        prev_vmax = float(np.nanpercentile(prev_vals, 99))
    else:
        prev_vmin, prev_vmax = 0.0, 1.0
    if not np.isfinite(prev_vmin) or not np.isfinite(prev_vmax) or prev_vmax <= prev_vmin:
        prev_vmin, prev_vmax = 0.0, 1.0
    err_vals = err_plot[np.isfinite(err_plot)]
    err_vmax = float(np.nanpercentile(err_vals, 99)) if err_vals.size else 1.0
    if not np.isfinite(err_vmax) or err_vmax <= 0:
        err_vmax = 1.0

    topo_cmap = ListedColormap(["#f8fafc", "#8b5e34", "#111827", "#7c3aed"])
    topo_norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], topo_cmap.N)
    co2_cmap = safe_cmap("turbo", "jet")
    co2_cmap.set_bad("#ffffff")
    delta_cmap = plt.get_cmap("coolwarm").copy()
    delta_cmap.set_bad("#ffffff")
    err_cmap = plt.get_cmap("magma").copy()
    err_cmap.set_bad("#ffffff")

    fig, axes = plt.subplots(1, 5, figsize=(19, 4.4), dpi=180, constrained_layout=True)
    height_value = metadata.get("height_m", None)
    if height_value is None:
        height_text = "height=unknown"
    else:
        height_m = float(height_value)
        if np.isfinite(height_m) and abs(height_m - round(height_m)) < 1e-3:
            height_text = f"height={int(round(height_m))} m"
        elif np.isfinite(height_m):
            height_text = f"height={height_m:.1f} m"
        else:
            height_text = "height=unknown"

    fig.suptitle(
        f"Delta target: CO2(t)-CO2(t-1) | {metadata['job']} | month {metadata['month']} | "
        f"time {metadata['time_index']} | {height_text} | "
        f"R={layer_metrics['R']:.3f}, R2={layer_metrics['R2']:.3f}, "
        f"MAE={layer_metrics['MAE']:.3f}, RMSE={layer_metrics['RMSE']:.3f}",
        fontsize=11,
        fontweight="bold",
    )
    im0 = axes[0].imshow(topo_layer, cmap=topo_cmap, norm=topo_norm, interpolation="nearest")
    axes[0].set_title("Topography")
    cbar0 = fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04, ticks=[0, 1, 2, 3])
    cbar0.ax.set_yticklabels(["0 non-topo", "1 terrain", "2 building", "3 other"])
    im1 = axes[1].imshow(prev_plot, cmap=co2_cmap, vmin=prev_vmin, vmax=prev_vmax, interpolation="nearest")
    axes[1].set_title("CO2 previous t-1")
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
    im2 = axes[2].imshow(pred_plot, cmap=delta_cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
    axes[2].set_title("Predicted delta")
    fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)
    im3 = axes[3].imshow(truth_plot, cmap=delta_cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
    axes[3].set_title("Truth delta")
    fig.colorbar(im3, ax=axes[3], fraction=0.046, pad=0.04)
    im4 = axes[4].imshow(err_plot, cmap=err_cmap, vmin=0.0, vmax=err_vmax, interpolation="nearest")
    axes[4].set_title("|pred delta - truth delta|")
    fig.colorbar(im4, ax=axes[4], fraction=0.046, pad=0.04)
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
    fig.text(
        0.5,
        0.01,
        f"Delta panels share symmetric range {vmin:.2f} to {vmax:.2f} ppm; masked cells are hidden. "
        "If predicted delta is near zero everywhere, the final CO2 is mostly persistence.",
        ha="center",
        fontsize=8.5,
        color="#475569",
    )
    fig.savefig(path)
    plt.close(fig)
    return vmin, vmax


# Save v28 diagnostic visual to disk.
def save_v28_diagnostic_visual(
    path: Path,
    prev_co2: np.ndarray,
    physics_delta: np.ndarray,
    model_correction: np.ndarray,
    pred_delta: np.ndarray,
    truth_delta: np.ndarray,
    mask: np.ndarray,
    metadata: dict[str, object],
    final_metrics: dict[str, float | int],
    physics_metrics: dict[str, float | int],
    correction_metrics: dict[str, float | int],
) -> tuple[float, float]:
    truth_correction = truth_delta - physics_delta
    err_plot = np.abs(pred_delta - truth_delta)

    prev_plot = prev_co2.copy()
    physics_plot = physics_delta.copy()
    model_corr_plot = model_correction.copy()
    truth_corr_plot = truth_correction.copy()
    pred_plot = pred_delta.copy()
    truth_plot = truth_delta.copy()
    for arr in (prev_plot, physics_plot, model_corr_plot, truth_corr_plot, pred_plot, truth_plot, err_plot):
        arr[mask <= 0] = np.nan

    delta_vals = np.concatenate(
        [
            physics_plot[np.isfinite(physics_plot)],
            model_corr_plot[np.isfinite(model_corr_plot)],
            truth_corr_plot[np.isfinite(truth_corr_plot)],
            pred_plot[np.isfinite(pred_plot)],
            truth_plot[np.isfinite(truth_plot)],
        ]
    )
    if delta_vals.size:
        abs_lim = float(np.nanpercentile(np.abs(delta_vals), 99))
    else:
        abs_lim = 1.0
    if not np.isfinite(abs_lim) or abs_lim <= 0:
        abs_lim = 1.0
    vmin, vmax = -abs_lim, abs_lim

    prev_vals = prev_plot[np.isfinite(prev_plot)]
    if prev_vals.size:
        prev_vmin = float(np.nanpercentile(prev_vals, 1))
        prev_vmax = float(np.nanpercentile(prev_vals, 99))
    else:
        prev_vmin, prev_vmax = 0.0, 1.0
    if not np.isfinite(prev_vmin) or not np.isfinite(prev_vmax) or prev_vmax <= prev_vmin:
        prev_vmin, prev_vmax = 0.0, 1.0

    err_vals = err_plot[np.isfinite(err_plot)]
    err_vmax = float(np.nanpercentile(err_vals, 99)) if err_vals.size else 1.0
    if not np.isfinite(err_vmax) or err_vmax <= 0:
        err_vmax = 1.0

    co2_cmap = safe_cmap("turbo", "jet")
    co2_cmap.set_bad("#ffffff")
    delta_cmap = plt.get_cmap("coolwarm").copy()
    delta_cmap.set_bad("#ffffff")
    err_cmap = plt.get_cmap("magma").copy()
    err_cmap.set_bad("#ffffff")

    fig, axes = plt.subplots(1, 7, figsize=(25, 4.2), dpi=180, constrained_layout=True)
    height_value = metadata.get("height_m", None)
    if height_value is None:
        height_text = "height=unknown"
    else:
        height_m = float(height_value)
        if np.isfinite(height_m) and abs(height_m - round(height_m)) < 1e-3:
            height_text = f"height={int(round(height_m))} m"
        elif np.isfinite(height_m):
            height_text = f"height={height_m:.1f} m"
        else:
            height_text = "height=unknown"

    fig.suptitle(
        f"V28 diagnostic | {metadata['job']} | month {metadata['month']} | time {metadata['time_index']} | "
        f"{height_text} | final delta R={final_metrics['R']:.3f}, MAE={final_metrics['MAE']:.3f}, RMSE={final_metrics['RMSE']:.3f} | "
        f"physics R={physics_metrics['R']:.3f} | correction-target R={correction_metrics['R']:.3f}",
        fontsize=10.5,
        fontweight="bold",
    )

    panels = [
        ("CO2 previous t-1", prev_plot, co2_cmap, prev_vmin, prev_vmax),
        ("Physics adv delta", physics_plot, delta_cmap, vmin, vmax),
        ("Model correction", model_corr_plot, delta_cmap, vmin, vmax),
        ("Truth correction", truth_corr_plot, delta_cmap, vmin, vmax),
        ("Final pred delta", pred_plot, delta_cmap, vmin, vmax),
        ("Truth delta", truth_plot, delta_cmap, vmin, vmax),
        ("|final delta error|", err_plot, err_cmap, 0.0, err_vmax),
    ]
    for ax, (title, arr, cmap, lo, hi) in zip(axes, panels):
        im = ax.imshow(arr, cmap=cmap, vmin=lo, vmax=hi, interpolation="nearest")
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.text(
        0.5,
        0.01,
        f"Delta-like panels share symmetric range {vmin:.2f} to {vmax:.2f} ppm. "
        "Final pred delta = physics adv delta + model correction; truth correction = truth delta - physics adv delta.",
        ha="center",
        fontsize=8.5,
        color="#475569",
    )
    fig.savefig(path)
    plt.close(fig)
    return vmin, vmax


# Entry point for the command-line workflow.
def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate V15 z-focused corrected-residual test split with 256x256 figures")
    parser.add_argument("--jobs-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--topography-path", default="/data/linfeng/palm/london_camden_2019_new/JOBS/cam07_175vm_topo_surf_N02.000.nc")
    parser.add_argument("--single-job-name", default=None, help="Optional label when --jobs-root points directly at one PALM job")
    parser.add_argument("--single-job-month", type=int, default=None, help="Optional month override for a direct single-job root")
    parser.add_argument("--train-fraction", type=float, default=0.70)
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--samples-per-job", type=int, default=3)
    parser.add_argument("--max-samples", type=int, default=0, help="Debug only: limit total test samples")
    parser.add_argument("--visual-samples", type=int, default=4)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--patch-d", type=int, default=16)
    parser.add_argument("--patch-h", type=int, default=256)
    parser.add_argument("--patch-w", type=int, default=256)
    parser.add_argument("--tile-h", type=int, default=64)
    parser.add_argument("--tile-w", type=int, default=64)
    parser.add_argument("--tile-batch-size", type=int, default=8)
    parser.add_argument("--inference-mode", choices=["direct", "tile"], default="direct")
    parser.add_argument("--visual-local-z", default="0,4,8,11,15")
    parser.add_argument("--fixed-z0", type=int, default=None, help="Optional fixed patch z start for reproducible domain tests")
    parser.add_argument("--eval-layer-min", type=int, default=None, help="Only sample/score global z >= this layer")
    parser.add_argument("--eval-layer-max", type=int, default=None, help="Only sample/score global z <= this layer")
    parser.add_argument("--eval-min-layer-overlap", type=int, default=8, help="Minimum evaluated layers required in each sampled patch")
    parser.add_argument(
        "--mask-truth-below",
        type=float,
        default=None,
        help="Optional evaluation-only mask for invalid cells stored as unrealistically low CO2, e.g. 100 for Richmond.",
    )
    parser.add_argument(
        "--eval-require-full-layer-range",
        action="store_true",
        help="Choose z0 so the whole 16-layer patch stays inside --eval-layer-min/max when possible",
    )
    parser.add_argument("--v13-sidecar-root", default=None, help="V13 sidecar root; used for normalization/global context")
    parser.add_argument("--v22-prev-sidecar-root", default=None, help="V22 previous-CO2 sidecar root; used for autoregressive normalization")
    parser.add_argument("--v14-normalization-root", default=None, help="V14 normalization root; overrides checkpoint metadata")
    parser.add_argument("--target-normalization-path", default=None, help="V16 target residual normalization JSON")
    parser.add_argument(
        "--target-normalization-mode",
        choices=["none", "month", "background_bin"],
        default=None,
        help="Override V16 target normalization mode",
    )
    parser.add_argument("--v13-global-size", type=int, default=0, help="Override full-domain context sidecar size")
    parser.add_argument("--v13-global-sample-size", type=int, default=0, help="Override local sampling size for global context")
    parser.add_argument(
        "--save-delta-visuals",
        action="store_true",
        help="For V22 autoregressive models, also save predicted/true delta_CO2 visualizations.",
    )
    parser.add_argument(
        "--delta-active-threshold",
        type=float,
        default=0.75,
        help="Threshold in ppm for active-delta diagnostics such as sign accuracy and active-region R.",
    )
    parser.add_argument(
        "--save-v28-diagnostic-visuals",
        action="store_true",
        help="For V28, save previous CO2, physics advection delta, model correction, truth correction, final delta, and error.",
    )
    parser.add_argument(
        "--force-fluid-mask-ones",
        action="store_true",
        help="Input ablation: replace fluid_mask/topography context channels with ones while keeping evaluation masking unchanged.",
    )
    parser.add_argument(
        "--disable-topography-metric-mask",
        action="store_true",
        help="Metric ablation: do not exclude non-fluid cells with the topography mask; layer-range and truth-threshold masks still apply.",
    )
    parser.add_argument(
        "--exclude-months",
        default="",
        help="Comma-separated months to skip during evaluation, e.g. 11,12",
    )
    args = parser.parse_args()
    exclude_months = set(parse_int_list(args.exclude_months))

    out_dir = Path(args.out_dir)
    vis_dir = out_dir / "visualizations"
    vis_dir.mkdir(parents=True, exist_ok=True)
    delta_vis_dir = out_dir / "delta_visualizations"
    if args.save_delta_visuals:
        delta_vis_dir.mkdir(parents=True, exist_ok=True)
    v28_diag_dir = out_dir / "v28_diagnostic_visualizations"
    if args.save_v28_diagnostic_visuals:
        v28_diag_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)
    patch_size = (args.patch_d, args.patch_h, args.patch_w)
    tile_size = (args.patch_d, args.tile_h, args.tile_w)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    in_channels = int(checkpoint.get("in_channels", 18))
    target_mode = str(checkpoint.get("target_mode", "absolute"))
    architecture = str(checkpoint.get("architecture", ""))
    model_variant = str(checkpoint.get("model_variant", "coarse_local"))
    use_global_context = bool(checkpoint.get("use_global_context", int(checkpoint.get("global_channels", 0)) > 0))
    is_zfocused = (
        architecture.startswith("v14_")
        or architecture.startswith("v15_")
        or architecture.startswith("v16_")
        or architecture.startswith("v17_")
        or architecture.startswith("v18_")
        or architecture.startswith("v19_")
        or architecture.startswith("v20_")
        or architecture.startswith("v21_")
        or architecture.startswith("v22_")
        or architecture.startswith("v28_")
        or architecture.startswith("v35_")
        or architecture.startswith("v37_")
        or architecture.startswith("v38_")
    )
    target_is_normalized = target_mode == "corrected_residual_normalized" or bool(checkpoint.get("target_normalization_path"))
    is_v28_advection_correction = target_mode == "autoregressive_advection_correction"
    is_autoregressive = target_mode in {"autoregressive_delta", "autoregressive_advection_correction"}
    if is_autoregressive and args.inference_mode != "direct":
        raise RuntimeError("V22 autoregressive evaluation currently supports --inference-mode direct only")
    is_v13 = (
        target_mode in {
            "corrected_residual",
            "corrected_residual_normalized",
            "autoregressive_delta",
            "autoregressive_advection_correction",
        }
        or architecture.startswith("v13_")
        or is_zfocused
    )
    eval_layer_min = args.eval_layer_min
    eval_layer_max = args.eval_layer_max
    if is_zfocused:
        if eval_layer_min is None and checkpoint.get("v14_layer_min") is not None:
            eval_layer_min = int(checkpoint["v14_layer_min"])
        if eval_layer_max is None and checkpoint.get("v14_layer_max") is not None:
            eval_layer_max = int(checkpoint["v14_layer_max"])
    if (eval_layer_min is None) != (eval_layer_max is None):
        raise RuntimeError("--eval-layer-min and --eval-layer-max must be provided together")
    if eval_layer_min is not None and eval_layer_max is not None and eval_layer_max < eval_layer_min:
        raise RuntimeError("--eval-layer-max must be >= --eval-layer-min")
    if is_v13 and args.inference_mode != "direct":
        raise RuntimeError("V13 evaluation currently supports --inference-mode direct only")
    v13_sidecar_root = args.v13_sidecar_root or checkpoint.get("v13_sidecar_root")
    norm_root = args.v14_normalization_root or checkpoint.get("v14_normalization_root") or v13_sidecar_root
    v13_norm = None
    target_norm = None
    target_norm_mode = args.target_normalization_mode or str(checkpoint.get("target_normalization_mode", "none"))
    target_norm_path = args.target_normalization_path or checkpoint.get("target_normalization_path")
    v13_global_size = int(args.v13_global_size)
    v13_global_sample_size = int(args.v13_global_sample_size or checkpoint.get("v13_global_sample_size", 64))
    if is_v13:
        if not v13_sidecar_root:
            raise RuntimeError("V13 checkpoint requires --v13-sidecar-root or checkpoint['v13_sidecar_root']")
        if not norm_root:
            raise RuntimeError("Corrected-residual checkpoint requires a normalization root")
        v13_norm = load_v13_normalization(str(norm_root))
        if target_is_normalized:
            target_norm = load_target_normalization(str(target_norm_path))
            if target_norm_mode in (None, "none"):
                target_norm_mode = str(target_norm.get("mode", "month"))
        if v13_global_size <= 0:
            root_manifest = os.path.join(str(v13_sidecar_root), "manifest.json")
            if os.path.exists(root_manifest):
                with open(root_manifest, "r", encoding="utf-8") as f:
                    v13_global_size = int(json.load(f).get("global_size", 80))
            else:
                v13_global_size = 80
    coord_channels = list(checkpoint.get("coord_channels", []))
    appended_channels = list(checkpoint.get("appended_channels", coord_channels))
    dropped_input_channels = [int(v) for v in checkpoint.get("dropped_input_channels", [])]
    v22_prev_sidecar_root = args.v22_prev_sidecar_root or checkpoint.get("v22_prev_sidecar_root")
    v22_prev_norm = load_v22_prev_normalization(str(v22_prev_sidecar_root)) if is_autoregressive and v22_prev_sidecar_root else None
    if is_autoregressive and v22_prev_norm is None:
        raise RuntimeError("V22 autoregressive checkpoint requires --v22-prev-sidecar-root or checkpoint['v22_prev_sidecar_root']")
    if model_variant == "v38_event_texture_context_v7" or "event_texture_context_v7" in architecture:
        model = V38EventTextureContextV7UNet3D(
            in_channels=in_channels,
            out_channels=1,
            base_channels=int(checkpoint.get("base_channels", 32)),
            global_channels=int(checkpoint.get("global_channels", 0)),
            global_feature_channels=int(checkpoint.get("global_feature_channels", 8)),
            context_correction_scale=float(checkpoint.get("high_residual_scale", 1.0)),
            high_delta_scale=float(checkpoint.get("high_residual_scale", 1.0)),
            min_high_gate=float(checkpoint.get("min_high_gate", 0.20)),
        ).to(device)
    elif model_variant == "v37_hard_pattern_context_v7" or "hard_pattern_context_v7" in architecture:
        model = V37HardPatternContextV7UNet3D(
            in_channels=in_channels,
            out_channels=1,
            base_channels=int(checkpoint.get("base_channels", 32)),
            global_channels=int(checkpoint.get("global_channels", 0)),
            global_feature_channels=int(checkpoint.get("global_feature_channels", 8)),
            context_correction_scale=float(checkpoint.get("high_residual_scale", 1.0)),
            high_delta_scale=float(checkpoint.get("high_residual_scale", 1.0)),
        ).to(device)
    elif model_variant == "v35_multitask_context_v7" or "multitask_context_v7" in architecture:
        model = V35MultiTaskContextV7UNet3D(
            in_channels=in_channels,
            out_channels=1,
            base_channels=int(checkpoint.get("base_channels", 32)),
            global_channels=int(checkpoint.get("global_channels", 0)),
            global_feature_channels=int(checkpoint.get("global_feature_channels", 8)),
            context_correction_scale=float(checkpoint.get("high_residual_scale", 1.0)),
        ).to(device)
    elif model_variant in {"v20_context_v7", "v21_context_v7", "v22_context_v7"} or "context_v7" in architecture:
        model = V20ContextV7UNet3D(
            in_channels=in_channels,
            out_channels=1,
            base_channels=int(checkpoint.get("base_channels", 32)),
            global_channels=int(checkpoint.get("global_channels", 0)),
            global_feature_channels=int(checkpoint.get("global_feature_channels", 8)),
            context_correction_scale=float(checkpoint.get("high_residual_scale", 1.0)),
        ).to(device)
    elif model_variant == "v7_style" or "v7_style" in architecture:
        model = V7StyleUNet3D(
            in_channels=in_channels,
            out_channels=1,
            base_channels=int(checkpoint.get("base_channels", 32)),
        ).to(device)
    else:
        model = UNet3D(
            in_channels=in_channels,
            out_channels=1,
            base_channels=int(checkpoint.get("base_channels", 24)),
            bg_channel_idx=checkpoint.get("bg_channel_idx", 1),
            gate_channel_idx=checkpoint.get("height_gate_channel_idx", None),
            coarse_pool=int(checkpoint.get("coarse_pool", 4)),
            min_texture_gate=float(checkpoint.get("min_texture_gate", 0.05)),
            learned_texture_gate=bool(checkpoint.get("learned_texture_gate", True)),
            high_residual_scale=float(checkpoint.get("high_residual_scale", 1.0)),
            global_channels=int(checkpoint.get("global_channels", 0)),
            global_feature_channels=int(checkpoint.get("global_feature_channels", 8)),
        ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    all_jobs = discover_jobs(args.jobs_root)
    job_files = []
    for name in all_jobs:
        jf = locate_job_files(args.jobs_root, name)
        if jf is not None:
            job_files.append(jf)
    if not job_files:
        jf = locate_direct_job_root(args.jobs_root, job_name=args.single_job_name, month=args.single_job_month)
        if jf is not None:
            job_files.append(jf)
    if exclude_months:
        before_count = len(job_files)
        job_files = [jf for jf in job_files if int(jf.month) not in exclude_months]
        print(
            f"Excluded months from evaluation: {sorted(exclude_months)} "
            f"jobs {before_count}->{len(job_files)}",
            flush=True,
        )
    if not job_files:
        raise RuntimeError("No valid job files found")

    with safe_open_dataset(args.topography_path) as dstopo:
        topo_values = to_numpy(dstopo["topo_all"]) if "topo_all" in dstopo else None

    samples: list[dict[str, object]] = []
    for jf in job_files:
        with safe_open_dataset(jf.out3d) as ds3d, safe_open_dataset(jf.av3d) as dsav:
            n_t = min(int(ds3d.sizes.get("time", 0)), int(dsav.sizes.get("time", 0)))
            times = split_time_indices(max(1, n_t), "test", args.train_fraction, args.val_fraction)
            if is_autoregressive:
                times = [t for t in times if int(t) > 0]
            target = dsav[TARGET_VAR].isel(time=times[0] if times else 0)
            z_name = "zu_3d" if "zu_3d" in target.dims else target.dims[0]
            nz = int(target.sizes[z_name])
            ny = int(target.sizes["y"])
            nx = int(target.sizes["x"])
        if not times:
            continue
        for _ in range(args.samples_per_job):
            if args.fixed_z0 is not None:
                z0_choice = max(0, min(int(args.fixed_z0), max(0, nz - args.patch_d)))
            elif eval_layer_min is not None and eval_layer_max is not None:
                z0_choice = choose_layer_focused_z_start(
                    rng,
                    nz,
                    args.patch_d,
                    eval_layer_min,
                    eval_layer_max,
                    args.eval_min_layer_overlap,
                    args.eval_require_full_layer_range,
                )
                if z0_choice is None:
                    continue
            else:
                z0_choice = choose_start(rng, nz, args.patch_d)
            samples.append(
                {
                    "jf": jf,
                    "time_index": rng.choice(times),
                    "z0": z0_choice,
                    "y0": choose_start(rng, ny, args.patch_h),
                    "x0": choose_start(rng, nx, args.patch_w),
                    "nz": nz,
                    "ny": ny,
                    "nx": nx,
                }
            )
    if args.max_samples > 0:
        samples = samples[: args.max_samples]

    stats = new_stats()
    sample_records: list[dict[str, object]] = []
    visual_records: list[dict[str, object]] = []
    delta_visual_records: list[dict[str, object]] = []
    v28_diag_records: list[dict[str, object]] = []
    local_z_values = parse_int_list(args.visual_local_z)

    for sample_idx, sample in enumerate(samples):
        jf = sample["jf"]
        t_idx = int(sample["time_index"])
        z0 = int(sample["z0"])
        y0 = int(sample["y0"])
        x0 = int(sample["x0"])
        z1 = z0 + args.patch_d
        y1 = y0 + args.patch_h
        x1 = x0 + args.patch_w
        patch_heights = np.arange(z0, z1, dtype=np.float32)
        pred_delta = None
        truth_delta = None
        prev_co2_for_visual = None
        physics_delta_for_visual = None
        model_correction_for_visual = None

        with safe_open_dataset(jf.chemistry) as dsch, safe_open_dataset(jf.dynamic) as dsdyn, safe_open_dataset(
            jf.static
        ) as dsst, safe_open_dataset(jf.out3d) as ds3d, safe_open_dataset(jf.av3d) as dsav:
            if args.inference_mode == "direct":
                x_full, truth = build_input_patch(dsch, dsdyn, dsst, ds3d, dsav, jf, t_idx, z0, y0, x0, patch_size)
                input_mask = build_topography_mask(topo_values, z0, z1, y0, y1, x0, x1, truth.shape)
                if is_v13:
                    assert v13_norm is not None
                    target = dsav[TARGET_VAR].isel(time=min(t_idx, int(dsav.sizes["time"]) - 1))
                    z_name = "zu_3d" if "zu_3d" in target.dims else target.dims[0]
                    target_z_coords = target.coords[z_name].isel({z_name: slice(z0, z1)}) if z_name in target.coords else None
                    if target_z_coords is not None:
                        target_heights = np.asarray(target_z_coords.values, dtype=np.float32)
                    else:
                        target_heights = np.arange(z0, z1, dtype=np.float32)
                    patch_heights = target_heights
                    target_seconds = float(target["time"].values) if "time" in target.coords else float(t_idx) * 1800.0
                    corrected_emission_full = corrected_emission_2d(dsch, target_seconds)
                    corrected_bg = corrected_bg_profile(dsdyn, target_seconds, target_heights)
                    corrected_emission_patch = corrected_emission_full[y0:y1, x0:x1].astype(np.float32, copy=False)
                    x_full, bg_3d = apply_v13_transform(
                        x_full,
                        input_mask,
                        corrected_emission_patch=corrected_emission_patch,
                        corrected_bg=corrected_bg,
                        z0=z0,
                        y0=y0,
                        x0=x0,
                        nz=int(sample["nz"]),
                        ny=int(sample["ny"]),
                        nx=int(sample["nx"]),
                        checkpoint=checkpoint,
                        normalization=v13_norm,
                        force_fluid_mask_ones=args.force_fluid_mask_ones,
                    )
                    prev_co2 = None
                    physics_delta = None
                    if is_autoregressive:
                        assert v22_prev_norm is not None
                        prev_co2 = load_co2_patch(dsav, t_idx - 1, z0, z1, y0, y1, x0, x1)
                        x_full = append_v22_prev_channel(x_full, prev_co2, v22_prev_norm)
                        if bool(checkpoint.get("v28_advection_features", False)):
                            x_full, physics_delta = append_v28_advection_features(
                                x_full,
                                prev_co2,
                                v13_norm,
                                checkpoint,
                            )
                    with torch.no_grad():
                        if use_global_context:
                            global_context = build_global_context(
                                corrected_emission=corrected_emission_full,
                                corrected_bg=corrected_bg,
                                ds3d=ds3d,
                                topo_values=None if args.force_fluid_mask_ones else topo_values,
                                t_idx=t_idx,
                                z0=z0,
                                z1=z1,
                                target_heights=target_z_coords,
                                ny=int(sample["ny"]),
                                nx=int(sample["nx"]),
                                out_size=v13_global_size,
                            )
                            global_context = normalize_v13_global_context(global_context, v13_norm)
                            if is_autoregressive:
                                assert v22_prev_norm is not None
                                prev_global = load_co2_global_lowres(
                                    dsav,
                                    t_idx - 1,
                                    z0,
                                    z1,
                                    int(sample["ny"]),
                                    int(sample["nx"]),
                                    v13_global_size,
                                )
                                global_context = append_v22_prev_global(global_context, prev_global, v22_prev_norm)
                            global_grid = v13_global_grid(x_full, list(v13_norm["local_channels"]), v13_global_sample_size)
                            x_model = keep_input_channels_for_checkpoint(x_full, checkpoint)
                            xb = torch.from_numpy(x_model[None]).float().to(device, non_blocking=True)
                            gb = torch.from_numpy(global_context[None]).float().to(device, non_blocking=True)
                            gridb = torch.from_numpy(global_grid[None]).float().to(device, non_blocking=True)
                            model_out = model(xb, global_context=gb, global_grid=gridb)
                        else:
                            x_model = keep_input_channels_for_checkpoint(x_full, checkpoint)
                            xb = torch.from_numpy(x_model[None]).float().to(device, non_blocking=True)
                            model_out = model(xb)
                        pred_residual = model_out["final"] if isinstance(model_out, dict) else model_out
                        pred_residual_np = pred_residual.detach().cpu().numpy()[0, 0]
                        if target_norm is not None:
                            selected_layers = np.ones_like(corrected_bg, dtype=bool)
                            if eval_layer_min is not None and eval_layer_max is not None:
                                global_layers = np.arange(z0, z1, dtype=np.int64)
                                selected_layers = (global_layers >= eval_layer_min) & (global_layers <= eval_layer_max)
                            bg_for_group = corrected_bg[selected_layers] if selected_layers.any() else corrected_bg
                            group_key = target_norm_group_key(target_norm, target_norm_mode, jf.month, bg_for_group)
                            target_mean, target_std = target_norm_mean_std(target_norm, group_key)
                            pred_residual_np = pred_residual_np * float(target_std) + float(target_mean)
                        if is_autoregressive:
                            assert prev_co2 is not None
                            if is_v28_advection_correction:
                                if physics_delta is None:
                                    raise RuntimeError("V28 advection correction checkpoint is missing physics delta during evaluation")
                                model_correction_for_visual = pred_residual_np.astype(np.float32, copy=False)
                                physics_delta_for_visual = physics_delta.astype(np.float32, copy=False)
                                pred_delta = (pred_residual_np + physics_delta).astype(np.float32, copy=False)
                            else:
                                pred_delta = pred_residual_np.astype(np.float32, copy=False)
                            truth_delta = (truth - prev_co2).astype(np.float32, copy=False)
                            prev_co2_for_visual = prev_co2.astype(np.float32, copy=False)
                            pred = pred_delta + prev_co2
                        else:
                            pred = pred_residual_np + bg_3d
                else:
                    x_full = drop_input_channels(x_full, dropped_input_channels)
                    if appended_channels or checkpoint.get("surface_gated_channels", []):
                        x_full = apply_v12_input_transform(
                            x_full,
                            input_mask,
                            z0=z0,
                            y0=y0,
                            x0=x0,
                            nz=int(sample["nz"]),
                            ny=int(sample["ny"]),
                            nx=int(sample["nx"]),
                            checkpoint=checkpoint,
                        )
                    x_model = keep_input_channels_for_checkpoint(x_full, checkpoint)
                    xb = torch.from_numpy(x_model[None]).float().to(device, non_blocking=True)
                    with torch.no_grad():
                        pred = model(xb).detach().cpu().numpy()[0, 0]
                    if target_mode == "residual":
                        pred = pred + x_full[1]
            else:
                pred_sum = np.zeros(patch_size, dtype=np.float32)
                pred_count = np.zeros(patch_size, dtype=np.float32)
                batch_x: list[np.ndarray] = []
                batch_meta: list[tuple[int, int]] = []
                bg_tiles: list[np.ndarray] = []

                def flush_batch() -> None:
                    if not batch_x:
                        return
                    xb_np = np.stack(batch_x, axis=0)
                    xb = torch.from_numpy(xb_np).float().to(device, non_blocking=True)
                    with torch.no_grad():
                        pred_b = model(xb).detach().cpu().numpy()[:, 0]
                    if target_mode == "residual":
                        pred_b = pred_b + np.stack(bg_tiles, axis=0)[:, 0]
                    for pred_patch, (yy, xx) in zip(pred_b, batch_meta):
                        pred_sum[:, yy : yy + args.tile_h, xx : xx + args.tile_w] += pred_patch
                        pred_count[:, yy : yy + args.tile_h, xx : xx + args.tile_w] += 1.0
                    batch_x.clear()
                    batch_meta.clear()
                    bg_tiles.clear()

                for yy in range(0, args.patch_h, args.tile_h):
                    for xx in range(0, args.patch_w, args.tile_w):
                        x_tile, _ = build_input_patch(
                            dsch, dsdyn, dsst, ds3d, dsav, jf, t_idx, z0, y0 + yy, x0 + xx, tile_size
                        )
                        tile_mask = build_topography_mask(
                            topo_values,
                            z0,
                            z1,
                            y0 + yy,
                            y0 + yy + args.tile_h,
                            x0 + xx,
                            x0 + xx + args.tile_w,
                            x_tile.shape[-3:],
                        )
                        x_tile = drop_input_channels(x_tile, dropped_input_channels)
                        if appended_channels or checkpoint.get("surface_gated_channels", []):
                            x_tile = apply_v12_input_transform(
                                x_tile,
                                tile_mask,
                                z0=z0,
                                y0=y0 + yy,
                                x0=x0 + xx,
                                nz=int(sample["nz"]),
                                ny=int(sample["ny"]),
                                nx=int(sample["nx"]),
                                checkpoint=checkpoint,
                            )
                        bg_tiles.append(x_tile[1:2])
                        x_tile = keep_input_channels_for_checkpoint(x_tile, checkpoint)
                        batch_x.append(x_tile)
                        batch_meta.append((yy, xx))
                        if len(batch_x) >= args.tile_batch_size:
                            flush_batch()
                flush_batch()

                target = dsav[TARGET_VAR].isel(time=min(t_idx, int(dsav.sizes["time"]) - 1))
                z_name = "zu_3d" if "zu_3d" in target.dims else target.dims[0]
                target_z_coords = target.coords[z_name].isel({z_name: slice(z0, z1)}) if z_name in target.coords else None
                if target_z_coords is not None:
                    patch_heights = np.asarray(target_z_coords.values, dtype=np.float32)
                truth = to_numpy(target.isel({z_name: slice(z0, z1), "y": slice(y0, y1), "x": slice(x0, x1)}))
                truth = np.nan_to_num(truth, nan=0.0).astype(np.float32)
                pred = pred_sum / np.maximum(pred_count, 1.0)

        mask = build_topography_mask(topo_values, z0, z1, y0, y1, x0, x1, truth.shape)
        if args.disable_topography_metric_mask:
            metric_base = np.ones_like(mask, dtype=np.float32)
        else:
            metric_base = mask
        metric_mask = restrict_mask_to_layer_range(metric_base, z0, eval_layer_min, eval_layer_max)
        if args.mask_truth_below is not None:
            metric_mask = metric_mask * (truth >= float(args.mask_truth_below)).astype(np.float32)
        sample_stats = new_stats()
        update_stats(stats, pred, truth, metric_mask)
        update_stats(sample_stats, pred, truth, metric_mask)
        sample_metrics = finalize_stats(sample_stats)
        sample_records.append(
            {
                "sample": sample_idx,
                "job": jf.name,
                "month": jf.month,
                "time_index": t_idx,
                "z0": z0,
                "y0": y0,
                "x0": x0,
                **sample_metrics,
            }
        )

        if sample_idx < args.visual_samples:
            for local_z in local_z_values:
                if local_z < 0 or local_z >= pred.shape[0]:
                    continue
                global_z = z0 + local_z
                if eval_layer_min is not None and eval_layer_max is not None:
                    if global_z < eval_layer_min or global_z > eval_layer_max:
                        continue
                height_m = float(patch_heights[local_z]) if local_z < len(patch_heights) else float("nan")
                if topo_values is not None and global_z < topo_values.shape[0]:
                    topo_layer = topo_values[global_z, y0:y1, x0:x1].astype(np.float32)
                else:
                    topo_layer = np.zeros((args.patch_h, args.patch_w), dtype=np.float32)
                layer_stats = new_stats()
                update_stats(layer_stats, pred[local_z], truth[local_z], metric_mask[local_z])
                layer_metrics = finalize_stats(layer_stats)
                png_path = (
                    vis_dir
                    / f"sample_{sample_idx:03d}_globalz{global_z:03d}_{jf.name}_t{t_idx:03d}_y{y0:03d}_x{x0:03d}.png"
                )
                vmin, vmax = save_visual(
                    png_path,
                    topo_layer=topo_layer,
                    pred=pred[local_z],
                    truth=truth[local_z],
                    mask=metric_mask[local_z],
                    metadata={
                        "job": jf.name,
                        "month": jf.month,
                        "time_index": t_idx,
                        "global_z": global_z,
                        "height_m": height_m,
                        "inference_label": (
                            f"{args.patch_h}x{args.patch_w} direct"
                            if args.inference_mode == "direct"
                            else f"{args.patch_h}x{args.patch_w} from {args.tile_h}x{args.tile_w} tiles"
                        ),
                    },
                    layer_metrics=layer_metrics,
                )
                visual_records.append(
                    {
                        "sample": sample_idx,
                        "job": jf.name,
                        "month": jf.month,
                        "time_index": t_idx,
                        "z0": z0,
                        "global_z": global_z,
                        "local_z": local_z,
                        "height_m": height_m,
                        "y0": y0,
                        "x0": x0,
                        "co2_vmin": vmin,
                        "co2_vmax": vmax,
                        "file": str(png_path),
                        **layer_metrics,
                    }
                )
                if (
                    args.save_delta_visuals
                    and is_autoregressive
                    and pred_delta is not None
                    and truth_delta is not None
                    and prev_co2_for_visual is not None
                ):
                    delta_stats = new_stats()
                    update_stats(delta_stats, pred_delta[local_z], truth_delta[local_z], metric_mask[local_z])
                    delta_metrics = finalize_stats(delta_stats)
                    delta_diag_metrics = delta_diagnostic_metrics(
                        pred_delta[local_z],
                        truth_delta[local_z],
                        metric_mask[local_z],
                        active_threshold=args.delta_active_threshold,
                    )
                    delta_png_path = (
                        delta_vis_dir
                        / f"sample_{sample_idx:03d}_globalz{global_z:03d}_{jf.name}_t{t_idx:03d}_y{y0:03d}_x{x0:03d}_delta.png"
                    )
                    delta_vmin, delta_vmax = save_delta_visual(
                        delta_png_path,
                        topo_layer=topo_layer,
                        prev_co2=prev_co2_for_visual[local_z],
                        pred_delta=pred_delta[local_z],
                        truth_delta=truth_delta[local_z],
                        mask=metric_mask[local_z],
                        metadata={
                            "job": jf.name,
                            "month": jf.month,
                            "time_index": t_idx,
                            "global_z": global_z,
                            "height_m": height_m,
                        },
                        layer_metrics=delta_metrics,
                    )
                    delta_visual_records.append(
                        {
                            "sample": sample_idx,
                            "job": jf.name,
                            "month": jf.month,
                            "time_index": t_idx,
                            "z0": z0,
                            "global_z": global_z,
                            "local_z": local_z,
                            "height_m": height_m,
                            "y0": y0,
                            "x0": x0,
                            "delta_vmin": delta_vmin,
                            "delta_vmax": delta_vmax,
                            "file": str(delta_png_path),
                            **delta_metrics,
                            **delta_diag_metrics,
                        }
                    )
                    if (
                        args.save_v28_diagnostic_visuals
                        and is_v28_advection_correction
                        and physics_delta_for_visual is not None
                        and model_correction_for_visual is not None
                    ):
                        physics_stats = new_stats()
                        update_stats(
                            physics_stats,
                            physics_delta_for_visual[local_z],
                            truth_delta[local_z],
                            metric_mask[local_z],
                        )
                        physics_metrics = finalize_stats(physics_stats)
                        correction_truth = truth_delta[local_z] - physics_delta_for_visual[local_z]
                        correction_stats = new_stats()
                        update_stats(
                            correction_stats,
                            model_correction_for_visual[local_z],
                            correction_truth,
                            metric_mask[local_z],
                        )
                        correction_metrics = finalize_stats(correction_stats)
                        diag_png_path = (
                            v28_diag_dir
                            / f"sample_{sample_idx:03d}_globalz{global_z:03d}_{jf.name}_t{t_idx:03d}_y{y0:03d}_x{x0:03d}_v28_diagnostic.png"
                        )
                        diag_vmin, diag_vmax = save_v28_diagnostic_visual(
                            diag_png_path,
                            prev_co2=prev_co2_for_visual[local_z],
                            physics_delta=physics_delta_for_visual[local_z],
                            model_correction=model_correction_for_visual[local_z],
                            pred_delta=pred_delta[local_z],
                            truth_delta=truth_delta[local_z],
                            mask=metric_mask[local_z],
                            metadata={
                                "job": jf.name,
                                "month": jf.month,
                                "time_index": t_idx,
                                "global_z": global_z,
                                "height_m": height_m,
                            },
                            final_metrics=delta_metrics,
                            physics_metrics=physics_metrics,
                            correction_metrics=correction_metrics,
                        )
                        v28_diag_records.append(
                            {
                                "sample": sample_idx,
                                "job": jf.name,
                                "month": jf.month,
                                "time_index": t_idx,
                                "z0": z0,
                                "global_z": global_z,
                                "height_m": height_m,
                                "local_z": local_z,
                                "y0": y0,
                                "x0": x0,
                                "final_R": delta_metrics["R"],
                                "final_R2": delta_metrics["R2"],
                                "final_MAE": delta_metrics["MAE"],
                                "final_RMSE": delta_metrics["RMSE"],
                                "physics_R": physics_metrics["R"],
                                "physics_R2": physics_metrics["R2"],
                                "physics_MAE": physics_metrics["MAE"],
                                "physics_RMSE": physics_metrics["RMSE"],
                                "correction_R": correction_metrics["R"],
                                "correction_R2": correction_metrics["R2"],
                                "correction_MAE": correction_metrics["MAE"],
                                "correction_RMSE": correction_metrics["RMSE"],
                                "diag_vmin": diag_vmin,
                                "diag_vmax": diag_vmax,
                                "file": str(diag_png_path),
                            }
                        )
        print(
            f"sample {sample_idx + 1:03d}/{len(samples):03d} {jf.name} t={t_idx} "
            f"z0={z0} y0={y0} x0={x0} MAE={sample_metrics['MAE']:.4f} RMSE={sample_metrics['RMSE']:.4f}",
            flush=True,
        )

    metrics = {
        "checkpoint": os.path.abspath(args.checkpoint),
        "checkpoint_epoch": int(checkpoint.get("epoch", -1)),
        "checkpoint_val_loss": float(checkpoint.get("val_loss", float("nan"))),
        "target_mode": target_mode,
        "input_channel_1": checkpoint.get("input_channel_1", "ls_forcing_right_CO2"),
        "dropped_input_channels": dropped_input_channels,
        "kept_input_channels": checkpoint.get("kept_input_channels", []),
        "kept_input_channel_indices": checkpoint.get("kept_input_channel_indices", []),
        "original_input_channels": checkpoint.get("original_input_channels", []),
        "coord_channels": coord_channels,
        "appended_channels": appended_channels,
        "architecture": checkpoint.get("architecture", "unknown"),
        "model_variant": checkpoint.get("model_variant", "unknown"),
        "use_global_context": use_global_context,
        "surface_gated_channels": checkpoint.get("surface_gated_channels", []),
        "height_gate_decay_levels": checkpoint.get("height_gate_decay_levels", None),
        "v28_advection_features": bool(checkpoint.get("v28_advection_features", False)),
        "v28_advection_correction_target": bool(checkpoint.get("v28_advection_correction_target", False)),
        "v28_advection_dx": checkpoint.get("v28_advection_dx", None),
        "v28_advection_dy": checkpoint.get("v28_advection_dy", None),
        "v28_advection_dz": checkpoint.get("v28_advection_dz", None),
        "v28_advection_dt": checkpoint.get("v28_advection_dt", None),
        "v28_advection_delta_scale": checkpoint.get("v28_advection_delta_scale", None),
        "v28_advection_gradient_scale": checkpoint.get("v28_advection_gradient_scale", None),
        "v28_advection_clip": checkpoint.get("v28_advection_clip", None),
        "force_fluid_mask_ones": bool(args.force_fluid_mask_ones),
        "disable_topography_metric_mask": bool(args.disable_topography_metric_mask),
        "excluded_months": sorted(exclude_months),
        "delta_active_threshold": float(args.delta_active_threshold),
        "v13_sidecar_root": os.path.abspath(str(v13_sidecar_root)) if v13_sidecar_root else None,
        "v22_prev_sidecar_root": os.path.abspath(str(v22_prev_sidecar_root)) if v22_prev_sidecar_root else None,
        "v14_normalization_root": os.path.abspath(str(norm_root)) if norm_root else None,
        "v14_layer_min": checkpoint.get("v14_layer_min", None),
        "v14_layer_max": checkpoint.get("v14_layer_max", None),
        "v14_min_layer_overlap": checkpoint.get("v14_min_layer_overlap", None),
        "eval_layer_min": eval_layer_min,
        "eval_layer_max": eval_layer_max,
        "eval_min_layer_overlap": args.eval_min_layer_overlap,
        "eval_require_full_layer_range": bool(args.eval_require_full_layer_range),
        "v13_global_size": v13_global_size if is_v13 else None,
        "v13_global_sample_size": v13_global_sample_size if is_v13 else None,
        "benchmark": (
            "V35 multitask active/sign delta test split, 256x256 crops, direct inference"
            if architecture.startswith("v35_")
            else
            "V28 physics-advection correction to previous-CO2 delta, 256x256 crops, direct inference"
            if is_v28_advection_correction
            else "V22 autoregressive previous-CO2 delta test split, 256x256 crops, direct inference"
            if is_autoregressive
            else
            "V15 z-focused corrected-time test split, 256x256 crops, direct inference"
            if architecture.startswith("v15_")
            else "V14 z-focused corrected-time test split, 256x256 crops, direct inference with full-domain low-res context"
            if is_zfocused
            else "V13 corrected-time test split, 256x256 crops, direct inference with full-domain low-res context"
            if is_v13
            else (
                "V12 test time split, 256x256 crops, direct 256x256 inference"
                if args.inference_mode == "direct"
                else "V12 test time split, 256x256 crops, prediction stitched from 64x64 tiles"
            )
        ),
        "test_samples": len(samples),
        "patch_size": list(patch_size),
        "tile_size": list(tile_size),
        "inference_mode": args.inference_mode,
        "train_fraction": args.train_fraction,
        "val_fraction": args.val_fraction,
        "overall": finalize_stats(stats),
        "samples": sample_records,
        "visualizations": visual_records,
        "delta_visualizations": delta_visual_records,
    }
    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    with open(out_dir / "metrics.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        writer.writerow(["checkpoint_epoch", metrics["checkpoint_epoch"]])
        writer.writerow(["checkpoint_val_loss", metrics["checkpoint_val_loss"]])
        writer.writerow(["test_samples", len(samples)])
        for key, value in metrics["overall"].items():
            writer.writerow([key, value])
    with open(out_dir / "sample_metrics.csv", "w", newline="", encoding="utf-8") as f:
        fieldnames = ["sample", "job", "month", "time_index", "z0", "y0", "x0", "valid_count", "R", "R2", "MAE", "RMSE"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(sample_records)
    with open(out_dir / "visualization_metrics.csv", "w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "sample",
            "job",
            "month",
            "time_index",
            "z0",
            "global_z",
            "height_m",
            "local_z",
            "y0",
            "x0",
            "valid_count",
            "R",
            "R2",
            "MAE",
            "RMSE",
            "co2_vmin",
            "co2_vmax",
            "file",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(visual_records)
    if args.save_delta_visuals:
        with open(out_dir / "delta_visualization_metrics.csv", "w", newline="", encoding="utf-8") as f:
            fieldnames = [
                "sample",
                "job",
                "month",
                "time_index",
                "z0",
                "global_z",
                "height_m",
                "local_z",
                "y0",
                "x0",
                "valid_count",
                "R",
                "R2",
                "MAE",
                "RMSE",
                "active_valid_count",
                "active_fraction",
                "active_R",
                "active_R2",
                "active_MAE",
                "active_RMSE",
                "sign_accuracy",
                "pos_precision",
                "pos_recall",
                "pos_f1",
                "neg_precision",
                "neg_recall",
                "neg_f1",
                "amplitude_ratio",
                "pred_delta_std",
                "truth_delta_std",
                "active_pred_std",
                "active_truth_std",
                "pred_abs_mean",
                "truth_abs_mean",
                "delta_vmin",
                "delta_vmax",
                "file",
            ]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(delta_visual_records)
        if delta_visual_records:
            summary_fields = [
                "group",
                "count",
                "mean_R",
                "mean_RMSE",
                "mean_active_R",
                "mean_active_RMSE",
                "mean_sign_accuracy",
                "mean_pos_f1",
                "mean_neg_f1",
                "mean_amplitude_ratio",
                "mean_active_fraction",
            ]

            def finite_mean(records: list[dict[str, object]], key: str) -> float:
                vals: list[float] = []
                for rec in records:
                    try:
                        value = float(rec.get(key, float("nan")))
                    except (TypeError, ValueError):
                        value = float("nan")
                    if math.isfinite(value):
                        vals.append(value)
                return float(np.mean(vals)) if vals else float("nan")

            def summary_row(name: str, records: list[dict[str, object]]) -> dict[str, object]:
                return {
                    "group": name,
                    "count": len(records),
                    "mean_R": finite_mean(records, "R"),
                    "mean_RMSE": finite_mean(records, "RMSE"),
                    "mean_active_R": finite_mean(records, "active_R"),
                    "mean_active_RMSE": finite_mean(records, "active_RMSE"),
                    "mean_sign_accuracy": finite_mean(records, "sign_accuracy"),
                    "mean_pos_f1": finite_mean(records, "pos_f1"),
                    "mean_neg_f1": finite_mean(records, "neg_f1"),
                    "mean_amplitude_ratio": finite_mean(records, "amplitude_ratio"),
                    "mean_active_fraction": finite_mean(records, "active_fraction"),
                }

            summary_rows = [summary_row("all", delta_visual_records)]
            for month in sorted({int(rec["month"]) for rec in delta_visual_records}):
                group = [rec for rec in delta_visual_records if int(rec["month"]) == month]
                summary_rows.append(summary_row(f"month_{month:02d}", group))
            for height in sorted({float(rec["height_m"]) for rec in delta_visual_records if math.isfinite(float(rec["height_m"]))}):
                group = [rec for rec in delta_visual_records if math.isclose(float(rec["height_m"]), height)]
                summary_rows.append(summary_row(f"height_{height:.1f}m", group))

            with open(out_dir / "delta_diagnostic_summary.csv", "w", newline="", encoding="utf-8") as sf:
                writer = csv.DictWriter(sf, fieldnames=summary_fields)
                writer.writeheader()
                writer.writerows(summary_rows)
    if args.save_v28_diagnostic_visuals:
        with open(out_dir / "v28_diagnostic_metrics.csv", "w", newline="", encoding="utf-8") as f:
            fieldnames = [
                "sample",
                "job",
                "month",
                "time_index",
                "z0",
                "global_z",
                "height_m",
                "local_z",
                "y0",
                "x0",
                "final_R",
                "final_R2",
                "final_MAE",
                "final_RMSE",
                "physics_R",
                "physics_R2",
                "physics_MAE",
                "physics_RMSE",
                "correction_R",
                "correction_R2",
                "correction_MAE",
                "correction_RMSE",
                "diag_vmin",
                "diag_vmax",
                "file",
            ]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(v28_diag_records)
    print(json.dumps({"overall": metrics["overall"], "test_samples": len(samples)}, indent=2), flush=True)
    print(f"Saved metrics: {out_dir / 'metrics.json'}", flush=True)
    print(f"Saved visualizations: {vis_dir}", flush=True)
    if args.save_delta_visuals:
        print(f"Saved delta visualizations: {delta_vis_dir}", flush=True)
    if args.save_v28_diagnostic_visuals:
        print(f"Saved V28 diagnostic visualizations: {v28_diag_dir}", flush=True)


if __name__ == "__main__":
    main()

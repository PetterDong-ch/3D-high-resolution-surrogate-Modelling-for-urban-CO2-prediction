#!/usr/bin/env python3
from __future__ import annotations

import argparse
import atexit
import bisect
import csv
import glob
import json
import math
import os
import random
import shutil
import sys
import time
from dataclasses import dataclass
from typing import Iterable

# Hide optional acceleration packages that are ABI-incompatible with the
# NumPy version in /data/cd25/.pylibs. Xarray will safely fall back to NumPy.
for _optional_module in ("bottleneck", "dask", "dask.array"):
    sys.modules.setdefault(_optional_module, None)

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import xarray as xr
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset, RandomSampler, Subset, WeightedRandomSampler

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


# cd /data/cd25
# PYTHONPATH=/data/cd25/.pylibs /usr/bin/python3 scripts/train_3d_unet.py --jobs-root /data/linfeng/palm/london_camden_2019_new/JOBS --epochs 20 --batch-size 1 --patch-d 16 --patch-h 64 --patch-w 64 --samples-per-job 64 --out-dir /data/cd25/processed/unet3d_runs

STATIC_VARS = [
    "albedo_type",
    "water_type",
    "pavement_type",
    "street_type",
    "vegetation_type",
    "evi_pft",
    "lswi_pft",
]

DYNAMIC_VARS = ["emission_values", "ls_forcing_right_CO2", "u", "v", "w", "p", "theta"]
TARGET_VAR = "kc_CO2"


# Find one matching input file.
def find_one(patterns: Iterable[str]) -> str | None:
    for pattern in patterns:
        matches = sorted(glob.glob(pattern))
        if matches:
            return matches[0]
    return None


# Extract the simulation month from a job path.
def parse_month_from_job(job_name: str) -> int:
    # Example: z18_camden201901 -> month 1
    for i in range(len(job_name) - 5):
        token = job_name[i : i + 6]
        if token.isdigit() and token.startswith("2019"):
            month = int(token[-2:])
            if 1 <= month <= 12:
                return month
    return 1


# Encode month as sine/cosine features.
def month_features(month: int) -> tuple[float, float]:
    angle = 2.0 * math.pi * (month - 1) / 12.0
    return math.sin(angle), math.cos(angle)


# Load optional target-normalization metadata from disk.
def load_target_normalization(path: str | None) -> dict[str, object] | None:
    if not path:
        return None
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing target-normalization stats: {path}")
    with open(path, "r", encoding="utf-8") as f:
        stats = json.load(f)
    groups = stats.get("groups")
    if not isinstance(groups, dict) or not groups:
        raise RuntimeError(f"Target-normalization stats have no groups: {path}")
    return stats


# Choose the month or background-bin normalization group.
def target_norm_group_key(
    stats: dict[str, object],
    mode: str,
    month: int,
    bg_values: np.ndarray | torch.Tensor | None = None,
) -> str:
    if mode == "month":
        return f"{int(month):02d}"
    if mode == "background_bin":
        if bg_values is None:
            raise RuntimeError("background_bin target normalization requires corrected background values")
        if isinstance(bg_values, torch.Tensor):
            value = float(bg_values.detach().float().mean().item())
        else:
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


# Encode time of day as sine/cosine features.
def time_of_day_features(time_value: object) -> tuple[float, float]:
    seconds_per_day = 86400.0
    seconds = 0.0

    try:
        arr = np.asarray(time_value)
        if np.issubdtype(arr.dtype, np.datetime64):
            ts = arr.astype("datetime64[s]")
            day = ts.astype("datetime64[D]")
            seconds = float((ts - day) / np.timedelta64(1, "s"))
        elif np.issubdtype(arr.dtype, np.timedelta64):
            seconds = float(arr / np.timedelta64(1, "s"))
        else:
            seconds = float(arr)
    except Exception:
        seconds = 0.0

    seconds = seconds % seconds_per_day
    angle = 2.0 * math.pi * seconds / seconds_per_day
    return math.sin(angle), math.cos(angle)


# Parse a list of integer command-line values.
def parse_int_list(text: str) -> tuple[int, ...]:
    values: list[int] = []
    for item in text.split(","):
        item = item.strip()
        if item:
            values.append(int(item))
    return tuple(values)


# Parse a list of channel or variable names.
def parse_name_list(text: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in text.split(",") if item.strip())


# Convert input data to a NumPy array.
def to_numpy(da: xr.DataArray) -> np.ndarray:
    return np.asarray(da.values, dtype=np.float32)


# Align a 3D array to the requested shape.
def align_3d_to_shape(arr: np.ndarray, target_shape: tuple[int, int, int]) -> np.ndarray:
    """Crop/pad a 3D array to a fixed (z, y, x) shape."""
    tz, ty, tx = target_shape
    z = min(arr.shape[0], tz)
    y = min(arr.shape[1], ty)
    x = min(arr.shape[2], tx)

    out = np.zeros(target_shape, dtype=np.float32)
    out[:z, :y, :x] = arr[:z, :y, :x]
    return out


# Interpolate vertical velocity from zw to zu levels.
def interp_w_patch_from_zw_to_zu(
    w_patch: xr.DataArray,
    target_z_coords: xr.DataArray | None,
    target_shape: tuple[int, int, int],
) -> np.ndarray:
    """Interpolate cropped w values from zw faces to target zu centers."""
    w_np = to_numpy(w_patch)
    if w_np.shape[0] < 2:
        return align_3d_to_shape(w_np, target_shape)

    if "zw_3d" not in w_patch.coords or target_z_coords is None:
        return 0.5 * (w_np[:-1] + w_np[1:])

    try:
        zw = np.asarray(w_patch.coords["zw_3d"].values, dtype=np.float64)
        zu = np.asarray(target_z_coords.values, dtype=np.float64)
    except Exception:
        return 0.5 * (w_np[:-1] + w_np[1:])

    if zw.ndim != 1 or zu.ndim != 1 or len(zw) < 2 or len(zu) == 0:
        return 0.5 * (w_np[:-1] + w_np[1:])

    if np.any(np.diff(zw) <= 0):
        order = np.argsort(zw)
        zw = zw[order]
        w_np = w_np[order]

    left_idx = np.searchsorted(zw, zu, side="right") - 1
    left_idx = np.clip(left_idx, 0, len(zw) - 2)
    right_idx = left_idx + 1

    denom = zw[right_idx] - zw[left_idx]
    weights = np.zeros_like(zu, dtype=np.float64)
    valid = denom != 0.0
    weights[valid] = (zu[valid] - zw[left_idx][valid]) / denom[valid]
    weights = np.clip(weights, 0.0, 1.0).astype(np.float32)

    left = w_np[left_idx]
    right = w_np[right_idx]
    return (1.0 - weights[:, None, None]) * left + weights[:, None, None] * right


_OPEN_DATASET_CACHE: dict[str, xr.Dataset] = {}
_KEEP_DATASETS_OPEN = os.environ.get("PALM_KEEP_DATASETS_OPEN", "0") == "1"


# Stores loaded dataset handles and metadata shared across samples.
class _CachedDatasetContext:
    # Store constructor arguments and initialize object state.
    def __init__(self, dataset: xr.Dataset) -> None:
        self.dataset = dataset

    # Internal helper for enter.
    def __enter__(self) -> xr.Dataset:
        return self.dataset

    # Internal helper for exit.
    def __exit__(self, exc_type: object, exc: object, tb: object) -> bool:
        return False


# Internal helper for close cached datasets.
def _close_cached_datasets() -> None:
    for ds in _OPEN_DATASET_CACHE.values():
        ds.close()
    _OPEN_DATASET_CACHE.clear()


atexit.register(_close_cached_datasets)


# Open a dataset with a safe error message.
def safe_open_dataset(path: str) -> xr.Dataset | _CachedDatasetContext:
    if not _KEEP_DATASETS_OPEN:
        return xr.open_dataset(path, decode_times=False)

    dataset = _OPEN_DATASET_CACHE.get(path)
    if dataset is None:
        dataset = xr.open_dataset(path, decode_times=False)
        _OPEN_DATASET_CACHE[path] = dataset
    return _CachedDatasetContext(dataset)


# Convert vertical velocity from zw to zu levels.
def convert_w_from_zw_to_zu(
    w_da: xr.DataArray,
    zu_coords: xr.DataArray | None,
) -> xr.DataArray:
    """Map w from zw_3d faces to zu_3d centers.

    Primary path uses interpolation with physical coordinates.
    Fallback path uses direct rename if dimensions already align.
    """
    if "zw_3d" not in w_da.dims:
        return w_da

    if zu_coords is not None:
        try:
            w_zu = w_da.interp(zw_3d=zu_coords)
            return w_zu
        except Exception:
            pass

    if w_da.sizes.get("zw_3d", 0) >= 2:
        left = w_da.isel(zw_3d=slice(0, -1)).rename({"zw_3d": "zu_3d"})
        right = w_da.isel(zw_3d=slice(1, None)).rename({"zw_3d": "zu_3d"})
        w_zu = 0.5 * (left + right)
        return w_zu

    return w_da.rename({"zw_3d": "zu_3d"})


# Compute masked Huber loss.
def masked_huber_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    delta: float = 1.0,
    weight: torch.Tensor | None = None,
) -> torch.Tensor:
    diff = pred - target
    abs_diff = diff.abs()
    quadratic = torch.minimum(abs_diff, torch.tensor(delta, device=pred.device))
    linear = abs_diff - quadratic
    loss = 0.5 * quadratic * quadratic + delta * linear

    loss_weight = mask if weight is None else mask * weight
    masked = loss * loss_weight
    denom = loss_weight.sum().clamp_min(1.0)
    return masked.sum() / denom


# Compute masked L1 loss.
def masked_l1_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    weight: torch.Tensor | None = None,
) -> torch.Tensor:
    loss_weight = mask if weight is None else mask * weight
    loss = (pred - target).abs() * loss_weight
    return loss.sum() / loss_weight.sum().clamp_min(1.0)


# Compute sign loss for concentration deltas.
def masked_delta_sign_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    min_abs: float,
    scale: float,
) -> torch.Tensor:
    """Softly penalize wrong-sign delta predictions on meaningful delta cells."""
    active = mask * (target.abs() >= float(min_abs)).to(dtype=mask.dtype)
    if float(active.sum().detach().item()) <= 0.0:
        return pred.new_tensor(0.0)
    denom_scale = max(float(scale), 1.0e-6) ** 2
    signed_margin = pred * target / denom_scale
    loss = F.softplus(-signed_margin) * active
    return loss.sum() / active.sum().clamp_min(1.0)


# Compute masked BCE-with-logits loss.
def masked_bce_with_logits_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    pos_weight: float = 1.0,
) -> torch.Tensor:
    weight = mask
    if float(weight.sum().detach().item()) <= 0.0:
        return logits.new_tensor(0.0)
    loss = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    if pos_weight != 1.0:
        class_weight = torch.where(target > 0.5, target.new_tensor(float(pos_weight)), target.new_tensor(1.0))
        weight = weight * class_weight
    return (loss * weight).sum() / weight.sum().clamp_min(1.0)


# Compute weights from target residual magnitude.
def residual_magnitude_weight(
    target: torch.Tensor,
    mask: torch.Tensor,
    alpha: float,
    scale: float,
    max_weight: float,
) -> torch.Tensor | None:
    if alpha <= 0.0:
        return None
    scaled = target.abs() / max(float(scale), 1.0e-6)
    weight = 1.0 + float(alpha) * scaled
    weight = torch.clamp(weight, min=1.0, max=max(float(max_weight), 1.0))
    return weight


# Compute texture weights from target gradients.
def target_gradient_texture_weight(
    target: torch.Tensor,
    mask: torch.Tensor,
    alpha: float,
    scale: float,
    max_weight: float,
) -> torch.Tensor | None:
    if alpha <= 0.0:
        return None

    texture = torch.zeros_like(target)
    counts = torch.zeros_like(target)
    for dim in (2, 3, 4):
        if target.shape[dim] <= 1:
            continue
        left = target.narrow(dim, 0, target.shape[dim] - 1)
        right = target.narrow(dim, 1, target.shape[dim] - 1)
        mask_left = mask.narrow(dim, 0, mask.shape[dim] - 1)
        mask_right = mask.narrow(dim, 1, mask.shape[dim] - 1)
        pair_mask = mask_left * mask_right
        diff = (right - left).abs() * pair_mask

        texture.narrow(dim, 0, target.shape[dim] - 1).add_(diff)
        texture.narrow(dim, 1, target.shape[dim] - 1).add_(diff)
        counts.narrow(dim, 0, target.shape[dim] - 1).add_(pair_mask)
        counts.narrow(dim, 1, target.shape[dim] - 1).add_(pair_mask)

    texture = texture / counts.clamp_min(1.0)
    scaled = texture / max(float(scale), 1.0e-6)
    weight = 1.0 + float(alpha) * scaled
    weight = torch.clamp(weight, min=1.0, max=max(float(max_weight), 1.0))
    return weight


# Compute masked correlation loss.
def masked_correlation_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    valid = mask > 0
    if valid.sum() <= 1:
        return pred.new_tensor(0.0)

    p = pred[valid]
    t = target[valid]
    p_centered = p - p.mean()
    t_centered = t - t.mean()
    p_var = (p_centered * p_centered).mean()
    t_var = (t_centered * t_centered).mean()
    if p_var <= eps or t_var <= eps:
        return pred.new_tensor(0.0)
    corr = (p_centered * t_centered).mean() / torch.sqrt(p_var * t_var + eps)
    return 1.0 - corr.clamp(-1.0, 1.0)


# Compute gradient loss along one axis.
def gradient_axis_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    dim: int,
) -> torch.Tensor:
    pred_left = pred.narrow(dim, 0, pred.shape[dim] - 1)
    pred_right = pred.narrow(dim, 1, pred.shape[dim] - 1)
    target_left = target.narrow(dim, 0, target.shape[dim] - 1)
    target_right = target.narrow(dim, 1, target.shape[dim] - 1)
    mask_left = mask.narrow(dim, 0, mask.shape[dim] - 1)
    mask_right = mask.narrow(dim, 1, mask.shape[dim] - 1)
    pair_mask = mask_left * mask_right

    grad_pred = pred_right - pred_left
    grad_target = target_right - target_left
    loss = (grad_pred - grad_target).abs() * pair_mask
    return loss.sum() / pair_mask.sum().clamp_min(1.0)


# Compute masked gradient loss.
def masked_gradient_loss(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if pred.shape[2] < 2 or pred.shape[3] < 2 or pred.shape[4] < 2:
        return pred.new_tensor(0.0)
    return (
        gradient_axis_loss(pred, target, mask, dim=2)
        + gradient_axis_loss(pred, target, mask, dim=3)
        + gradient_axis_loss(pred, target, mask, dim=4)
    ) / 3.0


# Compute multiscale spatial loss over valid cells.
def masked_spatial_multiscale_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    scales: tuple[int, ...],
    base_loss: str,
    huber_delta: float,
    min_valid_fraction: float,
) -> torch.Tensor:
    losses: list[torch.Tensor] = []
    for scale in scales:
        if scale <= 1:
            continue
        if pred.shape[3] < scale or pred.shape[4] < scale:
            continue

        kernel = (1, scale, scale)
        stride = kernel
        pooled_mask = F.avg_pool3d(mask, kernel_size=kernel, stride=stride)
        valid = (pooled_mask >= min_valid_fraction).to(mask.dtype)
        denom = pooled_mask.clamp_min(1.0e-6)
        pooled_pred = F.avg_pool3d(pred * mask, kernel_size=kernel, stride=stride) / denom
        pooled_target = F.avg_pool3d(target * mask, kernel_size=kernel, stride=stride) / denom

        if valid.sum() <= 0:
            continue
        if base_loss == "huber":
            losses.append(masked_huber_loss(pooled_pred, pooled_target, valid, delta=huber_delta))
        elif base_loss == "l1":
            losses.append(masked_l1_loss(pooled_pred, pooled_target, valid))
        else:
            raise ValueError(f"Unknown base_loss: {base_loss}")

    if not losses:
        return pred.new_tensor(0.0)
    return torch.stack(losses).mean()


# Compute low-pass spatial loss over valid cells.
def masked_spatial_lowpass(
    tensor: torch.Tensor,
    mask: torch.Tensor,
    pool: int,
    min_valid_fraction: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build a full-resolution low-frequency field from masked spatial pooling."""
    pool = max(1, int(pool))
    if pool <= 1 or tensor.shape[3] < pool or tensor.shape[4] < pool:
        return tensor, mask

    kernel = (1, pool, pool)
    pooled_mask = F.avg_pool3d(mask, kernel_size=kernel, stride=kernel)
    pooled_valid = (pooled_mask >= float(min_valid_fraction)).to(mask.dtype)
    pooled = F.avg_pool3d(tensor * mask, kernel_size=kernel, stride=kernel) / pooled_mask.clamp_min(1.0e-6)

    low = F.interpolate(pooled, size=tensor.shape[-3:], mode="trilinear", align_corners=False)
    valid = F.interpolate(pooled_valid, size=tensor.shape[-3:], mode="nearest") * mask
    return low, valid


# Compute variance-normalized L1 loss.
def variance_normalized_l1_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    min_std: float,
    eps: float,
) -> torch.Tensor:
    # Normalize each sample/height layer by the ground-truth horizontal
    # variability on valid cells. This discourages mean-only predictions on
    # low-variance layers while min_std keeps the scale bounded.
    abs_err = (pred - target).abs()
    valid_count = mask.sum(dim=(3, 4), keepdim=True)
    denom_count = valid_count.clamp_min(1.0)
    mean = (target * mask).sum(dim=(3, 4), keepdim=True) / denom_count
    centered = (target - mean) * mask
    var = (centered * centered).sum(dim=(3, 4), keepdim=True) / denom_count
    std = torch.sqrt(var + eps)
    scale = torch.clamp(std, min=min_std)
    normalized = abs_err / scale
    return (normalized * mask).sum() / mask.sum().clamp_min(1.0)


# Compute loss normalized per vertical slice.
def slice_normalized_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    base_loss: str,
    huber_delta: float,
    min_std: float,
    eps: float,
    weight: torch.Tensor | None = None,
) -> torch.Tensor:
    valid_count = mask.sum(dim=(3, 4), keepdim=True)
    denom_count = valid_count.clamp_min(1.0)
    mean = (target * mask).sum(dim=(3, 4), keepdim=True) / denom_count
    centered_target = (target - mean) * mask
    var = (centered_target * centered_target).sum(dim=(3, 4), keepdim=True) / denom_count
    scale = torch.sqrt(var + eps).clamp_min(float(min_std))
    pred_norm = (pred - mean) / scale
    target_norm = (target - mean) / scale
    if base_loss == "huber":
        return masked_huber_loss(pred_norm, target_norm, mask, delta=huber_delta, weight=weight)
    if base_loss == "l1":
        return masked_l1_loss(pred_norm, target_norm, mask, weight=weight)
    raise ValueError(f"Unknown base_loss: {base_loss}")


# Compute masked correlation loss per slice.
def masked_slice_correlation_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    eps: float,
    min_target_std: float,
    min_valid_fraction: float,
    slice_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    b, c, d, h, w = target.shape
    valid_count = mask.sum(dim=(3, 4), keepdim=True)
    min_valid = float(h * w) * float(min_valid_fraction)
    denom_count = valid_count.clamp_min(1.0)

    pred_mean = (pred * mask).sum(dim=(3, 4), keepdim=True) / denom_count
    target_mean = (target * mask).sum(dim=(3, 4), keepdim=True) / denom_count
    pred_centered = (pred - pred_mean) * mask
    target_centered = (target - target_mean) * mask
    pred_var = (pred_centered * pred_centered).sum(dim=(3, 4), keepdim=True) / denom_count
    target_var = (target_centered * target_centered).sum(dim=(3, 4), keepdim=True) / denom_count
    target_std = torch.sqrt(target_var + eps)

    usable = (valid_count >= min_valid) & (target_std >= float(min_target_std)) & (pred_var > eps)
    if usable.sum() <= 0:
        return pred.new_tensor(0.0)

    cov = (pred_centered * target_centered).sum(dim=(3, 4), keepdim=True) / denom_count
    corr = cov / torch.sqrt(pred_var * target_var + eps)
    loss = 1.0 - corr.clamp(-1.0, 1.0)
    if slice_weight is not None:
        weights = slice_weight.expand_as(loss).clamp_min(0.0)
        used_weights = weights[usable]
        if used_weights.sum() > eps:
            return (loss[usable] * used_weights).sum() / used_weights.sum().clamp_min(eps)
    return loss[usable].mean()


# Compute local masked correlation loss.
def masked_local_correlation_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    pool: int,
    eps: float,
    min_target_std: float,
    min_valid_fraction: float,
) -> torch.Tensor:
    """Local window correlation on horizontal patches.

    This is stricter than whole-slice correlation: it penalizes plume texture
    that has the right global tendency but is misplaced inside the 256x256 tile.
    """
    pool = max(2, int(pool))
    if pred.shape[3] < pool or pred.shape[4] < pool:
        return pred.new_tensor(0.0)

    kernel = (1, pool, pool)
    stride = kernel
    area = float(pool * pool)
    mask_sum = F.avg_pool3d(mask, kernel_size=kernel, stride=stride) * area
    min_valid = area * float(min_valid_fraction)
    denom = mask_sum.clamp_min(1.0)

    pred_sum = F.avg_pool3d(pred * mask, kernel_size=kernel, stride=stride) * area
    target_sum = F.avg_pool3d(target * mask, kernel_size=kernel, stride=stride) * area
    pred2_sum = F.avg_pool3d(pred * pred * mask, kernel_size=kernel, stride=stride) * area
    target2_sum = F.avg_pool3d(target * target * mask, kernel_size=kernel, stride=stride) * area
    cross_sum = F.avg_pool3d(pred * target * mask, kernel_size=kernel, stride=stride) * area

    pred_mean = pred_sum / denom
    target_mean = target_sum / denom
    pred_var = (pred2_sum / denom - pred_mean * pred_mean).clamp_min(0.0)
    target_var = (target2_sum / denom - target_mean * target_mean).clamp_min(0.0)
    target_std = torch.sqrt(target_var + eps)
    usable = (mask_sum >= min_valid) & (target_std >= float(min_target_std)) & (pred_var > eps)
    if usable.sum() <= 0:
        return pred.new_tensor(0.0)

    cov = cross_sum / denom - pred_mean * target_mean
    corr = cov / torch.sqrt(pred_var * target_var + eps)
    return (1.0 - corr.clamp(-1.0, 1.0))[usable].mean()


# Compute masked amplitude loss.
def masked_amplitude_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    min_target_std: float,
    eps: float,
) -> torch.Tensor:
    """Match per-sample/per-height horizontal standard deviation.

    This directly fights the observed V36 failure mode where predicted delta is
    systematically weaker and smoother than the truth.
    """
    valid_count = mask.sum(dim=(3, 4), keepdim=True)
    denom = valid_count.clamp_min(1.0)
    pred_mean = (pred * mask).sum(dim=(3, 4), keepdim=True) / denom
    target_mean = (target * mask).sum(dim=(3, 4), keepdim=True) / denom
    pred_var = (((pred - pred_mean) * mask) ** 2).sum(dim=(3, 4), keepdim=True) / denom
    target_var = (((target - target_mean) * mask) ** 2).sum(dim=(3, 4), keepdim=True) / denom
    pred_std = torch.sqrt(pred_var + eps)
    target_std = torch.sqrt(target_var + eps)
    usable = (valid_count > 1.0) & (target_std >= float(min_target_std))
    if usable.sum() <= 0:
        return pred.new_tensor(0.0)
    return (torch.log(pred_std[usable] + eps) - torch.log(target_std[usable] + eps)).abs().mean()


# Compute Huber loss on active plume cells.
def masked_active_huber_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    threshold: float,
    delta: float,
    weight: torch.Tensor | None = None,
) -> torch.Tensor:
    active_mask = mask * (target.abs() >= float(threshold)).to(dtype=mask.dtype)
    if active_mask.sum() <= 0:
        return pred.new_tensor(0.0)
    return masked_huber_loss(pred, target, active_mask, delta=delta, weight=weight)


# Build a height gate from coordinate input channels.
def height_gate_from_input(
    input_x: torch.Tensor | None,
    height_gate_channel_idx: int | None,
) -> torch.Tensor | None:
    if input_x is None or height_gate_channel_idx is None or height_gate_channel_idx < 0:
        return None
    if height_gate_channel_idx >= input_x.shape[1]:
        return None
    return input_x[:, height_gate_channel_idx : height_gate_channel_idx + 1].clamp(0.0, 1.0)


# Convert a height gate into per-slice loss weights.
def slice_weight_from_height_gate(
    height_gate: torch.Tensor | None,
    mask: torch.Tensor,
) -> torch.Tensor | None:
    if height_gate is None:
        return None
    valid_count = mask.sum(dim=(3, 4), keepdim=True).clamp_min(1.0)
    return (height_gate * mask).sum(dim=(3, 4), keepdim=True) / valid_count


# Compute extra loss weight for low vertical layers.
def low_layer_loss_weight(
    input_x: torch.Tensor | None,
    height_gate_channel_idx: int | None,
    alpha: float,
) -> torch.Tensor | None:
    if alpha <= 0.0:
        return None
    gate = height_gate_from_input(input_x, height_gate_channel_idx)
    if gate is None:
        return None
    return 1.0 + float(alpha) * gate


# Compute smoothness loss for upper layers.
def high_layer_smoothness_loss(
    pred: torch.Tensor,
    mask: torch.Tensor,
    input_x: torch.Tensor | None,
    height_gate_channel_idx: int | None,
    kernel_size: int,
) -> torch.Tensor:
    if input_x is None or height_gate_channel_idx is None or height_gate_channel_idx < 0:
        return pred.new_tensor(0.0)
    if height_gate_channel_idx >= input_x.shape[1]:
        return pred.new_tensor(0.0)

    gate = input_x[:, height_gate_channel_idx : height_gate_channel_idx + 1].clamp(0.0, 1.0)
    high_weight = (1.0 - gate) * mask
    if high_weight.sum() <= 0:
        return pred.new_tensor(0.0)

    k = max(3, int(kernel_size))
    if k % 2 == 0:
        k += 1
    pad = k // 2
    pred_pad = F.pad(pred, (pad, pad, pad, pad, 0, 0), mode="replicate")
    smooth = F.avg_pool3d(pred_pad, kernel_size=(1, k, k), stride=1)
    high_freq = (pred - smooth).abs()
    return (high_freq * high_weight).sum() / high_weight.sum().clamp_min(1.0)


# Combine masked loss terms for training.
def combined_masked_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    input_x: torch.Tensor | None,
    components: dict[str, torch.Tensor] | None,
    base_loss: str,
    huber_delta: float,
    gradient_loss_weight: float,
    multiscale_loss_weight: float,
    multiscale_scales: tuple[int, ...],
    multiscale_min_valid_fraction: float,
    smoothness_loss_weight: float,
    smoothness_kernel_size: int,
    height_gate_channel_idx: int | None,
    variance_loss_weight: float,
    variance_min_std: float,
    variance_eps: float,
    residual_weight_alpha: float,
    residual_weight_scale: float,
    residual_weight_max: float,
    target_gradient_weight_alpha: float,
    target_gradient_weight_scale: float,
    target_gradient_weight_max: float,
    low_layer_weight_alpha: float,
    normalized_loss_weight: float,
    normalized_min_std: float,
    normalized_huber_delta: float,
    normalized_eps: float,
    correlation_loss_weight: float,
    correlation_eps: float,
    correlation_min_target_std: float,
    correlation_min_valid_fraction: float,
    low_frequency_loss_weight: float,
    low_frequency_pool: int,
    low_frequency_min_valid_fraction: float,
    low_frequency_correlation_weight: float,
    high_frequency_loss_weight: float,
    high_frequency_huber_delta: float,
    local_correlation_loss_weight: float,
    local_correlation_pool: int,
    local_correlation_min_target_std: float,
    local_correlation_min_valid_fraction: float,
    amplitude_loss_weight: float,
    amplitude_min_target_std: float,
    active_delta_loss_weight: float,
    active_delta_threshold: float,
    sign_loss_weight: float,
    sign_loss_min_abs: float,
    sign_loss_scale: float,
    active_loss_weight: float,
    active_loss_threshold: float,
    active_loss_pos_weight: float,
    sign_class_loss_weight: float,
    sign_class_loss_min_abs: float,
    sign_class_loss_pos_weight: float,
    pattern_height_decay: bool,
) -> tuple[torch.Tensor, dict[str, float]]:
    residual_weight = residual_magnitude_weight(
        target,
        mask,
        alpha=residual_weight_alpha,
        scale=residual_weight_scale,
        max_weight=residual_weight_max,
    )
    texture_weight = target_gradient_texture_weight(
        target,
        mask,
        alpha=target_gradient_weight_alpha,
        scale=target_gradient_weight_scale,
        max_weight=target_gradient_weight_max,
    )
    layer_weight = low_layer_loss_weight(input_x, height_gate_channel_idx, low_layer_weight_alpha)
    height_gate = height_gate_from_input(input_x, height_gate_channel_idx)
    pattern_slice_weight = slice_weight_from_height_gate(height_gate, mask) if pattern_height_decay else None
    combined_weight = None
    for candidate_weight in (residual_weight, texture_weight, layer_weight):
        if candidate_weight is None:
            continue
        combined_weight = candidate_weight if combined_weight is None else combined_weight * candidate_weight

    if base_loss == "huber":
        base = masked_huber_loss(pred, target, mask, delta=huber_delta, weight=combined_weight)
    elif base_loss == "l1":
        base = masked_l1_loss(pred, target, mask, weight=combined_weight)
    else:
        raise ValueError(f"Unknown base_loss: {base_loss}")

    grad = masked_gradient_loss(pred, target, mask) if gradient_loss_weight > 0.0 else pred.new_tensor(0.0)
    multiscale = (
        masked_spatial_multiscale_loss(
            pred,
            target,
            mask,
            scales=multiscale_scales,
            base_loss=base_loss,
            huber_delta=huber_delta,
            min_valid_fraction=multiscale_min_valid_fraction,
        )
        if multiscale_loss_weight > 0.0
        else pred.new_tensor(0.0)
    )
    var_norm = (
        variance_normalized_l1_loss(pred, target, mask, min_std=variance_min_std, eps=variance_eps)
        if variance_loss_weight > 0.0
        else pred.new_tensor(0.0)
    )
    smoothness = (
        high_layer_smoothness_loss(
            pred,
            mask,
            input_x=input_x,
            height_gate_channel_idx=height_gate_channel_idx,
            kernel_size=smoothness_kernel_size,
        )
        if smoothness_loss_weight > 0.0
        else pred.new_tensor(0.0)
    )
    normalized = (
        slice_normalized_loss(
            pred,
            target,
            mask,
            base_loss=base_loss,
            huber_delta=normalized_huber_delta,
            min_std=normalized_min_std,
            eps=normalized_eps,
            weight=layer_weight,
        )
        if normalized_loss_weight > 0.0
        else pred.new_tensor(0.0)
    )
    correlation = (
        masked_slice_correlation_loss(
            pred,
            target,
            mask,
            eps=correlation_eps,
            min_target_std=correlation_min_target_std,
            min_valid_fraction=correlation_min_valid_fraction,
            slice_weight=pattern_slice_weight,
        )
        if correlation_loss_weight > 0.0
        else pred.new_tensor(0.0)
    )
    local_correlation = (
        masked_local_correlation_loss(
            pred,
            target,
            mask,
            pool=local_correlation_pool,
            eps=correlation_eps,
            min_target_std=local_correlation_min_target_std,
            min_valid_fraction=local_correlation_min_valid_fraction,
        )
        if local_correlation_loss_weight > 0.0
        else pred.new_tensor(0.0)
    )
    amplitude = (
        masked_amplitude_loss(
            pred,
            target,
            mask,
            min_target_std=amplitude_min_target_std,
            eps=correlation_eps,
        )
        if amplitude_loss_weight > 0.0
        else pred.new_tensor(0.0)
    )
    active_delta = (
        masked_active_huber_loss(
            pred,
            target,
            mask,
            threshold=active_delta_threshold,
            delta=huber_delta,
            weight=combined_weight,
        )
        if active_delta_loss_weight > 0.0
        else pred.new_tensor(0.0)
    )
    sign = (
        masked_delta_sign_loss(
            pred,
            target,
            mask,
            min_abs=sign_loss_min_abs,
            scale=sign_loss_scale,
        )
        if sign_loss_weight > 0.0
        else pred.new_tensor(0.0)
    )
    active_class = pred.new_tensor(0.0)
    sign_class = pred.new_tensor(0.0)
    if components is not None and active_loss_weight > 0.0 and "active_logit" in components:
        active_target = (target.abs() >= float(active_loss_threshold)).to(dtype=target.dtype)
        active_class = masked_bce_with_logits_loss(
            components["active_logit"],
            active_target,
            mask,
            pos_weight=active_loss_pos_weight,
        )
    if components is not None and sign_class_loss_weight > 0.0 and "sign_logit" in components:
        sign_active_mask = mask * (target.abs() >= float(sign_class_loss_min_abs)).to(dtype=mask.dtype)
        sign_target = (target > 0.0).to(dtype=target.dtype)
        sign_class = masked_bce_with_logits_loss(
            components["sign_logit"],
            sign_target,
            sign_active_mask,
            pos_weight=sign_class_loss_pos_weight,
        )
    low_frequency = pred.new_tensor(0.0)
    low_frequency_corr = pred.new_tensor(0.0)
    high_frequency = pred.new_tensor(0.0)
    if (
        low_frequency_loss_weight > 0.0
        or low_frequency_correlation_weight > 0.0
        or high_frequency_loss_weight > 0.0
    ):
        low_target, low_mask = masked_spatial_lowpass(
            target,
            mask,
            pool=low_frequency_pool,
            min_valid_fraction=low_frequency_min_valid_fraction,
        )
        low_pred = components.get("low") if components is not None else None
        high_pred = components.get("high") if components is not None else None
        if low_pred is not None:
            low_frequency = masked_huber_loss(low_pred, low_target, low_mask, delta=huber_delta)
            low_frequency_corr = masked_slice_correlation_loss(
                low_pred,
                low_target,
                low_mask,
                eps=correlation_eps,
                min_target_std=correlation_min_target_std,
                min_valid_fraction=correlation_min_valid_fraction,
                slice_weight=pattern_slice_weight,
            )
        if high_frequency_loss_weight > 0.0 and high_pred is None:
            # V7-style models do not expose a separate high-frequency head.
            # Fall back to the high-pass part of the final prediction so the
            # texture loss still directly supervises streaks and local plumes.
            low_pred_final, _ = masked_spatial_lowpass(
                pred,
                mask,
                pool=low_frequency_pool,
                min_valid_fraction=low_frequency_min_valid_fraction,
            )
            high_pred = pred - low_pred_final
        if high_pred is not None:
            high_target = target - low_target
            high_frequency = masked_huber_loss(
                high_pred,
                high_target,
                mask,
                delta=high_frequency_huber_delta,
                weight=combined_weight,
            )

    total = (
        base
        + gradient_loss_weight * grad
        + multiscale_loss_weight * multiscale
        + smoothness_loss_weight * smoothness
        + variance_loss_weight * var_norm
        + normalized_loss_weight * normalized
        + correlation_loss_weight * correlation
        + local_correlation_loss_weight * local_correlation
        + amplitude_loss_weight * amplitude
        + active_delta_loss_weight * active_delta
        + low_frequency_loss_weight * low_frequency
        + low_frequency_correlation_weight * low_frequency_corr
        + high_frequency_loss_weight * high_frequency
        + sign_loss_weight * sign
        + active_loss_weight * active_class
        + sign_class_loss_weight * sign_class
    )
    if residual_weight is None:
        residual_weight_mean = 1.0
    else:
        residual_weight_mean = float(((residual_weight * mask).sum() / mask.sum().clamp_min(1.0)).detach().item())
    if texture_weight is None:
        texture_weight_mean = 1.0
    else:
        texture_weight_mean = float(((texture_weight * mask).sum() / mask.sum().clamp_min(1.0)).detach().item())
    return total, {
        "base": float(base.detach().item()),
        "gradient": float(grad.detach().item()),
        "multiscale": float(multiscale.detach().item()),
        "smoothness": float(smoothness.detach().item()),
        "variance": float(var_norm.detach().item()),
        "normalized": float(normalized.detach().item()),
        "correlation": float(correlation.detach().item()),
        "local_correlation": float(local_correlation.detach().item()),
        "amplitude": float(amplitude.detach().item()),
        "active_delta": float(active_delta.detach().item()),
        "low_frequency": float(low_frequency.detach().item()),
        "low_frequency_corr": float(low_frequency_corr.detach().item()),
        "high_frequency": float(high_frequency.detach().item()),
        "sign": float(sign.detach().item()),
        "active_class": float(active_class.detach().item()),
        "sign_class": float(sign_class.detach().item()),
        "residual_weight_mean": residual_weight_mean,
        "texture_weight_mean": texture_weight_mean,
    }


# Stores PALM file paths for one simulation job.
@dataclass
class JobFiles:
    name: str
    month: int
    chemistry: str
    dynamic: str
    static: str
    out3d: str
    av3d: str


# Locate PALM job files needed for a sample.
def locate_job_files(jobs_root: str, job_name: str) -> JobFiles | None:
    job_dir = os.path.join(jobs_root, job_name)
    chemistry = find_one(
        [
            os.path.join(job_dir, "INPUT", "*_chemistry_N02"),
            os.path.join(job_dir, "INPUT", "*_chemistry_N02.nc"),
            os.path.join(job_dir, "INPUT", "*_chemistry_N02*"),
        ]
    )
    dynamic = find_one(
        [
            os.path.join(job_dir, "INPUT", "*_dynamic_N02"),
            os.path.join(job_dir, "INPUT", "*_dynamic_N02.nc"),
            os.path.join(job_dir, "INPUT", "*_dynamic"),
            os.path.join(job_dir, "INPUT", "*_dynamic.nc"),
            os.path.join(job_dir, "INPUT", "*_dynamic*"),
        ]
    )
    static = find_one(
        [
            os.path.join(job_dir, "INPUT", "*_static_N02"),
            os.path.join(job_dir, "INPUT", "*_static_N02.nc"),
            os.path.join(job_dir, "INPUT", "*_static_N02*"),
        ]
    )
    out3d = find_one(
        [
            os.path.join(job_dir, "OUTPUT", "*_3d_N02.nc"),
            os.path.join(job_dir, "OUTPUT", "*_3d_N02.*.nc"),
            os.path.join(job_dir, "OUTPUT", "*_3d_N02*"),
        ]
    )
    av3d = find_one(
        [
            os.path.join(job_dir, "OUTPUT", "*_av_3d_N02.nc"),
            os.path.join(job_dir, "OUTPUT", "*_av_3d_N02.*.nc"),
            os.path.join(job_dir, "OUTPUT", "*_av_3d_N02*"),
        ]
    )
    if not all([chemistry, dynamic, static, out3d, av3d]):
        return None

    return JobFiles(
        name=job_name,
        month=parse_month_from_job(job_name),
        chemistry=chemistry,
        dynamic=dynamic,
        static=static,
        out3d=out3d,
        av3d=av3d,
    )


# Loads samples for the Palm Patch data pipeline.
class PalmPatchDataset(Dataset):
    # Load split files and prepare dataset state.
    def __init__(
        self,
        jobs_root: str,
        job_names: list[str],
        split: str,
        patch_size: tuple[int, int, int],
        samples_per_job: int,
        seed: int,
        topography_path: str,
        train_fraction: float = 0.70,
        val_fraction: float = 0.15,
        z_min_start: int | None = None,
        z_max_start: int | None = None,
    ) -> None:
        super().__init__()
        self.patch_d, self.patch_h, self.patch_w = patch_size
        self.rng = random.Random(seed)
        self.split = split
        self.topography_path = topography_path
        self.train_fraction = train_fraction
        self.val_fraction = val_fraction
        self.z_min_start = z_min_start
        self.z_max_start = z_max_start

        files: list[JobFiles] = []
        for name in job_names:
            jf = locate_job_files(jobs_root, name)
            if jf is not None:
                files.append(jf)

        if not files:
            raise RuntimeError("No valid job files found")
        if not os.path.exists(topography_path):
            raise FileNotFoundError(f"Topography file not found: {topography_path}")

        self.entries: list[tuple[JobFiles, int]] = []
        for jf in files:
            with safe_open_dataset(jf.out3d) as ds3d, safe_open_dataset(jf.av3d) as dsav:
                n_t = min(int(ds3d.sizes.get("time", 0)), int(dsav.sizes.get("time", 0)))
            times = self._split_time_indices(max(1, n_t), split)
            if not times:
                continue
            for _ in range(samples_per_job):
                self.entries.append((jf, self.rng.choice(times)))

        if not self.entries:
            raise RuntimeError(f"No valid entries found for split={split}")

    # Return the number of available samples.
    def __len__(self) -> int:
        return len(self.entries)

    # Internal helper for split time indices.
    def _split_time_indices(self, n_t: int, split: str) -> list[int]:
        indices = list(range(n_t))
        n_train = max(1, int(round(n_t * self.train_fraction)))
        n_val = max(1, int(round(n_t * self.val_fraction)))
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

    # Internal helper for choose start.
    def _choose_start(self, n: int, patch: int) -> int:
        if n <= patch:
            return 0
        return self.rng.randrange(0, n - patch + 1)

    # Internal helper for choose z start.
    def _choose_z_start(self, n: int) -> int:
        if n <= self.patch_d:
            return 0
        max_allowed = n - self.patch_d
        lo = 0 if self.z_min_start is None else max(0, int(self.z_min_start))
        hi = max_allowed if self.z_max_start is None else min(max_allowed, int(self.z_max_start))
        if hi < lo:
            raise RuntimeError(
                f"Invalid z start range [{lo}, {hi}] for nz={n}, patch_d={self.patch_d}. "
                "Check --z-min-start/--z-max-start."
            )
        return self.rng.randrange(lo, hi + 1)

    # Internal helper for build topography mask.
    def _build_topography_mask(
        self,
        dstopo: xr.Dataset,
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
        y_dim = "y"
        x_dim = "x"
        topo_z = int(topo.sizes.get(z_dim, 0))
        topo_y = int(topo.sizes.get(y_dim, 0))
        topo_x = int(topo.sizes.get(x_dim, 0))
        if topo_z <= z0 or topo_y <= y0 or topo_x <= x0:
            return mask

        z_read1 = min(z1, topo_z)
        y_read1 = min(y1, topo_y)
        x_read1 = min(x1, topo_x)
        topo_patch = topo.isel({z_dim: slice(z0, z_read1), y_dim: slice(y0, y_read1), x_dim: slice(x0, x_read1)})
        topo_np = to_numpy(topo_patch)
        fluid = (topo_np == 0).astype(np.float32)
        mask[: fluid.shape[0], : fluid.shape[1], : fluid.shape[2]] = fluid
        return mask

    # Return one indexed sample in the format expected by the model.
    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        jf, t_idx = self.entries[index]

        with safe_open_dataset(jf.chemistry) as dsch, safe_open_dataset(jf.dynamic) as dsdyn, safe_open_dataset(
            jf.static
        ) as dsst, safe_open_dataset(jf.out3d) as ds3d, safe_open_dataset(jf.av3d) as dsav, safe_open_dataset(
            self.topography_path
        ) as dstopo:
            t3d = min(t_idx, int(ds3d.sizes["time"]) - 1)
            tav = min(t_idx, int(dsav.sizes["time"]) - 1)

            target = dsav[TARGET_VAR].isel(time=tav)
            z_name = "zu_3d" if "zu_3d" in target.dims else target.dims[0]
            y_name = "y"
            x_name = "x"

            nz = int(target.sizes[z_name])
            ny = int(target.sizes[y_name])
            nx = int(target.sizes[x_name])

            z0 = self._choose_z_start(nz)
            y0 = self._choose_start(ny, self.patch_h)
            x0 = self._choose_start(nx, self.patch_w)

            z1 = min(nz, z0 + self.patch_d)
            y1 = min(ny, y0 + self.patch_h)
            x1 = min(nx, x0 + self.patch_w)

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

                # CO2 background is stored at 50 m intervals; use the first
                # 20 levels to cover the lowest 1 km and map by height.
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

            for v in ["u", "v", "p", "theta"]:
                da = ds3d[v].isel(time=t3d)
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
            for v in STATIC_VARS:
                s2d = dsst[v].isel({"y": slice(y0, y1), "x": slice(x0, x1)})
                s2d_np = to_numpy(s2d)
                s3d = np.repeat(s2d_np[None, :, :], target_shape[0], axis=0)
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

            x_channels = dyn_arrays + static_arrays + [month_sin, month_cos, tod_sin, tod_cos]
            x = np.stack(x_channels, axis=0)
            y = to_numpy(target_patch)[None, ...]
            topo_mask = self._build_topography_mask(dstopo, z0, z1, y0, y1, x0, x1, target_shape)[None, ...]

        y_mask = np.isfinite(y).astype(np.float32) * topo_mask
        y = np.nan_to_num(y, nan=0.0)
        x = np.nan_to_num(x, nan=0.0)

        return (
            torch.from_numpy(x).float(),
            torch.from_numpy(y).float(),
            torch.from_numpy(y_mask).float(),
        )


# Loads samples for the Drop Input Channel data pipeline.
class DropInputChannelDataset(Dataset):
    # Load split files and prepare dataset state.
    def __init__(self, dataset: Dataset, channel_idx: int) -> None:
        super().__init__()
        self.dataset = dataset
        self.channel_idx = channel_idx

    # Return the number of available samples.
    def __len__(self) -> int:
        return len(self.dataset)

    # Return one indexed sample in the format expected by the model.
    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x, y, m = self.dataset[index]
        if x.shape[0] <= self.channel_idx:
            raise RuntimeError(f"Cannot drop channel {self.channel_idx}; sample only has {x.shape[0]} channels")
        x = torch.cat((x[: self.channel_idx], x[self.channel_idx + 1 :]), dim=0)
        return x, y, m


# Loads samples for the Drop Input Channels data pipeline.
class DropInputChannelsDataset(Dataset):
    # Load split files and prepare dataset state.
    def __init__(self, dataset: Dataset, channel_indices: Iterable[int]) -> None:
        super().__init__()
        self.dataset = dataset
        self.channel_indices = tuple(sorted({int(i) for i in channel_indices}))

    # Return the number of available samples.
    def __len__(self) -> int:
        return len(self.dataset)

    # Return one indexed sample in the format expected by the model.
    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x, y, m = self.dataset[index]
        if not self.channel_indices:
            return x, y, m
        keep = [i for i in range(x.shape[0]) if i not in self.channel_indices]
        if len(keep) == x.shape[0]:
            return x, y, m
        if not keep:
            raise RuntimeError("Cannot drop every input channel")
        return x[keep], y, m


# Loads samples for the Keep Input Channels data pipeline.
class KeepInputChannelsDataset(Dataset):
    """Keep a named subset of local input channels while preserving targets/context."""

    # Load split files and prepare dataset state.
    def __init__(self, dataset: Dataset, channel_names: Iterable[str]) -> None:
        super().__init__()
        self.dataset = dataset
        requested = [str(name).strip() for name in channel_names if str(name).strip()]
        if not requested:
            raise RuntimeError("At least one input channel name is required")
        if not hasattr(dataset, "effective_local_channels"):
            raise RuntimeError("--keep-input-channel-names requires effective_local_channels")

        base_names = list(getattr(dataset, "effective_local_channels"))
        missing = [name for name in requested if name not in base_names]
        if missing:
            raise RuntimeError(
                "Requested input channels are missing from dataset: "
                + ", ".join(missing)
                + f"; available={base_names}"
            )

        self.kept_input_channels = requested
        self.kept_input_channel_indices = [base_names.index(name) for name in requested]
        self.original_input_channels = base_names
        self.effective_local_channels = requested
        self.appended_channels = [
            name for name in list(getattr(dataset, "appended_channels", [])) if name in requested
        ]
        self.dropped_input_channels = list(getattr(dataset, "dropped_input_channels", []))
        self.height_gate_channel_idx = requested.index("height_gate") if "height_gate" in requested else None
        self.global_channels = list(getattr(dataset, "global_channels", []))
        self.indices = getattr(dataset, "indices", None)
        self.z0 = getattr(dataset, "z0", None)
        self.month = getattr(dataset, "month", None)

    # Return the number of available samples.
    def __len__(self) -> int:
        return len(self.dataset)

    # Internal helper for getattr.
    def __getattr__(self, name: str):
        return getattr(self.dataset, name)

    # Return one indexed sample in the format expected by the model.
    def __getitem__(self, index: int):
        item = self.dataset[index]
        if len(item) == 5:
            x, global_context, global_grid, y, m = item
            return x[self.kept_input_channel_indices], global_context, global_grid, y, m
        if len(item) == 3:
            x, y, m = item
            return x[self.kept_input_channel_indices], y, m
        raise RuntimeError(f"Unexpected dataset item with {len(item)} tensors")


# Loads samples for the Residual Target data pipeline.
class ResidualTargetDataset(Dataset):
    # Load split files and prepare dataset state.
    def __init__(self, dataset: Dataset, bg_channel_idx: int = 1) -> None:
        super().__init__()
        self.dataset = dataset
        self.bg_channel_idx = bg_channel_idx

    # Return the number of available samples.
    def __len__(self) -> int:
        return len(self.dataset)

    # Return one indexed sample in the format expected by the model.
    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x, y, m = self.dataset[index]
        if x.shape[0] <= self.bg_channel_idx:
            raise RuntimeError(f"Cannot build residual target; sample only has {x.shape[0]} channels")
        bg = x[self.bg_channel_idx : self.bg_channel_idx + 1]
        return x, y - bg, m


# Loads samples for the Cached Patch data pipeline.
class CachedPatchDataset(Dataset):
    # Load split files and prepare dataset state.
    def __init__(self, split_dir: str) -> None:
        super().__init__()
        self.split_dir = os.path.abspath(split_dir)
        manifest_path = os.path.join(self.split_dir, "manifest.json")
        if not os.path.exists(manifest_path):
            raise FileNotFoundError(f"Missing cache manifest: {manifest_path}")

        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)

        self.shards = manifest["shards"]
        if not self.shards:
            raise RuntimeError(f"No shards listed in {manifest_path}")

        self.cumulative: list[int] = []
        total = 0
        for shard in self.shards:
            total += int(shard["count"])
            self.cumulative.append(total)
        self.total = total
        self._arrays: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}

    # Return the number of available samples.
    def __len__(self) -> int:
        return self.total

    # Internal helper for load shard.
    def _load_shard(self, shard_idx: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        arrays = self._arrays.get(shard_idx)
        if arrays is not None:
            return arrays

        shard = self.shards[shard_idx]
        arrays = (
            np.load(os.path.join(self.split_dir, shard["x"]), mmap_mode="r"),
            np.load(os.path.join(self.split_dir, shard["y"]), mmap_mode="r"),
            np.load(os.path.join(self.split_dir, shard["mask"]), mmap_mode="r"),
        )
        self._arrays[shard_idx] = arrays
        return arrays

    # Return one indexed sample in the format expected by the model.
    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if index < 0:
            index += self.total
        if index < 0 or index >= self.total:
            raise IndexError(index)

        shard_idx = bisect.bisect_right(self.cumulative, index)
        prev = 0 if shard_idx == 0 else self.cumulative[shard_idx - 1]
        local_idx = index - prev
        x_arr, y_arr, m_arr = self._load_shard(shard_idx)

        # Copy from mmap so PyTorch can safely pin and move the batch.
        x = torch.from_numpy(np.array(x_arr[local_idx], dtype=np.float32, copy=True))
        y = torch.from_numpy(np.array(y_arr[local_idx], dtype=np.float32, copy=True))
        m = torch.from_numpy(np.array(m_arr[local_idx], dtype=np.float32, copy=True))
        return x, y, m


# Loads samples for the Coordinate Channel data pipeline.
class CoordinateChannelDataset(Dataset):
    # Load split files and prepare dataset state.
    def __init__(self, dataset: Dataset, metadata_path: str) -> None:
        super().__init__()
        self.dataset = dataset
        self.metadata_path = os.path.abspath(metadata_path)
        if not os.path.exists(self.metadata_path):
            raise FileNotFoundError(f"Missing coordinate metadata sidecar: {self.metadata_path}")

        meta = np.load(self.metadata_path, allow_pickle=False)
        required = ("z0", "y0", "x0", "nz", "ny", "nx")
        missing = [key for key in required if key not in meta.files]
        if missing:
            raise RuntimeError(f"Coordinate metadata missing fields: {missing}")
        self.z0 = meta["z0"].astype(np.int64, copy=False)
        self.y0 = meta["y0"].astype(np.int64, copy=False)
        self.x0 = meta["x0"].astype(np.int64, copy=False)
        self.nz = meta["nz"].astype(np.int64, copy=False)
        self.ny = meta["ny"].astype(np.int64, copy=False)
        self.nx = meta["nx"].astype(np.int64, copy=False)
        if len(self.z0) < len(self.dataset):
            raise RuntimeError(
                f"Coordinate metadata has {len(self.z0)} rows but dataset has {len(self.dataset)} samples"
            )

    # Return the number of available samples.
    def __len__(self) -> int:
        return len(self.dataset)

    # Internal helper for norm axis.
    @staticmethod
    def _norm_axis(start: int, total: int, length: int, dtype: torch.dtype) -> torch.Tensor:
        denom = max(int(total) - 1, 1)
        values = (torch.arange(length, dtype=dtype) + float(start)) / float(denom)
        return values.mul(2.0).sub(1.0)

    # Return one indexed sample in the format expected by the model.
    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x, y, m = self.dataset[index]
        d, h, w = x.shape[-3:]
        dtype = x.dtype
        z_values = self._norm_axis(int(self.z0[index]), int(self.nz[index]), d, dtype)
        y_values = self._norm_axis(int(self.y0[index]), int(self.ny[index]), h, dtype)
        x_values = self._norm_axis(int(self.x0[index]), int(self.nx[index]), w, dtype)

        x_coord = x_values.view(1, 1, w).expand(d, h, w)
        y_coord = y_values.view(1, h, 1).expand(d, h, w)
        z_coord = z_values.view(d, 1, 1).expand(d, h, w)
        coords = torch.stack((x_coord, y_coord, z_coord), dim=0)
        return torch.cat((x, coords), dim=0), y, m


# Loads samples for the V10Input data pipeline.
class V10InputDataset(Dataset):
    # Load split files and prepare dataset state.
    def __init__(
        self,
        dataset: Dataset,
        metadata_path: str,
        surface_channel_indices: tuple[int, ...],
        height_gate_decay_levels: float,
        append_coords: bool = True,
        append_fluid_mask: bool = True,
        append_height_gate: bool = True,
        gate_surface_channels: bool = True,
    ) -> None:
        super().__init__()
        self.dataset = dataset
        self.surface_channel_indices = tuple(surface_channel_indices)
        self.height_gate_decay_levels = float(height_gate_decay_levels)
        self.append_coords = append_coords
        self.append_fluid_mask = append_fluid_mask
        self.append_height_gate = append_height_gate
        self.gate_surface_channels = gate_surface_channels
        self.metadata_path = os.path.abspath(metadata_path)
        if not os.path.exists(self.metadata_path):
            raise FileNotFoundError(f"Missing V10 metadata sidecar: {self.metadata_path}")

        meta = np.load(self.metadata_path, allow_pickle=False)
        required = ("z0", "y0", "x0", "nz", "ny", "nx")
        missing = [key for key in required if key not in meta.files]
        if missing:
            raise RuntimeError(f"V10 metadata missing fields: {missing}")
        self.z0 = meta["z0"].astype(np.int64, copy=False)
        self.y0 = meta["y0"].astype(np.int64, copy=False)
        self.x0 = meta["x0"].astype(np.int64, copy=False)
        self.nz = meta["nz"].astype(np.int64, copy=False)
        self.ny = meta["ny"].astype(np.int64, copy=False)
        self.nx = meta["nx"].astype(np.int64, copy=False)
        if len(self.z0) < len(self.dataset):
            raise RuntimeError(f"V10 metadata has {len(self.z0)} rows but dataset has {len(self.dataset)} samples")

    # Return the number of available samples.
    def __len__(self) -> int:
        return len(self.dataset)

    # Internal helper for norm axis.
    @staticmethod
    def _norm_axis(start: int, total: int, length: int, dtype: torch.dtype) -> torch.Tensor:
        denom = max(int(total) - 1, 1)
        values = (torch.arange(length, dtype=dtype) + float(start)) / float(denom)
        return values.mul(2.0).sub(1.0)

    # Internal helper for height gate.
    def _height_gate(self, z0: int, d: int, dtype: torch.dtype) -> torch.Tensor:
        z_index = torch.arange(d, dtype=dtype) + float(z0)
        gate = torch.exp(-z_index / max(self.height_gate_decay_levels, 1.0))
        return gate.clamp(0.0, 1.0)

    # Return one indexed sample in the format expected by the model.
    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x, y, m = self.dataset[index]
        d, h, w = x.shape[-3:]
        dtype = x.dtype
        z0 = int(self.z0[index])
        y0 = int(self.y0[index])
        x0 = int(self.x0[index])
        nz = int(self.nz[index])
        ny = int(self.ny[index])
        nx = int(self.nx[index])

        gate_z = self._height_gate(z0, d, dtype)
        gate = gate_z.view(d, 1, 1).expand(d, h, w)
        if self.gate_surface_channels:
            x = x.clone()
            for channel_idx in self.surface_channel_indices:
                if 0 <= channel_idx < x.shape[0]:
                    x[channel_idx] = x[channel_idx] * gate

        extra_channels: list[torch.Tensor] = []
        if self.append_coords:
            z_values = self._norm_axis(z0, nz, d, dtype)
            y_values = self._norm_axis(y0, ny, h, dtype)
            x_values = self._norm_axis(x0, nx, w, dtype)
            x_coord = x_values.view(1, 1, w).expand(d, h, w)
            y_coord = y_values.view(1, h, 1).expand(d, h, w)
            z_coord = z_values.view(d, 1, 1).expand(d, h, w)
            extra_channels.extend([x_coord, y_coord, z_coord])
        if self.append_fluid_mask:
            extra_channels.append(m[0].to(dtype))
        if self.append_height_gate:
            extra_channels.append(gate)
        if extra_channels:
            x = torch.cat((x, torch.stack(extra_channels, dim=0)), dim=0)
        return x, y, m


# Loads samples for the V13Corrected Sidecar data pipeline.
class V13CorrectedSidecarDataset(Dataset):
    """Apply corrected-time V13 sidecar data to an existing cached patch dataset."""

    RAW_EXTRA_CHANNELS = {"x_norm", "y_norm", "z_norm", "fluid_mask", "height_gate"}

    # Load split files and prepare dataset state.
    def __init__(
        self,
        dataset: Dataset,
        sidecar_root: str,
        split: str,
        global_sample_size: int,
        height_gate_decay_levels: float,
        normalize_raw_extra_channels: bool = False,
    ) -> None:
        super().__init__()
        self.dataset = dataset
        self.sidecar_root = os.path.abspath(sidecar_root)
        self.split = split
        self.global_sample_size = int(global_sample_size)
        self.height_gate_decay_levels = float(height_gate_decay_levels)
        self.normalize_raw_extra_channels = bool(normalize_raw_extra_channels)

        split_dir = os.path.join(self.sidecar_root, split)
        manifest_path = os.path.join(split_dir, "manifest.json")
        norm_path = os.path.join(self.sidecar_root, "normalization.json")
        if not os.path.exists(manifest_path):
            raise FileNotFoundError(f"Missing V13 sidecar manifest: {manifest_path}")
        if not os.path.exists(norm_path):
            raise FileNotFoundError(f"Missing V13 normalization stats: {norm_path}")
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
        with open(norm_path, "r", encoding="utf-8") as f:
            norm = json.load(f)

        self.split_dir = split_dir
        self.shards = manifest["shards"]
        self.patch_size = tuple(int(v) for v in manifest["patch_size"])
        self.local_channels = list(manifest["local_channels"])
        self.global_channels = list(manifest["global_channels"])
        self.total = int(manifest["total"])
        self.metadata_root = manifest.get("metadata_root")
        if not self.metadata_root:
            raise RuntimeError("V13 sidecar manifest is missing metadata_root")
        meta_path = os.path.join(self.metadata_root, f"{split}_metadata.npz")
        if not os.path.exists(meta_path):
            raise FileNotFoundError(f"Missing V13 coordinate metadata: {meta_path}")
        meta = np.load(meta_path, allow_pickle=False)
        required = ("z0", "y0", "x0", "nz", "ny", "nx")
        missing = [key for key in required if key not in meta.files]
        if missing:
            raise RuntimeError(f"V13 metadata missing fields: {missing}")
        self.z0 = meta["z0"].astype(np.int64, copy=False)
        self.y0 = meta["y0"].astype(np.int64, copy=False)
        self.x0 = meta["x0"].astype(np.int64, copy=False)
        self.nz = meta["nz"].astype(np.int64, copy=False)
        self.ny = meta["ny"].astype(np.int64, copy=False)
        self.nx = meta["nx"].astype(np.int64, copy=False)
        if len(self.z0) < self.total:
            raise RuntimeError(f"V13 metadata has {len(self.z0)} rows but sidecar has {self.total} rows")
        if self.total > len(self.dataset):
            raise RuntimeError(f"V13 sidecar has {self.total} rows but base dataset has {len(self.dataset)} rows")
        if len(self.local_channels) != len(norm["local_mean"]):
            raise RuntimeError("V13 local normalization channel count does not match sidecar manifest")
        if len(self.global_channels) != len(norm["global_mean"]):
            raise RuntimeError("V13 global normalization channel count does not match sidecar manifest")

        self.local_mean = torch.tensor(norm["local_mean"], dtype=torch.float32).view(-1, 1, 1, 1)
        self.local_std = torch.tensor(norm["local_std"], dtype=torch.float32).clamp_min(1.0e-6).view(-1, 1, 1, 1)
        self.global_mean = torch.tensor(norm["global_mean"], dtype=torch.float32).view(-1, 1, 1, 1)
        self.global_std = torch.tensor(norm["global_std"], dtype=torch.float32).clamp_min(1.0e-6).view(-1, 1, 1, 1)
        if not self.normalize_raw_extra_channels:
            for i, name in enumerate(self.local_channels):
                if name in self.RAW_EXTRA_CHANNELS:
                    self.local_mean[i] = 0.0
                    self.local_std[i] = 1.0

        self.height_gate_channel_idx = self.local_channels.index("height_gate") if "height_gate" in self.local_channels else None
        self.appended_channels = [name for name in ("x_norm", "y_norm", "z_norm", "fluid_mask", "height_gate") if name in self.local_channels]
        self.dropped_input_channels = [14, 15]
        self._arrays: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        self.cumulative: list[int] = []
        total = 0
        for shard in self.shards:
            total += int(shard["count"])
            self.cumulative.append(total)
        if total != self.total:
            self.total = total

    # Return the number of available samples.
    def __len__(self) -> int:
        return self.total

    # Internal helper for load shard.
    def _load_shard(self, shard_idx: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        arrays = self._arrays.get(shard_idx)
        if arrays is not None:
            return arrays
        shard = self.shards[shard_idx]
        arrays = (
            np.load(os.path.join(self.split_dir, shard["corrected_emission"]), mmap_mode="r"),
            np.load(os.path.join(self.split_dir, shard["corrected_bg"]), mmap_mode="r"),
            np.load(os.path.join(self.split_dir, shard["global_context"]), mmap_mode="r"),
        )
        self._arrays[shard_idx] = arrays
        return arrays

    # Internal helper for drop month channels.
    @staticmethod
    def _drop_month_channels(x: torch.Tensor) -> torch.Tensor:
        keep = [i for i in range(x.shape[0]) if i not in (14, 15)]
        return x[keep]

    # Internal helper for norm axis.
    @staticmethod
    def _norm_axis(start: int, total: int, length: int, dtype: torch.dtype) -> torch.Tensor:
        denom = max(int(total) - 1, 1)
        values = (torch.arange(length, dtype=dtype) + float(start)) / float(denom)
        return values.mul(2.0).sub(1.0)

    # Internal helper for global grid.
    @staticmethod
    def _global_grid(d: int, out_h: int, out_w: int, y0: float, x0: float, full_h: float, full_w: float) -> torch.Tensor:
        z_values = torch.linspace(-1.0, 1.0, steps=d, dtype=torch.float32)
        y_values = ((torch.linspace(0.0, out_h - 1.0, steps=out_h, dtype=torch.float32) + y0) / max(full_h - 1.0, 1.0)) * 2.0 - 1.0
        x_values = ((torch.linspace(0.0, out_w - 1.0, steps=out_w, dtype=torch.float32) + x0) / max(full_w - 1.0, 1.0)) * 2.0 - 1.0
        zz = z_values.view(d, 1, 1).expand(d, out_h, out_w)
        yy = y_values.view(1, out_h, 1).expand(d, out_h, out_w)
        xx = x_values.view(1, 1, out_w).expand(d, out_h, out_w)
        return torch.stack((xx, yy, zz), dim=-1).contiguous()

    # Return one indexed sample in the format expected by the model.
    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if index < 0:
            index += self.total
        if index < 0 or index >= self.total:
            raise IndexError(index)

        x, y, m = self.dataset[index]
        shard_idx = bisect.bisect_right(self.cumulative, index)
        prev = 0 if shard_idx == 0 else self.cumulative[shard_idx - 1]
        local_idx = index - prev
        emission_arr, bg_arr, global_arr = self._load_shard(shard_idx)

        corrected_emission = torch.from_numpy(np.array(emission_arr[local_idx], dtype=np.float32, copy=True))
        corrected_bg = torch.from_numpy(np.array(bg_arr[local_idx], dtype=np.float32, copy=True))
        global_context = torch.from_numpy(np.array(global_arr[local_idx], dtype=np.float32, copy=True))

        d, h, w = x.shape[-3:]
        z0 = int(self.z0[index])
        y0 = int(self.y0[index])
        x0 = int(self.x0[index])
        nz = int(self.nz[index])
        ny = int(self.ny[index])
        nx = int(self.nx[index])
        if corrected_emission.shape != (h, w):
            raise RuntimeError(f"Corrected emission shape {tuple(corrected_emission.shape)} does not match patch {(h, w)}")
        if corrected_bg.shape[0] != d:
            raise RuntimeError(f"Corrected background length {corrected_bg.shape[0]} does not match patch depth {d}")

        x = x.clone()
        x[0] = corrected_emission.view(1, h, w).expand(d, h, w)
        bg_3d = corrected_bg.view(d, 1, 1).expand(d, h, w)
        x[1] = bg_3d
        x = self._drop_month_channels(x)

        dtype = x.dtype
        gate_z = torch.exp(-(torch.arange(d, dtype=dtype) + float(z0)) / max(self.height_gate_decay_levels, 1.0)).clamp(0.0, 1.0)
        gate = gate_z.view(d, 1, 1).expand(d, h, w)
        for channel_idx in (0, 7, 8, 9, 10, 11, 12, 13):
            if 0 <= channel_idx < x.shape[0]:
                x[channel_idx] = x[channel_idx] * gate

        z_values = self._norm_axis(z0, nz, d, dtype)
        y_values = self._norm_axis(y0, ny, h, dtype)
        x_values = self._norm_axis(x0, nx, w, dtype)
        x_coord = x_values.view(1, 1, w).expand(d, h, w)
        y_coord = y_values.view(1, h, 1).expand(d, h, w)
        z_coord = z_values.view(d, 1, 1).expand(d, h, w)
        x = torch.cat((x, torch.stack((x_coord, y_coord, z_coord, m[0].to(dtype), gate), dim=0)), dim=0)
        if x.shape[0] != len(self.local_channels):
            raise RuntimeError(f"V13 transformed channel count {x.shape[0]} does not match stats {len(self.local_channels)}")

        # V13 sidecar was built after month-channel removal and surface-height
        # gating; normalize to the train statistics computed from that exact
        # transformed input.
        x = (x - self.local_mean) / self.local_std
        global_context = (global_context - self.global_mean) / self.global_std

        target_residual = y - bg_3d.unsqueeze(0)
        global_grid = self._global_grid(
            d=d,
            out_h=self.global_sample_size,
            out_w=self.global_sample_size,
            y0=float(y0),
            x0=float(x0),
            full_h=float(ny),
            full_w=float(nx),
        )
        # Replace local-grid x/y coordinates using the raw coordinate channels
        # if they are present. This keeps the sidecar independent from metadata
        # files during training while still sampling the full-domain context at
        # the correct tile position.
        if "x_norm" in self.local_channels and "y_norm" in self.local_channels:
            x_coord = x[self.local_channels.index("x_norm")]
            y_coord = x[self.local_channels.index("y_norm")]
            xs = torch.linspace(0, w - 1, self.global_sample_size).round().long().clamp(0, w - 1)
            ys = torch.linspace(0, h - 1, self.global_sample_size).round().long().clamp(0, h - 1)
            xx = x_coord[:, ys][:, :, xs]
            yy = y_coord[:, ys][:, :, xs]
            zz = torch.linspace(-1.0, 1.0, steps=d, dtype=torch.float32).view(d, 1, 1).expand(d, self.global_sample_size, self.global_sample_size)
            global_grid = torch.stack((xx, yy, zz), dim=-1).contiguous()
        return x, global_context, global_grid, target_residual, m


# Loads samples for the V14Focused Sidecar data pipeline.
class V14FocusedSidecarDataset(Dataset):
    """Corrected residual dataset whose training signal is limited to selected global z layers."""

    RAW_EXTRA_CHANNELS = {"x_norm", "y_norm", "z_norm", "fluid_mask"}

    # Load split files and prepare dataset state.
    def __init__(
        self,
        dataset: Dataset,
        sidecar_root: str,
        normalization_root: str,
        split: str,
        global_sample_size: int,
        layer_min: int,
        layer_max: int,
        min_layer_overlap: int,
        use_global_context: bool = True,
        target_normalization_path: str | None = None,
        target_normalization_mode: str = "none",
        surface_channel_indices: tuple[int, ...] = (),
        height_gate_decay_levels: float = 8.0,
        append_height_gate: bool = False,
        exclude_months: tuple[int, ...] = (),
    ) -> None:
        super().__init__()
        self.dataset = dataset
        self.sidecar_root = os.path.abspath(sidecar_root)
        self.normalization_root = os.path.abspath(normalization_root)
        self.split = split
        self.global_sample_size = int(global_sample_size)
        self.layer_min = int(layer_min)
        self.layer_max = int(layer_max)
        self.min_layer_overlap = int(min_layer_overlap)
        self.use_global_context = bool(use_global_context)
        self.target_normalization_path = os.path.abspath(target_normalization_path) if target_normalization_path else None
        self.target_normalization_mode = str(target_normalization_mode)
        self.target_normalization = load_target_normalization(self.target_normalization_path)
        if self.target_normalization is None:
            self.target_normalization_mode = "none"
        elif self.target_normalization_mode == "none":
            self.target_normalization_mode = str(self.target_normalization.get("mode", "month"))
        self.surface_channel_indices = tuple(int(v) for v in surface_channel_indices)
        self.height_gate_decay_levels = float(height_gate_decay_levels)
        self.append_height_gate = bool(append_height_gate)
        self.exclude_months = set(int(v) for v in exclude_months)

        split_dir = os.path.join(self.sidecar_root, split)
        manifest_path = os.path.join(split_dir, "manifest.json")
        norm_path = os.path.join(self.normalization_root, "normalization.json")
        if not os.path.exists(manifest_path):
            raise FileNotFoundError(f"Missing corrected sidecar manifest: {manifest_path}")
        if not os.path.exists(norm_path):
            raise FileNotFoundError(f"Missing V14 normalization stats: {norm_path}")
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
        with open(norm_path, "r", encoding="utf-8") as f:
            norm = json.load(f)

        self.split_dir = split_dir
        self.shards = manifest["shards"]
        self.patch_size = tuple(int(v) for v in manifest["patch_size"])
        self.local_channels = list(norm["local_channels"])
        self.global_channels = list(norm["global_channels"]) if self.use_global_context else []
        self.total_source = int(manifest["total"])
        self.metadata_root = manifest.get("metadata_root")
        if not self.metadata_root:
            raise RuntimeError("Corrected sidecar manifest is missing metadata_root")
        meta_path = os.path.join(self.metadata_root, f"{split}_metadata.npz")
        if not os.path.exists(meta_path):
            raise FileNotFoundError(f"Missing coordinate metadata: {meta_path}")
        meta = np.load(meta_path, allow_pickle=False)
        required = ("z0", "y0", "x0", "nz", "ny", "nx")
        missing = [key for key in required if key not in meta.files]
        if missing:
            raise RuntimeError(f"Coordinate metadata missing fields: {missing}")
        if self.target_normalization is not None and "month" not in meta.files:
            raise RuntimeError("Coordinate metadata missing month field required by target normalization")
        self.z0 = meta["z0"].astype(np.int64, copy=False)
        self.y0 = meta["y0"].astype(np.int64, copy=False)
        self.x0 = meta["x0"].astype(np.int64, copy=False)
        self.nz = meta["nz"].astype(np.int64, copy=False)
        self.ny = meta["ny"].astype(np.int64, copy=False)
        self.nx = meta["nx"].astype(np.int64, copy=False)
        self.month = meta["month"].astype(np.int64, copy=False) if "month" in meta.files else np.ones_like(self.z0)
        if self.total_source > len(self.dataset):
            raise RuntimeError(f"Corrected sidecar has {self.total_source} rows but base dataset has {len(self.dataset)} rows")
        if len(self.z0) < self.total_source:
            raise RuntimeError(f"Coordinate metadata has {len(self.z0)} rows but sidecar has {self.total_source} rows")
        if len(self.local_channels) != len(norm["local_mean"]):
            raise RuntimeError("V14 local normalization channel count mismatch")
        if self.use_global_context and len(self.global_channels) != len(norm["global_mean"]):
            raise RuntimeError("V14 global normalization channel count mismatch")

        selected_path = os.path.join(self.normalization_root, f"{split}_selected_indices.npy")
        if os.path.exists(selected_path):
            selected = np.load(selected_path).astype(np.int64, copy=False)
            self.indices = [int(i) for i in selected if self._overlap_count(int(self.z0[int(i)]), self.patch_size[0]) >= self.min_layer_overlap]
        else:
            self.indices = [
                i
                for i in range(self.total_source)
                if self._overlap_count(int(self.z0[i]), self.patch_size[0]) >= self.min_layer_overlap
            ]
        if self.exclude_months:
            self.indices = [int(i) for i in self.indices if int(self.month[int(i)]) not in self.exclude_months]
        if not self.indices:
            raise RuntimeError(
                f"No {split} samples overlap global z={self.layer_min}-{self.layer_max} "
                f"by at least {self.min_layer_overlap} layers after excluding months "
                f"{sorted(self.exclude_months)}"
            )

        self.local_mean = torch.tensor(norm["local_mean"], dtype=torch.float32).view(-1, 1, 1, 1)
        self.local_std = torch.tensor(norm["local_std"], dtype=torch.float32).clamp_min(1.0e-6).view(-1, 1, 1, 1)
        if self.use_global_context:
            self.global_mean = torch.tensor(norm["global_mean"], dtype=torch.float32).view(-1, 1, 1, 1)
            self.global_std = torch.tensor(norm["global_std"], dtype=torch.float32).clamp_min(1.0e-6).view(-1, 1, 1, 1)
        else:
            self.global_mean = torch.empty((0, 1, 1, 1), dtype=torch.float32)
            self.global_std = torch.empty((0, 1, 1, 1), dtype=torch.float32)
        for i, name in enumerate(self.local_channels):
            if name in self.RAW_EXTRA_CHANNELS:
                self.local_mean[i] = 0.0
                self.local_std[i] = 1.0

        self.keep_month_channels = "month_sin" in self.local_channels and "month_cos" in self.local_channels
        self.effective_local_channels = list(self.local_channels)
        if self.append_height_gate and "height_gate" not in self.effective_local_channels:
            self.effective_local_channels.append("height_gate")
        self.height_gate_channel_idx = (
            self.effective_local_channels.index("height_gate") if "height_gate" in self.effective_local_channels else None
        )
        self.appended_channels = [name for name in ("x_norm", "y_norm", "z_norm", "fluid_mask") if name in self.local_channels]
        if self.append_height_gate and "height_gate" not in self.appended_channels:
            self.appended_channels.append("height_gate")
        self.dropped_input_channels = [] if self.keep_month_channels else [14, 15]
        self._arrays: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray | None]] = {}
        self.cumulative: list[int] = []
        total = 0
        for shard in self.shards:
            total += int(shard["count"])
            self.cumulative.append(total)

    # Return the number of available samples.
    def __len__(self) -> int:
        return len(self.indices)

    # Internal helper for overlap count.
    def _overlap_count(self, z0: int, depth: int) -> int:
        z1 = z0 + depth - 1
        return max(0, min(z1, self.layer_max) - max(z0, self.layer_min) + 1)

    # Internal helper for selected layer mask.
    def _selected_layer_mask(self, z0: int, depth: int, dtype: torch.dtype) -> torch.Tensor:
        global_z = torch.arange(depth, dtype=torch.int64) + int(z0)
        return ((global_z >= self.layer_min) & (global_z <= self.layer_max)).to(dtype)

    # Internal helper for normalize target residual.
    def _normalize_target_residual(
        self,
        target_residual: torch.Tensor,
        corrected_bg: torch.Tensor,
        selected_z: torch.Tensor,
        source_index: int,
    ) -> torch.Tensor:
        if self.target_normalization is None:
            return target_residual
        selected_bg = corrected_bg[selected_z.bool()] if bool(selected_z.bool().any()) else corrected_bg
        key = target_norm_group_key(
            self.target_normalization,
            self.target_normalization_mode,
            int(self.month[source_index]),
            selected_bg,
        )
        mean, std = target_norm_mean_std(self.target_normalization, key)
        return (target_residual - float(mean)) / float(std)

    # Internal helper for height gate.
    def _height_gate(self, z0: int, depth: int, height: int, width: int, dtype: torch.dtype) -> torch.Tensor:
        z_index = torch.arange(depth, dtype=dtype) + float(z0)
        gate_z = torch.exp(-z_index / max(self.height_gate_decay_levels, 1.0)).clamp(0.0, 1.0)
        return gate_z.view(depth, 1, 1).expand(depth, height, width)

    # Internal helper for load shard.
    def _load_shard(self, shard_idx: int) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
        arrays = self._arrays.get(shard_idx)
        if arrays is not None:
            return arrays
        shard = self.shards[shard_idx]
        arrays = (
            np.load(os.path.join(self.split_dir, shard["corrected_emission"]), mmap_mode="r"),
            np.load(os.path.join(self.split_dir, shard["corrected_bg"]), mmap_mode="r"),
            np.load(os.path.join(self.split_dir, shard["global_context"]), mmap_mode="r")
            if self.use_global_context
            else None,
        )
        self._arrays[shard_idx] = arrays
        return arrays

    # Internal helper for norm axis.
    @staticmethod
    def _norm_axis(start: int, total: int, length: int, dtype: torch.dtype) -> torch.Tensor:
        denom = max(int(total) - 1, 1)
        values = (torch.arange(length, dtype=dtype) + float(start)) / float(denom)
        return values.mul(2.0).sub(1.0)

    # Internal helper for drop month channels.
    @staticmethod
    def _drop_month_channels(x: torch.Tensor) -> torch.Tensor:
        keep = [i for i in range(x.shape[0]) if i not in (14, 15)]
        return x[keep]

    # Return one indexed sample in the format expected by the model.
    def __getitem__(
        self,
        index: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if index < 0:
            index += len(self.indices)
        if index < 0 or index >= len(self.indices):
            raise IndexError(index)
        source_index = self.indices[index]
        x, y, m = self.dataset[source_index]
        shard_idx = bisect.bisect_right(self.cumulative, source_index)
        prev = 0 if shard_idx == 0 else self.cumulative[shard_idx - 1]
        local_idx = source_index - prev
        emission_arr, bg_arr, global_arr = self._load_shard(shard_idx)

        corrected_emission = torch.from_numpy(np.array(emission_arr[local_idx], dtype=np.float32, copy=True))
        corrected_bg = torch.from_numpy(np.array(bg_arr[local_idx], dtype=np.float32, copy=True))
        global_context = (
            torch.from_numpy(np.array(global_arr[local_idx], dtype=np.float32, copy=True))
            if global_arr is not None
            else None
        )

        d, h, w = x.shape[-3:]
        z0 = int(self.z0[source_index])
        y0 = int(self.y0[source_index])
        x0 = int(self.x0[source_index])
        nz = int(self.nz[source_index])
        ny = int(self.ny[source_index])
        nx = int(self.nx[source_index])
        if corrected_emission.shape != (h, w):
            raise RuntimeError(f"Corrected emission shape {tuple(corrected_emission.shape)} does not match patch {(h, w)}")
        if corrected_bg.shape[0] != d:
            raise RuntimeError(f"Corrected background length {corrected_bg.shape[0]} does not match patch depth {d}")

        x = x.clone()
        x[0] = corrected_emission.view(1, h, w).expand(d, h, w)
        bg_3d = corrected_bg.view(d, 1, 1).expand(d, h, w)
        x[1] = bg_3d
        if not self.keep_month_channels:
            x = self._drop_month_channels(x)

        dtype = x.dtype
        z_values = self._norm_axis(z0, nz, d, dtype)
        y_values = self._norm_axis(y0, ny, h, dtype)
        x_values = self._norm_axis(x0, nx, w, dtype)
        x_coord = x_values.view(1, 1, w).expand(d, h, w)
        y_coord = y_values.view(1, h, 1).expand(d, h, w)
        z_coord = z_values.view(d, 1, 1).expand(d, h, w)
        extras = {"x_norm": x_coord, "y_norm": y_coord, "z_norm": z_coord, "fluid_mask": m[0].to(dtype)}
        extra_tensors = [extras[name] for name in self.local_channels[x.shape[0] :]]
        if extra_tensors:
            x = torch.cat((x, torch.stack(extra_tensors, dim=0)), dim=0)
        if x.shape[0] != len(self.local_channels):
            raise RuntimeError(f"V14 transformed channel count {x.shape[0]} does not match stats {len(self.local_channels)}")

        x = (x - self.local_mean) / self.local_std
        gate = self._height_gate(z0, d, h, w, x.dtype)
        if self.surface_channel_indices:
            x = x.clone()
            for channel_idx in self.surface_channel_indices:
                if 0 <= channel_idx < x.shape[0]:
                    x[channel_idx] = x[channel_idx] * gate
        if self.append_height_gate and "height_gate" not in self.local_channels:
            x = torch.cat((x, gate.unsqueeze(0)), dim=0)
        if global_context is not None:
            global_context = (global_context - self.global_mean) / self.global_std

        selected_z = self._selected_layer_mask(z0, d, m.dtype).view(1, d, 1, 1)
        focused_mask = m * selected_z
        target_residual = y - bg_3d.unsqueeze(0)
        target_residual = self._normalize_target_residual(target_residual, corrected_bg, selected_z.view(-1), source_index)

        if global_context is None:
            return x, target_residual, focused_mask

        xs = torch.linspace(0, w - 1, self.global_sample_size).round().long().clamp(0, w - 1)
        ys = torch.linspace(0, h - 1, self.global_sample_size).round().long().clamp(0, h - 1)
        xx = x_coord[:, ys][:, :, xs]
        yy = y_coord[:, ys][:, :, xs]
        zz = torch.linspace(-1.0, 1.0, steps=d, dtype=torch.float32).view(d, 1, 1).expand(d, self.global_sample_size, self.global_sample_size)
        global_grid = torch.stack((xx, yy, zz), dim=-1).contiguous()
        return x, global_context, global_grid, target_residual, focused_mask


# Loads samples for the V22Autoregressive Sidecar data pipeline.
class V22AutoregressiveSidecarDataset(V14FocusedSidecarDataset):
    """Use previous-timestep CO2 as input and train on CO2(t)-CO2(t-1)."""

    # Load split files and prepare dataset state.
    def __init__(self, *args, prev_sidecar_root: str, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if self.target_normalization is not None:
            raise RuntimeError("V22 autoregressive delta training should not use corrected-residual target normalization")
        self.prev_sidecar_root = os.path.abspath(prev_sidecar_root)
        split_dir = os.path.join(self.prev_sidecar_root, self.split)
        manifest_path = os.path.join(split_dir, "manifest.json")
        norm_path = os.path.join(self.prev_sidecar_root, "normalization.json")
        if not os.path.exists(manifest_path):
            raise FileNotFoundError(f"Missing V22 previous-CO2 sidecar manifest: {manifest_path}")
        if not os.path.exists(norm_path):
            raise FileNotFoundError(f"Missing V22 previous-CO2 normalization stats: {norm_path}")
        with open(manifest_path, "r", encoding="utf-8") as f:
            prev_manifest = json.load(f)
        with open(norm_path, "r", encoding="utf-8") as f:
            prev_norm = json.load(f)
        self.prev_split_dir = split_dir
        self.prev_shards = prev_manifest["shards"]
        self.prev_total = int(prev_manifest["total"])
        if self.prev_total < self.total_source:
            raise RuntimeError(f"V22 previous-CO2 sidecar has {self.prev_total} rows but corrected sidecar has {self.total_source}")

        self.prev_local_mean = float(prev_norm.get("local_prev_mean", 0.0))
        self.prev_local_std = max(float(prev_norm.get("local_prev_std", 1.0)), 1.0e-6)
        self.prev_global_mean = float(prev_norm.get("global_prev_mean", 0.0))
        self.prev_global_std = max(float(prev_norm.get("global_prev_std", 1.0)), 1.0e-6)
        self.prev_local_channel = str(prev_norm.get("local_prev_channel", "prev_kc_CO2"))
        self.prev_global_channel = str(prev_norm.get("global_prev_channel", "prev_kc_CO2_global"))

        self.prev_cumulative: list[int] = []
        total = 0
        has_prev_parts = []
        for shard in self.prev_shards:
            total += int(shard["count"])
            self.prev_cumulative.append(total)
            has_prev_parts.append(np.load(os.path.join(self.prev_split_dir, shard["has_prev"]), mmap_mode="r"))
        self.has_prev = np.concatenate([np.asarray(v, dtype=np.uint8) for v in has_prev_parts], axis=0)
        if len(self.has_prev) < self.total_source:
            raise RuntimeError("V22 previous-CO2 has_prev vector is shorter than the corrected sidecar")
        self.indices = [int(i) for i in self.indices if int(self.has_prev[int(i)]) > 0]
        if not self.indices:
            raise RuntimeError(f"No {self.split} samples have a previous timestep after layer filtering")

        self.effective_local_channels = list(self.effective_local_channels) + [self.prev_local_channel]
        self.global_channels = list(self.global_channels) + [self.prev_global_channel] if self.use_global_context else []
        self.appended_channels = list(self.appended_channels) + [self.prev_local_channel]
        self._prev_arrays: dict[int, tuple[np.ndarray, np.ndarray]] = {}

    # Internal helper for load prev shard.
    def _load_prev_shard(self, shard_idx: int) -> tuple[np.ndarray, np.ndarray]:
        arrays = self._prev_arrays.get(shard_idx)
        if arrays is not None:
            return arrays
        shard = self.prev_shards[shard_idx]
        arrays = (
            np.load(os.path.join(self.prev_split_dir, shard["prev_co2"]), mmap_mode="r"),
            np.load(os.path.join(self.prev_split_dir, shard["prev_global_co2"]), mmap_mode="r"),
        )
        self._prev_arrays[shard_idx] = arrays
        return arrays

    # Internal helper for prev for source index.
    def _prev_for_source_index(self, source_index: int) -> tuple[torch.Tensor, torch.Tensor]:
        shard_idx = bisect.bisect_right(self.prev_cumulative, source_index)
        prev = 0 if shard_idx == 0 else self.prev_cumulative[shard_idx - 1]
        local_idx = source_index - prev
        prev_arr, prev_global_arr = self._load_prev_shard(shard_idx)
        prev_co2 = torch.from_numpy(np.array(prev_arr[local_idx], dtype=np.float32, copy=True))
        prev_global = torch.from_numpy(np.array(prev_global_arr[local_idx], dtype=np.float32, copy=True))
        return prev_co2, prev_global

    # Return one indexed sample in the format expected by the model.
    def __getitem__(
        self,
        index: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if index < 0:
            index += len(self.indices)
        if index < 0 or index >= len(self.indices):
            raise IndexError(index)
        source_index = self.indices[index]
        prev_co2, prev_global = self._prev_for_source_index(source_index)
        base = super().__getitem__(index)
        shard_idx = bisect.bisect_right(self.cumulative, source_index)
        prev = 0 if shard_idx == 0 else self.cumulative[shard_idx - 1]
        local_idx = source_index - prev
        _, bg_arr, _ = self._load_shard(shard_idx)
        corrected_bg = torch.from_numpy(np.array(bg_arr[local_idx], dtype=np.float32, copy=True))
        bg_3d = corrected_bg.view(prev_co2.shape[0], 1, 1).expand_as(prev_co2)
        if len(base) == 3:
            x_base, target_residual, focused_mask = base
            prev_norm = ((prev_co2 - self.prev_local_mean) / self.prev_local_std).to(x_base.dtype)
            x_out = torch.cat((x_base, prev_norm.unsqueeze(0)), dim=0)
            target_delta = target_residual + bg_3d.unsqueeze(0) - prev_co2.unsqueeze(0)
            return x_out, target_delta, focused_mask

        x_base, global_context, global_grid, target_residual, focused_mask = base
        prev_norm = ((prev_co2 - self.prev_local_mean) / self.prev_local_std).to(x_base.dtype)
        x_out = torch.cat((x_base, prev_norm.unsqueeze(0)), dim=0)
        prev_global_norm = ((prev_global - self.prev_global_mean) / self.prev_global_std).to(global_context.dtype)
        global_out = torch.cat((global_context, prev_global_norm.unsqueeze(0)), dim=0)
        target_delta = target_residual + bg_3d.unsqueeze(0) - prev_co2.unsqueeze(0)
        return x_out, global_out, global_grid, target_delta, focused_mask


# Compute 3D finite differences for gradient terms.
def finite_difference_3d(field: torch.Tensor, spacing: float, dim: int) -> torch.Tensor:
    """Finite difference with non-periodic boundaries, preserving field shape."""
    spacing = max(float(spacing), 1.0e-6)
    out = torch.zeros_like(field)
    n = int(field.shape[dim])
    if n <= 1:
        return out

    first = [slice(None)] * field.ndim
    first[dim] = 0
    second = [slice(None)] * field.ndim
    second[dim] = 1
    out[tuple(first)] = (field[tuple(second)] - field[tuple(first)]) / spacing

    last = [slice(None)] * field.ndim
    last[dim] = n - 1
    before_last = [slice(None)] * field.ndim
    before_last[dim] = n - 2
    out[tuple(last)] = (field[tuple(last)] - field[tuple(before_last)]) / spacing

    if n > 2:
        middle = [slice(None)] * field.ndim
        middle[dim] = slice(1, n - 1)
        plus = [slice(None)] * field.ndim
        plus[dim] = slice(2, n)
        minus = [slice(None)] * field.ndim
        minus[dim] = slice(0, n - 2)
        out[tuple(middle)] = (field[tuple(plus)] - field[tuple(minus)]) / (2.0 * spacing)
    return out


# Loads samples for the V28Advection Sidecar data pipeline.
class V28AdvectionSidecarDataset(V22AutoregressiveSidecarDataset):
    """Append previous-CO2 gradient/advection features and optionally train correction to physics delta."""

    advection_channel_names = [
        "prev_dCdx",
        "prev_dCdy",
        "prev_dCdz",
        "adv_x_delta",
        "adv_y_delta",
        "adv_z_delta",
        "adv_total_delta",
    ]

    # Load split files and prepare dataset state.
    def __init__(
        self,
        *args,
        advection_dx: float,
        advection_dy: float,
        advection_dz: float,
        advection_dt: float,
        advection_delta_scale: float,
        advection_gradient_scale: float,
        advection_clip: float,
        advection_input_clip: float,
        correction_target: bool,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.advection_dx = float(advection_dx)
        self.advection_dy = float(advection_dy)
        self.advection_dz = float(advection_dz)
        self.advection_dt = float(advection_dt)
        self.advection_delta_scale = max(float(advection_delta_scale), 1.0e-6)
        self.advection_gradient_scale = max(float(advection_gradient_scale), 1.0e-6)
        self.advection_clip = float(advection_clip)
        self.advection_input_clip = float(advection_input_clip)
        self.correction_target = bool(correction_target)
        self.effective_local_channels = list(self.effective_local_channels) + list(self.advection_channel_names)
        self.appended_channels = list(self.appended_channels) + list(self.advection_channel_names)
        self._channel_to_idx = {name: idx for idx, name in enumerate(self.effective_local_channels)}

    # Internal helper for denormalized channel.
    def _denormalized_channel(self, x_out: torch.Tensor, name: str) -> torch.Tensor:
        if name not in self._channel_to_idx:
            raise RuntimeError(f"Missing channel required for V28 advection: {name}")
        idx = int(self._channel_to_idx[name])
        if idx >= len(self.local_mean):
            raise RuntimeError(f"Cannot denormalize appended channel {name}")
        return x_out[idx] * self.local_std[idx].to(x_out.dtype) + self.local_mean[idx].to(x_out.dtype)

    # Internal helper for advection from prev.
    def _advection_from_prev(self, x_out: torch.Tensor, prev_co2: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        u = self._denormalized_channel(x_out, "u")
        v = self._denormalized_channel(x_out, "v")
        w = self._denormalized_channel(x_out, "w")

        dc_dx = finite_difference_3d(prev_co2, self.advection_dx, dim=-1)
        dc_dy = finite_difference_3d(prev_co2, self.advection_dy, dim=-2)
        dc_dz = finite_difference_3d(prev_co2, self.advection_dz, dim=-3)

        adv_x = -u * dc_dx * self.advection_dt
        adv_y = -v * dc_dy * self.advection_dt
        adv_z = -w * dc_dz * self.advection_dt
        adv_total = adv_x + adv_y + adv_z
        if self.advection_clip > 0:
            adv_x = adv_x.clamp(-self.advection_clip, self.advection_clip)
            adv_y = adv_y.clamp(-self.advection_clip, self.advection_clip)
            adv_z = adv_z.clamp(-self.advection_clip, self.advection_clip)
            adv_total = adv_total.clamp(-self.advection_clip, self.advection_clip)

        features = torch.stack(
            (
                dc_dx / self.advection_gradient_scale,
                dc_dy / self.advection_gradient_scale,
                dc_dz / self.advection_gradient_scale,
                adv_x / self.advection_delta_scale,
                adv_y / self.advection_delta_scale,
                adv_z / self.advection_delta_scale,
                adv_total / self.advection_delta_scale,
            ),
            dim=0,
        ).to(x_out.dtype)
        features = torch.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
        if self.advection_input_clip > 0:
            features = features.clamp(-self.advection_input_clip, self.advection_input_clip)
        return adv_total.to(x_out.dtype), features

    # Return one indexed sample in the format expected by the model.
    def __getitem__(
        self,
        index: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if index < 0:
            source_index = self.indices[index + len(self.indices)]
        else:
            source_index = self.indices[index]
        prev_co2, _ = self._prev_for_source_index(int(source_index))
        base = super().__getitem__(index)

        if len(base) == 3:
            x_out, target_delta, focused_mask = base
            physics_delta, adv_features = self._advection_from_prev(x_out, prev_co2)
            x_out = torch.cat((x_out, adv_features), dim=0)
            if self.correction_target:
                target_delta = target_delta - physics_delta.unsqueeze(0)
            return x_out, target_delta, focused_mask

        x_out, global_out, global_grid, target_delta, focused_mask = base
        physics_delta, adv_features = self._advection_from_prev(x_out, prev_co2)
        x_out = torch.cat((x_out, adv_features), dim=0)
        if self.correction_target:
            target_delta = target_delta - physics_delta.unsqueeze(0)
        return x_out, global_out, global_grid, target_delta, focused_mask


# Split available jobs by month.
def split_months(split: str) -> set[int]:
    if split == "train":
        return set(range(1, 9))
    if split == "val":
        return {9, 10}
    if split == "test":
        return {11, 12}
    raise ValueError(f"Unknown split: {split}")


# Discover available PALM simulation jobs.
def discover_jobs(jobs_root: str) -> list[str]:
    names: list[str] = []
    for name in sorted(os.listdir(jobs_root)):
        full = os.path.join(jobs_root, name)
        if os.path.isdir(full) and (name.startswith("z") or name.startswith("Richmond_")):
            names.append(name)
    return names


# Tracks concentration-delta diagnostics during training.
class DeltaMonitor:
    """Lightweight validation monitor for autoregressive delta pattern quality."""

    # Store constructor arguments and initialize object state.
    def __init__(
        self,
        active_threshold: float,
        min_valid: int,
        hard_min_std: float,
        hard_min_active_fraction: float,
        eps: float = 1.0e-6,
    ) -> None:
        self.active_threshold = float(active_threshold)
        self.min_valid = int(min_valid)
        self.hard_min_std = float(hard_min_std)
        self.hard_min_active_fraction = float(hard_min_active_fraction)
        self.eps = float(eps)
        self.slice_r_sum = 0.0
        self.slice_r_count = 0
        self.hard_r_sum = 0.0
        self.hard_r_count = 0
        self.sign_sum = 0.0
        self.sign_count = 0
        self.amp_sum = 0.0
        self.amp_count = 0
        self.active_fraction_sum = 0.0
        self.active_fraction_count = 0

    # Update running metric or statistic accumulators.
    def update(self, pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> None:
        p = pred.detach()
        t = target.detach()
        m = mask.detach()
        if p.ndim != 5 or t.ndim != 5 or m.ndim != 5:
            return
        if p.shape[1] != 1:
            p = p[:, :1]
        if t.shape[1] != 1:
            t = t[:, :1]
        if m.shape[1] != 1:
            m = m[:, :1]

        p = p[:, 0].reshape(-1, p.shape[-2] * p.shape[-1])
        t = t[:, 0].reshape(-1, t.shape[-2] * t.shape[-1])
        m = (m[:, 0].reshape(-1, m.shape[-2] * m.shape[-1]) > 0.5).to(p.dtype)
        valid_count = m.sum(dim=1)
        valid = valid_count >= self.min_valid
        if not bool(valid.any()):
            return

        denom = valid_count.clamp_min(1.0)
        p_mean = (p * m).sum(dim=1) / denom
        t_mean = (t * m).sum(dim=1) / denom
        pc = (p - p_mean[:, None]) * m
        tc = (t - t_mean[:, None]) * m
        p_var = (pc * pc).sum(dim=1) / denom
        t_var = (tc * tc).sum(dim=1) / denom
        p_std = torch.sqrt(p_var + self.eps)
        t_std = torch.sqrt(t_var + self.eps)
        cov = (pc * tc).sum(dim=1) / denom
        r_valid = valid & (p_std > self.eps) & (t_std > self.eps)
        if bool(r_valid.any()):
            r = cov[r_valid] / (p_std[r_valid] * t_std[r_valid]).clamp_min(self.eps)
            r = torch.clamp(r, -1.0, 1.0)
            self.slice_r_sum += float(r.sum().item())
            self.slice_r_count += int(r.numel())
            active_cells = ((t.abs() >= self.active_threshold).to(p.dtype) * m).sum(dim=1)
            active_fraction = active_cells / denom
            hard = r_valid & ((t_std >= self.hard_min_std) | (active_fraction >= self.hard_min_active_fraction))
            if bool(hard.any()):
                hard_r = cov[hard] / (p_std[hard] * t_std[hard]).clamp_min(self.eps)
                hard_r = torch.clamp(hard_r, -1.0, 1.0)
                self.hard_r_sum += float(hard_r.sum().item())
                self.hard_r_count += int(hard_r.numel())
            amp = p_std[r_valid] / t_std[r_valid].clamp_min(self.eps)
            self.amp_sum += float(amp.sum().item())
            self.amp_count += int(amp.numel())
            self.active_fraction_sum += float(active_fraction[r_valid].sum().item())
            self.active_fraction_count += int(active_fraction[r_valid].numel())

        active = (t.abs() >= self.active_threshold).to(p.dtype) * m
        active_count = active.sum()
        if float(active_count.item()) > 0.0:
            sign_match = ((p >= 0.0) == (t >= 0.0)).to(p.dtype)
            self.sign_sum += float((sign_match * active).sum().item())
            self.sign_count += int(active_count.item())

    # Return a compact metrics summary.
    def summary(self) -> dict[str, float]:
        delta_r = self.slice_r_sum / self.slice_r_count if self.slice_r_count else float("nan")
        hard_delta_r = self.hard_r_sum / self.hard_r_count if self.hard_r_count else float("nan")
        sign_accuracy = self.sign_sum / self.sign_count if self.sign_count else float("nan")
        amplitude_ratio = self.amp_sum / self.amp_count if self.amp_count else float("nan")
        active_fraction = (
            self.active_fraction_sum / self.active_fraction_count if self.active_fraction_count else float("nan")
        )
        amp_penalty = abs(math.log(max(amplitude_ratio, self.eps))) if math.isfinite(amplitude_ratio) else 0.0
        hard_term = hard_delta_r if math.isfinite(hard_delta_r) else delta_r
        sign_term = sign_accuracy if math.isfinite(sign_accuracy) else 0.0
        pattern_score = hard_term + 0.20 * sign_term - 0.05 * amp_penalty
        return {
            "delta_r_mean": delta_r,
            "delta_r_count": float(self.slice_r_count),
            "hard_delta_r_mean": hard_delta_r,
            "hard_delta_r_count": float(self.hard_r_count),
            "sign_accuracy": sign_accuracy,
            "sign_count": float(self.sign_count),
            "amplitude_ratio": amplitude_ratio,
            "active_fraction": active_fraction,
            "pattern_score": pattern_score,
        }


# Run one training or validation epoch.
def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    desc: str,
    epoch: int,
    log_every: int,
    huber_delta: float,
    base_loss: str,
    gradient_loss_weight: float,
    multiscale_loss_weight: float,
    multiscale_scales: tuple[int, ...],
    multiscale_min_valid_fraction: float,
    smoothness_loss_weight: float,
    smoothness_kernel_size: int,
    height_gate_channel_idx: int | None,
    variance_loss_weight: float,
    variance_min_std: float,
    variance_eps: float,
    residual_weight_alpha: float,
    residual_weight_scale: float,
    residual_weight_max: float,
    target_gradient_weight_alpha: float,
    target_gradient_weight_scale: float,
    target_gradient_weight_max: float,
    low_layer_weight_alpha: float,
    normalized_loss_weight: float,
    normalized_min_std: float,
    normalized_huber_delta: float,
    normalized_eps: float,
    correlation_loss_weight: float,
    correlation_eps: float,
    correlation_min_target_std: float,
    correlation_min_valid_fraction: float,
    low_frequency_loss_weight: float,
    low_frequency_pool: int,
    low_frequency_min_valid_fraction: float,
    low_frequency_correlation_weight: float,
    high_frequency_loss_weight: float,
    high_frequency_huber_delta: float,
    local_correlation_loss_weight: float,
    local_correlation_pool: int,
    local_correlation_min_target_std: float,
    local_correlation_min_valid_fraction: float,
    amplitude_loss_weight: float,
    amplitude_min_target_std: float,
    active_delta_loss_weight: float,
    active_delta_threshold: float,
    sign_loss_weight: float,
    sign_loss_min_abs: float,
    sign_loss_scale: float,
    active_loss_weight: float,
    active_loss_threshold: float,
    active_loss_pos_weight: float,
    sign_class_loss_weight: float,
    sign_class_loss_min_abs: float,
    sign_class_loss_pos_weight: float,
    pattern_height_decay: bool,
    progress: str,
    collect_delta_monitor: bool = False,
    delta_monitor_active_threshold: float = 0.75,
    delta_monitor_min_valid: int = 256,
    delta_monitor_hard_min_std: float = 1.0,
    delta_monitor_hard_min_active_fraction: float = 0.25,
) -> float | tuple[float, dict[str, float]]:
    train_mode = optimizer is not None
    model.train(train_mode)

    total = 0.0
    total_base = 0.0
    total_gradient = 0.0
    total_multiscale = 0.0
    total_smoothness = 0.0
    total_variance = 0.0
    total_normalized = 0.0
    total_correlation = 0.0
    total_local_correlation = 0.0
    total_amplitude = 0.0
    total_active_delta = 0.0
    total_low_frequency = 0.0
    total_low_frequency_corr = 0.0
    total_high_frequency = 0.0
    total_sign = 0.0
    total_active_class = 0.0
    total_sign_class = 0.0
    total_residual_weight = 0.0
    total_texture_weight = 0.0
    count = 0
    n_steps = len(loader)
    epoch_start = time.time()
    delta_monitor = (
        DeltaMonitor(
            active_threshold=delta_monitor_active_threshold,
            min_valid=delta_monitor_min_valid,
            hard_min_std=delta_monitor_hard_min_std,
            hard_min_active_fraction=delta_monitor_hard_min_active_fraction,
        )
        if collect_delta_monitor and not train_mode
        else None
    )

    if progress == "tqdm":
        iterator = tqdm(loader, desc=desc, leave=False)
    else:
        iterator = loader
        if progress == "summary":
            phase = "train" if train_mode else "val"
            print(f"[{phase}] epoch={epoch:03d} start steps={n_steps}", flush=True)

    for step, batch in enumerate(iterator, start=1):
        if len(batch) == 5:
            x, global_context, global_grid, y, m = batch
        elif len(batch) == 3:
            x, y, m = batch
            global_context = None
            global_grid = None
        else:
            raise RuntimeError(f"Unexpected batch format with {len(batch)} tensors")
        x = x.to(device, non_blocking=True)
        if global_context is not None:
            global_context = global_context.to(device, non_blocking=True)
        if global_grid is not None:
            global_grid = global_grid.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        m = m.to(device, non_blocking=True)

        with torch.set_grad_enabled(train_mode):
            model_out = model(x, global_context=global_context, global_grid=global_grid, return_components=True)
            if isinstance(model_out, dict):
                pred = model_out["final"]
                components = model_out
            else:
                pred = model_out
                components = None
            loss, parts = combined_masked_loss(
                pred,
                y,
                m,
                input_x=x,
                components=components,
                base_loss=base_loss,
                huber_delta=huber_delta,
                gradient_loss_weight=gradient_loss_weight,
                multiscale_loss_weight=multiscale_loss_weight,
                multiscale_scales=multiscale_scales,
                multiscale_min_valid_fraction=multiscale_min_valid_fraction,
                smoothness_loss_weight=smoothness_loss_weight,
                smoothness_kernel_size=smoothness_kernel_size,
                height_gate_channel_idx=height_gate_channel_idx,
                variance_loss_weight=variance_loss_weight,
                variance_min_std=variance_min_std,
                variance_eps=variance_eps,
                residual_weight_alpha=residual_weight_alpha,
                residual_weight_scale=residual_weight_scale,
                residual_weight_max=residual_weight_max,
                target_gradient_weight_alpha=target_gradient_weight_alpha,
                target_gradient_weight_scale=target_gradient_weight_scale,
                target_gradient_weight_max=target_gradient_weight_max,
                low_layer_weight_alpha=low_layer_weight_alpha,
                normalized_loss_weight=normalized_loss_weight,
                normalized_min_std=normalized_min_std,
                normalized_huber_delta=normalized_huber_delta,
                normalized_eps=normalized_eps,
                correlation_loss_weight=correlation_loss_weight,
                correlation_eps=correlation_eps,
                correlation_min_target_std=correlation_min_target_std,
                correlation_min_valid_fraction=correlation_min_valid_fraction,
                low_frequency_loss_weight=low_frequency_loss_weight,
                low_frequency_pool=low_frequency_pool,
                low_frequency_min_valid_fraction=low_frequency_min_valid_fraction,
                low_frequency_correlation_weight=low_frequency_correlation_weight,
                high_frequency_loss_weight=high_frequency_loss_weight,
                high_frequency_huber_delta=high_frequency_huber_delta,
                local_correlation_loss_weight=local_correlation_loss_weight,
                local_correlation_pool=local_correlation_pool,
                local_correlation_min_target_std=local_correlation_min_target_std,
                local_correlation_min_valid_fraction=local_correlation_min_valid_fraction,
                amplitude_loss_weight=amplitude_loss_weight,
                amplitude_min_target_std=amplitude_min_target_std,
                active_delta_loss_weight=active_delta_loss_weight,
                active_delta_threshold=active_delta_threshold,
                sign_loss_weight=sign_loss_weight,
                sign_loss_min_abs=sign_loss_min_abs,
                sign_loss_scale=sign_loss_scale,
                active_loss_weight=active_loss_weight,
                active_loss_threshold=active_loss_threshold,
                active_loss_pos_weight=active_loss_pos_weight,
                sign_class_loss_weight=sign_class_loss_weight,
                sign_class_loss_min_abs=sign_class_loss_min_abs,
                sign_class_loss_pos_weight=sign_class_loss_pos_weight,
                pattern_height_decay=pattern_height_decay,
            )

        if train_mode:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        elif delta_monitor is not None:
            delta_monitor.update(pred, y, m)

        total += float(loss.item())
        total_base += parts["base"]
        total_gradient += parts["gradient"]
        total_multiscale += parts["multiscale"]
        total_smoothness += parts["smoothness"]
        total_variance += parts["variance"]
        total_normalized += parts["normalized"]
        total_correlation += parts["correlation"]
        total_local_correlation += parts["local_correlation"]
        total_amplitude += parts["amplitude"]
        total_active_delta += parts["active_delta"]
        total_low_frequency += parts["low_frequency"]
        total_low_frequency_corr += parts["low_frequency_corr"]
        total_high_frequency += parts["high_frequency"]
        total_sign += parts["sign"]
        total_active_class += parts["active_class"]
        total_sign_class += parts["sign_class"]
        total_residual_weight += parts["residual_weight_mean"]
        total_texture_weight += parts["texture_weight_mean"]
        count += 1
        if progress == "tqdm":
            iterator.set_postfix(
                loss=f"{loss.item():.4f}",
                base=f"{parts['base']:.3f}",
                grad=f"{parts['gradient']:.3f}",
                ms=f"{parts['multiscale']:.3f}",
                smooth=f"{parts['smoothness']:.3f}",
                var=f"{parts['variance']:.3f}",
                norm=f"{parts['normalized']:.3f}",
                corr=f"{parts['correlation']:.3f}",
                lcorr=f"{parts['local_correlation']:.3f}",
                amp=f"{parts['amplitude']:.3f}",
                adel=f"{parts['active_delta']:.3f}",
                low=f"{parts['low_frequency']:.3f}",
                high=f"{parts['high_frequency']:.3f}",
                sign=f"{parts['sign']:.3f}",
                act=f"{parts['active_class']:.3f}",
                scls=f"{parts['sign_class']:.3f}",
            )

        should_log_step = progress == "summary" and (step % max(1, log_every) == 0 or step == n_steps)
        if should_log_step:
            elapsed = time.time() - epoch_start
            avg_loss = total / count
            avg_base = total_base / count
            avg_gradient = total_gradient / count
            avg_multiscale = total_multiscale / count
            avg_smoothness = total_smoothness / count
            avg_variance = total_variance / count
            avg_normalized = total_normalized / count
            avg_correlation = total_correlation / count
            avg_local_correlation = total_local_correlation / count
            avg_amplitude = total_amplitude / count
            avg_active_delta = total_active_delta / count
            avg_low_frequency = total_low_frequency / count
            avg_low_frequency_corr = total_low_frequency_corr / count
            avg_high_frequency = total_high_frequency / count
            avg_sign = total_sign / count
            avg_active_class = total_active_class / count
            avg_sign_class = total_sign_class / count
            avg_residual_weight = total_residual_weight / count
            avg_texture_weight = total_texture_weight / count
            phase = "train" if train_mode else "val"
            print(
                f"[{phase}] epoch={epoch:03d} step={step:04d}/{n_steps:04d} "
                f"loss={loss.item():.6f} avg_loss={avg_loss:.6f} "
                f"base={avg_base:.6f} grad={avg_gradient:.6f} "
                f"multiscale={avg_multiscale:.6f} smooth={avg_smoothness:.6f} "
                f"var={avg_variance:.6f} norm={avg_normalized:.6f} corr={avg_correlation:.6f} "
                f"local_corr={avg_local_correlation:.6f} amp={avg_amplitude:.6f} "
                f"active_delta={avg_active_delta:.6f} "
                f"low={avg_low_frequency:.6f} low_corr={avg_low_frequency_corr:.6f} "
                f"high={avg_high_frequency:.6f} sign={avg_sign:.6f} "
                f"active_cls={avg_active_class:.6f} sign_cls={avg_sign_class:.6f} "
                f"res_w={avg_residual_weight:.3f} tex_w={avg_texture_weight:.3f} "
                f"elapsed={elapsed:.1f}s",
                flush=True,
            )

    if progress == "none":
        phase = "train" if train_mode else "val"
        elapsed = time.time() - epoch_start
        print(
            f"[{phase}] epoch={epoch:03d} done steps={n_steps} "
            f"avg_loss={total / max(1, count):.6f} elapsed={elapsed:.1f}s",
            flush=True,
        )

    avg_epoch_loss = total / max(1, count)
    if delta_monitor is not None:
        metrics = delta_monitor.summary()
        print(
            f"[monitor] epoch={epoch:03d} "
            f"delta_R={metrics['delta_r_mean']:.6f} "
            f"hard_delta_R={metrics['hard_delta_r_mean']:.6f} "
            f"sign_acc={metrics['sign_accuracy']:.6f} "
            f"amp_ratio={metrics['amplitude_ratio']:.6f} "
            f"active_frac={metrics['active_fraction']:.6f} "
            f"pattern_score={metrics['pattern_score']:.6f}",
            flush=True,
        )
        return avg_epoch_loss, metrics
    return avg_epoch_loss


# Return the current optimizer learning rate.
def current_lr(optimizer: torch.optim.Optimizer) -> float:
    return float(optimizer.param_groups[0]["lr"])


# Load flexible partial state dict from disk or cache.
def load_flexible_partial_state_dict(model: nn.Module, source_state: dict[str, torch.Tensor]) -> tuple[list[str], list[str], list[str]]:
    """Load matching checkpoint tensors and copy overlapping slices for changed channel counts."""
    target_state = model.state_dict()
    adapted: dict[str, torch.Tensor] = {}
    skipped: list[str] = []
    adapted_keys: list[str] = []
    for key, value in source_state.items():
        if key not in target_state:
            skipped.append(key)
            continue
        target = target_state[key]
        if tuple(value.shape) == tuple(target.shape):
            adapted[key] = value
            continue
        if value.ndim == target.ndim and value.ndim > 0:
            new_value = target.clone()
            slices = tuple(slice(0, min(int(a), int(b))) for a, b in zip(value.shape, target.shape))
            new_value[slices] = value[slices].to(dtype=target.dtype)
            adapted[key] = new_value
            adapted_keys.append(key)
            continue
        skipped.append(key)
    result = model.load_state_dict(adapted, strict=False)
    missing = list(result.missing_keys)
    unexpected = list(result.unexpected_keys) + skipped
    return missing, unexpected, adapted_keys


# Load texture sampler weights from disk or cache.
def load_texture_sampler_weights(
    sidecar_root: str,
    split: str,
    expected_len: int,
    weight_power: float,
    min_weight: float,
    max_weight: float,
) -> torch.Tensor:
    sidecar_path = os.path.join(sidecar_root, f"{split}_texture_stats.npz")
    if not os.path.exists(sidecar_path):
        raise FileNotFoundError(f"Missing texture sidecar: {sidecar_path}")
    data = np.load(sidecar_path, allow_pickle=False)
    if "sample_weight" in data.files:
        weights = data["sample_weight"].astype(np.float64, copy=True)
    elif "texture_score" in data.files:
        score = data["texture_score"].astype(np.float64, copy=False)
        score = np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)
        score = score - float(score.min())
        denom = float(score.max())
        norm = score / denom if denom > 0 else score
        weights = 1.0 + norm
    else:
        raise RuntimeError(f"Texture sidecar {sidecar_path} has neither sample_weight nor texture_score")

    if len(weights) < expected_len:
        raise RuntimeError(
            f"Texture sidecar {sidecar_path} has {len(weights)} rows but dataset has {expected_len} samples"
        )
    weights = weights[:expected_len]
    weights = np.nan_to_num(weights, nan=1.0, posinf=max_weight, neginf=min_weight)
    weights = np.clip(weights, float(min_weight), float(max_weight))
    if weight_power != 1.0:
        weights = np.power(weights, float(weight_power))
    weights = np.clip(weights, float(min_weight), float(max_weight))
    return torch.as_tensor(weights, dtype=torch.double)


# Entry point for the command-line workflow.
def main() -> None:
    parser = argparse.ArgumentParser(description="Train V22 autoregressive previous-CO2 residual model with global context")
    parser.add_argument("--jobs-root", default=None)
    parser.add_argument("--cache-root", default=None, help="Read preprocessed patch cache instead of raw PALM files")
    parser.add_argument("--metadata-root", default=None, help="Directory containing train_metadata.npz and val_metadata.npz")
    parser.add_argument("--v13-sidecar-root", default=None, help="Directory containing corrected-time V13 sidecar cache")
    parser.add_argument("--v22-prev-sidecar-root", default=None, help="Directory containing V22 previous-timestep CO2 sidecar cache")
    parser.add_argument("--v14-normalization-root", default=None, help="Directory containing V15 z-focused normalization.json")
    parser.add_argument("--v14-layer-min", type=int, default=1)
    parser.add_argument("--v14-layer-max", type=int, default=15)
    parser.add_argument("--v14-min-layer-overlap", type=int, default=8)
    parser.add_argument(
        "--target-normalization-path",
        default=None,
        help="JSON stats for normalized residual target, computed from train cache only",
    )
    parser.add_argument(
        "--target-normalization-mode",
        choices=["none", "month", "background_bin"],
        default="none",
        help="Group used to z-score corrected residual target",
    )
    parser.add_argument("--v13-global-sample-size", type=int, default=64, help="Local sampling size for full-domain context features")
    parser.add_argument(
        "--model-variant",
        choices=[
            "coarse_local",
            "v7_style",
            "v20_context_v7",
            "v21_context_v7",
            "v22_context_v7",
            "v35_multitask_context_v7",
            "v37_hard_pattern_context_v7",
            "v38_event_texture_context_v7",
        ],
        default="coarse_local",
        help=(
            "coarse_local keeps the coarse+local model; v7_style uses a direct V7-like U-Net; "
            "v20/v21_context_v7 keep the direct backbone and add a global-context correction head; "
            "v35_multitask_context_v7 also adds active/sign auxiliary delta heads; "
            "v37_hard_pattern_context_v7 makes the active head gate a high-frequency delta branch; "
            "v38_event_texture_context_v7 keeps that split but gives the event-texture branch a gate floor."
        ),
    )
    parser.add_argument("--no-global-context", action="store_true", help="Do not load or feed corrected global context sidecar arrays")
    parser.add_argument("--v13-normalize-raw-extra-channels", action="store_true", help="Also normalize x/y/z, fluid_mask, and height_gate channels")
    parser.add_argument("--add-coords", action="store_true", help="Append x_norm, y_norm, z_norm channels from metadata sidecar")
    parser.add_argument("--append-fluid-mask", action="store_true", help="Append the valid fluid/topography mask as an input channel")
    parser.add_argument("--append-height-gate", action="store_true", help="Append exp(-z/decay) height gate as an input channel")
    parser.add_argument("--height-gate-surface", action="store_true", help="Multiply surface/static channels by the height gate")
    parser.add_argument("--height-gate-decay-levels", type=float, default=40.0)
    parser.add_argument("--surface-channel-indices", default="0,7,8,9,10,11,12,13")
    parser.add_argument("--topography-path", default="/data/linfeng/palm/london_camden_2019_new/JOBS/cam07_175vm_topo_surf_N02.000.nc")
    parser.add_argument("--train-fraction", type=float, default=0.70)
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--train-samples-per-epoch", type=int, default=512)
    parser.add_argument("--val-samples", type=int, default=128)
    parser.add_argument("--drop-bg-co2", action="store_true", help="Drop cached/dataset ls_forcing_right_CO2 input channel")
    parser.add_argument("--drop-month-channels", action="store_true", help="Drop cached month sin/cos input channels")
    parser.add_argument("--texture-sidecar-root", default=None, help="Directory containing train_texture_stats.npz")
    parser.add_argument("--texture-aware-sampler", action="store_true", help="Sample cached train patches by texture sidecar weights")
    parser.add_argument("--texture-weight-power", type=float, default=1.0)
    parser.add_argument("--texture-min-weight", type=float, default=0.25)
    parser.add_argument("--texture-max-weight", type=float, default=12.0)
    parser.add_argument(
        "--exclude-months",
        default="",
        help="Comma-separated months to remove from cached train/val metadata, e.g. 11,12",
    )
    parser.add_argument(
        "--keep-input-channel-names",
        default="",
        help=(
            "Comma-separated local input channel names to keep after all sidecar/autoregressive "
            "features are appended. Empty keeps every channel."
        ),
    )
    parser.add_argument("--v28-advection-features", action="store_true", help="Append previous-CO2 gradient and wind-advection channels")
    parser.add_argument(
        "--v28-advection-correction-target",
        action="store_true",
        help="Train the network on delta_CO2 - physics_advection_delta instead of raw delta_CO2",
    )
    parser.add_argument("--v28-advection-dx", type=float, default=5.0, help="Horizontal x grid spacing in metres for dC/dx")
    parser.add_argument("--v28-advection-dy", type=float, default=5.0, help="Horizontal y grid spacing in metres for dC/dy")
    parser.add_argument("--v28-advection-dz", type=float, default=10.0, help="Vertical z grid spacing in metres for dC/dz")
    parser.add_argument(
        "--v28-advection-dt",
        type=float,
        default=300.0,
        help="Effective seconds used to convert -u.grad(C) into a delta-CO2 baseline",
    )
    parser.add_argument("--v28-advection-delta-scale", type=float, default=5.0, help="Scale for advection-delta input channels")
    parser.add_argument("--v28-advection-gradient-scale", type=float, default=0.2, help="Scale for previous-CO2 gradient input channels")
    parser.add_argument("--v28-advection-clip", type=float, default=20.0, help="Clip physics advection delta in ppm; <=0 disables")
    parser.add_argument("--v28-advection-input-clip", type=float, default=8.0, help="Clip scaled V28 advection input channels; <=0 disables")
    parser.add_argument("--target-mode", choices=["residual", "absolute"], default="residual")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--samples-per-job", type=int, default=64)
    parser.add_argument("--patch-d", type=int, default=16)
    parser.add_argument("--patch-h", type=int, default=256)
    parser.add_argument("--patch-w", type=int, default=256)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--base-channels", type=int, default=24)
    parser.add_argument("--coarse-pool", type=int, default=8)
    parser.add_argument("--global-feature-channels", type=int, default=8)
    parser.add_argument("--high-residual-scale", type=float, default=1.0)
    parser.add_argument("--lr-scheduler", choices=["none", "plateau"], default="plateau")
    parser.add_argument("--lr-factor", type=float, default=0.5)
    parser.add_argument("--lr-patience", type=int, default=8)
    parser.add_argument("--min-lr", type=float, default=1e-6)
    parser.add_argument("--early-stopping-patience", type=int, default=25)
    parser.add_argument("--early-stopping-min-delta", type=float, default=0.0)
    parser.add_argument("--disable-early-stopping", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument(
        "--progress",
        choices=["summary", "tqdm", "none"],
        default="none",
        help="none logs only epoch completion; summary logs every --log-every steps; tqdm shows per-batch progress bars.",
    )
    parser.add_argument(
        "--delta-monitor-checkpoints",
        action="store_true",
        help="Compute validation delta-pattern metrics and save extra monitor checkpoints.",
    )
    parser.add_argument("--delta-monitor-active-threshold", type=float, default=0.75)
    parser.add_argument("--delta-monitor-min-valid", type=int, default=256)
    parser.add_argument("--delta-monitor-hard-min-std", type=float, default=1.0)
    parser.add_argument("--delta-monitor-hard-min-active-fraction", type=float, default=0.25)
    parser.add_argument("--base-loss", choices=["huber", "l1"], default="huber")
    parser.add_argument("--huber-delta", type=float, default=5.0)
    parser.add_argument("--gradient-loss-weight", type=float, default=0.02)
    parser.add_argument("--multiscale-loss-weight", type=float, default=0.02)
    parser.add_argument("--multiscale-scales", default="8,16,32,64")
    parser.add_argument("--multiscale-min-valid-fraction", type=float, default=0.5)
    parser.add_argument("--smoothness-loss-weight", type=float, default=0.0)
    parser.add_argument("--smoothness-kernel-size", type=int, default=9)
    parser.add_argument("--variance-loss-weight", type=float, default=0.0)
    parser.add_argument("--variance-min-std", type=float, default=2.0)
    parser.add_argument("--variance-eps", type=float, default=1e-6)
    parser.add_argument("--residual-weight-alpha", type=float, default=0.0)
    parser.add_argument("--residual-weight-scale", type=float, default=5.0)
    parser.add_argument("--residual-weight-max", type=float, default=4.0)
    parser.add_argument("--target-gradient-weight-alpha", type=float, default=0.0)
    parser.add_argument("--target-gradient-weight-scale", type=float, default=1.0)
    parser.add_argument("--target-gradient-weight-max", type=float, default=3.0)
    parser.add_argument("--low-layer-weight-alpha", type=float, default=0.0)
    parser.add_argument("--normalized-loss-weight", type=float, default=0.0)
    parser.add_argument("--normalized-min-std", type=float, default=0.75)
    parser.add_argument("--normalized-huber-delta", type=float, default=1.0)
    parser.add_argument("--normalized-eps", type=float, default=1e-6)
    parser.add_argument("--correlation-loss-weight", type=float, default=0.05)
    parser.add_argument("--correlation-eps", type=float, default=1e-6)
    parser.add_argument("--correlation-min-target-std", type=float, default=0.5)
    parser.add_argument("--correlation-min-valid-fraction", type=float, default=0.5)
    parser.add_argument(
        "--pattern-height-decay",
        action="store_true",
        help="Weight pattern/correlation losses by the appended exp(-z/decay) height gate.",
    )
    parser.add_argument("--low-frequency-loss-weight", type=float, default=0.0)
    parser.add_argument("--low-frequency-pool", type=int, default=8)
    parser.add_argument("--low-frequency-min-valid-fraction", type=float, default=0.5)
    parser.add_argument("--low-frequency-correlation-weight", type=float, default=0.0)
    parser.add_argument("--high-frequency-loss-weight", type=float, default=0.0)
    parser.add_argument("--high-frequency-huber-delta", type=float, default=2.0)
    parser.add_argument("--local-correlation-loss-weight", type=float, default=0.0)
    parser.add_argument("--local-correlation-pool", type=int, default=32)
    parser.add_argument("--local-correlation-min-target-std", type=float, default=0.5)
    parser.add_argument("--local-correlation-min-valid-fraction", type=float, default=0.5)
    parser.add_argument("--amplitude-loss-weight", type=float, default=0.0)
    parser.add_argument("--amplitude-min-target-std", type=float, default=0.5)
    parser.add_argument("--active-delta-loss-weight", type=float, default=0.0)
    parser.add_argument("--active-delta-threshold", type=float, default=0.75)
    parser.add_argument(
        "--sign-loss-weight",
        type=float,
        default=0.0,
        help="Weight for wrong-sign delta penalty on cells where |target delta| exceeds --sign-loss-min-abs.",
    )
    parser.add_argument("--sign-loss-min-abs", type=float, default=0.5)
    parser.add_argument("--sign-loss-scale", type=float, default=2.0)
    parser.add_argument("--active-loss-weight", type=float, default=0.0)
    parser.add_argument("--active-loss-threshold", type=float, default=0.75)
    parser.add_argument("--active-loss-pos-weight", type=float, default=2.0)
    parser.add_argument("--sign-class-loss-weight", type=float, default=0.0)
    parser.add_argument("--sign-class-loss-min-abs", type=float, default=0.75)
    parser.add_argument("--sign-class-loss-pos-weight", type=float, default=1.0)
    parser.add_argument("--min-high-gate", type=float, default=0.20)
    parser.add_argument("--min-texture-gate", type=float, default=0.05)
    parser.add_argument("--disable-learned-texture-gate", action="store_true")
    parser.add_argument("--init-weights", default=None, help="Path to a checkpoint whose model weights initialize a fresh run")
    parser.add_argument(
        "--init-allow-partial",
        action="store_true",
        help="Allow partial checkpoint initialization when adding new V20 context parameters.",
    )
    parser.add_argument("--resume", default=None, help="Path to a checkpoint to resume from")
    parser.add_argument("--out-dir", default="/data/cd25/processed/unet3d_runs")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    os.makedirs(args.out_dir, exist_ok=True)

    patch_size = (args.patch_d, args.patch_h, args.patch_w)
    multiscale_scales = parse_int_list(args.multiscale_scales)
    surface_channel_indices = parse_int_list(args.surface_channel_indices)
    exclude_months = parse_int_list(args.exclude_months)

    if args.cache_root:
        train_ds = CachedPatchDataset(os.path.join(args.cache_root, "train"))
        val_ds = CachedPatchDataset(os.path.join(args.cache_root, "val"))
        print(f"Using patch cache: {os.path.abspath(args.cache_root)}", flush=True)
    else:
        if not args.jobs_root:
            raise RuntimeError("--jobs-root is required when --cache-root is not set")
        jobs = discover_jobs(args.jobs_root)
        if not jobs:
            raise RuntimeError("No job folders found")
        print(f"Discovered {len(jobs)} job folders under {args.jobs_root}", flush=True)

        train_ds = PalmPatchDataset(
            jobs_root=args.jobs_root,
            job_names=jobs,
            split="train",
            patch_size=patch_size,
            samples_per_job=args.samples_per_job,
            seed=args.seed,
            topography_path=args.topography_path,
            train_fraction=args.train_fraction,
            val_fraction=args.val_fraction,
        )
        val_ds = PalmPatchDataset(
            jobs_root=args.jobs_root,
            job_names=jobs,
            split="val",
            patch_size=patch_size,
            samples_per_job=max(8, args.samples_per_job // 4),
            seed=args.seed + 1,
            topography_path=args.topography_path,
            train_fraction=args.train_fraction,
            val_fraction=args.val_fraction,
        )
    print(
        f"Dataset ready: train_samples={len(train_ds)} val_samples={len(val_ds)} "
        f"patch={patch_size}",
        flush=True,
    )

    coord_channels: list[str] = []
    appended_channels: list[str] = []
    dropped_input_channels: list[int] = []
    kept_input_channels: list[str] = []
    kept_input_channel_indices: list[int] = []
    original_input_channels: list[str] = []
    height_gate_channel_idx: int | None = None
    using_v13_sidecar = args.v13_sidecar_root is not None
    using_v22_autoregressive = args.v22_prev_sidecar_root is not None
    global_channels = 0
    if using_v13_sidecar:
        if not args.cache_root:
            raise RuntimeError("--v13-sidecar-root currently requires --cache-root")
        if not args.v14_normalization_root:
            raise RuntimeError("--v14-normalization-root is required for V15 z-focused training")
        if args.v28_advection_features and not using_v22_autoregressive:
            raise RuntimeError("--v28-advection-features requires --v22-prev-sidecar-root")
        sidecar_dataset_cls = (
            V28AdvectionSidecarDataset
            if args.v28_advection_features
            else V22AutoregressiveSidecarDataset
            if using_v22_autoregressive
            else V14FocusedSidecarDataset
        )
        sidecar_extra_kwargs = {"prev_sidecar_root": args.v22_prev_sidecar_root} if using_v22_autoregressive else {}
        if args.v28_advection_features:
            sidecar_extra_kwargs.update(
                {
                    "advection_dx": args.v28_advection_dx,
                    "advection_dy": args.v28_advection_dy,
                    "advection_dz": args.v28_advection_dz,
                    "advection_dt": args.v28_advection_dt,
                    "advection_delta_scale": args.v28_advection_delta_scale,
                    "advection_gradient_scale": args.v28_advection_gradient_scale,
                    "advection_clip": args.v28_advection_clip,
                    "advection_input_clip": args.v28_advection_input_clip,
                    "correction_target": args.v28_advection_correction_target,
                }
            )
        train_ds = sidecar_dataset_cls(
            train_ds,
            sidecar_root=args.v13_sidecar_root,
            normalization_root=args.v14_normalization_root,
            split="train",
            global_sample_size=args.v13_global_sample_size,
            layer_min=args.v14_layer_min,
            layer_max=args.v14_layer_max,
            min_layer_overlap=args.v14_min_layer_overlap,
            use_global_context=not args.no_global_context,
            target_normalization_path=args.target_normalization_path,
            target_normalization_mode=args.target_normalization_mode,
            surface_channel_indices=surface_channel_indices if args.height_gate_surface else (),
            height_gate_decay_levels=args.height_gate_decay_levels,
            append_height_gate=args.append_height_gate,
            exclude_months=exclude_months,
            **sidecar_extra_kwargs,
        )
        val_ds = sidecar_dataset_cls(
            val_ds,
            sidecar_root=args.v13_sidecar_root,
            normalization_root=args.v14_normalization_root,
            split="val",
            global_sample_size=args.v13_global_sample_size,
            layer_min=args.v14_layer_min,
            layer_max=args.v14_layer_max,
            min_layer_overlap=args.v14_min_layer_overlap,
            use_global_context=not args.no_global_context,
            target_normalization_path=args.target_normalization_path,
            target_normalization_mode=args.target_normalization_mode,
            surface_channel_indices=surface_channel_indices if args.height_gate_surface else (),
            height_gate_decay_levels=args.height_gate_decay_levels,
            append_height_gate=args.append_height_gate,
            exclude_months=exclude_months,
            **sidecar_extra_kwargs,
        )
        height_gate_channel_idx = train_ds.height_gate_channel_idx
        appended_channels = list(train_ds.appended_channels)
        coord_channels = [name for name in ("x_norm", "y_norm", "z_norm") if name in appended_channels]
        dropped_input_channels = list(train_ds.dropped_input_channels)
        global_channels = len(train_ds.global_channels)
        corrected_target_label = (
            "autoregressive advection correction = kc_CO2(t)-kc_CO2(t-1)-physics_adv_delta"
            if args.v28_advection_features and args.v28_advection_correction_target
            else "autoregressive delta CO2 = kc_CO2(t)-kc_CO2(t-1)"
            if using_v22_autoregressive
            else "normalized corrected residual"
            if args.target_normalization_path
            else "corrected residual"
        )
        print(
            f"Training target mode: V22 previous-CO2 {corrected_target_label} on global z={args.v14_layer_min}-{args.v14_layer_max}",
            flush=True,
        )
        print(
            f"Corrected sidecar root: {os.path.abspath(args.v13_sidecar_root)} "
            f"V22 normalization root: {os.path.abspath(args.v14_normalization_root)} "
            f"global_channels={train_ds.global_channels} "
            f"global_sample_size={args.v13_global_sample_size} "
            f"use_global_context={not args.no_global_context} "
            f"min_layer_overlap={args.v14_min_layer_overlap}",
            flush=True,
        )
        if exclude_months:
            print(f"Excluded months from train/val metadata: {list(exclude_months)}", flush=True)
        if using_v22_autoregressive:
            print(f"V22 previous-CO2 sidecar root: {os.path.abspath(args.v22_prev_sidecar_root)}", flush=True)
        if args.v28_advection_features:
            print(
                "V28 physics advection: "
                f"features=1 correction_target={args.v28_advection_correction_target} "
                f"dx/dy/dz=({args.v28_advection_dx}, {args.v28_advection_dy}, {args.v28_advection_dz}) m "
                f"effective_dt={args.v28_advection_dt} s "
                f"delta_scale={args.v28_advection_delta_scale} "
                f"gradient_scale={args.v28_advection_gradient_scale} "
                f"advection_clip={args.v28_advection_clip} "
                f"input_clip={args.v28_advection_input_clip}",
                flush=True,
            )
        print(f"V22 selected samples: train={len(train_ds)} val={len(val_ds)}", flush=True)
        print(f"V22 local transformed channels={train_ds.effective_local_channels}", flush=True)
        print(
            f"V22 surface decay: enabled={args.height_gate_surface} "
            f"channels={surface_channel_indices if args.height_gate_surface else ()} "
            f"decay_levels={args.height_gate_decay_levels} append_height_gate={args.append_height_gate}",
            flush=True,
        )
        if args.target_normalization_path:
            print(
                f"V16 target normalization: mode={train_ds.target_normalization_mode} "
                f"path={os.path.abspath(args.target_normalization_path)}",
                flush=True,
            )
    elif args.target_mode == "residual":
        if args.drop_bg_co2:
            raise RuntimeError("--drop-bg-co2 cannot be used with --target-mode residual")
        train_ds = ResidualTargetDataset(train_ds, bg_channel_idx=1)
        val_ds = ResidualTargetDataset(val_ds, bg_channel_idx=1)
        print("Training target mode: residual = kc_CO2 - ls_forcing_right_CO2", flush=True)
    else:
        print("Training target mode: absolute kc_CO2", flush=True)

    if args.drop_bg_co2:
        train_ds = DropInputChannelDataset(train_ds, channel_idx=1)
        val_ds = DropInputChannelDataset(val_ds, channel_idx=1)
        print("Dropping input channel 1: ls_forcing_right_CO2", flush=True)

    if args.drop_month_channels and not using_v13_sidecar:
        month_channels = [14, 15]
        train_ds = DropInputChannelsDataset(train_ds, month_channels)
        val_ds = DropInputChannelsDataset(val_ds, month_channels)
        dropped_input_channels.extend(month_channels)
        print("Dropping input channels 14,15: month_sin, month_cos", flush=True)

    use_v10_input_wrapper = (
        args.add_coords or args.append_fluid_mask or args.append_height_gate or args.height_gate_surface
    ) and not using_v13_sidecar
    if use_v10_input_wrapper:
        if not args.metadata_root:
            raise RuntimeError("--metadata-root is required for legacy input transforms")
        train_meta = os.path.join(args.metadata_root, "train_metadata.npz")
        val_meta = os.path.join(args.metadata_root, "val_metadata.npz")
        original_channels = train_ds[0][0].shape[0]
        train_ds = V10InputDataset(
            train_ds,
            train_meta,
            surface_channel_indices=surface_channel_indices,
            height_gate_decay_levels=args.height_gate_decay_levels,
            append_coords=args.add_coords,
            append_fluid_mask=args.append_fluid_mask,
            append_height_gate=args.append_height_gate,
            gate_surface_channels=args.height_gate_surface,
        )
        val_ds = V10InputDataset(
            val_ds,
            val_meta,
            surface_channel_indices=surface_channel_indices,
            height_gate_decay_levels=args.height_gate_decay_levels,
            append_coords=args.add_coords,
            append_fluid_mask=args.append_fluid_mask,
            append_height_gate=args.append_height_gate,
            gate_surface_channels=args.height_gate_surface,
        )
        if args.add_coords:
            coord_channels = ["x_norm", "y_norm", "z_norm"]
            appended_channels.extend(coord_channels)
        if args.append_fluid_mask:
            appended_channels.append("fluid_mask")
        if args.append_height_gate:
            appended_channels.append("height_gate")
            height_gate_channel_idx = original_channels + len(appended_channels) - 1
        print(f"Legacy input transforms: appended_channels={appended_channels}", flush=True)
        print(
            f"Legacy surface gating: enabled={args.height_gate_surface} "
            f"surface_channels={surface_channel_indices} decay_levels={args.height_gate_decay_levels}",
            flush=True,
        )
        print(f"Legacy metadata root: {os.path.abspath(args.metadata_root)}", flush=True)

    keep_input_channel_names = parse_name_list(args.keep_input_channel_names)
    if keep_input_channel_names:
        original_input_channels = list(getattr(train_ds, "effective_local_channels", []))
        train_ds = KeepInputChannelsDataset(train_ds, keep_input_channel_names)
        val_ds = KeepInputChannelsDataset(val_ds, keep_input_channel_names)
        kept_input_channels = list(train_ds.kept_input_channels)
        kept_input_channel_indices = list(train_ds.kept_input_channel_indices)
        appended_channels = list(train_ds.appended_channels)
        coord_channels = [name for name in ("x_norm", "y_norm", "z_norm") if name in appended_channels]
        height_gate_channel_idx = train_ds.height_gate_channel_idx
        global_channels = len(train_ds.global_channels)
        print(
            "RF-guided local input pruning: "
            f"kept={kept_input_channels} "
            f"indices={kept_input_channel_indices} "
            f"original_count={len(original_input_channels)} kept_count={len(kept_input_channels)}",
            flush=True,
        )

    train_sampler = None
    train_shuffle = True
    if args.cache_root:
        train_samples_per_epoch = min(args.train_samples_per_epoch, len(train_ds))
        if args.texture_aware_sampler:
            if not args.texture_sidecar_root:
                raise RuntimeError("--texture-sidecar-root is required with --texture-aware-sampler")
            train_weights = load_texture_sampler_weights(
                args.texture_sidecar_root,
                split="train",
                expected_len=len(train_ds),
                weight_power=args.texture_weight_power,
                min_weight=args.texture_min_weight,
                max_weight=args.texture_max_weight,
            )
            train_sampler = WeightedRandomSampler(
                train_weights,
                num_samples=train_samples_per_epoch,
                replacement=True,
            )
            print(
                f"Texture-aware training sampler: sidecar={os.path.abspath(args.texture_sidecar_root)} "
                f"num_samples={train_samples_per_epoch} replacement=1 "
                f"weight_min={float(train_weights.min()):.4g} "
                f"weight_max={float(train_weights.max()):.4g} "
                f"weight_mean={float(train_weights.mean()):.4g}",
                flush=True,
            )
        else:
            train_sampler = RandomSampler(train_ds, replacement=False, num_samples=train_samples_per_epoch)
        train_shuffle = False
        val_pool_size = len(val_ds)
        val_samples = min(args.val_samples, val_pool_size)
        val_generator = torch.Generator()
        val_seed = args.seed + 100003
        val_generator.manual_seed(val_seed)
        val_indices = torch.randperm(val_pool_size, generator=val_generator)[:val_samples].tolist()
        val_ds = Subset(val_ds, val_indices)
        print(
            f"Cached training epoch samples: {train_samples_per_epoch} "
            f"random patches from pool={len(train_ds)}",
            flush=True,
        )
        print(
            f"Cached validation epoch samples: {val_samples} "
            f"fixed random patches from pool={val_pool_size} seed={val_seed}",
            flush=True,
        )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=train_shuffle,
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    sample = train_ds[0]
    sample_x = sample[0]
    sample_global = sample[1] if len(sample) == 5 else None
    in_channels = int(sample_x.shape[0])
    print(f"Sample channels={in_channels}, sample shape={tuple(sample_x.shape)}", flush=True)
    if sample_global is not None:
        print(f"Sample global context channels={int(sample_global.shape[0])}, shape={tuple(sample_global.shape)}", flush=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}", flush=True)
    bg_channel_idx = None if args.drop_bg_co2 else 1
    checkpoint_target_mode = (
        "autoregressive_advection_correction"
        if (using_v22_autoregressive and args.v28_advection_features and args.v28_advection_correction_target)
        else "autoregressive_delta"
        if using_v22_autoregressive
        else
        "corrected_residual_normalized"
        if (using_v13_sidecar and args.target_normalization_path)
        else "corrected_residual"
        if using_v13_sidecar
        else args.target_mode
    )
    if args.model_variant == "v7_style":
        model = V7StyleUNet3D(
            in_channels=in_channels,
            out_channels=1,
            base_channels=args.base_channels,
        ).to(device)
        architecture_target = "normalized_corrected_residual" if args.target_normalization_path else "corrected_residual"
        architecture_name = f"v20_z{args.v14_layer_min:03d}_{args.v14_layer_max:03d}_v7_style_{architecture_target}"
        if global_channels:
            print("Model variant v7_style ignores global context features.", flush=True)
    elif args.model_variant == "v38_event_texture_context_v7":
        model = V38EventTextureContextV7UNet3D(
            in_channels=in_channels,
            out_channels=1,
            base_channels=args.base_channels,
            global_channels=global_channels,
            global_feature_channels=args.global_feature_channels,
            context_correction_scale=args.high_residual_scale,
            high_delta_scale=args.high_residual_scale,
            min_high_gate=args.min_high_gate,
        ).to(device)
        architecture_target = "normalized_corrected_residual" if args.target_normalization_path else "corrected_residual"
        architecture_target = (
            "autoregressive_advection_correction"
            if (using_v22_autoregressive and args.v28_advection_features and args.v28_advection_correction_target)
            else "autoregressive_delta"
            if using_v22_autoregressive
            else architecture_target
        )
        architecture_name = (
            f"v38_z{args.v14_layer_min:03d}_{args.v14_layer_max:03d}_"
            f"event_texture_context_v7_{architecture_target}"
        )
        if not global_channels:
            print("Warning: v38_event_texture_context_v7 is running without global context channels.", flush=True)
    elif args.model_variant == "v37_hard_pattern_context_v7":
        model = V37HardPatternContextV7UNet3D(
            in_channels=in_channels,
            out_channels=1,
            base_channels=args.base_channels,
            global_channels=global_channels,
            global_feature_channels=args.global_feature_channels,
            context_correction_scale=args.high_residual_scale,
            high_delta_scale=args.high_residual_scale,
        ).to(device)
        architecture_target = "normalized_corrected_residual" if args.target_normalization_path else "corrected_residual"
        architecture_target = (
            "autoregressive_advection_correction"
            if (using_v22_autoregressive and args.v28_advection_features and args.v28_advection_correction_target)
            else "autoregressive_delta"
            if using_v22_autoregressive
            else architecture_target
        )
        architecture_name = (
            f"v37_z{args.v14_layer_min:03d}_{args.v14_layer_max:03d}_"
            f"hard_pattern_context_v7_{architecture_target}"
        )
        if not global_channels:
            print("Warning: v37_hard_pattern_context_v7 is running without global context channels.", flush=True)
    elif args.model_variant == "v35_multitask_context_v7":
        model = V35MultiTaskContextV7UNet3D(
            in_channels=in_channels,
            out_channels=1,
            base_channels=args.base_channels,
            global_channels=global_channels,
            global_feature_channels=args.global_feature_channels,
            context_correction_scale=args.high_residual_scale,
        ).to(device)
        architecture_target = "normalized_corrected_residual" if args.target_normalization_path else "corrected_residual"
        architecture_target = (
            "autoregressive_advection_correction"
            if (using_v22_autoregressive and args.v28_advection_features and args.v28_advection_correction_target)
            else "autoregressive_delta"
            if using_v22_autoregressive
            else architecture_target
        )
        architecture_name = (
            f"v35_z{args.v14_layer_min:03d}_{args.v14_layer_max:03d}_"
            f"multitask_context_v7_{architecture_target}"
        )
        if not global_channels:
            print("Warning: v35_multitask_context_v7 is running without global context channels.", flush=True)
    elif args.model_variant in {"v20_context_v7", "v21_context_v7", "v22_context_v7"}:
        model = V20ContextV7UNet3D(
            in_channels=in_channels,
            out_channels=1,
            base_channels=args.base_channels,
            global_channels=global_channels,
            global_feature_channels=args.global_feature_channels,
            context_correction_scale=args.high_residual_scale,
        ).to(device)
        architecture_target = "normalized_corrected_residual" if args.target_normalization_path else "corrected_residual"
        architecture_target = (
            "autoregressive_advection_correction"
            if (using_v22_autoregressive and args.v28_advection_features and args.v28_advection_correction_target)
            else "autoregressive_delta"
            if using_v22_autoregressive
            else architecture_target
        )
        version_prefix = "v28" if args.v28_advection_features else "v22"
        architecture_name = f"{version_prefix}_z{args.v14_layer_min:03d}_{args.v14_layer_max:03d}_context_v7_{architecture_target}"
        if not global_channels:
            print("Warning: context_v7 is running without global context channels.", flush=True)
    else:
        model = UNet3D(
            in_channels=in_channels,
            out_channels=1,
            base_channels=args.base_channels,
            bg_channel_idx=bg_channel_idx,
            gate_channel_idx=height_gate_channel_idx,
            coarse_pool=args.coarse_pool,
            min_texture_gate=args.min_texture_gate,
            learned_texture_gate=not args.disable_learned_texture_gate,
            high_residual_scale=args.high_residual_scale,
            global_channels=global_channels,
            global_feature_channels=args.global_feature_channels,
        ).to(device)
        architecture_target = "normalized_corrected_residual" if args.target_normalization_path else "corrected_residual"
        architecture_name = f"v20_z{args.v14_layer_min:03d}_{args.v14_layer_max:03d}_{architecture_target}_global_context"
    print(f"Model variant: {args.model_variant} architecture={architecture_name}", flush=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = None
    if args.lr_scheduler == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=args.lr_factor,
            patience=args.lr_patience,
            min_lr=args.min_lr,
        )

    start_epoch = 1
    best_val = float("inf")
    epochs_without_improvement = 0
    best_delta_r = -float("inf")
    best_hard_delta_r = -float("inf")
    best_sign_accuracy = -float("inf")
    best_pattern_score = -float("inf")
    monitor_best: dict[str, dict[str, float]] = {}
    best_path = os.path.join(args.out_dir, "best_model.pt")
    last_path = os.path.join(args.out_dir, "last_model.pt")
    history_path = os.path.join(args.out_dir, "history.csv")
    monitor_history_path = os.path.join(args.out_dir, "monitor_history.csv")
    monitor_best_path = os.path.join(args.out_dir, "monitor_best.json")
    best_delta_r_path = os.path.join(args.out_dir, "best_delta_r_model.pt")
    best_hard_delta_r_path = os.path.join(args.out_dir, "best_hard_delta_r_model.pt")
    best_sign_accuracy_path = os.path.join(args.out_dir, "best_sign_accuracy_model.pt")
    best_pattern_score_path = os.path.join(args.out_dir, "best_pattern_score_model.pt")
    print(f"Checkpoint path: {best_path}", flush=True)
    if args.delta_monitor_checkpoints:
        print(
            "Delta monitor checkpoints enabled: "
            f"delta_R->{best_delta_r_path}, hard_delta_R->{best_hard_delta_r_path}, "
            f"sign->{best_sign_accuracy_path}, pattern_score->{best_pattern_score_path}",
            flush=True,
        )
    print(
        f"Loss config: base={args.base_loss}, huber_delta={args.huber_delta}, "
        f"gradient_weight={args.gradient_loss_weight}, "
        f"multiscale_weight={args.multiscale_loss_weight}, "
        f"multiscale_scales={multiscale_scales}, "
        f"multiscale_min_valid_fraction={args.multiscale_min_valid_fraction}, "
        f"smoothness_weight={args.smoothness_loss_weight}, "
        f"smoothness_kernel={args.smoothness_kernel_size}, "
        f"variance_weight={args.variance_loss_weight}, "
        f"variance_min_std={args.variance_min_std}, variance_eps={args.variance_eps}, "
        f"residual_weight_alpha={args.residual_weight_alpha}, "
        f"residual_weight_scale={args.residual_weight_scale}, "
        f"residual_weight_max={args.residual_weight_max}, "
        f"target_gradient_weight_alpha={args.target_gradient_weight_alpha}, "
        f"target_gradient_weight_scale={args.target_gradient_weight_scale}, "
        f"target_gradient_weight_max={args.target_gradient_weight_max}, "
        f"low_layer_weight_alpha={args.low_layer_weight_alpha}, "
        f"normalized_weight={args.normalized_loss_weight}, "
        f"normalized_min_std={args.normalized_min_std}, "
        f"normalized_huber_delta={args.normalized_huber_delta}, "
        f"correlation_weight={args.correlation_loss_weight}, "
        f"correlation_min_target_std={args.correlation_min_target_std}, "
        f"pattern_height_decay={args.pattern_height_decay}, "
        f"low_frequency_weight={args.low_frequency_loss_weight}, "
        f"low_frequency_pool={args.low_frequency_pool}, "
        f"low_frequency_corr_weight={args.low_frequency_correlation_weight}, "
        f"high_frequency_weight={args.high_frequency_loss_weight}, "
        f"high_frequency_huber_delta={args.high_frequency_huber_delta}, "
        f"local_correlation_weight={args.local_correlation_loss_weight}, "
        f"local_correlation_pool={args.local_correlation_pool}, "
        f"local_correlation_min_target_std={args.local_correlation_min_target_std}, "
        f"amplitude_weight={args.amplitude_loss_weight}, "
        f"amplitude_min_target_std={args.amplitude_min_target_std}, "
        f"active_delta_weight={args.active_delta_loss_weight}, "
        f"active_delta_threshold={args.active_delta_threshold}, "
        f"sign_loss_weight={args.sign_loss_weight}, "
        f"sign_loss_min_abs={args.sign_loss_min_abs}, "
        f"sign_loss_scale={args.sign_loss_scale}, "
        f"active_loss_weight={args.active_loss_weight}, "
        f"active_loss_threshold={args.active_loss_threshold}, "
        f"active_loss_pos_weight={args.active_loss_pos_weight}, "
        f"sign_class_loss_weight={args.sign_class_loss_weight}, "
        f"sign_class_loss_min_abs={args.sign_class_loss_min_abs}, "
        f"sign_class_loss_pos_weight={args.sign_class_loss_pos_weight}, "
        f"min_high_gate={args.min_high_gate}, "
        f"v28_advection_features={args.v28_advection_features}, "
        f"v28_advection_correction_target={args.v28_advection_correction_target}, "
        f"v28_advection_dt={args.v28_advection_dt}, "
        f"v28_advection_spacing=({args.v28_advection_dx},{args.v28_advection_dy},{args.v28_advection_dz})",
        flush=True,
    )
    print(
        f"LR scheduler: {args.lr_scheduler} "
        f"(factor={args.lr_factor}, patience={args.lr_patience}, min_lr={args.min_lr})",
        flush=True,
    )
    if args.disable_early_stopping:
        print("Early stopping: disabled", flush=True)
    else:
        print(
            f"Early stopping: patience={args.early_stopping_patience}, "
            f"min_delta={args.early_stopping_min_delta}",
            flush=True,
        )

    if args.init_weights and args.resume:
        raise RuntimeError("--init-weights and --resume are mutually exclusive")

    if args.init_weights:
        init_path = os.path.abspath(args.init_weights)
        print(f"Initializing model weights from checkpoint: {init_path}", flush=True)
        checkpoint = torch.load(init_path, map_location=device)
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        if args.init_allow_partial:
            missing_keys, unexpected_keys, adapted_keys = load_flexible_partial_state_dict(model, state_dict)
            print(
                f"Flexible partial init: missing_keys={len(missing_keys)} "
                f"unexpected_or_skipped_keys={len(unexpected_keys)} adapted_shape_keys={len(adapted_keys)}",
                flush=True,
            )
            if adapted_keys:
                print("Flexible partial init adapted keys: " + ", ".join(adapted_keys[:30]), flush=True)
            if missing_keys:
                print("Flexible partial init missing keys: " + ", ".join(missing_keys[:30]), flush=True)
            if unexpected_keys:
                print("Flexible partial init unexpected/skipped keys: " + ", ".join(unexpected_keys[:30]), flush=True)
        else:
            model.load_state_dict(state_dict)
        print(
            f"Initialized from epoch={checkpoint.get('epoch', 'unknown')} "
            f"val_loss={checkpoint.get('val_loss', 'unknown')} "
            f"best_val={checkpoint.get('best_val', 'unknown')}; "
            "optimizer, scheduler, history, and best_val start fresh for this run.",
            flush=True,
        )

    if args.resume:
        resume_path = os.path.abspath(args.resume)
        print(f"Resuming from checkpoint: {resume_path}", flush=True)
        checkpoint = torch.load(resume_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])

        if "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if scheduler is not None and "scheduler_state_dict" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        start_epoch = int(checkpoint.get("epoch", 0)) + 1
        best_val = float(checkpoint.get("best_val", checkpoint.get("val_loss", float("inf"))))
        epochs_without_improvement = int(checkpoint.get("epochs_without_improvement", 0))
        print(
            f"Resume state: previous_epoch={start_epoch - 1:03d} "
            f"best_val={best_val:.6f} lr={current_lr(optimizer):.8g}",
            flush=True,
        )

        if not os.path.exists(best_path):
            torch.save(checkpoint, best_path)

    if start_epoch > args.epochs:
        print(
            f"Nothing to do: resume checkpoint already reached epoch {start_epoch - 1:03d}, "
            f"which is >= requested epochs {args.epochs:03d}",
            flush=True,
        )
        return

    if not os.path.exists(history_path) or start_epoch == 1:
        with open(history_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "epoch",
                    "train_loss",
                    "val_loss",
                    "best_val",
                    "lr",
                    "delta_monitor_r",
                    "delta_monitor_hard_r",
                    "delta_monitor_sign_accuracy",
                    "delta_monitor_amplitude_ratio",
                    "delta_monitor_pattern_score",
                    "base_loss",
                    "gradient_loss_weight",
                    "multiscale_loss_weight",
                    "multiscale_scales",
                    "smoothness_loss_weight",
                    "variance_loss_weight",
                    "residual_weight_alpha",
                    "residual_weight_scale",
                    "residual_weight_max",
                    "target_gradient_weight_alpha",
                    "target_gradient_weight_scale",
                    "target_gradient_weight_max",
                    "low_layer_weight_alpha",
                    "normalized_loss_weight",
                    "normalized_min_std",
                    "correlation_loss_weight",
                    "pattern_height_decay",
                    "low_frequency_loss_weight",
                    "low_frequency_pool",
                    "low_frequency_correlation_weight",
                    "high_frequency_loss_weight",
                    "local_correlation_loss_weight",
                    "local_correlation_pool",
                    "amplitude_loss_weight",
                    "active_delta_loss_weight",
                    "active_delta_threshold",
                    "sign_loss_weight",
                    "sign_loss_min_abs",
                    "sign_loss_scale",
                    "active_loss_weight",
                    "active_loss_threshold",
                    "active_loss_pos_weight",
                    "sign_class_loss_weight",
                    "sign_class_loss_min_abs",
                    "sign_class_loss_pos_weight",
                    "improved",
                    "epochs_without_improvement",
                ]
            )

    if args.delta_monitor_checkpoints and (not os.path.exists(monitor_history_path) or start_epoch == 1):
        with open(monitor_history_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "epoch",
                    "val_loss",
                    "delta_r_mean",
                    "delta_r_count",
                    "hard_delta_r_mean",
                    "hard_delta_r_count",
                    "sign_accuracy",
                    "sign_count",
                    "amplitude_ratio",
                    "active_fraction",
                    "pattern_score",
                ]
            )

    for epoch in range(start_epoch, args.epochs + 1):
        print(f"Start epoch {epoch:03d}/{args.epochs:03d}", flush=True)
        train_loss = run_epoch(
            model,
            train_loader,
            optimizer,
            device,
            desc=f"train {epoch:03d}",
            epoch=epoch,
            log_every=args.log_every,
            huber_delta=args.huber_delta,
            base_loss=args.base_loss,
            gradient_loss_weight=args.gradient_loss_weight,
            multiscale_loss_weight=args.multiscale_loss_weight,
            multiscale_scales=multiscale_scales,
            multiscale_min_valid_fraction=args.multiscale_min_valid_fraction,
            smoothness_loss_weight=args.smoothness_loss_weight,
            smoothness_kernel_size=args.smoothness_kernel_size,
            height_gate_channel_idx=height_gate_channel_idx,
            variance_loss_weight=args.variance_loss_weight,
            variance_min_std=args.variance_min_std,
            variance_eps=args.variance_eps,
            residual_weight_alpha=args.residual_weight_alpha,
            residual_weight_scale=args.residual_weight_scale,
            residual_weight_max=args.residual_weight_max,
            target_gradient_weight_alpha=args.target_gradient_weight_alpha,
            target_gradient_weight_scale=args.target_gradient_weight_scale,
            target_gradient_weight_max=args.target_gradient_weight_max,
            low_layer_weight_alpha=args.low_layer_weight_alpha,
            normalized_loss_weight=args.normalized_loss_weight,
            normalized_min_std=args.normalized_min_std,
            normalized_huber_delta=args.normalized_huber_delta,
            normalized_eps=args.normalized_eps,
            correlation_loss_weight=args.correlation_loss_weight,
            correlation_eps=args.correlation_eps,
            correlation_min_target_std=args.correlation_min_target_std,
            correlation_min_valid_fraction=args.correlation_min_valid_fraction,
            low_frequency_loss_weight=args.low_frequency_loss_weight,
            low_frequency_pool=args.low_frequency_pool,
            low_frequency_min_valid_fraction=args.low_frequency_min_valid_fraction,
            low_frequency_correlation_weight=args.low_frequency_correlation_weight,
            high_frequency_loss_weight=args.high_frequency_loss_weight,
            high_frequency_huber_delta=args.high_frequency_huber_delta,
            local_correlation_loss_weight=args.local_correlation_loss_weight,
            local_correlation_pool=args.local_correlation_pool,
            local_correlation_min_target_std=args.local_correlation_min_target_std,
            local_correlation_min_valid_fraction=args.local_correlation_min_valid_fraction,
            amplitude_loss_weight=args.amplitude_loss_weight,
            amplitude_min_target_std=args.amplitude_min_target_std,
            active_delta_loss_weight=args.active_delta_loss_weight,
            active_delta_threshold=args.active_delta_threshold,
            sign_loss_weight=args.sign_loss_weight,
            sign_loss_min_abs=args.sign_loss_min_abs,
            sign_loss_scale=args.sign_loss_scale,
            active_loss_weight=args.active_loss_weight,
            active_loss_threshold=args.active_loss_threshold,
            active_loss_pos_weight=args.active_loss_pos_weight,
            sign_class_loss_weight=args.sign_class_loss_weight,
            sign_class_loss_min_abs=args.sign_class_loss_min_abs,
            sign_class_loss_pos_weight=args.sign_class_loss_pos_weight,
            pattern_height_decay=args.pattern_height_decay,
            progress=args.progress,
        )
        val_result = run_epoch(
            model,
            val_loader,
            None,
            device,
            desc=f"val   {epoch:03d}",
            epoch=epoch,
            log_every=args.log_every,
            huber_delta=args.huber_delta,
            base_loss=args.base_loss,
            gradient_loss_weight=args.gradient_loss_weight,
            multiscale_loss_weight=args.multiscale_loss_weight,
            multiscale_scales=multiscale_scales,
            multiscale_min_valid_fraction=args.multiscale_min_valid_fraction,
            smoothness_loss_weight=args.smoothness_loss_weight,
            smoothness_kernel_size=args.smoothness_kernel_size,
            height_gate_channel_idx=height_gate_channel_idx,
            variance_loss_weight=args.variance_loss_weight,
            variance_min_std=args.variance_min_std,
            variance_eps=args.variance_eps,
            residual_weight_alpha=args.residual_weight_alpha,
            residual_weight_scale=args.residual_weight_scale,
            residual_weight_max=args.residual_weight_max,
            target_gradient_weight_alpha=args.target_gradient_weight_alpha,
            target_gradient_weight_scale=args.target_gradient_weight_scale,
            target_gradient_weight_max=args.target_gradient_weight_max,
            low_layer_weight_alpha=args.low_layer_weight_alpha,
            normalized_loss_weight=args.normalized_loss_weight,
            normalized_min_std=args.normalized_min_std,
            normalized_huber_delta=args.normalized_huber_delta,
            normalized_eps=args.normalized_eps,
            correlation_loss_weight=args.correlation_loss_weight,
            correlation_eps=args.correlation_eps,
            correlation_min_target_std=args.correlation_min_target_std,
            correlation_min_valid_fraction=args.correlation_min_valid_fraction,
            low_frequency_loss_weight=args.low_frequency_loss_weight,
            low_frequency_pool=args.low_frequency_pool,
            low_frequency_min_valid_fraction=args.low_frequency_min_valid_fraction,
            low_frequency_correlation_weight=args.low_frequency_correlation_weight,
            high_frequency_loss_weight=args.high_frequency_loss_weight,
            high_frequency_huber_delta=args.high_frequency_huber_delta,
            local_correlation_loss_weight=args.local_correlation_loss_weight,
            local_correlation_pool=args.local_correlation_pool,
            local_correlation_min_target_std=args.local_correlation_min_target_std,
            local_correlation_min_valid_fraction=args.local_correlation_min_valid_fraction,
            amplitude_loss_weight=args.amplitude_loss_weight,
            amplitude_min_target_std=args.amplitude_min_target_std,
            active_delta_loss_weight=args.active_delta_loss_weight,
            active_delta_threshold=args.active_delta_threshold,
            sign_loss_weight=args.sign_loss_weight,
            sign_loss_min_abs=args.sign_loss_min_abs,
            sign_loss_scale=args.sign_loss_scale,
            active_loss_weight=args.active_loss_weight,
            active_loss_threshold=args.active_loss_threshold,
            active_loss_pos_weight=args.active_loss_pos_weight,
            sign_class_loss_weight=args.sign_class_loss_weight,
            sign_class_loss_min_abs=args.sign_class_loss_min_abs,
            sign_class_loss_pos_weight=args.sign_class_loss_pos_weight,
            pattern_height_decay=args.pattern_height_decay,
            progress=args.progress,
            collect_delta_monitor=args.delta_monitor_checkpoints,
            delta_monitor_active_threshold=args.delta_monitor_active_threshold,
            delta_monitor_min_valid=args.delta_monitor_min_valid,
            delta_monitor_hard_min_std=args.delta_monitor_hard_min_std,
            delta_monitor_hard_min_active_fraction=args.delta_monitor_hard_min_active_fraction,
        )
        if isinstance(val_result, tuple):
            val_loss, monitor_metrics = val_result
        else:
            val_loss = val_result
            monitor_metrics = {}

        lr_before = current_lr(optimizer)
        if scheduler is not None:
            scheduler.step(val_loss)
        lr_after = current_lr(optimizer)

        improved = val_loss < (best_val - args.early_stopping_min_delta)
        if improved:
            best_val = val_loss
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        print(
            f"epoch={epoch:03d} train_loss={train_loss:.6f} val_loss={val_loss:.6f} "
            f"best_val={best_val:.6f} lr={lr_after:.8g} improved={int(improved)} "
            f"no_improve={epochs_without_improvement}",
            flush=True,
        )
        if lr_after != lr_before:
            print(f"LR reduced: {lr_before:.8g} -> {lr_after:.8g}", flush=True)

        with open(history_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    epoch,
                    f"{train_loss:.10f}",
                    f"{val_loss:.10f}",
                    f"{best_val:.10f}",
                    f"{lr_after:.10g}",
                    f"{monitor_metrics.get('delta_r_mean', float('nan')):.10f}",
                    f"{monitor_metrics.get('hard_delta_r_mean', float('nan')):.10f}",
                    f"{monitor_metrics.get('sign_accuracy', float('nan')):.10f}",
                    f"{monitor_metrics.get('amplitude_ratio', float('nan')):.10f}",
                    f"{monitor_metrics.get('pattern_score', float('nan')):.10f}",
                    args.base_loss,
                    f"{args.gradient_loss_weight:.10g}",
                    f"{args.multiscale_loss_weight:.10g}",
                    ",".join(str(x) for x in multiscale_scales),
                    f"{args.smoothness_loss_weight:.10g}",
                    f"{args.variance_loss_weight:.10g}",
                    f"{args.residual_weight_alpha:.10g}",
                    f"{args.residual_weight_scale:.10g}",
                    f"{args.residual_weight_max:.10g}",
                    f"{args.target_gradient_weight_alpha:.10g}",
                    f"{args.target_gradient_weight_scale:.10g}",
                    f"{args.target_gradient_weight_max:.10g}",
                    f"{args.low_layer_weight_alpha:.10g}",
                    f"{args.normalized_loss_weight:.10g}",
                    f"{args.normalized_min_std:.10g}",
                    f"{args.correlation_loss_weight:.10g}",
                    int(args.pattern_height_decay),
                    f"{args.low_frequency_loss_weight:.10g}",
                    args.low_frequency_pool,
                    f"{args.low_frequency_correlation_weight:.10g}",
                    f"{args.high_frequency_loss_weight:.10g}",
                    f"{args.local_correlation_loss_weight:.10g}",
                    args.local_correlation_pool,
                    f"{args.amplitude_loss_weight:.10g}",
                    f"{args.active_delta_loss_weight:.10g}",
                    f"{args.active_delta_threshold:.10g}",
                    f"{args.sign_loss_weight:.10g}",
                    f"{args.sign_loss_min_abs:.10g}",
                    f"{args.sign_loss_scale:.10g}",
                    f"{args.active_loss_weight:.10g}",
                    f"{args.active_loss_threshold:.10g}",
                    f"{args.active_loss_pos_weight:.10g}",
                    f"{args.sign_class_loss_weight:.10g}",
                    f"{args.sign_class_loss_min_abs:.10g}",
                    f"{args.sign_class_loss_pos_weight:.10g}",
                    int(improved),
                    epochs_without_improvement,
                ]
            )
        if args.delta_monitor_checkpoints:
            with open(monitor_history_path, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(
                    [
                        epoch,
                        f"{val_loss:.10f}",
                        f"{monitor_metrics.get('delta_r_mean', float('nan')):.10f}",
                        f"{monitor_metrics.get('delta_r_count', float('nan')):.0f}",
                        f"{monitor_metrics.get('hard_delta_r_mean', float('nan')):.10f}",
                        f"{monitor_metrics.get('hard_delta_r_count', float('nan')):.0f}",
                        f"{monitor_metrics.get('sign_accuracy', float('nan')):.10f}",
                        f"{monitor_metrics.get('sign_count', float('nan')):.0f}",
                        f"{monitor_metrics.get('amplitude_ratio', float('nan')):.10f}",
                        f"{monitor_metrics.get('active_fraction', float('nan')):.10f}",
                        f"{monitor_metrics.get('pattern_score', float('nan')):.10f}",
                    ]
                )

        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
                "val_loss": val_loss,
                "best_val": best_val,
                "train_loss": train_loss,
                "in_channels": in_channels,
                "patch_size": list(patch_size),
                "architecture": architecture_name,
                "model_variant": args.model_variant,
                "base_channels": args.base_channels,
                "coarse_pool": args.coarse_pool,
                "global_channels": global_channels,
                "global_feature_channels": args.global_feature_channels,
                "use_global_context": not args.no_global_context,
                "v13_sidecar_root": os.path.abspath(args.v13_sidecar_root) if args.v13_sidecar_root else None,
                "v22_prev_sidecar_root": os.path.abspath(args.v22_prev_sidecar_root) if args.v22_prev_sidecar_root else None,
                "v14_normalization_root": os.path.abspath(args.v14_normalization_root) if args.v14_normalization_root else None,
                "v14_layer_min": args.v14_layer_min,
                "v14_layer_max": args.v14_layer_max,
                "v14_min_layer_overlap": args.v14_min_layer_overlap,
                "v13_global_sample_size": args.v13_global_sample_size,
                "high_residual_scale": args.high_residual_scale,
                "min_high_gate": args.min_high_gate,
                "min_texture_gate": args.min_texture_gate,
                "learned_texture_gate": not args.disable_learned_texture_gate,
                "bg_channel_idx": bg_channel_idx,
                "height_gate_channel_idx": height_gate_channel_idx,
                "base_loss": args.base_loss,
                "huber_delta": args.huber_delta,
                "gradient_loss_weight": args.gradient_loss_weight,
                "multiscale_loss_weight": args.multiscale_loss_weight,
                "multiscale_scales": list(multiscale_scales),
                "multiscale_min_valid_fraction": args.multiscale_min_valid_fraction,
                "smoothness_loss_weight": args.smoothness_loss_weight,
                "smoothness_kernel_size": args.smoothness_kernel_size,
                "variance_loss_weight": args.variance_loss_weight,
                "variance_min_std": args.variance_min_std,
                "variance_eps": args.variance_eps,
                "residual_weight_alpha": args.residual_weight_alpha,
                "residual_weight_scale": args.residual_weight_scale,
                "residual_weight_max": args.residual_weight_max,
                "target_gradient_weight_alpha": args.target_gradient_weight_alpha,
                "target_gradient_weight_scale": args.target_gradient_weight_scale,
                "target_gradient_weight_max": args.target_gradient_weight_max,
                "low_layer_weight_alpha": args.low_layer_weight_alpha,
                "normalized_loss_weight": args.normalized_loss_weight,
                "normalized_min_std": args.normalized_min_std,
                "normalized_huber_delta": args.normalized_huber_delta,
                "normalized_eps": args.normalized_eps,
                "correlation_loss_weight": args.correlation_loss_weight,
                "correlation_eps": args.correlation_eps,
                "correlation_min_target_std": args.correlation_min_target_std,
                "correlation_min_valid_fraction": args.correlation_min_valid_fraction,
                "pattern_height_decay": args.pattern_height_decay,
                "low_frequency_loss_weight": args.low_frequency_loss_weight,
                "low_frequency_pool": args.low_frequency_pool,
                "low_frequency_min_valid_fraction": args.low_frequency_min_valid_fraction,
                "low_frequency_correlation_weight": args.low_frequency_correlation_weight,
                "high_frequency_loss_weight": args.high_frequency_loss_weight,
                "high_frequency_huber_delta": args.high_frequency_huber_delta,
                "local_correlation_loss_weight": args.local_correlation_loss_weight,
                "local_correlation_pool": args.local_correlation_pool,
                "local_correlation_min_target_std": args.local_correlation_min_target_std,
                "local_correlation_min_valid_fraction": args.local_correlation_min_valid_fraction,
                "amplitude_loss_weight": args.amplitude_loss_weight,
                "amplitude_min_target_std": args.amplitude_min_target_std,
                "active_delta_loss_weight": args.active_delta_loss_weight,
                "active_delta_threshold": args.active_delta_threshold,
                "sign_loss_weight": args.sign_loss_weight,
                "sign_loss_min_abs": args.sign_loss_min_abs,
                "sign_loss_scale": args.sign_loss_scale,
                "active_loss_weight": args.active_loss_weight,
                "active_loss_threshold": args.active_loss_threshold,
                "active_loss_pos_weight": args.active_loss_pos_weight,
                "sign_class_loss_weight": args.sign_class_loss_weight,
                "sign_class_loss_min_abs": args.sign_class_loss_min_abs,
                "sign_class_loss_pos_weight": args.sign_class_loss_pos_weight,
                "v28_advection_features": args.v28_advection_features,
                "v28_advection_correction_target": args.v28_advection_correction_target,
                "v28_advection_dx": args.v28_advection_dx,
                "v28_advection_dy": args.v28_advection_dy,
                "v28_advection_dz": args.v28_advection_dz,
                "v28_advection_dt": args.v28_advection_dt,
                "v28_advection_delta_scale": args.v28_advection_delta_scale,
                "v28_advection_gradient_scale": args.v28_advection_gradient_scale,
                "v28_advection_clip": args.v28_advection_clip,
                "v28_advection_input_clip": args.v28_advection_input_clip,
                "target_mode": checkpoint_target_mode,
                "target_normalization_path": os.path.abspath(args.target_normalization_path) if args.target_normalization_path else None,
                "target_normalization_mode": args.target_normalization_mode,
                "input_channel_1": "corrected_ls_forcing_right_CO2" if using_v13_sidecar else "ls_forcing_right_CO2",
                "autoregressive_input_channel": "prev_kc_CO2" if using_v22_autoregressive else None,
                "autoregressive_target": (
                    "kc_CO2(t)-kc_CO2(t-1)-physics_advection_delta"
                    if (using_v22_autoregressive and args.v28_advection_features and args.v28_advection_correction_target)
                    else "kc_CO2(t)-kc_CO2(t-1)"
                    if using_v22_autoregressive
                    else None
                ),
                "dropped_input_channels": dropped_input_channels,
                "kept_input_channels": kept_input_channels,
                "kept_input_channel_indices": kept_input_channel_indices,
                "original_input_channels": original_input_channels,
                "coord_channels": coord_channels,
                "appended_channels": appended_channels,
                "surface_gated_channels": list(surface_channel_indices) if args.height_gate_surface else [],
                "height_gate_decay_levels": args.height_gate_decay_levels,
                "metadata_root": os.path.abspath(args.metadata_root) if args.metadata_root else None,
                "texture_sidecar_root": os.path.abspath(args.texture_sidecar_root) if args.texture_sidecar_root else None,
                "texture_aware_sampler": args.texture_aware_sampler,
                "excluded_months": list(exclude_months),
                "lr": lr_after,
                "epochs_without_improvement": epochs_without_improvement,
            },
            last_path,
        )

        if args.delta_monitor_checkpoints and monitor_metrics:
            current_best_values = {
                "best_delta_r": best_delta_r,
                "best_hard_delta_r": best_hard_delta_r,
                "best_sign_accuracy": best_sign_accuracy,
                "best_pattern_score": best_pattern_score,
            }
            monitor_candidates = [
                ("delta_r_mean", "best_delta_r", best_delta_r_path),
                ("hard_delta_r_mean", "best_hard_delta_r", best_hard_delta_r_path),
                ("sign_accuracy", "best_sign_accuracy", best_sign_accuracy_path),
                ("pattern_score", "best_pattern_score", best_pattern_score_path),
            ]
            for metric_name, best_name, path in monitor_candidates:
                value = float(monitor_metrics.get(metric_name, float("nan")))
                if math.isfinite(value) and value > current_best_values[best_name]:
                    shutil.copyfile(last_path, path)
                    current_best_values[best_name] = value
                    monitor_best[best_name] = {"epoch": epoch, "value": value}
                    print(
                        f"Saved monitor checkpoint {best_name}: "
                        f"epoch={epoch:03d} {metric_name}={value:.6f} -> {path}",
                        flush=True,
                    )
            best_delta_r = current_best_values["best_delta_r"]
            best_hard_delta_r = current_best_values["best_hard_delta_r"]
            best_sign_accuracy = current_best_values["best_sign_accuracy"]
            best_pattern_score = current_best_values["best_pattern_score"]
            with open(monitor_best_path, "w", encoding="utf-8") as f:
                json.dump(monitor_best, f, indent=2)

        if improved:
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
                    "val_loss": val_loss,
                    "best_val": best_val,
                    "train_loss": train_loss,
                    "in_channels": in_channels,
                    "patch_size": list(patch_size),
                    "architecture": architecture_name,
                    "model_variant": args.model_variant,
                    "base_channels": args.base_channels,
                    "coarse_pool": args.coarse_pool,
                    "global_channels": global_channels,
                    "global_feature_channels": args.global_feature_channels,
                    "use_global_context": not args.no_global_context,
                    "v13_sidecar_root": os.path.abspath(args.v13_sidecar_root) if args.v13_sidecar_root else None,
                    "v22_prev_sidecar_root": os.path.abspath(args.v22_prev_sidecar_root) if args.v22_prev_sidecar_root else None,
                    "v14_normalization_root": os.path.abspath(args.v14_normalization_root) if args.v14_normalization_root else None,
                    "v14_layer_min": args.v14_layer_min,
                    "v14_layer_max": args.v14_layer_max,
                    "v14_min_layer_overlap": args.v14_min_layer_overlap,
                    "v13_global_sample_size": args.v13_global_sample_size,
                    "high_residual_scale": args.high_residual_scale,
                    "min_high_gate": args.min_high_gate,
                    "min_texture_gate": args.min_texture_gate,
                    "learned_texture_gate": not args.disable_learned_texture_gate,
                    "bg_channel_idx": bg_channel_idx,
                    "height_gate_channel_idx": height_gate_channel_idx,
                    "base_loss": args.base_loss,
                    "huber_delta": args.huber_delta,
                    "gradient_loss_weight": args.gradient_loss_weight,
                    "multiscale_loss_weight": args.multiscale_loss_weight,
                    "multiscale_scales": list(multiscale_scales),
                    "multiscale_min_valid_fraction": args.multiscale_min_valid_fraction,
                    "smoothness_loss_weight": args.smoothness_loss_weight,
                    "smoothness_kernel_size": args.smoothness_kernel_size,
                "variance_loss_weight": args.variance_loss_weight,
                "variance_min_std": args.variance_min_std,
                "variance_eps": args.variance_eps,
                "residual_weight_alpha": args.residual_weight_alpha,
                "residual_weight_scale": args.residual_weight_scale,
                "residual_weight_max": args.residual_weight_max,
                "target_gradient_weight_alpha": args.target_gradient_weight_alpha,
                "target_gradient_weight_scale": args.target_gradient_weight_scale,
                "target_gradient_weight_max": args.target_gradient_weight_max,
                "low_layer_weight_alpha": args.low_layer_weight_alpha,
                "normalized_loss_weight": args.normalized_loss_weight,
                "normalized_min_std": args.normalized_min_std,
                "normalized_huber_delta": args.normalized_huber_delta,
                "normalized_eps": args.normalized_eps,
                "correlation_loss_weight": args.correlation_loss_weight,
                "correlation_eps": args.correlation_eps,
                "correlation_min_target_std": args.correlation_min_target_std,
                "correlation_min_valid_fraction": args.correlation_min_valid_fraction,
                "pattern_height_decay": args.pattern_height_decay,
                "low_frequency_loss_weight": args.low_frequency_loss_weight,
                "low_frequency_pool": args.low_frequency_pool,
                "low_frequency_min_valid_fraction": args.low_frequency_min_valid_fraction,
                "low_frequency_correlation_weight": args.low_frequency_correlation_weight,
                "high_frequency_loss_weight": args.high_frequency_loss_weight,
                "high_frequency_huber_delta": args.high_frequency_huber_delta,
                "local_correlation_loss_weight": args.local_correlation_loss_weight,
                "local_correlation_pool": args.local_correlation_pool,
                "local_correlation_min_target_std": args.local_correlation_min_target_std,
                "local_correlation_min_valid_fraction": args.local_correlation_min_valid_fraction,
                "amplitude_loss_weight": args.amplitude_loss_weight,
                "amplitude_min_target_std": args.amplitude_min_target_std,
                "active_delta_loss_weight": args.active_delta_loss_weight,
                "active_delta_threshold": args.active_delta_threshold,
                "sign_loss_weight": args.sign_loss_weight,
                "sign_loss_min_abs": args.sign_loss_min_abs,
                "sign_loss_scale": args.sign_loss_scale,
                "active_loss_weight": args.active_loss_weight,
                "active_loss_threshold": args.active_loss_threshold,
                "active_loss_pos_weight": args.active_loss_pos_weight,
                "sign_class_loss_weight": args.sign_class_loss_weight,
                "sign_class_loss_min_abs": args.sign_class_loss_min_abs,
                "sign_class_loss_pos_weight": args.sign_class_loss_pos_weight,
                "v28_advection_features": args.v28_advection_features,
                "v28_advection_correction_target": args.v28_advection_correction_target,
                "v28_advection_dx": args.v28_advection_dx,
                "v28_advection_dy": args.v28_advection_dy,
                "v28_advection_dz": args.v28_advection_dz,
                "v28_advection_dt": args.v28_advection_dt,
                "v28_advection_delta_scale": args.v28_advection_delta_scale,
                "v28_advection_gradient_scale": args.v28_advection_gradient_scale,
                "v28_advection_clip": args.v28_advection_clip,
                "v28_advection_input_clip": args.v28_advection_input_clip,
                "target_mode": checkpoint_target_mode,
                "target_normalization_path": os.path.abspath(args.target_normalization_path) if args.target_normalization_path else None,
                "target_normalization_mode": args.target_normalization_mode,
                    "input_channel_1": "corrected_ls_forcing_right_CO2" if using_v13_sidecar else "ls_forcing_right_CO2",
                    "autoregressive_input_channel": "prev_kc_CO2" if using_v22_autoregressive else None,
                    "autoregressive_target": (
                        "kc_CO2(t)-kc_CO2(t-1)-physics_advection_delta"
                        if (using_v22_autoregressive and args.v28_advection_features and args.v28_advection_correction_target)
                        else "kc_CO2(t)-kc_CO2(t-1)"
                        if using_v22_autoregressive
                        else None
                    ),
                    "dropped_input_channels": dropped_input_channels,
                    "kept_input_channels": kept_input_channels,
                    "kept_input_channel_indices": kept_input_channel_indices,
                    "original_input_channels": original_input_channels,
                    "coord_channels": coord_channels,
                    "appended_channels": appended_channels,
                    "surface_gated_channels": list(surface_channel_indices) if args.height_gate_surface else [],
                    "height_gate_decay_levels": args.height_gate_decay_levels,
                    "metadata_root": os.path.abspath(args.metadata_root) if args.metadata_root else None,
                    "texture_sidecar_root": os.path.abspath(args.texture_sidecar_root) if args.texture_sidecar_root else None,
                    "texture_aware_sampler": args.texture_aware_sampler,
                    "excluded_months": list(exclude_months),
                    "lr": lr_after,
                    "epochs_without_improvement": epochs_without_improvement,
                },
                best_path,
            )
            print(f"Saved new best checkpoint: {best_path}", flush=True)

        if (
            not args.disable_early_stopping
            and args.early_stopping_patience > 0
            and epochs_without_improvement >= args.early_stopping_patience
        ):
            print(
                f"Early stopping triggered at epoch {epoch:03d}: "
                f"no validation improvement for {epochs_without_improvement} epochs.",
                flush=True,
            )
            break

    print(f"Saved best checkpoint to: {best_path}")


if __name__ == "__main__":
    main()

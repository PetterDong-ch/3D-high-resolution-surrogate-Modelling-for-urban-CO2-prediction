from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F


# Load json from disk or cache.
def load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


# Build normalized coordinate values for one patch axis.
def norm_axis(start: int, total: int, length: int) -> np.ndarray:
    denom = max(int(total) - 1, 1)
    values = (np.arange(length, dtype=np.float32) + float(start)) / float(denom)
    return values * 2.0 - 1.0


# Compute NumPy 3D finite differences for cached arrays.
def finite_difference_3d_np(field: np.ndarray, spacing: float, axis: int) -> np.ndarray:
    spacing = max(float(spacing), 1.0e-6)
    arr = np.asarray(field, dtype=np.float32)
    out = np.zeros_like(arr, dtype=np.float32)
    n = arr.shape[axis]
    if n <= 1:
        return out

    first = [slice(None)] * arr.ndim
    first[axis] = 0
    second = [slice(None)] * arr.ndim
    second[axis] = 1
    out[tuple(first)] = (arr[tuple(second)] - arr[tuple(first)]) / spacing

    last = [slice(None)] * arr.ndim
    last[axis] = n - 1
    before = [slice(None)] * arr.ndim
    before[axis] = n - 2
    out[tuple(last)] = (arr[tuple(last)] - arr[tuple(before)]) / spacing

    if n > 2:
        mid = [slice(None)] * arr.ndim
        mid[axis] = slice(1, n - 1)
        plus = [slice(None)] * arr.ndim
        plus[axis] = slice(2, n)
        minus = [slice(None)] * arr.ndim
        minus[axis] = slice(0, n - 2)
        out[tuple(mid)] = (arr[tuple(plus)] - arr[tuple(minus)]) / (2.0 * spacing)
    return out


# Assemble the physical input channels for Stage 2.
def assemble_stage2_physical_input(sample: dict[str, np.ndarray], channels: tuple[str, ...]) -> dict[str, np.ndarray]:
    met = np.asarray(sample["met_pred"], dtype=np.float32)
    _, depth, height, width = met.shape

    emission = np.asarray(sample["emission_2d"], dtype=np.float32)[None, :, :]
    emission_3d = np.broadcast_to(emission, (depth, height, width))
    bg = np.asarray(sample["bg_profile"], dtype=np.float32)[:, None, None]
    bg_3d = np.broadcast_to(bg, (depth, height, width))
    scalar = np.asarray(sample["scalar"], dtype=np.float32)
    scalar_3d = np.broadcast_to(scalar[:, None, None, None], (scalar.shape[0], depth, height, width))

    z0 = int(np.asarray(sample["z0"]).item())
    y0 = int(np.asarray(sample["y0"]).item())
    x0 = int(np.asarray(sample["x0"]).item())
    nz = int(np.asarray(sample["nz"]).item())
    ny = int(np.asarray(sample["ny"]).item())
    nx = int(np.asarray(sample["nx"]).item())
    z_grid = norm_axis(z0, nz, depth)[:, None, None]
    y_grid = norm_axis(y0, ny, height)[None, :, None]
    x_grid = norm_axis(x0, nx, width)[None, None, :]

    physical = {
        "emission_values": emission_3d,
        "ls_forcing_right_CO2": bg_3d,
        "u": met[0],
        "v": met[1],
        "p": np.zeros((depth, height, width), dtype=np.float32),
        "theta": met[2],
        "w": met[3],
        "month_sin": scalar_3d[0],
        "month_cos": scalar_3d[1],
        "tod_sin": scalar_3d[2],
        "tod_cos": scalar_3d[3],
        "x_norm": np.broadcast_to(x_grid, (depth, height, width)),
        "y_norm": np.broadcast_to(y_grid, (depth, height, width)),
        "z_norm": np.broadcast_to(z_grid, (depth, height, width)),
    }
    return {name: physical[name].astype(np.float32, copy=False) for name in channels if name in physical}


# Downsample full-domain context fields to the model grid.
def downsample_context_stack(stack: np.ndarray, out_hw: int) -> torch.Tensor:
    """Resize a C,D,H,W context stack to C,D,out_hw,out_hw."""
    arr = np.ascontiguousarray(stack, dtype=np.float32)
    c, d, h, w = arr.shape
    tensor = torch.from_numpy(arr).view(c * d, 1, h, w)
    resized = F.interpolate(tensor, size=(int(out_hw), int(out_hw)), mode="bilinear", align_corners=False)
    return resized.view(c, d, int(out_hw), int(out_hw))


# Build an identity sampling grid for local context.
def identity_global_grid(depth: int, height: int, width: int) -> torch.Tensor:
    z_values = torch.linspace(-1.0, 1.0, steps=int(depth), dtype=torch.float32)
    y_values = torch.linspace(-1.0, 1.0, steps=int(height), dtype=torch.float32)
    x_values = torch.linspace(-1.0, 1.0, steps=int(width), dtype=torch.float32)
    zz = z_values.view(depth, 1, 1).expand(depth, height, width)
    yy = y_values.view(1, height, 1).expand(depth, height, width)
    xx = x_values.view(1, 1, width).expand(depth, height, width)
    return torch.stack((xx, yy, zz), dim=-1).contiguous()


# Build the sampling grid for full-domain context.
def full_domain_global_grid(z0: int, y0: int, x0: int, nz: int, ny: int, nx: int, depth: int, height: int, width: int) -> torch.Tensor:
    """Grid-sample coordinates mapping the local patch into a full-domain context."""
    z_values = torch.as_tensor(norm_axis(z0, nz, depth), dtype=torch.float32)
    y_values = torch.as_tensor(norm_axis(y0, ny, height), dtype=torch.float32)
    x_values = torch.as_tensor(norm_axis(x0, nx, width), dtype=torch.float32)
    zz = z_values.view(depth, 1, 1).expand(depth, height, width)
    yy = y_values.view(1, height, 1).expand(depth, height, width)
    xx = x_values.view(1, 1, width).expand(depth, height, width)
    return torch.stack((xx, yy, zz), dim=-1).contiguous()

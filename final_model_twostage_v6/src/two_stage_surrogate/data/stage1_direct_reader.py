from __future__ import annotations

import json
import os
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from netCDF4 import Dataset

from .alignment import linear_time_match, values_to_seconds


TOPOGRAPHY_PATH = os.environ.get(
    "CAMDEN_TOPOGRAPHY_PATH",
    str(Path(os.environ.get("CAMDEN_JOBS_ROOT", "external_data/camden/JOBS")) / "cam07_175vm_topo_surf_N02.000.nc"),
)


# Groups Stage 1 input and target arrays loaded from disk.
@dataclass(frozen=True)
class Stage1Arrays:
    sample_key: str
    geometry_3d: np.ndarray
    surface_2d: np.ndarray
    profile: np.ndarray
    scalar: np.ndarray
    theta_reference: np.ndarray
    target_uv: np.ndarray
    target_w: np.ndarray
    target_theta_prime: np.ndarray
    target_theta: np.ndarray
    mask: np.ndarray
    metadata: dict[str, Any]


# Internal helper for array.
def _array(value: Any) -> np.ndarray:
    arr = np.asanyarray(value)
    if np.ma.isMaskedArray(arr):
        arr = arr.astype(np.float32).filled(np.nan)
    return np.asarray(arr, dtype=np.float32)


# Internal helper for read time seconds.
def _read_time_seconds(path: str) -> np.ndarray:
    with Dataset(path, "r") as ds:
        var = ds.variables["time"]
        return values_to_seconds(_array(var[:]), getattr(var, "units", "seconds"))


# Internal helper for linear read profile.
def _linear_read_profile(ds: Dataset, name: str, left: int, right: int, weight_right: float) -> np.ndarray:
    var = ds.variables[name]
    left_value = _array(var[left])
    right_value = _array(var[right])
    return (1.0 - weight_right) * left_value + weight_right * right_value


# Internal helper for resize nearest 2d.
def _resize_nearest_2d(arr: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    if arr.shape == (out_h, out_w):
        return arr.astype(np.float32, copy=False)
    y_idx = np.linspace(0, arr.shape[0] - 1, out_h).round().astype(np.int64)
    x_idx = np.linspace(0, arr.shape[1] - 1, out_w).round().astype(np.int64)
    return arr[y_idx[:, None], x_idx[None, :]].astype(np.float32, copy=False)


# Internal helper for static patch 600 to output.
def _static_patch_600_to_output(
    static_arr: np.ndarray,
    y0: int,
    x0: int,
    patch_h: int,
    patch_w: int,
    output_h: int = 800,
    output_w: int = 800,
) -> np.ndarray:
    src_h, src_w = static_arr.shape
    sy0 = int(round(y0 * src_h / output_h))
    sx0 = int(round(x0 * src_w / output_w))
    sy1 = int(round((y0 + patch_h) * src_h / output_h))
    sx1 = int(round((x0 + patch_w) * src_w / output_w))
    sy1 = max(sy0 + 1, min(src_h, sy1))
    sx1 = max(sx0 + 1, min(src_w, sx1))
    crop = _array(static_arr[sy0:sy1, sx0:sx1])
    return _resize_nearest_2d(crop, patch_h, patch_w)


# Internal helper for topography 3d patch.
def _topography_3d_patch(topo_path: str, z0: int, y0: int, x0: int, dz: int, dy: int, dx: int) -> np.ndarray:
    with Dataset(topo_path, "r") as ds:
        topo = ds.variables["topo_all"]
        topo_z = len(ds.dimensions["z"])
        out = np.zeros((dz, dy, dx), dtype=np.float32)
        take = max(0, min(dz, topo_z - z0))
        if take > 0:
            out[:take] = _array(topo[z0 : z0 + take, y0 : y0 + dy, x0 : x0 + dx])
    return out


# Internal helper for topography surface patch.
def _topography_surface_patch(topo_path: str, y0: int, x0: int, dy: int, dx: int) -> tuple[np.ndarray, np.ndarray]:
    with Dataset(topo_path, "r") as ds:
        zt = _array(ds.variables["zt"][y0 : y0 + dy, x0 : x0 + dx])
        topo = _array(ds.variables["topo_all"][:, y0 : y0 + dy, x0 : x0 + dx])
    topo_2d = np.max(topo, axis=0)
    return topo_2d, zt


# Internal helper for interpolate profile to.
def _interpolate_profile_to(values: np.ndarray, source_z: np.ndarray, target_z: np.ndarray) -> np.ndarray:
    return np.interp(target_z, source_z, values).astype(np.float32)


# Internal helper for interpolate w to zu.
def _interpolate_w_to_zu(w_zw: np.ndarray, zw: np.ndarray, target_zu: np.ndarray) -> np.ndarray:
    z_count, h, w = w_zw.shape
    flat = w_zw.reshape(z_count, -1)
    out = np.empty((target_zu.size, flat.shape[1]), dtype=np.float32)
    for idx in range(flat.shape[1]):
        out[:, idx] = np.interp(target_zu, zw, flat[:, idx])
    return out.reshape(target_zu.size, h, w)


# Internal helper for norm coord.
def _norm_coord(start: int, size: int, total: int) -> np.ndarray:
    idx = np.arange(start, start + size, dtype=np.float32)
    return 2.0 * (idx / max(1, total - 1)) - 1.0


# Internal helper for tod encoding.
def _tod_encoding(seconds: float) -> tuple[float, float]:
    angle = 2.0 * math.pi * ((seconds % 86400.0) / 86400.0)
    return math.sin(angle), math.cos(angle)


# Internal helper for month encoding.
def _month_encoding(month: int) -> tuple[float, float]:
    angle = 2.0 * math.pi * ((month - 1) / 12.0)
    return math.sin(angle), math.cos(angle)


# Load manifest records from disk or cache.
def load_manifest_records(manifest_path: str | Path, split: str = "train") -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    jobs = {job["job"]: job for job in manifest["jobs"]}
    records = [record for record in manifest["records"] if record["split"] == split]
    return records, jobs


# Read stage1 sample from disk.
def read_stage1_sample(
    record: dict[str, Any],
    job_files: dict[str, Any],
    *,
    patch_h: int,
    patch_w: int,
    topo_path: str = TOPOGRAPHY_PATH,
) -> Stage1Arrays:
    z0 = int(record["z0"])
    y0 = int(record["y0"])
    x0 = int(record["x0"])
    dz = int(record["dz"])
    dy = min(patch_h, int(record["dy"]))
    dx = min(patch_w, int(record["dx"]))
    t_idx = int(record["output_time_index"])

    with Dataset(job_files["out3d"], "r") as out_ds:
        output_seconds = values_to_seconds(_array(out_ds.variables["time"][:]), getattr(out_ds.variables["time"], "units", "seconds"))
        zu = _array(out_ds.variables["zu_3d"][:])
        zw = _array(out_ds.variables["zw_3d"][:])
        target_zu = zu[z0 : z0 + dz]
        u = _array(out_ds.variables["u"][t_idx, z0 : z0 + dz, y0 : y0 + dy, x0 : x0 + dx])
        v = _array(out_ds.variables["v"][t_idx, z0 : z0 + dz, y0 : y0 + dy, x0 : x0 + dx])
        theta = _array(out_ds.variables["theta"][t_idx, z0 : z0 + dz, y0 : y0 + dy, x0 : x0 + dx])
        w_all = _array(out_ds.variables["w"][t_idx, :, y0 : y0 + dy, x0 : x0 + dx])
        w = _interpolate_w_to_zu(w_all, zw, target_zu)
        output_nz = len(out_ds.dimensions["zu_3d"])
        output_ny = len(out_ds.dimensions["y"])
        output_nx = len(out_ds.dimensions["x"])
        out_sec = float(output_seconds[t_idx])

    with Dataset(job_files["dynamic"], "r") as dyn_ds:
        dyn_seconds = values_to_seconds(_array(dyn_ds.variables["time"][:]), getattr(dyn_ds.variables["time"], "units", "seconds"))
        match = linear_time_match(out_sec, dyn_seconds)
        dyn_z = _array(dyn_ds.variables["z"][:])
        dyn_zw = _array(dyn_ds.variables["zw"][:])
        wr = match.weight_right
        profile_u = _linear_read_profile(dyn_ds, "ls_forcing_right_u", match.left_index, match.right_index, wr)
        profile_v = _linear_read_profile(dyn_ds, "ls_forcing_right_v", match.left_index, match.right_index, wr)
        profile_w_raw = _linear_read_profile(dyn_ds, "ls_forcing_right_w", match.left_index, match.right_index, wr)
        profile_w = _interpolate_profile_to(profile_w_raw, dyn_zw, dyn_z)
        profile_pt = _linear_read_profile(dyn_ds, "ls_forcing_right_pt", match.left_index, match.right_index, wr)
        profile_qv = _linear_read_profile(dyn_ds, "ls_forcing_right_qv", match.left_index, match.right_index, wr)
        profile_co2 = _linear_read_profile(dyn_ds, "ls_forcing_right_CO2", match.left_index, match.right_index, wr)
        theta_reference_1d = _interpolate_profile_to(profile_pt, dyn_z, target_zu)

    profile = np.stack([profile_u, profile_v, profile_w, profile_pt, profile_qv, profile_co2], axis=0)
    theta_reference = theta_reference_1d[None, :, None, None]
    theta_prime = theta - theta_reference_1d[:, None, None]

    topo_3d = _topography_3d_patch(topo_path, z0, y0, x0, dz, dy, dx)
    fluid_mask = (topo_3d == 0).astype(np.float32)
    building_voxel_mask = (topo_3d != 0).astype(np.float32)
    finite_mask = np.isfinite(u) & np.isfinite(v) & np.isfinite(w) & np.isfinite(theta_prime)
    mask = (fluid_mask > 0) & finite_mask

    z_grid = _norm_coord(z0, dz, output_nz)[:, None, None]
    y_grid = _norm_coord(y0, dy, output_ny)[None, :, None]
    x_grid = _norm_coord(x0, dx, output_nx)[None, None, :]
    geometry = np.stack(
        [
            fluid_mask,
            np.broadcast_to(x_grid, (dz, dy, dx)),
            np.broadcast_to(y_grid, (dz, dy, dx)),
            np.broadcast_to(z_grid, (dz, dy, dx)),
            building_voxel_mask,
        ],
        axis=0,
    ).astype(np.float32)

    topo_2d, building_height = _topography_surface_patch(topo_path, y0, x0, dy, dx)
    with Dataset(job_files["static"], "r") as static_ds:
        surface = [
            topo_2d,
            building_height,
            _static_patch_600_to_output(static_ds.variables["vegetation_type"], y0, x0, dy, dx),
            _static_patch_600_to_output(static_ds.variables["pavement_type"], y0, x0, dy, dx),
            _static_patch_600_to_output(static_ds.variables["water_type"], y0, x0, dy, dx),
            _static_patch_600_to_output(static_ds.variables["albedo_type"], y0, x0, dy, dx),
            _static_patch_600_to_output(static_ds.variables["evi_pft"], y0, x0, dy, dx),
            _static_patch_600_to_output(static_ds.variables["lswi_pft"], y0, x0, dy, dx),
        ]
    surface_2d = np.stack(surface, axis=0).astype(np.float32)

    month_sin, month_cos = _month_encoding(int(record["month"]))
    tod_sin, tod_cos = _tod_encoding(out_sec)
    scalar = np.asarray([month_sin, month_cos, tod_sin, tod_cos], dtype=np.float32)

    target_uv = np.stack([u, v], axis=0).astype(np.float32)
    target_w = w[None].astype(np.float32)
    target_theta_prime = theta_prime.astype(np.float32)
    target_theta = theta.astype(np.float32)
    target_theta_prime = target_theta_prime[None]
    target_theta = target_theta[None]
    mask_out = mask[None].astype(np.float32)

    metadata = {
        "job": record["job"],
        "month": int(record["month"]),
        "time_index": t_idx,
        "output_seconds": out_sec,
        "z0": z0,
        "y0": y0,
        "x0": x0,
        "dz": dz,
        "dy": dy,
        "dx": dx,
        "valid_fraction": float(mask_out.mean()),
        "dynamic_left_index": match.left_index,
        "dynamic_right_index": match.right_index,
        "dynamic_weight_right": match.weight_right,
    }

    return Stage1Arrays(
        sample_key=str(record["sample_key"]),
        geometry_3d=np.nan_to_num(geometry, nan=0.0, posinf=0.0, neginf=0.0),
        surface_2d=np.nan_to_num(surface_2d, nan=0.0, posinf=0.0, neginf=0.0),
        profile=np.nan_to_num(profile, nan=0.0, posinf=0.0, neginf=0.0),
        scalar=scalar,
        theta_reference=np.nan_to_num(theta_reference.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0),
        target_uv=np.nan_to_num(target_uv, nan=0.0, posinf=0.0, neginf=0.0),
        target_w=np.nan_to_num(target_w, nan=0.0, posinf=0.0, neginf=0.0),
        target_theta_prime=np.nan_to_num(target_theta_prime, nan=0.0, posinf=0.0, neginf=0.0),
        target_theta=np.nan_to_num(target_theta, nan=0.0, posinf=0.0, neginf=0.0),
        mask=mask_out,
        metadata=metadata,
    )

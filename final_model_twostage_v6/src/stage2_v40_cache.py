from __future__ import annotations

import bisect
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset


STAGE2_TO_V40_CHANNEL = {
    "emission_values": "emission_values",
    "ls_forcing_right_CO2": "ls_forcing_right_CO2",
    "u": "stage1_u",
    "v": "stage1_v",
    "theta": "stage1_theta",
    "w": "stage1_w",
    "month_sin": "month_sin",
    "month_cos": "month_cos",
    "tod_sin": "tod_sin",
    "tod_cos": "tod_cos",
    "x_norm": "x_norm",
    "y_norm": "y_norm",
    "z_norm": "z_norm",
}

V40_STAGE1_MET_CHANNELS = (
    "emission_values",
    "ls_forcing_right_CO2",
    "u",
    "v",
    "p",
    "theta",
    "w",
    "month_sin",
    "month_cos",
    "tod_sin",
    "tod_cos",
    "x_norm",
    "y_norm",
    "z_norm",
    "prev_kc_CO2",
    "prev_dCdx",
    "prev_dCdy",
    "prev_dCdz",
)


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


# Reads Stage 2 shard metadata and cached tensor arrays.
class Stage2ShardReader:
    # Load manifests and initialize shard caches.
    def __init__(self, cache_root: Path, split: str) -> None:
        self.cache_root = Path(cache_root)
        self.split = split
        self.manifest = load_json(self.cache_root / "cache_manifest.json")
        self.channels = tuple(self.manifest["channels"])
        split_info = self.manifest["splits"][split]
        self.shards = split_info["shards"]
        self.cumulative: list[int] = []
        total = 0
        for shard in self.shards:
            total += int(shard["count"])
            self.cumulative.append(total)
        self.total = total
        self._loaded_index: int | None = None
        self._loaded: dict[str, np.ndarray] | None = None
        self._light_index: int | None = None
        self._light: dict[str, np.ndarray] | None = None

    # Return the number of available samples.
    def __len__(self) -> int:
        return self.total

    # Map a global sample index to its shard and row offsets.
    def local_index(self, index: int) -> tuple[int, int]:
        if index < 0:
            index += self.total
        if index < 0 or index >= self.total:
            raise IndexError(index)
        shard_idx = bisect.bisect_right(self.cumulative, index)
        prev = 0 if shard_idx == 0 else self.cumulative[shard_idx - 1]
        return shard_idx, index - prev

    # Load a shard and cache it for repeated access.
    def load_shard(self, shard_idx: int) -> dict[str, np.ndarray]:
        if self._loaded_index == shard_idx and self._loaded is not None:
            return self._loaded
        path = self.cache_root / self.shards[shard_idx]["path"]
        loaded = np.load(path, allow_pickle=False)
        self._loaded_index = shard_idx
        self._loaded = {name: loaded[name] for name in loaded.files}
        return self._loaded

    # Return one sample by global index.
    def sample(self, index: int) -> dict[str, np.ndarray]:
        shard_idx, local_idx = self.local_index(index)
        data = self.load_shard(shard_idx)
        return {name: data[name][local_idx] for name in data.keys()}

    # Load the lightweight shard fields needed for indexing.
    def load_light_shard(self, shard_idx: int) -> dict[str, np.ndarray]:
        if self._light_index == shard_idx and self._light is not None:
            return self._light
        path = self.cache_root / self.shards[shard_idx]["path"]
        loaded = np.load(path, allow_pickle=False)
        self._light_index = shard_idx
        self._light = {
            "z0": loaded["z0"],
        }
        return self._light

    # Return the starting vertical index for one sample.
    def z0_at(self, index: int) -> int:
        shard_idx, local_idx = self.local_index(index)
        data = self.load_light_shard(shard_idx)
        return int(np.asarray(data["z0"][local_idx]).item())


# Reads previous-timestep CO2 sidecar shards.
class PrevCo2SidecarReader:
    # Load manifests and initialize shard caches.
    def __init__(self, sidecar_root: Path, split: str) -> None:
        self.sidecar_root = Path(sidecar_root)
        self.split = split
        root_manifest = load_json(self.sidecar_root / "manifest.json")
        self.manifest = root_manifest["splits"][split]
        self.norm = load_json(self.sidecar_root / "normalization.json")
        self.shards = self.manifest["shards"]
        self.cumulative: list[int] = []
        total = 0
        for shard in self.shards:
            total += int(shard["count"])
            self.cumulative.append(total)
        self.total = total
        self._loaded_index: int | None = None
        self._loaded: dict[str, np.ndarray] | None = None
        self._has_prev_index: int | None = None
        self._has_prev: np.ndarray | None = None

    # Map a global sample index to its shard and row offsets.
    def local_index(self, index: int) -> tuple[int, int]:
        if index < 0:
            index += self.total
        if index < 0 or index >= self.total:
            raise IndexError(index)
        shard_idx = bisect.bisect_right(self.cumulative, index)
        prev = 0 if shard_idx == 0 else self.cumulative[shard_idx - 1]
        return shard_idx, index - prev

    # Load a shard and cache it for repeated access.
    def load_shard(self, shard_idx: int) -> dict[str, np.ndarray]:
        if self._loaded_index == shard_idx and self._loaded is not None:
            return self._loaded
        path = self.sidecar_root / self.shards[shard_idx]["path"]
        loaded = np.load(path, allow_pickle=False)
        self._loaded_index = shard_idx
        self._loaded = {name: loaded[name] for name in loaded.files}
        return self._loaded

    # Return one sample by global index.
    def sample(self, index: int) -> tuple[np.ndarray, int]:
        shard_idx, local_idx = self.local_index(index)
        data = self.load_shard(shard_idx)
        return np.asarray(data["prev_co2"][local_idx], dtype=np.float32), int(data["has_prev"][local_idx])

    # Return whether a previous-timestep CO2 profile exists for one sample.
    def has_prev_at(self, index: int) -> int:
        shard_idx, local_idx = self.local_index(index)
        if self._has_prev_index != shard_idx or self._has_prev is None:
            path = self.sidecar_root / self.shards[shard_idx]["path"]
            loaded = np.load(path, allow_pickle=False)
            self._has_prev = loaded["has_prev"]
            self._has_prev_index = shard_idx
        return int(self._has_prev[local_idx])


# Loads samples for the Stage2 V40 Local data pipeline.
class Stage2V40LocalDataset(Dataset):
    """V40-like autoregressive delta dataset using Stage1-predicted met fields.

    The dataset keeps the V40 keep-prev local channel contract but replaces
    PALM-resolved u/v/theta/w with Stage1 predictions from the Stage2 cache.
    It intentionally returns no global context so the model cannot consume
    PALM meteorology through a hidden side branch.
    """

    # Load split files and prepare dataset state.
    def __init__(
        self,
        cache_root: str | Path,
        prev_sidecar_root: str | Path,
        split: str,
        *,
        layer_min: int = 1,
        layer_max: int = 10,
        min_layer_overlap: int = 1,
        dx: float = 10.0,
        dy: float = 10.0,
        dz: float = 10.0,
        channels: tuple[str, ...] = V40_STAGE1_MET_CHANNELS,
    ) -> None:
        self.cache = Stage2ShardReader(Path(cache_root), split)
        self.prev = PrevCo2SidecarReader(Path(prev_sidecar_root), split)
        self.total_source = min(self.cache.total, self.prev.total)
        self.split = split
        self.channels = tuple(channels)
        self.layer_min = int(layer_min)
        self.layer_max = int(layer_max)
        self.min_layer_overlap = int(min_layer_overlap)
        self.dx = float(dx)
        self.dy = float(dy)
        self.dz = float(dz)
        self.target_mode = "autoregressive_delta_from_stage1_met"

        stage2_stats = self.cache.manifest["normalization"]
        stage2_names = tuple(self.cache.manifest["channels"])
        stage2_mean = {name: float(stage2_stats["x_mean"][idx]) for idx, name in enumerate(stage2_names)}
        stage2_std = {name: max(float(stage2_stats["x_std"][idx]), 1.0e-6) for idx, name in enumerate(stage2_names)}
        prev_norm = self.prev.norm
        self.x_mean: list[float] = []
        self.x_std: list[float] = []
        for name in self.channels:
            if name in STAGE2_TO_V40_CHANNEL:
                source = STAGE2_TO_V40_CHANNEL[name]
                self.x_mean.append(stage2_mean[source])
                self.x_std.append(stage2_std[source])
            elif name == "p":
                self.x_mean.append(0.0)
                self.x_std.append(1.0)
            elif name == "prev_kc_CO2":
                self.x_mean.append(float(prev_norm["prev_kc_CO2_mean"]))
                self.x_std.append(max(float(prev_norm["prev_kc_CO2_std"]), 1.0e-6))
            elif name in ("prev_dCdx", "prev_dCdy", "prev_dCdz"):
                self.x_mean.append(float(prev_norm[f"{name}_mean"]))
                self.x_std.append(max(float(prev_norm[f"{name}_std"]), 1.0e-6))
            else:
                raise KeyError(f"Unknown V40-stage1 channel: {name}")
        self.x_mean_tensor = torch.tensor(self.x_mean, dtype=torch.float32).view(-1, 1, 1, 1)
        self.x_std_tensor = torch.tensor(self.x_std, dtype=torch.float32).view(-1, 1, 1, 1).clamp_min(1.0e-6)

        indices: list[int] = []
        depth = int(self.cache.manifest.get("patch_shape", [16, 256, 256])[0])
        for idx in range(self.total_source):
            has_prev = self.prev.has_prev_at(idx)
            if not has_prev:
                continue
            z0 = self.cache.z0_at(idx)
            global_z = np.arange(z0, z0 + depth, dtype=np.int32)
            overlap = int(((global_z >= self.layer_min) & (global_z <= self.layer_max)).sum())
            if overlap >= self.min_layer_overlap:
                indices.append(idx)
        if not indices:
            raise RuntimeError(f"No usable {split} samples after previous-CO2/layer filtering")
        self.indices = indices

    # Return the number of available samples.
    def __len__(self) -> int:
        return len(self.indices)

    # Internal helper for focused mask.
    def _focused_mask(self, mask: np.ndarray, z0: int) -> np.ndarray:
        out = np.asarray(mask, dtype=np.float32).copy()
        depth = out.shape[1]
        global_z = np.arange(z0, z0 + depth, dtype=np.int32)
        keep = ((global_z >= self.layer_min) & (global_z <= self.layer_max)).astype(np.float32)
        out *= keep[None, :, None, None]
        return out

    # Return one indexed sample in the format expected by the model.
    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        source_index = self.indices[index]
        sample = self.cache.sample(source_index)
        prev_co2, has_prev = self.prev.sample(source_index)
        if not has_prev:
            raise RuntimeError("Filtered index unexpectedly has no previous CO2")

        physical_base = assemble_stage2_physical_input(sample, self.channels)
        dc_dx = finite_difference_3d_np(prev_co2, self.dx, axis=2)
        dc_dy = finite_difference_3d_np(prev_co2, self.dy, axis=1)
        dc_dz = finite_difference_3d_np(prev_co2, self.dz, axis=0)
        physical_base["prev_kc_CO2"] = prev_co2
        physical_base["prev_dCdx"] = dc_dx
        physical_base["prev_dCdy"] = dc_dy
        physical_base["prev_dCdz"] = dc_dz

        x = np.stack([physical_base[name] for name in self.channels], axis=0).astype(np.float32, copy=False)
        x_t = (torch.from_numpy(x) - self.x_mean_tensor) / self.x_std_tensor

        bg = np.asarray(sample["bg_profile"], dtype=np.float32)[:, None, None]
        current_co2 = np.asarray(sample["target"], dtype=np.float32)[0] + bg
        target_delta = current_co2 - prev_co2
        y_t = torch.from_numpy(target_delta[None].astype(np.float32, copy=False))
        mask = self._focused_mask(np.asarray(sample["mask"], dtype=np.float32), int(np.asarray(sample["z0"]).item()))
        m_t = torch.from_numpy(mask.astype(np.float32, copy=False))
        return x_t, y_t, m_t

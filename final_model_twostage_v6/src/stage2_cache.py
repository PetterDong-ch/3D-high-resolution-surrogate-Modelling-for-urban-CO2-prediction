from __future__ import annotations

import bisect
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset


BASE_CHANNELS = (
    "emission_values",
    "ls_forcing_right_CO2",
    "stage1_u",
    "stage1_v",
    "stage1_theta",
    "stage1_w",
    "topo_all",
    "building_height",
    "vegetation_type",
    "pavement_type",
    "water_type",
    "albedo_type",
    "evi_pft",
    "lswi_pft",
    "month_sin",
    "month_cos",
    "tod_sin",
    "tod_cos",
    "x_norm",
    "y_norm",
    "z_norm",
    "fluid_mask",
)


# Accumulates channel-wise mean and variance statistics.
class ChannelStats:
    # Store constructor arguments and initialize object state.
    def __init__(self) -> None:
        self.sum: np.ndarray | None = None
        self.sumsq: np.ndarray | None = None
        self.count: np.ndarray | None = None

    # Update running metric or statistic accumulators.
    def update(self, arr: np.ndarray, mask: np.ndarray | None = None) -> None:
        arr64 = np.asarray(arr, dtype=np.float64)
        channels = arr64.shape[0]
        flat = arr64.reshape(channels, -1)
        if mask is None:
            valid = np.isfinite(flat)
        else:
            mask_flat = np.asarray(mask, dtype=bool).reshape(1, -1)
            valid = np.broadcast_to(mask_flat, flat.shape) & np.isfinite(flat)
        values = np.where(valid, flat, 0.0)
        counts = valid.sum(axis=1).astype(np.float64)
        if self.sum is None:
            self.sum = np.zeros(channels, dtype=np.float64)
            self.sumsq = np.zeros(channels, dtype=np.float64)
            self.count = np.zeros(channels, dtype=np.float64)
        self.sum += values.sum(axis=1)
        self.sumsq += (values * values).sum(axis=1)
        self.count += counts

    # Return the final accumulated statistics.
    def finalize(self) -> tuple[np.ndarray, np.ndarray]:
        if self.sum is None or self.sumsq is None or self.count is None:
            raise RuntimeError("Cannot finalize empty stats")
        count = np.maximum(self.count, 1.0)
        mean = self.sum / count
        var = np.maximum(self.sumsq / count - mean * mean, 1.0e-12)
        std = np.sqrt(var)
        return mean.astype(np.float32), std.astype(np.float32)


# Build normalized coordinate values for one patch axis.
def norm_axis(start: int, total: int, length: int) -> np.ndarray:
    denom = max(int(total) - 1, 1)
    values = (np.arange(length, dtype=np.float32) + float(start)) / float(denom)
    return values * 2.0 - 1.0


# Assemble the input tensor for Stage 2.
def assemble_stage2_input(sample: dict[str, np.ndarray], channels: tuple[str, ...] = BASE_CHANNELS) -> np.ndarray:
    met = np.asarray(sample["met_pred"], dtype=np.float32)
    _, depth, height, width = met.shape

    emission = np.asarray(sample["emission_2d"], dtype=np.float32)[None, :, :]
    emission_3d = np.broadcast_to(emission, (depth, height, width))
    bg = np.asarray(sample["bg_profile"], dtype=np.float32)[:, None, None]
    bg_3d = np.broadcast_to(bg, (depth, height, width))
    surface = np.asarray(sample["surface_2d"], dtype=np.float32)
    surface_3d = np.broadcast_to(surface[:, None, :, :], (surface.shape[0], depth, height, width))
    scalar = np.asarray(sample["scalar"], dtype=np.float32)
    scalar_3d = np.broadcast_to(scalar[:, None, None, None], (scalar.shape[0], depth, height, width))
    mask = np.asarray(sample["mask"], dtype=np.float32)[0]

    z0 = int(np.asarray(sample["z0"]).item())
    y0 = int(np.asarray(sample["y0"]).item())
    x0 = int(np.asarray(sample["x0"]).item())
    nz = int(np.asarray(sample["nz"]).item())
    ny = int(np.asarray(sample["ny"]).item())
    nx = int(np.asarray(sample["nx"]).item())
    z_grid = norm_axis(z0, nz, depth)[:, None, None]
    y_grid = norm_axis(y0, ny, height)[None, :, None]
    x_grid = norm_axis(x0, nx, width)[None, None, :]

    available = {
        "emission_values": emission_3d,
        "ls_forcing_right_CO2": bg_3d,
        "stage1_u": met[0],
        "stage1_v": met[1],
        "stage1_theta": met[2],
        "stage1_w": met[3],
        "topo_all": surface_3d[0],
        "building_height": surface_3d[1],
        "vegetation_type": surface_3d[2],
        "pavement_type": surface_3d[3],
        "water_type": surface_3d[4],
        "albedo_type": surface_3d[5],
        "evi_pft": surface_3d[6],
        "lswi_pft": surface_3d[7],
        "month_sin": scalar_3d[0],
        "month_cos": scalar_3d[1],
        "tod_sin": scalar_3d[2],
        "tod_cos": scalar_3d[3],
        "x_norm": np.broadcast_to(x_grid, (depth, height, width)),
        "y_norm": np.broadcast_to(y_grid, (depth, height, width)),
        "z_norm": np.broadcast_to(z_grid, (depth, height, width)),
        "fluid_mask": mask,
    }
    return np.stack([available[name] for name in channels], axis=0).astype(np.float32, copy=False)


# Load stage2 manifest from disk or cache.
def load_stage2_manifest(cache_root: str | Path) -> dict[str, Any]:
    path = Path(cache_root) / "cache_manifest.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing Stage 2 cache manifest: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


# Loads samples for the Stage2Cache data pipeline.
class Stage2CacheDataset(Dataset):
    # Load split files and prepare dataset state.
    def __init__(self, cache_root: str | Path, split: str, normalize: bool = True) -> None:
        self.cache_root = Path(cache_root)
        self.manifest = load_stage2_manifest(self.cache_root)
        if split not in self.manifest["splits"]:
            raise KeyError(f"Split {split!r} not present in {self.cache_root}")
        self.split = split
        self.normalize = normalize
        self.channels = tuple(self.manifest["channels"])
        self.target_mode = str(self.manifest["target_mode"])
        self.stats = self.manifest["normalization"]
        self.x_mean = torch.tensor(self.stats["x_mean"], dtype=torch.float32).view(-1, 1, 1, 1)
        self.x_std = torch.tensor(self.stats["x_std"], dtype=torch.float32).clamp_min(1.0e-6).view(-1, 1, 1, 1)
        self.y_mean = float(self.stats["target_mean"][0])
        self.y_std = max(float(self.stats["target_std"][0]), 1.0e-6)

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

    # Return the number of available samples.
    def __len__(self) -> int:
        return self.total

    # Internal helper for local index.
    def _local_index(self, index: int) -> tuple[int, int]:
        if index < 0:
            index += self.total
        if index < 0 or index >= self.total:
            raise IndexError(index)
        shard_idx = bisect.bisect_right(self.cumulative, index)
        prev = 0 if shard_idx == 0 else self.cumulative[shard_idx - 1]
        return shard_idx, index - prev

    # Internal helper for load shard.
    def _load_shard(self, shard_idx: int) -> dict[str, np.ndarray]:
        if self._loaded_index == shard_idx and self._loaded is not None:
            return self._loaded
        shard_path = self.cache_root / self.shards[shard_idx]["path"]
        loaded = np.load(shard_path, allow_pickle=False)
        self._loaded_index = shard_idx
        self._loaded = {name: loaded[name] for name in loaded.files}
        return self._loaded

    # Return one indexed sample in the format expected by the model.
    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        shard_idx, local_idx = self._local_index(index)
        data = self._load_shard(shard_idx)
        sample = {name: data[name][local_idx] for name in data.keys() if name not in ("sample_key", "metadata_json")}
        x = torch.from_numpy(assemble_stage2_input(sample, self.channels))
        y = torch.from_numpy(np.asarray(sample["target"], dtype=np.float32))
        mask = torch.from_numpy(np.asarray(sample["mask"], dtype=np.float32))
        bg = torch.from_numpy(np.asarray(sample["bg_profile"], dtype=np.float32))
        if self.normalize:
            x = (x - self.x_mean) / self.x_std
            y = (y - self.y_mean) / self.y_std
        return {
            "x": x,
            "target": y,
            "mask": mask,
            "bg_profile": bg,
            "z0": torch.tensor(int(np.asarray(sample["z0"]).item()), dtype=torch.long),
            "sample_key": str(data["sample_key"][local_idx]),
            "metadata_json": str(data["metadata_json"][local_idx]),
        }

    # Convert normalized targets back to physical units.
    def denormalize_target(self, target: torch.Tensor) -> torch.Tensor:
        return target * self.y_std + self.y_mean

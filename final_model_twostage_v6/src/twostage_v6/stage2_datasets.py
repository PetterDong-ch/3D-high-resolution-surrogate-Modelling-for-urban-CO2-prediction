from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .stage2_constants import (
    STAGE2_TO_V40_CHANNEL,
    V40_STAGE1_GLOBAL_CONTEXT_CHANNELS,
    V40_STAGE1_MET_CHANNELS,
)
from .stage2_readers import FullDomainContextSidecarReader, PrevCo2SidecarReader, Stage2ShardReader
from .stage2_utils import (
    assemble_stage2_physical_input,
    downsample_context_stack,
    finite_difference_3d_np,
    full_domain_global_grid,
    identity_global_grid,
)


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


# Loads samples for the Stage2 V40 Global Context data pipeline.
class Stage2V40GlobalContextDataset(Stage2V40LocalDataset):
    """V40-style dataset with a Stage1-met low-resolution context branch.

    The local branch is identical to :class:`Stage2V40LocalDataset`. The global
    branch intentionally avoids PALM-resolved meteorology: it is assembled from
    the same Stage1-predicted met fields, emissions, background CO2, fluid mask,
    and previous CO2 available in the Stage2 cache.
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
        global_channels: tuple[str, ...] = V40_STAGE1_GLOBAL_CONTEXT_CHANNELS,
        global_size: int = 80,
        global_sidecar_root: str | Path | None = None,
    ) -> None:
        super().__init__(
            cache_root,
            prev_sidecar_root,
            split,
            layer_min=layer_min,
            layer_max=layer_max,
            min_layer_overlap=min_layer_overlap,
            dx=dx,
            dy=dy,
            dz=dz,
            channels=channels,
        )
        self.global_channels = tuple(global_channels)
        self.global_size = int(global_size)
        self.global_sidecar = FullDomainContextSidecarReader(global_sidecar_root, split) if global_sidecar_root is not None else None
        if self.global_sidecar is not None:
            if self.global_sidecar.channels != self.global_channels:
                raise ValueError(
                    f"Global sidecar channels {self.global_sidecar.channels} do not match expected {self.global_channels}"
                )
            if self.global_sidecar.sample_context_index.shape[0] < self.total_source:
                raise ValueError(
                    f"Global sidecar sample map is shorter than Stage2 source split: "
                    f"{self.global_sidecar.sample_context_index.shape[0]} < {self.total_source}"
                )
            shape = self.global_sidecar.context_shape
            if len(shape) != 4 or shape[0] != len(self.global_channels) or shape[2] != self.global_size or shape[3] != self.global_size:
                raise ValueError(f"Unexpected global sidecar shape: {shape}")

        stage2_stats = self.cache.manifest["normalization"]
        stage2_names = tuple(self.cache.manifest["channels"])
        stage2_mean = {name: float(stage2_stats["x_mean"][idx]) for idx, name in enumerate(stage2_names)}
        stage2_std = {name: max(float(stage2_stats["x_std"][idx]), 1.0e-6) for idx, name in enumerate(stage2_names)}
        prev_norm = self.prev.norm

        global_mean: list[float] = []
        global_std: list[float] = []
        if self.global_sidecar is not None:
            global_mean = [float(v) for v in self.global_sidecar.normalization["mean"]]
            global_std = [max(float(v), 1.0e-6) for v in self.global_sidecar.normalization["std"]]
        else:
            for name in self.global_channels:
                if name in STAGE2_TO_V40_CHANNEL:
                    source = STAGE2_TO_V40_CHANNEL[name]
                    global_mean.append(stage2_mean[source])
                    global_std.append(stage2_std[source])
                elif name == "w":
                    source = STAGE2_TO_V40_CHANNEL[name]
                    global_mean.append(stage2_mean[source])
                    global_std.append(stage2_std[source])
                elif name == "p":
                    global_mean.append(0.0)
                    global_std.append(1.0)
                elif name == "fluid_mask":
                    global_mean.append(0.0)
                    global_std.append(1.0)
                elif name == "prev_kc_CO2":
                    global_mean.append(float(prev_norm["prev_kc_CO2_mean"]))
                    global_std.append(max(float(prev_norm["prev_kc_CO2_std"]), 1.0e-6))
                else:
                    raise KeyError(f"Unknown V40-stage1 global channel: {name}")

        self.global_mean_tensor = torch.tensor(global_mean, dtype=torch.float32).view(-1, 1, 1, 1)
        self.global_std_tensor = torch.tensor(global_std, dtype=torch.float32).view(-1, 1, 1, 1).clamp_min(1.0e-6)

    # Internal helper for global context from physical.
    def _global_context_from_physical(
        self,
        physical: dict[str, np.ndarray],
        prev_co2: np.ndarray,
        mask: np.ndarray,
    ) -> torch.Tensor:
        fluid = np.asarray(mask, dtype=np.float32)[0]
        context_physical = dict(physical)
        context_physical["fluid_mask"] = fluid
        context_physical["prev_kc_CO2"] = np.asarray(prev_co2, dtype=np.float32)
        stack = np.stack([context_physical[name] for name in self.global_channels], axis=0)
        global_t = downsample_context_stack(stack, self.global_size)
        return (global_t - self.global_mean_tensor) / self.global_std_tensor

    # Return one indexed sample in the format expected by the model.
    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
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

        raw_mask = np.asarray(sample["mask"], dtype=np.float32)
        m_t = torch.from_numpy(
            self._focused_mask(raw_mask, int(np.asarray(sample["z0"]).item())).astype(np.float32, copy=False)
        )
        if self.global_sidecar is None:
            global_context = self._global_context_from_physical(physical_base, prev_co2, raw_mask)
        else:
            global_physical = self.global_sidecar.sample(source_index)
            global_context = (global_physical - self.global_mean_tensor) / self.global_std_tensor
        _, depth, height, width = x_t.shape
        if self.global_sidecar is None:
            global_grid = identity_global_grid(depth, height, width)
        else:
            global_grid = full_domain_global_grid(
                int(np.asarray(sample["z0"]).item()),
                int(np.asarray(sample["y0"]).item()),
                int(np.asarray(sample["x0"]).item()),
                int(np.asarray(sample["nz"]).item()),
                int(np.asarray(sample["ny"]).item()),
                int(np.asarray(sample["nx"]).item()),
                depth,
                height,
                width,
            )
        return x_t, global_context, global_grid, y_t, m_t

from __future__ import annotations

import bisect
import json
import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


# Load optional target-normalization metadata from disk.
def load_target_normalization(path: str | None) -> dict[str, object] | None:
    """Load optional residual-target normalization metadata."""

    if not path:
        return None
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


# Choose the month or background-bin normalization group.
def target_norm_group_key(
    stats: dict[str, object],
    mode: str,
    month: int,
    bg_values: np.ndarray | torch.Tensor | None = None,
) -> str:
    """Choose the target-normalization group for older residual datasets."""

    if mode == "month":
        return f"{int(month):02d}"
    if mode == "background_bin":
        if bg_values is None:
            raise RuntimeError("background_bin normalization requires background values")
        value = float(bg_values.float().mean().item()) if isinstance(bg_values, torch.Tensor) else float(np.asarray(bg_values).mean())
        bins = stats.get("background_bins")
        if not isinstance(bins, list) or len(bins) < 2:
            raise RuntimeError("background_bin normalization requires background_bins")
        for idx, (lo, hi) in enumerate(zip(bins[:-1], bins[1:])):
            if float(lo) <= value < float(hi) or (idx == len(bins) - 2 and value <= float(hi)):
                return f"bin_{idx:02d}"
        return "global"
    raise RuntimeError(f"Unknown target normalization mode: {mode}")


# Return safe mean and standard deviation values for a normalization group.
def target_norm_mean_std(stats: dict[str, object], key: str) -> tuple[float, float]:
    """Return mean/std for the chosen normalization group with safe defaults."""

    groups = stats.get("groups", {})
    group = groups.get(key) if isinstance(groups, dict) else None
    if not isinstance(group, dict):
        group = stats.get("global", {})
    return float(group.get("mean", 0.0)), max(float(group.get("std", 1.0)), 1.0e-6)

# Reads mmap patch-cache shards as PyTorch samples.
class PatchCache(Dataset):
    # Store constructor arguments and initialize object state.
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


# Adds corrected context channels, targets, and masks to the base patch cache.
class _ContextDataset(Dataset):
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
            raise FileNotFoundError(f"Missing V40 normalization stats: {norm_path}")

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
            raise RuntimeError("V40 local normalization channel count mismatch")
        if self.use_global_context and len(self.global_channels) != len(norm["global_mean"]):
            raise RuntimeError("V40 global normalization channel count mismatch")

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
            raise RuntimeError(f"V40 transformed channel count {x.shape[0]} does not match stats {len(self.local_channels)}")

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


# Extends the context dataset with previous-timestep CO2 inputs.
class _PreviousCO2Dataset(_ContextDataset):
    """Use previous-timestep CO2 as input and train on CO2(t)-CO2(t-1)."""

    # Load split files and prepare dataset state.
    def __init__(self, *args, prev_sidecar_root: str, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if self.target_normalization is not None:
            raise RuntimeError("V40 autoregressive delta training should not use corrected-residual target normalization")

        self.prev_sidecar_root = os.path.abspath(prev_sidecar_root)
        split_dir = os.path.join(self.prev_sidecar_root, self.split)
        manifest_path = os.path.join(split_dir, "manifest.json")
        norm_path = os.path.join(self.prev_sidecar_root, "normalization.json")
        if not os.path.exists(manifest_path):
            raise FileNotFoundError(f"Missing V40 previous-CO2 sidecar manifest: {manifest_path}")
        if not os.path.exists(norm_path):
            raise FileNotFoundError(f"Missing V40 previous-CO2 normalization stats: {norm_path}")
        with open(manifest_path, "r", encoding="utf-8") as f:
            prev_manifest = json.load(f)
        with open(norm_path, "r", encoding="utf-8") as f:
            prev_norm = json.load(f)
        self.prev_split_dir = split_dir
        self.prev_shards = prev_manifest["shards"]
        self.prev_total = int(prev_manifest["total"])
        if self.prev_total < self.total_source:
            raise RuntimeError(f"V40 previous-CO2 sidecar has {self.prev_total} rows but corrected sidecar has {self.total_source}")

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
            raise RuntimeError("V40 previous-CO2 has_prev vector is shorter than the corrected sidecar")
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


# Keeps the frozen reproduction dataset interface compatible with the final model.
class _FeatureDataset(_PreviousCO2Dataset):
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
            raise RuntimeError(f"Missing channel required for V40 advection: {name}")
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


# Public dataset wrapper used by the final V40 scripts.
class FinalV40Dataset(_FeatureDataset):
    """The single public dataset for the final V40 training contract.

    It reads the patch cache, context, previous-CO2 and coordinate sidecars,
    builds autoregressive gradient/advection features, and returns only the
    requested final input channels.
    """

    # Load split files and prepare dataset state.
    def __init__(self, *args, keep_channels: tuple[str, ...], **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.keep_channels = tuple(keep_channels)
        positions = {name: idx for idx, name in enumerate(self.effective_local_channels)}
        missing = [name for name in self.keep_channels if name not in positions]
        if missing:
            raise ValueError(f"Final V40 channels missing from prepared data: {missing}")
        self.keep_indices = [positions[name] for name in self.keep_channels]
        self.effective_local_channels = list(self.keep_channels)

    # Return one indexed sample in the format expected by the model.
    def __getitem__(self, index: int):
        item = super().__getitem__(index)
        if len(item) == 3:
            x, target, mask = item
            return x[self.keep_indices], target, mask
        x, global_context, global_grid, target, mask = item
        return x[self.keep_indices], global_context, global_grid, target, mask

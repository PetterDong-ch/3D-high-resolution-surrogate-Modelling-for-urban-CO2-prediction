from __future__ import annotations

import bisect
from pathlib import Path

import numpy as np
import torch

from .stage2_utils import load_json


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


# Reads low-resolution full-domain context sidecars.
class FullDomainContextSidecarReader:
    """Reads a de-duplicated full-domain context sidecar aligned to Stage2 samples."""

    # Load manifests and initialize shard caches.
    def __init__(self, sidecar_root: str | Path, split: str) -> None:
        self.sidecar_root = Path(sidecar_root)
        self.split = split
        self.root_manifest = load_json(self.sidecar_root / "manifest.json")
        self.manifest = self.root_manifest["splits"][split]
        self.normalization = load_json(self.sidecar_root / "normalization.json")
        self.channels = tuple(self.root_manifest["global_channels"])
        self.context_shape = tuple(int(v) for v in self.root_manifest["global_shape"])
        self.sample_context_index = np.load(self.sidecar_root / self.manifest["sample_context_index"], mmap_mode="r")
        self.shards = self.manifest["shards"]
        self.cumulative: list[int] = []
        total = 0
        for shard in self.shards:
            total += int(shard["count"])
            self.cumulative.append(total)
        self.total_contexts = total
        self._loaded_index: int | None = None
        self._loaded: np.ndarray | None = None

    # Map a source sample index to the matching context row.
    def context_index_for_source(self, source_index: int) -> int:
        if source_index < 0 or source_index >= self.sample_context_index.shape[0]:
            raise IndexError(source_index)
        return int(self.sample_context_index[source_index])

    # Map a global sample index to its shard and row offsets.
    def local_index(self, context_index: int) -> tuple[int, int]:
        if context_index < 0:
            raise IndexError("Context sidecar does not contain this Stage2 sample")
        if context_index >= self.total_contexts:
            raise IndexError(context_index)
        shard_idx = bisect.bisect_right(self.cumulative, context_index)
        prev = 0 if shard_idx == 0 else self.cumulative[shard_idx - 1]
        return shard_idx, context_index - prev

    # Load a shard and cache it for repeated access.
    def load_shard(self, shard_idx: int) -> np.ndarray:
        if self._loaded_index == shard_idx and self._loaded is not None:
            return self._loaded
        path = self.sidecar_root / self.shards[shard_idx]["path"]
        self._loaded_index = shard_idx
        self._loaded = np.load(path, mmap_mode="r")
        return self._loaded

    # Return one sample by global index.
    def sample(self, source_index: int) -> torch.Tensor:
        context_index = self.context_index_for_source(source_index)
        shard_idx, local_idx = self.local_index(context_index)
        arr = self.load_shard(shard_idx)[local_idx]
        return torch.from_numpy(np.asarray(arr, dtype=np.float32))


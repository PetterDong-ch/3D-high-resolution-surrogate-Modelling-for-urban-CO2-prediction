from __future__ import annotations

import bisect
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset


INPUT_FIELDS = ("geometry_3d", "surface_2d", "profile", "scalar")
TARGET_FIELDS = ("target_uv", "target_w", "target_theta_prime")
PHYSICAL_FIELDS = ("theta_reference", "target_theta")


# Load cache manifest from disk or cache.
def load_cache_manifest(cache_root: str | Path) -> dict[str, Any]:
    cache_root = Path(cache_root)
    manifest_path = cache_root / "cache_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing Stage 1 cache manifest: {manifest_path}")
    return json.loads(manifest_path.read_text(encoding="utf-8"))


# Internal helper for stats array.
def _stats_array(stats: dict[str, Any], field: str, kind: str) -> np.ndarray:
    key = f"{field}_{kind}"
    if key not in stats:
        raise KeyError(f"Missing normalization statistic: {key}")
    return np.asarray(stats[key], dtype=np.float32)


# Internal helper for normalize.
def _normalize(arr: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return ((arr.astype(np.float32, copy=False) - mean) / np.maximum(std, 1e-6)).astype(np.float32, copy=False)


# Internal helper for to tensor.
def _to_tensor(arr: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.asarray(arr, dtype=np.float32))


# Loads samples for the Stage1Cache data pipeline.
class Stage1CacheDataset(Dataset):
    """Reads sharded Stage 1 arrays and applies train-set normalization.

    The cache stores physical arrays. Normalization is intentionally applied at
    read time so the same cache can be reused for ablations.
    """

    # Load split files and prepare dataset state.
    def __init__(
        self,
        cache_root: str | Path,
        *,
        split: str,
        normalize_inputs: bool = True,
        normalize_targets: bool = True,
        cache_last_shard: bool = True,
    ) -> None:
        self.cache_root = Path(cache_root)
        self.manifest = load_cache_manifest(self.cache_root)
        if split not in self.manifest["splits"]:
            raise KeyError(f"Split {split!r} is not present in {self.cache_root}")
        self.split = split
        self.normalize_inputs = normalize_inputs
        self.normalize_targets = normalize_targets
        self.cache_last_shard = cache_last_shard
        self.stats = self.manifest["normalization"]

        split_info = self.manifest["splits"][split]
        self.shards = split_info["shards"]
        self.cumulative_counts: list[int] = []
        total = 0
        for shard in self.shards:
            total += int(shard["count"])
            self.cumulative_counts.append(total)
        self.count = total
        self._loaded_shard_index: int | None = None
        self._loaded_shard: dict[str, np.ndarray] | None = None

    # Return the number of available samples.
    def __len__(self) -> int:
        return self.count

    # Internal helper for load shard.
    def _load_shard(self, shard_index: int) -> dict[str, np.ndarray]:
        if self.cache_last_shard and self._loaded_shard_index == shard_index and self._loaded_shard is not None:
            return self._loaded_shard
        shard_path = self.cache_root / self.shards[shard_index]["path"]
        loaded = np.load(shard_path, allow_pickle=False)
        data = {name: loaded[name] for name in loaded.files}
        if self.cache_last_shard:
            self._loaded_shard_index = shard_index
            self._loaded_shard = data
        return data

    # Internal helper for local index.
    def _local_index(self, index: int) -> tuple[int, int]:
        if index < 0:
            index += self.count
        if index < 0 or index >= self.count:
            raise IndexError(index)
        shard_index = bisect.bisect_right(self.cumulative_counts, index)
        previous_total = 0 if shard_index == 0 else self.cumulative_counts[shard_index - 1]
        return shard_index, index - previous_total

    # Return one indexed sample in the format expected by the model.
    def __getitem__(self, index: int) -> dict[str, Any]:
        shard_index, local_index = self._local_index(index)
        shard = self._load_shard(shard_index)

        item: dict[str, Any] = {}
        for field in INPUT_FIELDS:
            arr = shard[field][local_index]
            if self.normalize_inputs:
                arr = _normalize(arr, _stats_array(self.stats, field, "mean"), _stats_array(self.stats, field, "std"))
            item[field] = _to_tensor(arr)

        for field in TARGET_FIELDS:
            arr = shard[field][local_index]
            if self.normalize_targets:
                arr = _normalize(arr, _stats_array(self.stats, field, "mean"), _stats_array(self.stats, field, "std"))
            item[field] = _to_tensor(arr)

        for field in PHYSICAL_FIELDS:
            item[field] = _to_tensor(shard[field][local_index])

        item["mask"] = _to_tensor(shard["mask"][local_index])
        item["sample_key"] = str(shard["sample_key"][local_index])
        item["metadata_json"] = str(shard["metadata_json"][local_index])
        return item

    # Return target-stat tensors for denormalization.
    def target_stat_tensors(self, device: torch.device | str = "cpu") -> dict[str, torch.Tensor]:
        out: dict[str, torch.Tensor] = {}
        for field in TARGET_FIELDS:
            out[f"{field}_mean"] = torch.as_tensor(_stats_array(self.stats, field, "mean"), device=device)
            out[f"{field}_std"] = torch.as_tensor(_stats_array(self.stats, field, "std"), device=device)
        return out


# Convert normalized targets back to physical units.
def denormalize_target(field: str, value: torch.Tensor, stats: dict[str, torch.Tensor]) -> torch.Tensor:
    return value * stats[f"{field}_std"] + stats[f"{field}_mean"]


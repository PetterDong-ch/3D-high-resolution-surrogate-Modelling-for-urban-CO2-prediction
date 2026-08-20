from __future__ import annotations

import os

import numpy as np
import torch
from torch.utils.data import Dataset

from .io import read_json


# Find the incoming-background CO2 feature in the profile-cache manifest.
def background_feature_index(profile_cache_root: str) -> int:
    manifest_path = os.path.join(profile_cache_root, "manifest.json")
    manifest = read_json(manifest_path)
    feature_names = manifest.get("local_feature_names", [])
    try:
        return int(feature_names.index("ls_forcing_right_CO2_mean"))
    except ValueError as exc:
        raise RuntimeError("Missing ls_forcing_right_CO2_mean in average profile cache.") from exc


# Compute per-height target normalization statistics from training data.
def compute_target_norm_stats(profile_cache_root: str, min_std: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    data = np.load(os.path.join(profile_cache_root, "train_profiles.npz"), allow_pickle=False)
    local = data["local"].astype(np.float32, copy=False)
    concentration = data["concentration"].astype(np.float32, copy=False)
    bg_idx = background_feature_index(profile_cache_root)
    target = concentration - local[:, bg_idx, :]
    mask = data["delta_mask"].astype(np.float32, copy=False) > 0.5
    depth = target.shape[1]
    mean = np.zeros(depth, dtype=np.float32)
    std = np.ones(depth, dtype=np.float32)
    valid_fraction = mask.mean(axis=0).astype(np.float32)
    for z in range(depth):
        vals = target[mask[:, z], z]
        if len(vals) > 0:
            mean[z] = float(vals.mean())
            std[z] = max(float(vals.std()), float(min_std))
    return mean, std, valid_fraction


# Build vertical loss weights for low and sparsely valid layers.
def make_layer_weights(
    depth: int,
    valid_fraction: np.ndarray,
    low_alpha: float,
    low_tau: float,
    valid_power: float,
) -> np.ndarray:
    z = np.arange(depth, dtype=np.float32)
    low = 1.0 + float(low_alpha) * np.exp(-z / max(float(low_tau), 1.0e-6))
    valid = np.maximum(valid_fraction.astype(np.float32), 1.0e-6)
    valid_comp = (float(valid.max()) / valid) ** float(valid_power)
    weights = low * valid_comp
    weights = weights / max(float(weights.mean()), 1.0e-6)
    return weights.astype(np.float32)


# Loads normalized background-enhancement profiles for Task 2.
class BackgroundEnhancementV3Dataset(Dataset):
    # Load split files and prepare dataset state.
    def __init__(
        self,
        profile_cache_root: str,
        split: str,
        target_mean: np.ndarray,
        target_std: np.ndarray,
        use_global_context: bool = True,
    ) -> None:
        super().__init__()
        self.profile_cache_root = os.path.abspath(profile_cache_root)
        self.split = str(split)
        self.use_global_context = bool(use_global_context)
        data_path = os.path.join(self.profile_cache_root, f"{self.split}_profiles.npz")
        norm_path = os.path.join(self.profile_cache_root, "normalization.json")
        if not os.path.exists(data_path):
            raise FileNotFoundError(data_path)
        if not os.path.exists(norm_path):
            raise FileNotFoundError(norm_path)
        norm = read_json(norm_path)
        data = np.load(data_path, allow_pickle=False)

        local = data["local"].astype(np.float32, copy=False)
        bg_idx = background_feature_index(self.profile_cache_root)
        self.background_profile = local[:, bg_idx, :].astype(np.float32, copy=False)
        global_context = data["global_context"].astype(np.float32, copy=False)
        local_mean = np.asarray(norm["local_feature_mean"], dtype=np.float32)[None, :, None]
        local_std = np.maximum(np.asarray(norm["local_feature_std"], dtype=np.float32), 1.0e-6)[None, :, None]
        global_mean = np.asarray(norm["global_feature_mean"], dtype=np.float32)[None, :, None]
        global_std = np.maximum(np.asarray(norm["global_feature_std"], dtype=np.float32), 1.0e-6)[None, :, None]

        self.local = ((local - local_mean) / local_std).astype(np.float32)
        if self.use_global_context:
            self.global_context = ((global_context - global_mean) / global_std).astype(np.float32)
        else:
            self.global_context = np.zeros((local.shape[0], 0, local.shape[-1]), dtype=np.float32)

        self.delta_mask = data["delta_mask"].astype(np.float32, copy=False)
        self.concentration = data["concentration"].astype(np.float32, copy=False)
        self.delta = (self.concentration - self.background_profile).astype(np.float32)
        self.meta = data["meta"].astype(np.int64, copy=False)
        self.target_mean = np.asarray(target_mean, dtype=np.float32)
        self.target_std = np.maximum(np.asarray(target_std, dtype=np.float32), 1.0e-6)
        self.target_norm = ((self.delta - self.target_mean[None, :]) / self.target_std[None, :]).astype(np.float32)
        usable = self.delta_mask.sum(axis=1) > 0.0
        self.indices = np.nonzero(usable)[0].astype(np.int64)
        if len(self.indices) == 0:
            raise RuntimeError(f"No usable samples for split={self.split}")

    # Return the number of available samples.
    def __len__(self) -> int:
        return int(len(self.indices))

    # Return one indexed sample in the format expected by the model.
    def __getitem__(self, index: int):
        i = int(self.indices[index])
        return (
            torch.from_numpy(self.local[i]),
            torch.from_numpy(self.global_context[i]),
            torch.from_numpy(self.target_norm[i].astype(np.float32, copy=False)),
            torch.from_numpy(self.delta_mask[i].astype(np.float32, copy=False)),
            torch.from_numpy(self.delta[i].astype(np.float32, copy=False)),
            torch.from_numpy(self.concentration[i].astype(np.float32, copy=False)),
            torch.from_numpy(self.background_profile[i].astype(np.float32, copy=False)),
            torch.from_numpy(self.meta[i]),
        )

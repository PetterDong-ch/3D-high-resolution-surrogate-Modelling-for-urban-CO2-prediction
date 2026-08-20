#!/usr/bin/env python3

"""Build V30 texture-aware sampling weights for autoregressive delta training.

This script does not regenerate patch cache. It scans cached target CO2,
topography/fluid masks, and previous-timestep CO2 sidecars, then assigns higher
sampling weights to patches whose raw delta field has large amplitude and
strong spatial gradients.
"""

from __future__ import annotations

import argparse
import bisect
import csv
import json
import os
from dataclasses import dataclass

import numpy as np


# Reads and indexes Shard data.
@dataclass
class ShardReader:
    split_dir: str
    shards: list[dict]
    keys: tuple[str, ...]

    # Internal helper for post init.
    def __post_init__(self) -> None:
        self.cumulative: list[int] = []
        total = 0
        for shard in self.shards:
            total += int(shard["count"])
            self.cumulative.append(total)
        self._arrays: dict[int, dict[str, np.ndarray]] = {}

    # Return the number of available samples.
    def __len__(self) -> int:
        return self.cumulative[-1] if self.cumulative else 0

    # Internal helper for load shard.
    def _load_shard(self, shard_idx: int) -> dict[str, np.ndarray]:
        arrays = self._arrays.get(shard_idx)
        if arrays is not None:
            return arrays
        shard = self.shards[shard_idx]
        arrays = {
            key: np.load(os.path.join(self.split_dir, shard[key]), mmap_mode="r")
            for key in self.keys
        }
        self._arrays[shard_idx] = arrays
        return arrays

    # Return a stored statistic by name.
    def get(self, index: int, key: str) -> np.ndarray:
        shard_idx = bisect.bisect_right(self.cumulative, int(index))
        prev = 0 if shard_idx == 0 else self.cumulative[shard_idx - 1]
        local_idx = int(index) - prev
        return self._load_shard(shard_idx)[key][local_idx]


# Read a JSON metadata or configuration file.
def read_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# Count overlap between a patch and selected layers.
def overlap_count(z0: int, depth: int, layer_min: int, layer_max: int) -> int:
    z1 = int(z0) + int(depth) - 1
    return max(0, min(z1, int(layer_max)) - max(int(z0), int(layer_min)) + 1)


# Build a mask for selected vertical layers.
def selected_layer_mask(z0: int, depth: int, layer_min: int, layer_max: int) -> np.ndarray:
    global_z = np.arange(depth, dtype=np.int64) + int(z0)
    return ((global_z >= int(layer_min)) & (global_z <= int(layer_max))).astype(np.float32)


# Compute the mean over valid cells.
def valid_mean(values: np.ndarray, mask: np.ndarray) -> float:
    denom = float(mask.sum())
    if denom <= 0.0:
        return 0.0
    return float((values * mask).sum() / denom)


# Compute standard deviation over valid cells.
def masked_std(values: np.ndarray, mask: np.ndarray) -> float:
    denom = float(mask.sum())
    if denom <= 0.0:
        return 0.0
    mean = float((values * mask).sum() / denom)
    var = float((((values - mean) ** 2) * mask).sum() / denom)
    return float(np.sqrt(max(var, 0.0)))


# Compute the mean absolute difference.
def mean_abs_diff(delta: np.ndarray, mask: np.ndarray, axis: int) -> float:
    if delta.shape[axis] <= 1:
        return 0.0
    left = [slice(None)] * delta.ndim
    right = [slice(None)] * delta.ndim
    left[axis] = slice(0, delta.shape[axis] - 1)
    right[axis] = slice(1, delta.shape[axis])
    diff = np.abs(delta[tuple(right)] - delta[tuple(left)])
    pair_mask = mask[tuple(right)] * mask[tuple(left)]
    denom = float(pair_mask.sum())
    if denom <= 0.0:
        return 0.0
    return float((diff * pair_mask).sum() / denom)


# Normalize values with robust scale safeguards.
def robust_normalize(values: np.ndarray, percentile: float, clip: float) -> tuple[np.ndarray, float]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return np.zeros_like(values, dtype=np.float64), 1.0
    scale = float(np.percentile(finite, percentile))
    if scale <= 1.0e-12:
        scale = float(finite.max()) if float(finite.max()) > 1.0e-12 else 1.0
    norm = np.clip(values.astype(np.float64) / scale, 0.0, float(clip))
    return norm, scale


# Generate one train or validation profile-cache split.
def build_split(args: argparse.Namespace, split: str) -> None:
    cache_split = os.path.join(args.cache_root, split)
    prev_split = os.path.join(args.prev_sidecar_root, split)
    meta_path = os.path.join(args.metadata_root, f"{split}_metadata.npz")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"Missing metadata: {meta_path}")

    cache_manifest = read_json(os.path.join(cache_split, "manifest.json"))
    prev_manifest = read_json(os.path.join(prev_split, "manifest.json"))
    cache_reader = ShardReader(cache_split, cache_manifest["shards"], ("y", "mask"))
    prev_reader = ShardReader(prev_split, prev_manifest["shards"], ("prev_co2", "has_prev"))

    meta = np.load(meta_path, allow_pickle=False)
    depth = int(cache_manifest["patch_size"][0])
    total = min(len(cache_reader), len(prev_reader), len(meta["z0"]))
    exclude_months = {int(v) for v in args.exclude_months}

    selected: list[int] = []
    for idx in range(total):
        if int(meta["month"][idx]) in exclude_months:
            continue
        if overlap_count(int(meta["z0"][idx]), depth, args.layer_min, args.layer_max) < args.min_layer_overlap:
            continue
        if int(prev_reader.get(idx, "has_prev")) <= 0:
            continue
        selected.append(idx)
        if args.max_samples > 0 and len(selected) >= args.max_samples:
            break

    n = len(selected)
    source_index = np.asarray(selected, dtype=np.int64)
    delta_abs_mean = np.zeros(n, dtype=np.float32)
    delta_std = np.zeros(n, dtype=np.float32)
    grad_xy_mean = np.zeros(n, dtype=np.float32)
    grad_z_mean = np.zeros(n, dtype=np.float32)
    high_delta_fraction = np.zeros(n, dtype=np.float32)
    valid_fraction = np.zeros(n, dtype=np.float32)
    valid_count = np.zeros(n, dtype=np.int64)

    month = np.asarray([int(meta["month"][idx]) for idx in selected], dtype=np.int16)
    time_index = np.asarray([int(meta["time_index"][idx]) for idx in selected], dtype=np.int16)
    z0 = np.asarray([int(meta["z0"][idx]) for idx in selected], dtype=np.int16)
    y0 = np.asarray([int(meta["y0"][idx]) for idx in selected], dtype=np.int16)
    x0 = np.asarray([int(meta["x0"][idx]) for idx in selected], dtype=np.int16)

    for out_idx, idx in enumerate(selected):
        y = np.asarray(cache_reader.get(idx, "y")[0], dtype=np.float32)
        mask = np.asarray(cache_reader.get(idx, "mask")[0], dtype=np.float32)
        prev = np.asarray(prev_reader.get(idx, "prev_co2"), dtype=np.float32)
        zmask = selected_layer_mask(int(meta["z0"][idx]), depth, args.layer_min, args.layer_max).reshape(depth, 1, 1)
        mask = mask * zmask
        delta = y - prev

        vc = int(mask.sum())
        valid_count[out_idx] = vc
        valid_fraction[out_idx] = float(vc) / float(mask.size)
        if vc <= 0:
            continue
        abs_delta = np.abs(delta)
        delta_abs_mean[out_idx] = valid_mean(abs_delta, mask)
        delta_std[out_idx] = masked_std(delta, mask)
        gx = mean_abs_diff(delta, mask, axis=-1)
        gy = mean_abs_diff(delta, mask, axis=-2)
        gz = mean_abs_diff(delta, mask, axis=-3)
        grad_xy_mean[out_idx] = 0.5 * (gx + gy)
        grad_z_mean[out_idx] = gz
        high_delta_fraction[out_idx] = valid_mean((abs_delta >= float(args.high_delta_threshold)).astype(np.float32), mask)

        if args.log_every > 0 and (out_idx + 1) % args.log_every == 0:
            print(f"{split}: {out_idx + 1}/{n} texture samples", flush=True)

    grad_score_raw = grad_xy_mean + float(args.z_gradient_weight) * grad_z_mean
    norm_abs, scale_abs = robust_normalize(delta_abs_mean, args.norm_percentile, args.score_clip)
    norm_std, scale_std = robust_normalize(delta_std, args.norm_percentile, args.score_clip)
    norm_grad, scale_grad = robust_normalize(grad_score_raw, args.norm_percentile, args.score_clip)
    norm_frac, scale_frac = robust_normalize(high_delta_fraction, args.norm_percentile, args.score_clip)

    texture_score = (
        float(args.abs_weight) * norm_abs
        + float(args.std_weight) * norm_std
        + float(args.gradient_weight) * norm_grad
        + float(args.high_delta_fraction_weight) * norm_frac
    )
    sample_weight = 1.0 + float(args.sampler_alpha) * texture_score
    mean_weight = float(sample_weight.mean()) if sample_weight.size else 1.0
    if mean_weight > 1.0e-12:
        sample_weight = sample_weight / mean_weight
    sample_weight = np.clip(sample_weight, float(args.min_weight), float(args.max_weight)).astype(np.float32)

    os.makedirs(args.out_dir, exist_ok=True)
    out_npz = os.path.join(args.out_dir, f"{split}_texture_stats.npz")
    np.savez_compressed(
        out_npz,
        source_index=source_index,
        sample_weight=sample_weight,
        texture_score=texture_score.astype(np.float32),
        delta_abs_mean=delta_abs_mean,
        delta_std=delta_std,
        grad_xy_mean=grad_xy_mean,
        grad_z_mean=grad_z_mean,
        high_delta_fraction=high_delta_fraction,
        valid_fraction=valid_fraction,
        valid_count=valid_count,
        month=month,
        time_index=time_index,
        z0=z0,
        y0=y0,
        x0=x0,
        scale_abs=np.asarray(scale_abs, dtype=np.float32),
        scale_std=np.asarray(scale_std, dtype=np.float32),
        scale_grad=np.asarray(scale_grad, dtype=np.float32),
        scale_high_delta_fraction=np.asarray(scale_frac, dtype=np.float32),
    )

    preview_path = os.path.join(args.out_dir, f"{split}_texture_preview.csv")
    with open(preview_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "rank",
                "source_index",
                "sample_weight",
                "texture_score",
                "delta_abs_mean",
                "delta_std",
                "grad_xy_mean",
                "grad_z_mean",
                "high_delta_fraction",
                "month",
                "time_index",
                "z0",
                "y0",
                "x0",
            ],
        )
        writer.writeheader()
        order = np.argsort(sample_weight)[::-1][: min(200, n)]
        for rank, i in enumerate(order, start=1):
            writer.writerow(
                {
                    "rank": rank,
                    "source_index": int(source_index[i]),
                    "sample_weight": float(sample_weight[i]),
                    "texture_score": float(texture_score[i]),
                    "delta_abs_mean": float(delta_abs_mean[i]),
                    "delta_std": float(delta_std[i]),
                    "grad_xy_mean": float(grad_xy_mean[i]),
                    "grad_z_mean": float(grad_z_mean[i]),
                    "high_delta_fraction": float(high_delta_fraction[i]),
                    "month": int(month[i]),
                    "time_index": int(time_index[i]),
                    "z0": int(z0[i]),
                    "y0": int(y0[i]),
                    "x0": int(x0[i]),
                }
            )

    print(
        f"Saved {split} texture stats: {out_npz} "
        f"n={n} weight_min={float(sample_weight.min()) if n else 0:.4g} "
        f"weight_max={float(sample_weight.max()) if n else 0:.4g} "
        f"weight_mean={float(sample_weight.mean()) if n else 0:.4g} "
        f"scale_abs={scale_abs:.4g} scale_std={scale_std:.4g} scale_grad={scale_grad:.4g}",
        flush=True,
    )


# Entry point for the command-line workflow.
def main() -> None:
    parser = argparse.ArgumentParser(description="Build V30 high-delta/high-gradient sampler sidecar")
    parser.add_argument("--cache-root", required=True)
    parser.add_argument("--prev-sidecar-root", required=True)
    parser.add_argument("--metadata-root", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--splits", default="train,val")
    parser.add_argument("--layer-min", type=int, default=1)
    parser.add_argument("--layer-max", type=int, default=10)
    parser.add_argument("--min-layer-overlap", type=int, default=8)
    parser.add_argument("--exclude-months", default="11,12")
    parser.add_argument("--high-delta-threshold", type=float, default=3.0)
    parser.add_argument("--z-gradient-weight", type=float, default=0.25)
    parser.add_argument("--norm-percentile", type=float, default=90.0)
    parser.add_argument("--score-clip", type=float, default=3.0)
    parser.add_argument("--abs-weight", type=float, default=0.50)
    parser.add_argument("--std-weight", type=float, default=0.35)
    parser.add_argument("--gradient-weight", type=float, default=1.00)
    parser.add_argument("--high-delta-fraction-weight", type=float, default=0.50)
    parser.add_argument("--sampler-alpha", type=float, default=1.50)
    parser.add_argument("--min-weight", type=float, default=0.25)
    parser.add_argument("--max-weight", type=float, default=8.0)
    parser.add_argument("--max-samples", type=int, default=0, help="Debug limit per split; 0 means all")
    parser.add_argument("--log-every", type=int, default=200)
    args = parser.parse_args()

    args.exclude_months = tuple(
        int(v.strip()) for v in str(args.exclude_months).split(",") if v.strip()
    )
    splits = [v.strip() for v in str(args.splits).split(",") if v.strip()]
    for split in splits:
        build_split(args, split)


if __name__ == "__main__":
    main()

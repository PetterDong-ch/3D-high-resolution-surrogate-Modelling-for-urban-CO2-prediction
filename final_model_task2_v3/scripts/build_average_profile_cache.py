#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Iterable

import numpy as np


LOCAL_CHANNELS = [
    "emission_values",
    "ls_forcing_right_CO2",
    "u",
    "v",
    "p",
    "theta",
    "w",
    "albedo_type",
    "water_type",
    "pavement_type",
    "street_type",
    "vegetation_type",
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
]

DEFAULT_KEEP_LOCAL = [
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
    "fluid_mask",
]

PREVIOUS_TIMESTEP_CHANNEL_HINTS = (
    "prev",
    "previous",
    "adv_x_delta",
    "adv_y_delta",
    "adv_z_delta",
    "adv_total_delta",
)


# Parse a comma-separated command-line list.
def parse_csv_list(text: str | None) -> list[str]:
    if text is None or not str(text).strip():
        return []
    return [part.strip() for part in str(text).split(",") if part.strip()]


# Read a JSON metadata or configuration file.
def read_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# Write a JSON metadata or configuration file.
def write_json(path: str, data: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


# Convert masked 3D fields into vertical mean/std profiles.
def masked_mean_std(values: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(values, dtype=np.float32)
    valid = (mask > 0.5) & np.isfinite(values)
    counts = valid.sum(axis=(2, 3)).astype(np.float32)
    sums = np.where(valid, values, 0.0).sum(axis=(2, 3), dtype=np.float64).astype(np.float32)
    sq_sums = np.where(valid, values * values, 0.0).sum(axis=(2, 3), dtype=np.float64).astype(np.float32)
    mean = np.zeros_like(counts, dtype=np.float32)
    std = np.zeros_like(counts, dtype=np.float32)
    good = counts > 0.0
    mean[good] = sums[good] / counts[good]
    var = np.zeros_like(counts, dtype=np.float32)
    var[good] = np.maximum(sq_sums[good] / counts[good] - mean[good] * mean[good], 0.0)
    std[good] = np.sqrt(var[good])
    return mean, std


# Average valid CO2 cells into a vertical profile target.
def masked_profile_mean(values: np.ndarray, mask: np.ndarray, min_valid_cells: int) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(values, dtype=np.float32)
    valid = (mask > 0.5) & np.isfinite(values)
    counts = valid.sum(axis=(2, 3)).astype(np.float32)
    sums = np.where(valid, values, 0.0).sum(axis=(2, 3), dtype=np.float64).astype(np.float32)
    out = np.zeros_like(counts, dtype=np.float32)
    good = counts >= float(min_valid_cells)
    out[good] = sums[good] / np.maximum(counts[good], 1.0)
    return out, good.astype(np.float32)


# Build normalized coordinate values for one patch axis.
def norm_axis(start: int, total: int, length: int) -> np.ndarray:
    denom = max(int(total) - 1, 1)
    return ((np.arange(length, dtype=np.float32) + float(start)) / float(denom)) * 2.0 - 1.0


# Load patch metadata used to align profile rows with source samples.
def load_metadata(context_manifest: dict, split: str, total: int) -> dict[str, np.ndarray]:
    meta_root = context_manifest.get("metadata_root")
    if not meta_root:
        raise RuntimeError("Context sidecar manifest has no metadata_root")
    meta_path = os.path.join(meta_root, f"{split}_metadata.npz")
    meta = np.load(meta_path, allow_pickle=False)
    out = {
        "z0": meta["z0"].astype(np.int64, copy=False),
        "y0": meta["y0"].astype(np.int64, copy=False),
        "x0": meta["x0"].astype(np.int64, copy=False),
        "nz": meta["nz"].astype(np.int64, copy=False),
        "ny": meta["ny"].astype(np.int64, copy=False),
        "nx": meta["nx"].astype(np.int64, copy=False),
        "month": meta["month"].astype(np.int64, copy=False) if "month" in meta.files else np.ones(total, dtype=np.int64),
        "time_index": (
            meta["time_index"].astype(np.int64, copy=False)
            if "time_index" in meta.files
            else np.full(total, -1, dtype=np.int64)
        ),
    }
    return out


# Reconstruct coordinate-feature profile summaries from metadata.
def full_axis_mean_std(meta: dict[str, np.ndarray], indices: np.ndarray, axis_name: str, depth: int, height: int, width: int) -> tuple[np.ndarray, np.ndarray]:
    count = len(indices)
    mean = np.zeros((count, depth), dtype=np.float32)
    std = np.zeros((count, depth), dtype=np.float32)
    for row, source_index in enumerate(indices):
        if axis_name == "x_norm":
            vals = norm_axis(int(meta["x0"][source_index]), int(meta["nx"][source_index]), width)
            mean[row, :] = float(vals.mean())
            std[row, :] = float(vals.std())
        elif axis_name == "y_norm":
            vals = norm_axis(int(meta["y0"][source_index]), int(meta["ny"][source_index]), height)
            mean[row, :] = float(vals.mean())
            std[row, :] = float(vals.std())
        elif axis_name == "z_norm":
            vals = norm_axis(int(meta["z0"][source_index]), int(meta["nz"][source_index]), depth)
            mean[row, :] = vals
            std[row, :] = 0.0
        else:
            raise ValueError(axis_name)
    return mean, std


# Generate one train or validation profile-cache split.
def build_split(
    split: str,
    cache_root: str,
    context_sidecar_root: str,
    prev_sidecar_root: str,
    out_dir: str,
    keep_local_channels: Iterable[str],
    min_valid_cells: int,
    max_shards: int | None,
) -> dict:
    cache_dir = os.path.join(cache_root, split)
    context_dir = os.path.join(context_sidecar_root, split)
    prev_dir = os.path.join(prev_sidecar_root, split) if prev_sidecar_root else None

    cache_manifest = read_json(os.path.join(cache_dir, "manifest.json"))
    context_manifest = read_json(os.path.join(context_dir, "manifest.json"))
    norm = read_json(os.path.join(context_sidecar_root, "normalization.json"))
    prev_manifest = read_json(os.path.join(prev_dir, "manifest.json")) if prev_dir else None

    cache_shards = cache_manifest["shards"]
    context_shards = context_manifest["shards"]
    prev_shards = prev_manifest["shards"] if prev_manifest else None
    if max_shards is not None:
        cache_shards = cache_shards[: int(max_shards)]
        context_shards = context_shards[: int(max_shards)]
        prev_shards = prev_shards[: int(max_shards)] if prev_shards else None

    keep = list(keep_local_channels)
    local_channels = list(norm["local_channels"])
    global_channels = list(norm["global_channels"])
    if local_channels != LOCAL_CHANNELS:
        raise RuntimeError(f"Unexpected local channel order: {local_channels}")
    bad = [name for name in keep if any(hint in name for hint in PREVIOUS_TIMESTEP_CHANNEL_HINTS)]
    if bad:
        raise RuntimeError(f"Previous-timestep channels are forbidden in profile-model inputs: {bad}")
    missing = [name for name in keep if name not in local_channels]
    if missing:
        raise RuntimeError(f"Requested local channels not present: {missing}")

    total = int(sum(int(shard["count"]) for shard in cache_shards))
    meta = load_metadata(context_manifest, split, int(context_manifest["total"]))
    patch_d, patch_h, patch_w = (int(v) for v in cache_manifest["patch_size"])

    local_chunks: list[np.ndarray] = []
    global_chunks: list[np.ndarray] = []
    conc_chunks: list[np.ndarray] = []
    prev_chunks: list[np.ndarray] = []
    delta_chunks: list[np.ndarray] = []
    conc_mask_chunks: list[np.ndarray] = []
    delta_mask_chunks: list[np.ndarray] = []
    meta_chunks: list[np.ndarray] = []

    local_feature_names = []
    for name in keep:
        local_feature_names.extend([f"{name}_mean", f"{name}_std"])
    global_feature_names = []
    for name in global_channels:
        global_feature_names.extend([f"global_{name}_mean", f"global_{name}_std"])

    source_offset = 0
    for shard_i, cache_shard in enumerate(cache_shards):
        context_shard = context_shards[shard_i]
        prev_shard = prev_shards[shard_i] if prev_shards else None
        count = int(cache_shard["count"])
        source_indices = np.arange(source_offset, source_offset + count, dtype=np.int64)
        source_offset += count

        x_arr = np.load(os.path.join(cache_dir, cache_shard["x"]), mmap_mode="r")
        y_arr = np.load(os.path.join(cache_dir, cache_shard["y"]), mmap_mode="r")
        mask_arr = np.load(os.path.join(cache_dir, cache_shard["mask"]), mmap_mode="r")
        emission_arr = np.load(os.path.join(context_dir, context_shard["corrected_emission"]), mmap_mode="r")
        bg_arr = np.load(os.path.join(context_dir, context_shard["corrected_bg"]), mmap_mode="r")
        global_arr = np.load(os.path.join(context_dir, context_shard["global_context"]), mmap_mode="r")

        mask = np.asarray(mask_arr[:count, 0], dtype=np.float32)
        y = np.asarray(y_arr[:count, 0], dtype=np.float32)
        conc_profile, conc_mask = masked_profile_mean(y, mask, min_valid_cells)

        local_features = np.zeros((count, len(local_feature_names), patch_d), dtype=np.float32)
        feature_pos = 0
        for channel_name in keep:
            if channel_name == "emission_values":
                values = np.broadcast_to(
                    np.asarray(emission_arr[:count], dtype=np.float32)[:, None, :, :],
                    (count, patch_d, patch_h, patch_w),
                )
                mean, std = masked_mean_std(values, mask)
            elif channel_name == "ls_forcing_right_CO2":
                mean = np.asarray(bg_arr[:count], dtype=np.float32)
                std = np.zeros_like(mean, dtype=np.float32)
            elif channel_name in {"x_norm", "y_norm", "z_norm"}:
                mean, std = full_axis_mean_std(meta, source_indices, channel_name, patch_d, patch_h, patch_w)
            elif channel_name == "fluid_mask":
                mean = mask.mean(axis=(2, 3), dtype=np.float64).astype(np.float32)
                std = mask.std(axis=(2, 3), dtype=np.float64).astype(np.float32)
            else:
                channel_index = local_channels.index(channel_name)
                if channel_index >= x_arr.shape[1]:
                    raise RuntimeError(f"{channel_name} is not present in source x array")
                values = np.asarray(x_arr[:count, channel_index], dtype=np.float32)
                mean, std = masked_mean_std(values, mask)
            local_features[:, feature_pos, :] = mean
            local_features[:, feature_pos + 1, :] = std
            feature_pos += 2

        global_context = np.asarray(global_arr[:count], dtype=np.float32)
        global_mean = global_context.mean(axis=(3, 4), dtype=np.float64).astype(np.float32)
        global_std = global_context.std(axis=(3, 4), dtype=np.float64).astype(np.float32)
        global_features = np.empty((count, len(global_feature_names), patch_d), dtype=np.float32)
        pos = 0
        for gi in range(len(global_channels)):
            global_features[:, pos, :] = global_mean[:, gi, :]
            global_features[:, pos + 1, :] = global_std[:, gi, :]
            pos += 2

        prev_profile = np.zeros_like(conc_profile, dtype=np.float32)
        delta_profile = np.zeros_like(conc_profile, dtype=np.float32)
        delta_mask = np.zeros_like(conc_mask, dtype=np.float32)
        if prev_dir and prev_shard:
            prev_co2 = np.load(os.path.join(prev_dir, prev_shard["prev_co2"]), mmap_mode="r")
            has_prev = np.asarray(np.load(os.path.join(prev_dir, prev_shard["has_prev"]), mmap_mode="r")[:count], dtype=np.uint8)
            prev_values = np.asarray(prev_co2[:count], dtype=np.float32)
            prev_profile, prev_mask = masked_profile_mean(prev_values, mask, min_valid_cells)
            delta_profile = conc_profile - prev_profile
            delta_mask = conc_mask * prev_mask * (has_prev[:, None] > 0).astype(np.float32)

        meta_rows = np.stack(
            [
                source_indices,
                meta["month"][source_indices],
                meta["time_index"][source_indices],
                meta["z0"][source_indices],
                meta["y0"][source_indices],
                meta["x0"][source_indices],
            ],
            axis=1,
        ).astype(np.int64)

        local_chunks.append(local_features)
        global_chunks.append(global_features)
        conc_chunks.append(conc_profile.astype(np.float32))
        prev_chunks.append(prev_profile.astype(np.float32))
        delta_chunks.append(delta_profile.astype(np.float32))
        conc_mask_chunks.append(conc_mask.astype(np.float32))
        delta_mask_chunks.append(delta_mask.astype(np.float32))
        meta_chunks.append(meta_rows)

        if (shard_i + 1) % 50 == 0 or (shard_i + 1) == len(cache_shards):
            print(f"{split}: processed {shard_i + 1}/{len(cache_shards)} shards ({source_offset} samples)", flush=True)

    local_all = np.concatenate(local_chunks, axis=0)
    global_all = np.concatenate(global_chunks, axis=0)
    conc_all = np.concatenate(conc_chunks, axis=0)
    prev_all = np.concatenate(prev_chunks, axis=0)
    delta_all = np.concatenate(delta_chunks, axis=0)
    conc_mask_all = np.concatenate(conc_mask_chunks, axis=0)
    delta_mask_all = np.concatenate(delta_mask_chunks, axis=0)
    meta_all = np.concatenate(meta_chunks, axis=0)

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{split}_profiles.npz")
    np.savez(
        out_path,
        local=local_all.astype(np.float32),
        global_context=global_all.astype(np.float32),
        concentration=conc_all.astype(np.float32),
        previous_concentration=prev_all.astype(np.float32),
        delta=delta_all.astype(np.float32),
        concentration_mask=conc_mask_all.astype(np.float32),
        delta_mask=delta_mask_all.astype(np.float32),
        meta=meta_all.astype(np.int64),
        local_feature_names=np.asarray(local_feature_names),
        global_feature_names=np.asarray(global_feature_names),
    )
    print(f"Saved {split} profile cache: {out_path}", flush=True)
    return {
        "split": split,
        "samples": int(local_all.shape[0]),
        "local_shape": list(local_all.shape),
        "global_shape": list(global_all.shape),
        "path": out_path,
    }


# Compute feature normalization statistics from profile arrays.
def feature_norm(arr: np.ndarray) -> tuple[list[float], list[float]]:
    mean = arr.mean(axis=(0, 2), dtype=np.float64).astype(np.float32)
    std = arr.std(axis=(0, 2), dtype=np.float64).astype(np.float32)
    std = np.maximum(std, 1.0e-6)
    return mean.tolist(), std.tolist()


# Entry point for the command-line workflow.
def main() -> None:
    parser = argparse.ArgumentParser(description="Build the compact average-profile cache.")
    project_root = Path(__file__).resolve().parents[1]
    parser.add_argument("--cache-root", default=str(project_root / "external_data" / "spatial_patch_cache"))
    parser.add_argument("--context-sidecar-root", default=str(project_root / "external_data" / "context_sidecar"))
    parser.add_argument("--prev-sidecar-root", default=str(project_root / "external_data" / "previous_co2_sidecar"))
    parser.add_argument("--out-dir", default=str(project_root / "generated" / "average_profile_cache"))
    parser.add_argument("--keep-local-channels", default=",".join(DEFAULT_KEEP_LOCAL))
    parser.add_argument("--min-valid-cells", type=int, default=128)
    parser.add_argument("--max-shards", type=int, default=0, help="Debug only. 0 means all shards.")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    out_dir = os.path.abspath(args.out_dir)
    if os.path.exists(out_dir) and args.overwrite:
        shutil.rmtree(out_dir)
    os.makedirs(out_dir, exist_ok=True)
    keep = parse_csv_list(args.keep_local_channels) or list(DEFAULT_KEEP_LOCAL)
    max_shards = int(args.max_shards) if int(args.max_shards) > 0 else None

    summaries = []
    for split in ("train", "val"):
        summaries.append(
            build_split(
                split=split,
                cache_root=args.cache_root,
                context_sidecar_root=args.context_sidecar_root,
                prev_sidecar_root=args.prev_sidecar_root,
                out_dir=out_dir,
                keep_local_channels=keep,
                min_valid_cells=args.min_valid_cells,
                max_shards=max_shards,
            )
        )

    train_data = np.load(os.path.join(out_dir, "train_profiles.npz"), allow_pickle=False)
    local_mean, local_std = feature_norm(train_data["local"])
    global_mean, global_std = feature_norm(train_data["global_context"])
    manifest = {
        "cache_type": "task2_average_profile_cache",
        "source_cache_root": os.path.abspath(args.cache_root),
        "source_context_sidecar_root": os.path.abspath(args.context_sidecar_root),
        "source_prev_sidecar_root": os.path.abspath(args.prev_sidecar_root),
        "keep_local_channels": keep,
        "local_feature_names": [str(v) for v in train_data["local_feature_names"].tolist()],
        "global_feature_names": [str(v) for v in train_data["global_feature_names"].tolist()],
        "splits": summaries,
        "min_valid_cells": int(args.min_valid_cells),
        "note": "Previous timestep CO2 is not an input feature. It is stored only for mean_delta target reconstruction.",
    }
    normalization = {
        "local_feature_mean": local_mean,
        "local_feature_std": local_std,
        "global_feature_mean": global_mean,
        "global_feature_std": global_std,
    }
    write_json(os.path.join(out_dir, "manifest.json"), manifest)
    write_json(os.path.join(out_dir, "normalization.json"), normalization)
    print(f"Saved manifest: {os.path.join(out_dir, 'manifest.json')}", flush=True)


if __name__ == "__main__":
    main()

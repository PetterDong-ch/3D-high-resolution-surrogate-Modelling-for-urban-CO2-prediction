from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Iterable

import numpy as np


# Stores the slice coordinates for one extracted patch.
@dataclass(frozen=True)
class PatchSlice:
    z0: int
    y0: int
    x0: int
    dz: int
    dy: int
    dx: int

    # Return slice objects for this patch.
    @property
    def slices(self) -> tuple[slice, slice, slice]:
        return (
            slice(self.z0, self.z0 + self.dz),
            slice(self.y0, self.y0 + self.dy),
            slice(self.x0, self.x0 + self.dx),
        )


# Compute starts for the workflow.
def compute_starts(length: int, patch: int, stride: int) -> list[int]:
    """Compute patch starts and include the final edge-aligned patch."""

    if length <= 0 or patch <= 0 or stride <= 0:
        raise ValueError("length, patch and stride must be positive")
    if patch > length:
        return [0]

    starts = list(range(0, length - patch + 1, stride))
    final = length - patch
    if starts[-1] != final:
        starts.append(final)
    return starts


# Iterate over patch slices for a full domain.
def iter_patch_slices(
    shape: tuple[int, int, int],
    patch_shape: tuple[int, int, int],
    strides: tuple[int, int, int],
) -> list[PatchSlice]:
    """Return deterministic 3D patch slices for a volume."""

    z_starts = compute_starts(shape[0], patch_shape[0], strides[0])
    y_starts = compute_starts(shape[1], patch_shape[1], strides[1])
    x_starts = compute_starts(shape[2], patch_shape[2], strides[2])
    return [
        PatchSlice(z, y, x, patch_shape[0], patch_shape[1], patch_shape[2])
        for z, y, x in product(z_starts, y_starts, x_starts)
    ]


# Extract patches from a full-domain array.
def extract_patches(volume: np.ndarray, patch_slices: Iterable[PatchSlice]) -> list[np.ndarray]:
    if volume.ndim != 3:
        raise ValueError("volume must have shape [Z,H,W]")
    return [volume[patch.slices].copy() for patch in patch_slices]


# Reconstruct a full-domain array from patch predictions.
def reconstruct_from_patches(
    patches: Iterable[np.ndarray],
    patch_slices: Iterable[PatchSlice],
    output_shape: tuple[int, int, int],
) -> np.ndarray:
    """Reconstruct a volume by normalized overlap averaging."""

    out = np.zeros(output_shape, dtype=np.float64)
    weight = np.zeros(output_shape, dtype=np.float64)
    for patch, patch_slice in zip(patches, patch_slices, strict=True):
        target = patch_slice.slices
        expected = tuple(s.stop - s.start for s in target)
        if patch.shape != expected:
            raise ValueError(f"patch shape {patch.shape} does not match slice shape {expected}")
        out[target] += patch.astype(np.float64, copy=False)
        weight[target] += 1.0

    if np.any(weight == 0.0):
        raise ValueError("patches do not cover the requested output shape")
    return (out / weight).astype(np.float32)



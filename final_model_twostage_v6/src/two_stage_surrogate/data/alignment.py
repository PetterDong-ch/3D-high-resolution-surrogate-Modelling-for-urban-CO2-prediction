from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable

import numpy as np


# Stores the result of matching a requested time to available data.
@dataclass(frozen=True)
class LinearTimeMatch:
    """Linear interpolation metadata for matching one time axis to another."""

    target_seconds: float
    left_index: int
    right_index: int
    left_seconds: float
    right_seconds: float
    weight_right: float
    nearest_index: int
    nearest_seconds: float

    # Convert this record to a dictionary.
    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


# Convert time values to seconds.
def values_to_seconds(values: Iterable[float], units: str | None) -> np.ndarray:
    """Convert a simple NetCDF time coordinate to seconds.

    The PALM files inspected so far use either seconds or hour. This helper is
    deliberately conservative: unknown units are treated as seconds and should
    be flagged in the manifest audit rather than silently reinterpreted.
    """

    arr = np.asarray(list(values), dtype=np.float64)
    unit_text = (units or "").lower()
    if "hour" in unit_text:
        return arr * 3600.0
    if "minute" in unit_text:
        return arr * 60.0
    return arr


# Find the nearest available time index.
def nearest_time_index(target_seconds: float, source_seconds: np.ndarray) -> int:
    if source_seconds.ndim != 1 or source_seconds.size == 0:
        raise ValueError("source_seconds must be a non-empty 1D array")
    return int(np.abs(source_seconds - float(target_seconds)).argmin())


# Interpolate or match values across time.
def linear_time_match(target_seconds: float, source_seconds: np.ndarray) -> LinearTimeMatch:
    """Return indices and interpolation weight for target_seconds on source_seconds."""

    if source_seconds.ndim != 1 or source_seconds.size == 0:
        raise ValueError("source_seconds must be a non-empty 1D array")
    if np.any(np.diff(source_seconds) < 0):
        raise ValueError("source_seconds must be sorted ascending")

    target = float(target_seconds)
    nearest = nearest_time_index(target, source_seconds)
    right = int(np.searchsorted(source_seconds, target, side="left"))
    if right <= 0:
        left = right = 0
    elif right >= source_seconds.size:
        left = right = int(source_seconds.size - 1)
    else:
        left = right - 1

    left_seconds = float(source_seconds[left])
    right_seconds = float(source_seconds[right])
    denom = right_seconds - left_seconds
    weight_right = 0.0 if denom == 0.0 else (target - left_seconds) / denom
    weight_right = float(np.clip(weight_right, 0.0, 1.0))

    return LinearTimeMatch(
        target_seconds=target,
        left_index=int(left),
        right_index=int(right),
        left_seconds=left_seconds,
        right_seconds=right_seconds,
        weight_right=weight_right,
        nearest_index=nearest,
        nearest_seconds=float(source_seconds[nearest]),
    )


# Return the maximum allowed time gap in seconds.
def max_time_gap_seconds(target_seconds: float, source_seconds: np.ndarray) -> float:
    """Distance in seconds from a target time to its nearest source time."""

    idx = nearest_time_index(target_seconds, source_seconds)
    return abs(float(source_seconds[idx]) - float(target_seconds))



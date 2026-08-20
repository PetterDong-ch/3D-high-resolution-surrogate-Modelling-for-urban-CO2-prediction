from __future__ import annotations

import math

import numpy as np


# Compute Pearson correlation for finite prediction pairs.
def pearson_r(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    good = np.isfinite(y_true) & np.isfinite(y_pred)
    if int(good.sum()) < 2:
        return float("nan")
    yt = y_true[good] - float(y_true[good].mean())
    yp = y_pred[good] - float(y_pred[good].mean())
    denom = math.sqrt(float(np.dot(yt, yt)) * float(np.dot(yp, yp)))
    if denom <= 1.0e-12:
        return float("nan")
    return float(np.dot(yt, yp) / denom)


# Compute count, R, R2, MAE, and RMSE.
def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    good = np.isfinite(y_true) & np.isfinite(y_pred)
    if int(good.sum()) == 0:
        return {"count": 0, "R": float("nan"), "R2": float("nan"), "MAE": float("nan"), "RMSE": float("nan")}
    yt = y_true[good]
    yp = y_pred[good]
    err = yp - yt
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err * err)))
    ss_res = float(np.sum(err * err))
    ss_tot = float(np.sum((yt - float(yt.mean())) ** 2))
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 1.0e-12 else float("nan")
    return {"count": int(good.sum()), "R": pearson_r(yt, yp), "R2": r2, "MAE": mae, "RMSE": rmse}

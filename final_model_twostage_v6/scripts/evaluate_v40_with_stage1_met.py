#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402
from torch.utils.data import DataLoader, Subset  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
V40_ROOT = PROJECT_ROOT / "runtime"
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(V40_ROOT))

from models.unet3d import V38EventTextureContextV7UNet3D  # noqa: E402
from twostage_v6.stage2_datasets import (  # noqa: E402
    Stage2V40GlobalContextDataset,
    Stage2V40LocalDataset,
)
from twostage_v6.stage2_constants import (  # noqa: E402
    V40_STAGE1_GLOBAL_CONTEXT_CHANNELS,
    V40_STAGE1_MET_CHANNELS,
)


# Compute metrics on finite prediction pairs.
def finite_metrics(pred: np.ndarray, truth: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    valid = (mask > 0) & np.isfinite(pred) & np.isfinite(truth)
    if not np.any(valid):
        return {"valid_count": 0, "R": float("nan"), "R2": float("nan"), "MAE": float("nan"), "RMSE": float("nan")}
    p = pred[valid].astype(np.float64, copy=False)
    t = truth[valid].astype(np.float64, copy=False)
    err = p - t
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err * err)))
    ss_res = float(np.sum(err * err))
    ss_tot = float(np.sum((t - float(np.mean(t))) ** 2))
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")
    p_std = float(np.std(p))
    t_std = float(np.std(t))
    r = float(np.corrcoef(p, t)[0, 1]) if p_std > 0 and t_std > 0 else float("nan")
    return {"valid_count": int(valid.sum()), "R": r, "R2": r2, "MAE": mae, "RMSE": rmse}


# Accumulates streaming regression metrics without storing every prediction.
class StreamingMetrics:
    # Store constructor arguments and initialize object state.
    def __init__(self) -> None:
        self.n = 0
        self.sum_p = 0.0
        self.sum_t = 0.0
        self.sum_p2 = 0.0
        self.sum_t2 = 0.0
        self.sum_pt = 0.0
        self.sum_abs = 0.0
        self.sum_sqerr = 0.0

    # Update running metric or statistic accumulators.
    def update(self, pred: np.ndarray, truth: np.ndarray, mask: np.ndarray) -> None:
        valid = (mask > 0) & np.isfinite(pred) & np.isfinite(truth)
        if not np.any(valid):
            return
        p = pred[valid].astype(np.float64, copy=False)
        t = truth[valid].astype(np.float64, copy=False)
        err = p - t
        self.n += int(p.size)
        self.sum_p += float(p.sum())
        self.sum_t += float(t.sum())
        self.sum_p2 += float((p * p).sum())
        self.sum_t2 += float((t * t).sum())
        self.sum_pt += float((p * t).sum())
        self.sum_abs += float(np.abs(err).sum())
        self.sum_sqerr += float((err * err).sum())

    # Return the final accumulated metrics.
    def finish(self) -> dict[str, float]:
        if self.n <= 0:
            return {"valid_count": 0, "R": float("nan"), "R2": float("nan"), "MAE": float("nan"), "RMSE": float("nan")}
        n = float(self.n)
        mae = self.sum_abs / n
        rmse = math.sqrt(self.sum_sqerr / n)
        cov = self.sum_pt - self.sum_p * self.sum_t / n
        var_p = self.sum_p2 - self.sum_p * self.sum_p / n
        var_t = self.sum_t2 - self.sum_t * self.sum_t / n
        r = cov / math.sqrt(var_p * var_t) if var_p > 0 and var_t > 0 else float("nan")
        r2 = 1.0 - self.sum_sqerr / var_t if var_t > 0 else float("nan")
        return {"valid_count": self.n, "R": r, "R2": r2, "MAE": mae, "RMSE": rmse}


# Load a model from a saved checkpoint.
def model_from_checkpoint(checkpoint: dict[str, Any], device: torch.device) -> torch.nn.Module:
    cfg = checkpoint.get("model_config", {})
    model = V38EventTextureContextV7UNet3D(
        in_channels=int(cfg.get("in_channels", len(V40_STAGE1_MET_CHANNELS))),
        out_channels=1,
        base_channels=int(cfg.get("base_channels", 32)),
        global_channels=int(cfg.get("global_channels", len(V40_STAGE1_GLOBAL_CONTEXT_CHANNELS))),
        global_feature_channels=int(cfg.get("global_feature_channels", 8)),
        context_correction_scale=float(checkpoint.get("high_residual_scale", 1.0)),
        high_delta_scale=float(checkpoint.get("high_residual_scale", 1.0)),
        min_high_gate=float(checkpoint.get("min_high_gate", 0.20)),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


# Recover previous CO2 from normalized input channels.
def denormalized_prev_from_x(dataset: Stage2V40LocalDataset, x: torch.Tensor) -> torch.Tensor:
    idx = dataset.channels.index("prev_kc_CO2")
    mean = float(dataset.x_mean[idx])
    std = float(dataset.x_std[idx])
    return x[:, idx : idx + 1] * std + mean


# Convert tensors or arrays to plain NumPy arrays.
def to_plain_array(arr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    masked = np.ma.asarray(arr)
    data = np.asarray(masked.filled(np.nan), dtype=np.float32)
    valid = np.isfinite(data) & ~np.ma.getmaskarray(masked)
    return data, valid


# Create an RGB gradient for visualization.
def gradient_rgb(values: np.ndarray, stops: list[tuple[float, tuple[int, int, int]]]) -> np.ndarray:
    values = np.clip(values.astype(np.float32), 0.0, 1.0)
    rgb = np.zeros(values.shape + (3,), dtype=np.float32)
    for (x0, c0), (x1, c1) in zip(stops[:-1], stops[1:]):
        zone = (values >= x0) & (values <= x1)
        if not np.any(zone):
            continue
        weight = (values[zone] - x0) / max(x1 - x0, 1.0e-6)
        c0a = np.asarray(c0, dtype=np.float32)
        c1a = np.asarray(c1, dtype=np.float32)
        rgb[zone] = c0a * (1.0 - weight[:, None]) + c1a * weight[:, None]
    rgb[values <= stops[0][0]] = stops[0][1]
    rgb[values >= stops[-1][0]] = stops[-1][1]
    return np.clip(rgb, 0, 255).astype(np.uint8)


# Build color stops for the custom colormap.
def color_stops(scheme: str) -> list[tuple[float, tuple[int, int, int]]]:
    if scheme == "coolwarm":
        return [(0.0, (49, 76, 180)), (0.5, (247, 247, 247)), (1.0, (180, 4, 38))]
    if scheme == "magma":
        return [(0.0, (0, 0, 4)), (0.35, (85, 18, 105)), (0.70, (220, 73, 80)), (1.0, (252, 253, 191))]
    # A compact turbo-like ramp. Close enough for comparison figures without
    # importing matplotlib, which is broken in this mixed NumPy environment.
    return [
        (0.0, (48, 18, 59)),
        (0.15, (40, 87, 183)),
        (0.35, (31, 164, 221)),
        (0.50, (50, 220, 120)),
        (0.65, (190, 225, 55)),
        (0.82, (245, 125, 32)),
        (1.0, (165, 0, 38)),
    ]


# Colorize a 2D field for preview images.
def colorize(arr: np.ndarray, scheme: str, vmin: float | None, vmax: float | None) -> Image.Image:
    data, valid = to_plain_array(arr)
    if scheme == "topography":
        rgb = np.full(data.shape + (3,), 255, dtype=np.uint8)
        rounded = np.rint(np.nan_to_num(data, nan=-1)).astype(np.int32)
        rgb[rounded == 0] = (248, 248, 248)
        rgb[rounded == 1] = (154, 103, 55)
        rgb[rounded == 2] = (10, 20, 35)
        rgb[rounded == 3] = (125, 55, 220)
        rgb[~valid] = (255, 255, 255)
        return Image.fromarray(np.flipud(rgb), mode="RGB")

    finite = data[valid]
    if finite.size == 0:
        scaled = np.zeros(data.shape, dtype=np.float32)
    else:
        lo = float(np.nanmin(finite)) if vmin is None else float(vmin)
        hi = float(np.nanmax(finite)) if vmax is None else float(vmax)
        if hi <= lo:
            hi = lo + 1.0
        scaled = (data - lo) / (hi - lo)
    rgb = gradient_rgb(np.nan_to_num(scaled, nan=0.0), color_stops(scheme))
    rgb[~valid] = (255, 255, 255)
    return Image.fromarray(np.flipud(rgb), mode="RGB")


# Load the font used for generated figures.
def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ]
    for path in candidates:
        try:
            return ImageFont.truetype(path, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


# Draw centered text on an image.
def draw_centered(draw: ImageDraw.ImageDraw, xy: tuple[int, int, int, int], text: str, fnt: ImageFont.ImageFont, fill: tuple[int, int, int]) -> None:
    left, top, right, bottom = xy
    bbox = draw.textbbox((0, 0), text, font=fnt)
    x = left + max((right - left - (bbox[2] - bbox[0])) // 2, 0)
    y = top + max((bottom - top - (bbox[3] - bbox[1])) // 2, 0)
    draw.text((x, y), text, font=fnt, fill=fill)


# Draw a continuous colorbar for a figure.
def draw_continuous_colorbar(
    canvas: Image.Image,
    draw: ImageDraw.ImageDraw,
    *,
    x: int,
    y: int,
    h: int,
    scheme: str,
    vmin: float | None,
    vmax: float | None,
    tick_font: ImageFont.ImageFont,
) -> None:
    if vmin is None or vmax is None or not np.isfinite(vmin) or not np.isfinite(vmax):
        return
    if vmax <= vmin:
        vmax = vmin + 1.0
    w = 18
    values = np.linspace(1.0, 0.0, h, dtype=np.float32)[:, None]
    rgb = gradient_rgb(np.broadcast_to(values, (h, w)), color_stops(scheme))
    canvas.paste(Image.fromarray(rgb, mode="RGB"), (x, y))
    draw.rectangle((x, y, x + w, y + h), outline=(35, 35, 35), width=1)
    ticks = [(0.0, vmax), (0.5, (vmin + vmax) / 2.0), (1.0, vmin)]
    for frac, value in ticks:
        yy = int(y + frac * h)
        draw.line((x + w, yy, x + w + 5, yy), fill=(35, 35, 35), width=1)
        draw.text((x + w + 8, yy - 7), f"{value:.1f}", fill=(35, 35, 35), font=tick_font)


# Draw the topography legend for preview figures.
def draw_topography_legend(
    draw: ImageDraw.ImageDraw,
    *,
    x: int,
    y: int,
    h: int,
    tick_font: ImageFont.ImageFont,
) -> None:
    colors = [(248, 248, 248), (154, 103, 55), (10, 20, 35), (125, 55, 220)]
    labels = ["0 non-topo", "1 terrain", "2 building", "3 other"]
    box_h = max(h // 4, 1)
    w = 18
    for i, (color, label) in enumerate(zip(colors, labels)):
        yy = y + h - (i + 1) * box_h
        draw.rectangle((x, yy, x + w, yy + box_h), fill=color, outline=(50, 50, 50))
        draw.text((x + w + 8, yy + box_h // 2 - 7), label, fill=(35, 35, 35), font=tick_font)


# Convert a label into a filesystem-safe name.
def safe_name(text: str) -> str:
    keep = []
    for ch in text:
        keep.append(ch if ch.isalnum() or ch in "._-" else "_")
    return "".join(keep)


# Load visual arrays from disk or cache.
def load_visual_arrays(dataset: Stage2V40LocalDataset, index: int) -> dict[str, Any]:
    source_index = dataset.indices[index]
    sample = dataset.cache.sample(source_index)
    prev_co2, _ = dataset.prev.sample(source_index)
    metadata_raw = sample.get("metadata_json", "{}")
    if isinstance(metadata_raw, np.ndarray):
        metadata_raw = str(metadata_raw.item())
    meta = json.loads(str(metadata_raw))
    topo = np.asarray(sample["surface_2d"][0], dtype=np.float32)
    z0 = int(np.asarray(sample["z0"]).item())
    return {"sample": sample, "prev": prev_co2, "meta": meta, "topo": topo, "z0": z0}


# Save figure to disk.
def save_figure(
    out_path: Path,
    title: str,
    panels: list[tuple[str, np.ndarray, str, float | None, float | None]],
    *,
    figsize: tuple[float, float] = (18, 4.2),
    note: str = "Masked cells are hidden. CO2 values are in ppm.",
) -> None:
    del figsize
    title_font = font(25, bold=True)
    panel_font = font(19, bold=False)
    tick_font = font(13, bold=False)
    note_font = font(13, bold=False)
    panel_w = 460
    panel_h = 460
    colorbar_w = 96
    left_margin = 34
    top_margin = 18
    title_h = 42
    label_h = 34
    gap = 38
    width = left_margin * 2 + len(panels) * (panel_w + colorbar_w) + (len(panels) - 1) * gap
    height = top_margin + title_h + label_h + panel_h + 42
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    draw_centered(draw, (left_margin, top_margin, width - left_margin, top_margin + title_h), title[:260], title_font, (25, 32, 42))
    x0 = left_margin
    y_img = top_margin + title_h + label_h
    for label, arr, cm, vmin, vmax in panels:
        scheme = "topography" if label.lower().startswith("topography") else cm
        resample = Image.Resampling.NEAREST
        img = colorize(arr, scheme, vmin, vmax).resize((panel_w, panel_h), resample)
        draw_centered(draw, (x0, top_margin + title_h, x0 + panel_w, y_img), label, panel_font, (30, 30, 30))
        canvas.paste(img, (x0, y_img))
        draw.rectangle((x0, y_img, x0 + panel_w, y_img + panel_h), outline=(45, 45, 45), width=1)
        cb_x = x0 + panel_w + 12
        if scheme == "topography":
            draw_topography_legend(draw, x=cb_x, y=y_img + 18, h=panel_h - 36, tick_font=tick_font)
        else:
            draw_continuous_colorbar(canvas, draw, x=cb_x, y=y_img + 18, h=panel_h - 36, scheme=scheme, vmin=vmin, vmax=vmax, tick_font=tick_font)
        x0 += panel_w + colorbar_w + gap
    draw.text((left_margin, height - 24), note[:320], fill=(90, 98, 110), font=note_font)
    canvas.save(out_path)


# Save visualizations to disk.
def save_visualizations(
    *,
    model: torch.nn.Module,
    dataset: Stage2V40LocalDataset,
    indices: list[int],
    local_zs: list[int],
    out_dir: Path,
    device: torch.device,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    direct_dir = out_dir / "visualizations"
    delta_dir = out_dir / "delta_visualizations"
    direct_dir.mkdir(parents=True, exist_ok=True)
    delta_dir.mkdir(parents=True, exist_ok=True)
    direct_rows: list[dict[str, Any]] = []
    delta_rows: list[dict[str, Any]] = []
    for vis_i, ds_index in enumerate(indices):
        batch = dataset[ds_index]
        if len(batch) == 5:
            x, global_context, global_grid, y, m = batch
            global_context = global_context.unsqueeze(0).to(device)
            global_grid = global_grid.unsqueeze(0).to(device)
        else:
            x, y, m = batch
            global_context = None
            global_grid = None
        with torch.no_grad():
            xb = x.unsqueeze(0).to(device)
            out = model(xb, global_context=global_context, global_grid=global_grid, return_components=True)
            pred_delta = out["final"] if isinstance(out, dict) else out
        pred_delta_np = pred_delta.squeeze(0).squeeze(0).detach().cpu().numpy()
        true_delta_np = y.squeeze(0).numpy()
        mask_np = m.squeeze(0).numpy()
        aux = load_visual_arrays(dataset, ds_index)
        prev_np = aux["prev"]
        pred_co2 = prev_np + pred_delta_np
        truth_co2 = prev_np + true_delta_np
        meta = aux["meta"]
        z0 = aux["z0"]
        topo = aux["topo"]
        for local_z in local_zs:
            if local_z < 0 or local_z >= pred_delta_np.shape[0]:
                continue
            mask2 = mask_np[local_z] > 0
            if not np.any(mask2):
                continue
            topo_panel = np.where(mask2, 0.0, topo).astype(np.float32, copy=False)
            direct_stats = finite_metrics(pred_co2[local_z], truth_co2[local_z], mask_np[local_z])
            delta_stats = finite_metrics(pred_delta_np[local_z], true_delta_np[local_z], mask_np[local_z])
            global_z = z0 + local_z
            height_m = global_z * 10
            job = str(meta.get("job", "job"))
            month = meta.get("month", "")
            time_index = meta.get("time_index", "na")
            y0 = int(meta.get("y0", -1)) if str(meta.get("y0", "")).lstrip("-").isdigit() else -1
            x0 = int(meta.get("x0", -1)) if str(meta.get("x0", "")).lstrip("-").isdigit() else -1
            stem = safe_name(
                f"sample_{vis_i:03d}_globalz{global_z:03d}_{job}_t{int(time_index):03d}_y{y0:03d}_x{x0:03d}"
                if str(time_index).lstrip("-").isdigit()
                else f"sample_{vis_i:03d}_globalz{global_z:03d}_{job}_t{time_index}_y{y0:03d}_x{x0:03d}"
            )
            truth_slice = np.ma.masked_where(~mask2, truth_co2[local_z])
            pred_slice = np.ma.masked_where(~mask2, pred_co2[local_z])
            err_slice = np.ma.masked_where(~mask2, np.abs(pred_co2[local_z] - truth_co2[local_z]))
            co2_vals = np.concatenate([pred_co2[local_z][mask2], truth_co2[local_z][mask2]])
            lo = float(np.nanpercentile(co2_vals, 1))
            hi = float(np.nanpercentile(co2_vals, 99))
            if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
                lo, hi = float(np.nanmin(co2_vals)), float(np.nanmax(co2_vals))
            if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
                lo, hi = 0.0, 1.0
            err_vals = np.abs(pred_co2[local_z] - truth_co2[local_z])[mask2]
            err_vmax = float(np.nanpercentile(err_vals, 99)) if err_vals.size else 1.0
            if not np.isfinite(err_vmax) or err_vmax <= 0:
                err_vmax = 1.0
            save_figure(
                direct_dir / f"{stem}.png",
                (
                    f"{job} | month {month} | time {time_index} | height={height_m} m | 256x256 direct | "
                    f"R={direct_stats['R']:.3f}, R2={direct_stats['R2']:.3f}, "
                    f"MAE={direct_stats['MAE']:.3f}, RMSE={direct_stats['RMSE']:.3f}"
                ),
                [
                    ("Topography", topo_panel, "topography", 0.0, 3.0),
                    ("CO2 prediction", pred_slice, "turbo", lo, hi),
                    ("CO2 ground truth", truth_slice, "turbo", lo, hi),
                    ("|prediction - truth|", err_slice, "magma", 0.0, err_vmax),
                ],
                note=f"CO2 prediction/truth share range {lo:.2f}-{hi:.2f}; topography-masked cells are hidden.",
            )
            delta_vals = np.concatenate([pred_delta_np[local_z][mask2], true_delta_np[local_z][mask2]])
            d_abs = float(np.nanpercentile(np.abs(delta_vals), 99)) if delta_vals.size else 1.0
            if not np.isfinite(d_abs) or d_abs <= 0:
                d_abs = 1.0
            delta_err_vals = np.abs(pred_delta_np[local_z] - true_delta_np[local_z])[mask2]
            delta_err_vmax = float(np.nanpercentile(delta_err_vals, 99)) if delta_err_vals.size else 1.0
            if not np.isfinite(delta_err_vmax) or delta_err_vmax <= 0:
                delta_err_vmax = 1.0
            prev_vals = prev_np[local_z][mask2]
            prev_lo = float(np.nanpercentile(prev_vals, 1)) if prev_vals.size else 0.0
            prev_hi = float(np.nanpercentile(prev_vals, 99)) if prev_vals.size else 1.0
            if not np.isfinite(prev_lo) or not np.isfinite(prev_hi) or prev_hi <= prev_lo:
                prev_lo, prev_hi = 0.0, 1.0
            save_figure(
                delta_dir / f"{stem}.png",
                (
                    f"Delta target: CO2(t)-CO2(t-1) | {job} | month {month} | time {time_index} | "
                    f"height={height_m} m | R={delta_stats['R']:.3f}, R2={delta_stats['R2']:.3f}, "
                    f"MAE={delta_stats['MAE']:.3f}, RMSE={delta_stats['RMSE']:.3f}"
                ),
                [
                    ("Topography", topo_panel, "topography", 0.0, 3.0),
                    ("CO2 previous t-1", np.ma.masked_where(~mask2, prev_np[local_z]), "turbo", prev_lo, prev_hi),
                    ("Predicted delta", np.ma.masked_where(~mask2, pred_delta_np[local_z]), "coolwarm", -d_abs, d_abs),
                    ("Truth delta", np.ma.masked_where(~mask2, true_delta_np[local_z]), "coolwarm", -d_abs, d_abs),
                    ("|pred delta - truth delta|", np.ma.masked_where(~mask2, np.abs(pred_delta_np[local_z] - true_delta_np[local_z])), "magma", 0.0, delta_err_vmax),
                ],
                note=(
                    f"Delta panels share symmetric range {-d_abs:.2f} to {d_abs:.2f} ppm; masked cells are hidden. "
                    "If predicted delta is near zero everywhere, the final CO2 is mostly persistence."
                ),
            )
            direct_row = {
                "dataset_index": ds_index,
                "job": job,
                "month": month,
                "time_index": time_index,
                "global_z": global_z,
                "height_m": height_m,
                "R": direct_stats["R"],
                "R2": direct_stats["R2"],
                "MAE": direct_stats["MAE"],
                "RMSE": direct_stats["RMSE"],
                "vmin": lo,
                "vmax": hi,
                "path": str(direct_dir / f"{stem}.png"),
            }
            delta_row = {
                "dataset_index": ds_index,
                "job": job,
                "month": month,
                "time_index": time_index,
                "global_z": global_z,
                "height_m": height_m,
                "R": delta_stats["R"],
                "R2": delta_stats["R2"],
                "MAE": delta_stats["MAE"],
                "RMSE": delta_stats["RMSE"],
                "vmin": -d_abs,
                "vmax": d_abs,
                "path": str(delta_dir / f"{stem}.png"),
            }
            combined_row = (
                {
                    "dataset_index": ds_index,
                    "job": job,
                    "month": month,
                    "time_index": time_index,
                    "global_z": global_z,
                    "height_m": height_m,
                    "direct_R": direct_stats["R"],
                    "direct_R2": direct_stats["R2"],
                    "direct_MAE": direct_stats["MAE"],
                    "direct_RMSE": direct_stats["RMSE"],
                    "delta_R": delta_stats["R"],
                    "delta_R2": delta_stats["R2"],
                    "delta_MAE": delta_stats["MAE"],
                    "delta_RMSE": delta_stats["RMSE"],
                }
            )
            direct_rows.append({**combined_row, **direct_row})
            delta_rows.append({**combined_row, **delta_row})
    return direct_rows, delta_rows


# Write csv to disk.
def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


# Entry point for the command-line workflow.
def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate V40-style Stage1-met CO2 delta model on the Stage2 cache.")
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--prev-sidecar-root", type=Path, required=True)
    parser.add_argument("--global-sidecar-root", type=Path, help="Optional full-domain Stage1-met context sidecar.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--split", choices=["train", "val"], default="val")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--visual-samples", type=int, default=10)
    parser.add_argument("--visual-local-z", default="0,4,8")
    parser.add_argument("--visual-only", action="store_true", help="Only regenerate visualization PNGs and visualization_metrics.csv.")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--layer-min", type=int, default=1)
    parser.add_argument("--layer-max", type=int, default=10)
    parser.add_argument("--min-layer-overlap", type=int, default=8)
    parser.add_argument("--dx", type=float, default=5.0)
    parser.add_argument("--dy", type=float, default=5.0)
    parser.add_argument("--dz", type=float, default=10.0)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else args.device)
    if args.device == "auto" and not torch.cuda.is_available():
        device = torch.device("cpu")

    checkpoint = torch.load(args.checkpoint, map_location=device)
    model = model_from_checkpoint(checkpoint, device)
    dataset = Stage2V40GlobalContextDataset(
        args.cache_root,
        args.prev_sidecar_root,
        args.split,
        layer_min=args.layer_min,
        layer_max=args.layer_max,
        min_layer_overlap=args.min_layer_overlap,
        dx=args.dx,
        dy=args.dy,
        dz=args.dz,
        global_sidecar_root=args.global_sidecar_root,
    )
    total = len(dataset)

    if args.visual_only:
        if args.visual_samples <= 0:
            raise ValueError("--visual-only requires --visual-samples > 0")
        rng = np.random.default_rng(args.seed + 991)
        vis_indices = np.sort(rng.choice(total, size=min(args.visual_samples, total), replace=False)).astype(int).tolist()
        local_zs = [int(v) for v in args.visual_local_z.split(",") if v.strip()]
        visual_rows, delta_visual_rows = save_visualizations(
            model=model,
            dataset=dataset,
            indices=vis_indices,
            local_zs=local_zs,
            out_dir=args.out_dir,
            device=device,
        )
        write_csv(args.out_dir / "visualization_metrics.csv", visual_rows)
        write_csv(args.out_dir / "delta_visualization_metrics.csv", delta_visual_rows)
        print(f"Saved {len(visual_rows)} direct PNGs to {args.out_dir / 'visualizations'}", flush=True)
        print(f"Saved {len(delta_visual_rows)} delta PNGs to {args.out_dir / 'delta_visualizations'}", flush=True)
        return

    if args.max_samples > 0:
        rng = np.random.default_rng(args.seed)
        eval_indices = np.sort(rng.choice(total, size=min(args.max_samples, total), replace=False)).astype(int).tolist()
    else:
        eval_indices = list(range(total))
    loader = DataLoader(
        Subset(dataset, eval_indices),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=args.num_workers > 0,
    )

    direct_metrics = StreamingMetrics()
    delta_metrics = StreamingMetrics()
    sample_rows: list[dict[str, Any]] = []
    seen = 0
    with torch.no_grad():
        for batch_i, batch in enumerate(loader, start=1):
            if len(batch) == 5:
                x, global_context, global_grid, y, m = batch
                global_context = global_context.to(device, non_blocking=True)
                global_grid = global_grid.to(device, non_blocking=True)
            else:
                x, y, m = batch
                global_context = None
                global_grid = None
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            m = m.to(device, non_blocking=True)
            out = model(x, global_context=global_context, global_grid=global_grid, return_components=True)
            pred_delta = out["final"] if isinstance(out, dict) else out
            prev = denormalized_prev_from_x(dataset, x)
            pred_direct = prev + pred_delta
            truth_direct = prev + y
            pd = pred_delta.detach().cpu().numpy()
            td = y.detach().cpu().numpy()
            pp = pred_direct.detach().cpu().numpy()
            tt = truth_direct.detach().cpu().numpy()
            mm = m.detach().cpu().numpy()
            direct_metrics.update(pp, tt, mm)
            delta_metrics.update(pd, td, mm)
            for j in range(pd.shape[0]):
                ds_index = eval_indices[seen + j]
                sample_direct = finite_metrics(pp[j, 0], tt[j, 0], mm[j, 0])
                sample_delta = finite_metrics(pd[j, 0], td[j, 0], mm[j, 0])
                meta = load_visual_arrays(dataset, ds_index)["meta"]
                sample_rows.append(
                    {
                        "dataset_index": ds_index,
                        "job": meta.get("job", ""),
                        "month": meta.get("month", ""),
                        "time_index": meta.get("time_index", ""),
                        "direct_R": sample_direct["R"],
                        "direct_R2": sample_direct["R2"],
                        "direct_MAE": sample_direct["MAE"],
                        "direct_RMSE": sample_direct["RMSE"],
                        "delta_R": sample_delta["R"],
                        "delta_R2": sample_delta["R2"],
                        "delta_MAE": sample_delta["MAE"],
                        "delta_RMSE": sample_delta["RMSE"],
                    }
                )
            seen += pd.shape[0]
            if batch_i % 25 == 0 or seen >= len(eval_indices):
                print(f"evaluated {seen}/{len(eval_indices)} samples", flush=True)

    metrics = {
        "checkpoint": str(args.checkpoint),
        "split": args.split,
        "global_sidecar_root": str(args.global_sidecar_root) if args.global_sidecar_root is not None else None,
        "evaluated_samples": len(eval_indices),
        "dataset_samples": total,
        "direct": direct_metrics.finish(),
        "delta": delta_metrics.finish(),
    }
    (args.out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    with (args.out_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["target", "valid_count", "R", "R2", "MAE", "RMSE"])
        for name in ("direct", "delta"):
            row = metrics[name]
            writer.writerow([name, row["valid_count"], row["R"], row["R2"], row["MAE"], row["RMSE"]])
    write_csv(args.out_dir / "sample_metrics.csv", sample_rows)

    if args.visual_samples > 0:
        rng = np.random.default_rng(args.seed + 991)
        vis_indices = np.sort(rng.choice(total, size=min(args.visual_samples, total), replace=False)).astype(int).tolist()
        local_zs = [int(v) for v in args.visual_local_z.split(",") if v.strip()]
        visual_rows, delta_visual_rows = save_visualizations(
            model=model,
            dataset=dataset,
            indices=vis_indices,
            local_zs=local_zs,
            out_dir=args.out_dir,
            device=device,
        )
        write_csv(args.out_dir / "visualization_metrics.csv", visual_rows)
        write_csv(args.out_dir / "delta_visualization_metrics.csv", delta_visual_rows)

    print(json.dumps(metrics, indent=2), flush=True)
    print(f"Saved evaluation to {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()

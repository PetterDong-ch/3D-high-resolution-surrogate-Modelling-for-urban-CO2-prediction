#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from two_stage_surrogate.data.stage1_cache import Stage1CacheDataset, denormalize_target  # noqa: E402
from two_stage_surrogate.models import LocalFNOStage1, Stage1ModelConfig  # noqa: E402


VARIABLES = ("u", "v", "w", "theta_prime", "theta")
PLOT_LABELS = {
    "u": "u wind",
    "v": "v wind",
    "w": "w wind",
    "theta_prime": "theta anomaly",
    "theta": "theta",
}
UNITS = {
    "u": "m s$^{-1}$",
    "v": "m s$^{-1}$",
    "w": "m s$^{-1}$",
    "theta_prime": "K",
    "theta": "K",
}


# Internal helper for load model.
def _load_model(checkpoint_path: Path, device: torch.device) -> LocalFNOStage1:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model_cfg = checkpoint["model_config"]
    model_cfg.setdefault("predict_w", True)
    model = LocalFNOStage1(Stage1ModelConfig(**model_cfg)).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


# Internal helper for to batch.
def _to_batch(item: dict[str, Any], device: torch.device) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in item.items():
        if torch.is_tensor(value):
            out[key] = value.unsqueeze(0).to(device)
        else:
            out[key] = value
    return out


# Internal helper for metadata.
def _metadata(item: dict[str, Any]) -> dict[str, Any]:
    return json.loads(item["metadata_json"])


# Internal helper for masked flat.
def _masked_flat(values: np.ndarray, mask: np.ndarray) -> np.ndarray:
    valid = np.isfinite(values) & mask.astype(bool)
    return values[valid].astype(np.float64, copy=False)


# Internal helper for metrics.
def _metrics(pred: np.ndarray, truth: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    valid = np.isfinite(pred) & np.isfinite(truth) & mask.astype(bool)
    if not np.any(valid):
        return {"valid_count": 0, "R": float("nan"), "MAE": float("nan"), "RMSE": float("nan")}
    p = pred[valid].astype(np.float64, copy=False)
    t = truth[valid].astype(np.float64, copy=False)
    diff = p - t
    mae = float(np.mean(np.abs(diff)))
    rmse = float(np.sqrt(np.mean(diff * diff)))
    if p.size < 2 or np.std(p) < 1e-12 or np.std(t) < 1e-12:
        corr = float("nan")
    else:
        corr = float(np.corrcoef(p, t)[0, 1])
    return {"valid_count": int(p.size), "R": corr, "MAE": mae, "RMSE": rmse}


# Internal helper for safe score.
def _safe_score(values: list[float]) -> float:
    clean = [v for v in values if np.isfinite(v)]
    if not clean:
        return -999.0
    return float(np.mean(clean))


# Internal helper for predict physical.
def _predict_physical(
    model: LocalFNOStage1,
    item: dict[str, Any],
    target_stats: dict[str, torch.Tensor],
    device: torch.device,
) -> dict[str, np.ndarray]:
    batch = _to_batch(item, device)
    with torch.no_grad():
        pred_norm = model(
            geometry_3d=batch["geometry_3d"],
            surface_2d=batch["surface_2d"],
            profile=batch["profile"],
            scalar=batch["scalar"],
            theta_reference=None,
        )
        uv_pred = denormalize_target("target_uv", pred_norm["uv"], target_stats)
        w_pred = denormalize_target("target_w", pred_norm["w"], target_stats)
        theta_prime_pred = denormalize_target("target_theta_prime", pred_norm["theta_prime"], target_stats)
        theta_pred = batch["theta_reference"] + theta_prime_pred

        uv_truth = denormalize_target("target_uv", batch["target_uv"], target_stats)
        w_truth = denormalize_target("target_w", batch["target_w"], target_stats)
        theta_prime_truth = denormalize_target("target_theta_prime", batch["target_theta_prime"], target_stats)

    return {
        "u_pred": uv_pred[0, 0].detach().cpu().numpy(),
        "u_truth": uv_truth[0, 0].detach().cpu().numpy(),
        "v_pred": uv_pred[0, 1].detach().cpu().numpy(),
        "v_truth": uv_truth[0, 1].detach().cpu().numpy(),
        "w_pred": w_pred[0, 0].detach().cpu().numpy(),
        "w_truth": w_truth[0, 0].detach().cpu().numpy(),
        "theta_prime_pred": theta_prime_pred[0, 0].detach().cpu().numpy(),
        "theta_prime_truth": theta_prime_truth[0, 0].detach().cpu().numpy(),
        "theta_pred": theta_pred[0, 0].detach().cpu().numpy(),
        "theta_truth": batch["target_theta"][0, 0].detach().cpu().numpy(),
        "mask": batch["mask"][0, 0].detach().cpu().numpy(),
    }


# Internal helper for scan case.
def _scan_case(
    dataset: Stage1CacheDataset,
    model: LocalFNOStage1,
    target_stats: dict[str, torch.Tensor],
    device: torch.device,
    index: int,
    z_choices: list[int],
) -> tuple[list[dict[str, Any]], dict[str, np.ndarray]]:
    item = dataset[index]
    arrays = _predict_physical(model, item, target_stats, device)
    meta = _metadata(item)
    rows: list[dict[str, Any]] = []
    for local_z in z_choices:
        if local_z < 0 or local_z >= arrays["mask"].shape[0]:
            continue
        mask = arrays["mask"][local_z] > 0.5
        per_var: dict[str, dict[str, float]] = {}
        for var in VARIABLES:
            per_var[var] = _metrics(arrays[f"{var}_pred"][local_z], arrays[f"{var}_truth"][local_z], mask)
        score = (
            0.35 * _safe_score([per_var["u"]["R"]])
            + 0.35 * _safe_score([per_var["v"]["R"]])
            + 0.25 * _safe_score([per_var["theta_prime"]["R"]])
            + 0.05 * _safe_score([per_var["w"]["R"]])
        )
        valid_fraction_slice = float(np.mean(mask))
        if valid_fraction_slice < 0.08:
            score -= 0.8
        rows.append(
            {
                "dataset_index": index,
                "sample_key": item["sample_key"],
                "job": meta.get("job"),
                "month": int(meta.get("month", -1)),
                "time_index": int(meta.get("time_index", -1)),
                "z0": int(meta.get("z0", 0)),
                "local_z": int(local_z),
                "global_z": int(meta.get("z0", 0)) + int(local_z),
                "height_m": 10.0 * (int(meta.get("z0", 0)) + int(local_z)),
                "y0": int(meta.get("y0", -1)),
                "x0": int(meta.get("x0", -1)),
                "valid_fraction_3d": float(meta.get("valid_fraction", float("nan"))),
                "valid_fraction_slice": valid_fraction_slice,
                "score": float(score),
                "u_R": per_var["u"]["R"],
                "u_MAE": per_var["u"]["MAE"],
                "u_RMSE": per_var["u"]["RMSE"],
                "v_R": per_var["v"]["R"],
                "v_MAE": per_var["v"]["MAE"],
                "v_RMSE": per_var["v"]["RMSE"],
                "w_R": per_var["w"]["R"],
                "w_MAE": per_var["w"]["MAE"],
                "w_RMSE": per_var["w"]["RMSE"],
                "theta_prime_R": per_var["theta_prime"]["R"],
                "theta_prime_MAE": per_var["theta_prime"]["MAE"],
                "theta_prime_RMSE": per_var["theta_prime"]["RMSE"],
                "theta_R": per_var["theta"]["R"],
                "theta_MAE": per_var["theta"]["MAE"],
                "theta_RMSE": per_var["theta"]["RMSE"],
            }
        )
    return rows, arrays


# Internal helper for font.
def _font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ]
    for path in candidates:
        if Path(path).exists():
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


# Internal helper for lerp color.
def _lerp_color(c0: tuple[int, int, int], c1: tuple[int, int, int], t: np.ndarray) -> np.ndarray:
    out = np.empty((*t.shape, 3), dtype=np.float32)
    for i in range(3):
        out[..., i] = c0[i] + (c1[i] - c0[i]) * t
    return out


# Internal helper for scalar to rgb.
def _scalar_to_rgb(
    arr: np.ndarray,
    mask: np.ndarray,
    *,
    vmin: float,
    vmax: float,
    diverging: bool,
) -> Image.Image:
    arr = np.asarray(arr, dtype=np.float32)
    mask_bool = mask.astype(bool) & np.isfinite(arr)
    scaled = np.clip((arr - vmin) / max(vmax - vmin, 1e-9), 0.0, 1.0)
    if diverging:
        low = (49, 97, 186)
        mid = (245, 245, 245)
        high = (180, 30, 30)
        rgb = np.empty((*arr.shape, 3), dtype=np.float32)
        lo_part = scaled <= 0.5
        hi_part = ~lo_part
        rgb[lo_part] = _lerp_color(low, mid, scaled[lo_part] / 0.5)
        rgb[hi_part] = _lerp_color(mid, high, (scaled[hi_part] - 0.5) / 0.5)
    else:
        anchors = [
            (68, 1, 84),
            (59, 82, 139),
            (33, 145, 140),
            (94, 201, 98),
            (253, 231, 37),
        ]
        pos = scaled * (len(anchors) - 1)
        idx = np.clip(np.floor(pos).astype(int), 0, len(anchors) - 2)
        frac = pos - idx
        rgb = np.empty((*arr.shape, 3), dtype=np.float32)
        for i in range(len(anchors) - 1):
            sel = idx == i
            if np.any(sel):
                rgb[sel] = _lerp_color(anchors[i], anchors[i + 1], frac[sel])
    rgb = np.asarray(np.clip(rgb, 0, 255), dtype=np.uint8)
    rgb[~mask_bool] = np.array([255, 255, 255], dtype=np.uint8)
    return Image.fromarray(rgb, mode="RGB")


# Internal helper for add colorbar.
def _add_colorbar(
    draw: ImageDraw.ImageDraw,
    x: int,
    y: int,
    h: int,
    *,
    vmin: float,
    vmax: float,
    diverging: bool,
    font: ImageFont.ImageFont,
) -> None:
    gradient = np.linspace(vmax, vmin, h, dtype=np.float32)[:, None]
    bar = _scalar_to_rgb(gradient, np.ones_like(gradient, dtype=bool), vmin=vmin, vmax=vmax, diverging=diverging)
    draw._image.paste(bar.resize((16, h)), (x, y))
    draw.rectangle([x, y, x + 15, y + h], outline=(60, 60, 60), width=1)
    draw.text((x + 20, y - 2), f"{vmax:.2g}", fill=(30, 30, 30), font=font)
    draw.text((x + 20, y + h - 12), f"{vmin:.2g}", fill=(30, 30, 30), font=font)


# Internal helper for range pair.
def _range_pair(truth: np.ndarray, pred: np.ndarray, mask: np.ndarray, robust: bool = True) -> tuple[float, float]:
    t = _masked_flat(truth, mask)
    p = _masked_flat(pred, mask)
    values = np.concatenate([t, p]) if t.size and p.size else np.array([], dtype=np.float64)
    if values.size == 0:
        return 0.0, 1.0
    if robust and values.size > 20:
        lo, hi = np.nanpercentile(values, [1, 99])
    else:
        lo, hi = float(np.nanmin(values)), float(np.nanmax(values))
    if not np.isfinite(lo) or not np.isfinite(hi) or abs(hi - lo) < 1e-9:
        center = float(np.nanmean(values)) if values.size else 0.0
        lo, hi = center - 1.0, center + 1.0
    return float(lo), float(hi)


# Internal helper for error range.
def _error_range(err: np.ndarray, mask: np.ndarray) -> tuple[float, float]:
    values = _masked_flat(err, mask)
    if values.size == 0:
        return -1.0, 1.0
    mag = float(np.nanpercentile(np.abs(values), 99))
    if not np.isfinite(mag) or mag < 1e-9:
        mag = 1.0
    return -mag, mag


# Internal helper for save visualization.
def _save_visualization(
    out_path: Path,
    arrays: dict[str, np.ndarray],
    row: dict[str, Any],
) -> None:
    z = int(row["local_z"])
    mask = arrays["mask"][z] > 0.5
    title_font = _font(28, bold=True)
    header_font = _font(21, bold=True)
    label_font = _font(19, bold=True)
    metric_font = _font(15, bold=False)
    tick_font = _font(12, bold=False)
    panel = 256
    scale = 2
    panel_out = panel * scale
    left = 190
    top = 118
    gap_x = 92
    gap_y = 58
    header_h = 30
    canvas_w = left + 3 * panel_out + 2 * gap_x + 95
    canvas_h = top + len(VARIABLES) * (panel_out + gap_y) + 30
    image = Image.new("RGB", (canvas_w, canvas_h), "white")
    draw = ImageDraw.Draw(image)
    title = (
        f"Stage 1 selected case | {row['job']} | month {row['month']} | "
        f"time {row['time_index']} | height={row['height_m']:.0f} m | "
        f"y={row['y0']}, x={row['x0']} | score={row['score']:.3f}"
    )
    draw.text((canvas_w // 2, 24), title, fill=(20, 20, 20), font=title_font, anchor="ma")
    for col, text in enumerate(("truth", "prediction", "prediction - truth")):
        x = left + col * (panel_out + gap_x) + panel_out // 2
        draw.text((x, top - header_h), text, fill=(20, 20, 20), font=header_font, anchor="ma")
    for row_idx, var in enumerate(VARIABLES):
        truth = arrays[f"{var}_truth"][z]
        pred = arrays[f"{var}_pred"][z]
        err = pred - truth
        vmin, vmax = _range_pair(truth, pred, mask)
        emin, emax = _error_range(err, mask)
        row_y = top + row_idx * (panel_out + gap_y)
        draw.text(
            (22, row_y + panel_out // 2 - 10),
            f"{PLOT_LABELS[var]}\n({UNITS[var]})",
            fill=(20, 20, 20),
            font=label_font,
            anchor="lm",
            spacing=6,
        )
        panels = [
            (truth, vmin, vmax, var != "theta"),
            (pred, vmin, vmax, var != "theta"),
            (err, emin, emax, True),
        ]
        for col_idx, (arr, lo, hi, diverging) in enumerate(panels):
            panel_img = _scalar_to_rgb(arr, mask, vmin=lo, vmax=hi, diverging=diverging)
            panel_img = panel_img.transpose(Image.Transpose.FLIP_TOP_BOTTOM).resize((panel_out, panel_out), Image.Resampling.BILINEAR)
            x = left + col_idx * (panel_out + gap_x)
            image.paste(panel_img, (x, row_y))
            draw.rectangle([x, row_y, x + panel_out, row_y + panel_out], outline=(60, 60, 60), width=2)
            _add_colorbar(draw, x + panel_out + 8, row_y, panel_out, vmin=lo, vmax=hi, diverging=diverging, font=tick_font)
        draw.text(
            (left + panel_out + gap_x + panel_out // 2, row_y + panel_out + 10),
            f"R={row[f'{var}_R']:.3f}, MAE={row[f'{var}_MAE']:.3f}, RMSE={row[f'{var}_RMSE']:.3f}",
            fill=(40, 40, 40),
            font=metric_font,
            anchor="ma",
        )
    image.save(out_path)


# Internal helper for write csv.
def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


# Internal helper for select rows.
def _select_rows(rows: list[dict[str, Any]], save_count: int, *, mode: str) -> list[dict[str, Any]]:
    rows = sorted(rows, key=lambda r: (float(r["score"]), float(r["valid_fraction_slice"])), reverse=True)
    if mode == "top":
        selected: list[dict[str, Any]] = []
        used_locations: set[tuple[int, int, int, int]] = set()
        for row in rows:
            location_key = (int(row["month"]), int(row["time_index"]), int(row["y0"]) // 256, int(row["x0"]) // 256)
            if location_key in used_locations:
                continue
            selected.append(row)
            used_locations.add(location_key)
            if len(selected) >= save_count:
                return selected
        return rows[:save_count]
    if mode != "diverse-month":
        raise ValueError(f"Unknown selection mode: {mode}")
    selected: list[dict[str, Any]] = []
    used_months: set[int] = set()
    used_locations: set[tuple[int, int, int]] = set()
    for row in rows:
        month = int(row["month"])
        location_key = (month, int(row["y0"]) // 256, int(row["x0"]) // 256)
        if month in used_months:
            continue
        if location_key in used_locations:
            continue
        selected.append(row)
        used_months.add(month)
        used_locations.add(location_key)
        if len(selected) >= save_count:
            return selected
    for row in rows:
        location_key = (int(row["month"]), int(row["y0"]) // 256, int(row["x0"]) // 256)
        if location_key in used_locations:
            continue
        selected.append(row)
        used_locations.add(location_key)
        if len(selected) >= save_count:
            return selected
    return selected[:save_count]


# Entry point for the command-line workflow.
def main() -> None:
    parser = argparse.ArgumentParser(description="Generate additional Stage 1 validation visualizations.")
    parser.add_argument("--cache-root", type=Path, default=PROJECT_ROOT / "generated" / "stage1_cache")
    parser.add_argument("--checkpoint", type=Path, default=PROJECT_ROOT / "checkpoints" / "stage1_local_fno_best_model.pt")
    parser.add_argument("--out-dir", type=Path, default=PROJECT_ROOT / "artifacts" / "stage1_case_visualizations_more")
    parser.add_argument("--split", default="val")
    parser.add_argument("--candidate-count", type=int, default=360)
    parser.add_argument("--save-count", type=int, default=12)
    parser.add_argument("--seed", type=int, default=20260813)
    parser.add_argument("--z-choices", default="3,5,7,10,12,15")
    parser.add_argument("--selection-mode", choices=("diverse-month", "top"), default="diverse-month")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else args.device)
    if args.device == "cpu":
        device = torch.device("cpu")

    dataset = Stage1CacheDataset(args.cache_root, split=args.split)
    model = _load_model(args.checkpoint, device)
    target_stats = dataset.target_stat_tensors(device)
    z_choices = [int(x) for x in args.z_choices.split(",") if x.strip()]

    rng = np.random.default_rng(args.seed)
    if args.candidate_count >= len(dataset):
        candidate_indices = np.arange(len(dataset), dtype=int)
    else:
        candidate_indices = rng.choice(len(dataset), size=args.candidate_count, replace=False)

    all_rows: list[dict[str, Any]] = []
    arrays_cache: dict[tuple[int, int], dict[str, np.ndarray]] = {}
    for n, index in enumerate(candidate_indices, start=1):
        rows, arrays = _scan_case(dataset, model, target_stats, device, int(index), z_choices)
        all_rows.extend(rows)
        for row in rows:
            arrays_cache[(int(index), int(row["local_z"]))] = arrays
        if n == 1 or n % 50 == 0 or n == len(candidate_indices):
            print(f"scanned {n}/{len(candidate_indices)} candidate samples", flush=True)

    selected = _select_rows(all_rows, args.save_count, mode=args.selection_mode)
    _write_csv(args.out_dir / "stage1_candidate_slice_metrics.csv", all_rows)
    _write_csv(args.out_dir / "stage1_selected_case_metrics.csv", selected)
    (args.out_dir / "stage1_selected_case_metrics.json").write_text(json.dumps(selected, indent=2), encoding="utf-8")

    for case_no, row in enumerate(selected):
        index = int(row["dataset_index"])
        local_z = int(row["local_z"])
        arrays = arrays_cache.get((index, local_z))
        if arrays is None:
            _, arrays = _scan_case(dataset, model, target_stats, device, index, [local_z])
        png_name = (
            f"stage1_more_case_{case_no:03d}_m{int(row['month']):02d}_t{int(row['time_index']):03d}_"
            f"h{int(round(float(row['height_m']))):03d}_y{int(row['y0']):04d}_x{int(row['x0']):04d}.png"
        )
        _save_visualization(args.out_dir / png_name, arrays, row)
        row["png"] = png_name

    _write_csv(args.out_dir / "stage1_selected_case_metrics.csv", selected)
    (args.out_dir / "stage1_selected_case_metrics.json").write_text(json.dumps(selected, indent=2), encoding="utf-8")
    print(f"Saved {len(selected)} visualizations to {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()

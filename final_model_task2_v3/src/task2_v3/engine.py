from __future__ import annotations

import csv
import os

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler

from .io import write_json
from .metrics import regression_metrics


# Create a DataLoader with the requested sampling behavior.
def make_loader(dataset: Dataset, samples: int | None, batch_size: int, workers: int, shuffle: bool) -> DataLoader:
    if samples is not None and samples > 0:
        if shuffle:
            sampler = RandomSampler(dataset, replacement=True, num_samples=int(samples))
        else:
            sampler = SequentialSampler(range(min(int(samples), len(dataset))))
    else:
        sampler = RandomSampler(dataset) if shuffle else SequentialSampler(dataset)
    return DataLoader(dataset, batch_size=batch_size, sampler=sampler, num_workers=workers, pin_memory=torch.cuda.is_available())


# Run validation or inference and collect metrics.
@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    target_mean: np.ndarray,
    target_std: np.ndarray,
    out_dir: str | None = None,
) -> dict[str, object]:
    model.eval()
    pred_target_rows: list[np.ndarray] = []
    truth_target_rows: list[np.ndarray] = []
    pred_conc_rows: list[np.ndarray] = []
    truth_conc_rows: list[np.ndarray] = []
    mask_rows: list[np.ndarray] = []
    meta_rows: list[np.ndarray] = []
    mean = target_mean.astype(np.float32)[None, :]
    std = target_std.astype(np.float32)[None, :]
    for batch in loader:
        local, global_context, target_norm, mask, true_target, conc, bg, meta = batch
        pred_norm = model(local.to(device), global_context.to(device)).cpu().numpy()
        pred_target = pred_norm * std + mean
        true_target_np = true_target.numpy()
        conc_np = conc.numpy()
        bg_np = bg.numpy()
        mask_np = mask.numpy()
        pred_conc = bg_np + pred_target
        pred_target_rows.append(pred_target)
        truth_target_rows.append(true_target_np)
        pred_conc_rows.append(pred_conc)
        truth_conc_rows.append(conc_np)
        mask_rows.append(mask_np)
        meta_rows.append(meta.numpy())

    pred_target = np.concatenate(pred_target_rows, axis=0)
    truth_target = np.concatenate(truth_target_rows, axis=0)
    pred_conc = np.concatenate(pred_conc_rows, axis=0)
    truth_conc = np.concatenate(truth_conc_rows, axis=0)
    masks = np.concatenate(mask_rows, axis=0).astype(bool)
    metas = np.concatenate(meta_rows, axis=0)
    metrics = {
        "concentration": regression_metrics(truth_conc[masks], pred_conc[masks]),
        "target_enhancement": regression_metrics(truth_target[masks], pred_target[masks]),
    }
    per_layer = []
    for z in range(pred_target.shape[1]):
        m = masks[:, z]
        row = {"local_z": z}
        row.update({f"concentration_{k}": v for k, v in regression_metrics(truth_conc[m, z], pred_conc[m, z]).items()})
        row.update({f"target_enhancement_{k}": v for k, v in regression_metrics(truth_target[m, z], pred_target[m, z]).items()})
        per_layer.append(row)
    metrics["per_layer"] = per_layer

    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        write_json(os.path.join(out_dir, "metrics.json"), metrics)
        with open(os.path.join(out_dir, "metrics.csv"), "w", newline="", encoding="utf-8") as f:
            fieldnames = ["metric_group", "count", "R", "R2", "MAE", "RMSE"]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for group in ("concentration", "target_enhancement"):
                row = {"metric_group": group}
                row.update(metrics[group])
                writer.writerow(row)
        with open(os.path.join(out_dir, "per_layer_metrics.csv"), "w", newline="", encoding="utf-8") as f:
            fieldnames = sorted(per_layer[0].keys())
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(per_layer)
        with open(os.path.join(out_dir, "profile_prediction_preview.csv"), "w", newline="", encoding="utf-8") as f:
            fieldnames = [
                "row",
                "source_index",
                "month",
                "time_index",
                "z0",
                "y0",
                "x0",
                "local_z",
                "valid",
                "truth_concentration",
                "pred_concentration",
                "truth_target",
                "pred_target",
            ]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            written = 0
            for i in range(min(pred_target.shape[0], 5000)):
                for z in range(pred_target.shape[1]):
                    writer.writerow(
                        {
                            "row": i,
                            "source_index": int(metas[i, 0]),
                            "month": int(metas[i, 1]),
                            "time_index": int(metas[i, 2]),
                            "z0": int(metas[i, 3]),
                            "y0": int(metas[i, 4]),
                            "x0": int(metas[i, 5]),
                            "local_z": z,
                            "valid": int(masks[i, z]),
                            "truth_concentration": float(truth_conc[i, z]),
                            "pred_concentration": float(pred_conc[i, z]),
                            "truth_target": float(truth_target[i, z]),
                            "pred_target": float(pred_target[i, z]),
                        }
                    )
                    written += 1
                    if written >= 5000:
                        break
                if written >= 5000:
                    break
    return metrics

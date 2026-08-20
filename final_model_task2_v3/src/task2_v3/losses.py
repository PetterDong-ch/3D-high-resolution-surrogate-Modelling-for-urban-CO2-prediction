from __future__ import annotations

import torch


# Compute masked weighted Huber loss for normalized profile enhancement.
def weighted_huber_loss(
    pred_norm: torch.Tensor,
    target_norm: torch.Tensor,
    mask: torch.Tensor,
    layer_weights: torch.Tensor,
    huber_delta: float,
    gradient_weight: float,
) -> torch.Tensor:
    weights = mask * layer_weights[None, :]
    err = pred_norm - target_norm
    abs_err = err.abs()
    delta = float(huber_delta)
    huber = torch.where(abs_err <= delta, 0.5 * err * err, delta * (abs_err - 0.5 * delta))
    denom = weights.sum().clamp_min(1.0e-6)
    base = (huber * weights).sum() / denom
    if gradient_weight <= 0.0 or pred_norm.shape[-1] < 2:
        return base
    pair_mask = (mask[:, 1:] > 0.5) & (mask[:, :-1] > 0.5)
    pair_weights = 0.5 * (layer_weights[1:] + layer_weights[:-1])[None, :] * pair_mask.float()
    pred_dz = pred_norm[:, 1:] - pred_norm[:, :-1]
    target_dz = target_norm[:, 1:] - target_norm[:, :-1]
    dz_err = pred_dz - target_dz
    abs_dz = dz_err.abs()
    dz_huber = torch.where(abs_dz <= delta, 0.5 * dz_err * dz_err, delta * (abs_dz - 0.5 * delta))
    grad = (dz_huber * pair_weights).sum() / pair_weights.sum().clamp_min(1.0e-6)
    return base + float(gradient_weight) * grad

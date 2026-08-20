from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn


# Stores Stage 1 loss weights and thresholds.
@dataclass(frozen=True)
class Stage1LossConfig:
    uv_weight: float = 1.0
    w_weight: float = 1.0
    theta_prime_weight: float = 1.0
    gradient_weight: float = 0.05
    predict_w: bool = True
    active_w_weight: float = 0.0
    active_w_threshold: float = 0.05
    active_w_boost: float = 4.0


# Average values over valid masked cells only.
def masked_mean(value: Tensor, mask: Tensor, eps: float = 1e-6) -> Tensor:
    mask = mask.to(dtype=value.dtype)
    while mask.ndim < value.ndim:
        mask = mask.unsqueeze(1)
    mask = mask.expand_as(value)
    return (value * mask).sum() / mask.sum().clamp_min(eps)


# Compute masked mean-squared error.
def masked_mse(pred: Tensor, target: Tensor, mask: Tensor) -> Tensor:
    return masked_mean((pred - target).square(), mask)


# Compute L1 loss on field gradients.
def gradient_l1(pred: Tensor, target: Tensor, mask: Tensor) -> Tensor:
    loss = pred.new_tensor(0.0)
    terms = 0
    for dim in (-3, -2, -1):
        pred_grad = pred.diff(dim=dim)
        target_grad = target.diff(dim=dim)
        grad_mask = mask
        if dim == -3:
            grad_mask = mask[:, :, 1:, :, :] * mask[:, :, :-1, :, :]
        elif dim == -2:
            grad_mask = mask[:, :, :, 1:, :] * mask[:, :, :, :-1, :]
        elif dim == -1:
            grad_mask = mask[:, :, :, :, 1:] * mask[:, :, :, :, :-1]
        loss = loss + masked_mean((pred_grad - target_grad).abs(), grad_mask)
        terms += 1
    return loss / max(1, terms)


# Computes the Stage1Loss training objective.
class Stage1Loss(nn.Module):
    """Masked Stage 1 loss with separate output-family weights.

    Targets are expected to be independently normalized before this loss is
    called. That keeps u/v, w and theta-prime on comparable numerical scales.
    """

    # Store constructor arguments and initialize object state.
    def __init__(self, config: Stage1LossConfig | dict[str, Any]) -> None:
        super().__init__()
        if isinstance(config, dict):
            config = Stage1LossConfig(**config)
        self.config = config

    # Run the forward pass for this module.
    def forward(self, pred: dict[str, Tensor], target: dict[str, Tensor], mask: Tensor) -> tuple[Tensor, dict[str, Tensor]]:
        uv_loss = masked_mse(pred["uv"], target["uv"], mask)
        use_w = self.config.predict_w and "w" in pred and "w" in target

        if use_w and self.config.active_w_weight > 0:
            active = (target["w"].abs() >= self.config.active_w_threshold).to(target["w"].dtype)
            w_mask = mask * (1.0 + self.config.active_w_boost * active)
            w_loss = masked_mean((pred["w"] - target["w"]).square(), w_mask)
        elif use_w:
            w_loss = masked_mse(pred["w"], target["w"], mask)
        else:
            w_loss = uv_loss.new_tensor(0.0)

        theta_loss = masked_mse(pred["theta_prime"], target["theta_prime"], mask)
        grad_terms = [
            gradient_l1(pred["uv"], target["uv"], mask),
            gradient_l1(pred["theta_prime"], target["theta_prime"], mask),
        ]
        if use_w:
            grad_terms.append(gradient_l1(pred["w"], target["w"], mask))
        grad_loss = sum(grad_terms) / max(1, len(grad_terms))

        total = (
            self.config.uv_weight * uv_loss
            + self.config.theta_prime_weight * theta_loss
            + self.config.gradient_weight * grad_loss
        )
        if use_w:
            total = total + self.config.w_weight * w_loss
        parts = {
            "loss": total.detach(),
            "uv": uv_loss.detach(),
            "theta_prime": theta_loss.detach(),
            "gradient": grad_loss.detach(),
        }
        if use_w:
            parts["w"] = w_loss.detach()
        return total, parts


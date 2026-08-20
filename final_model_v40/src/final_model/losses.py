from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


# Stores the weights for each term in the V40 loss.
@dataclass(frozen=True)
class V40LossWeights:
    huber_delta: float = 2.0
    gradient: float = 0.03
    correlation: float = 0.04
    local_correlation: float = 0.05
    amplitude: float = 0.05
    active_delta: float = 0.35
    low_frequency: float = 0.04
    low_frequency_correlation: float = 0.02
    high_frequency: float = 0.04
    sign: float = 0.015
    active_aux: float = 0.02
    sign_aux: float = 0.02
    active_threshold: float = 0.75
    sign_min_abs: float = 0.75
    sign_scale: float = 2.0


# Average values over valid masked cells only.
def masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mean over valid atmospheric cells only."""

    return (x * mask).sum() / mask.sum().clamp_min(1.0)


# Compute masked Huber loss over valid cells only.
def masked_huber(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    delta: float,
    weight: torch.Tensor | None = None,
) -> torch.Tensor:
    """Robust pointwise loss under the validity mask."""

    diff = pred - target
    abs_diff = diff.abs()
    quadratic = torch.minimum(abs_diff, diff.new_tensor(float(delta)))
    loss = 0.5 * quadratic.square() + float(delta) * (abs_diff - quadratic)
    loss_weight = mask if weight is None else mask * weight
    return (loss * loss_weight).sum() / loss_weight.sum().clamp_min(1.0)


# Compute residual-magnitude loss weights.
def residual_weight(target: torch.Tensor, alpha: float = 0.5, scale: float = 4.0, max_weight: float = 4.0) -> torch.Tensor:
    """Increase weight for larger true CO2 increments."""

    weight = 1.0 + float(alpha) * target.abs() / max(float(scale), 1.0e-6)
    return weight.clamp(1.0, float(max_weight))


# Compute gradient-based loss weights.
def gradient_weight(target: torch.Tensor, mask: torch.Tensor, alpha: float = 0.35, scale: float = 1.0, max_weight: float = 2.5) -> torch.Tensor:
    """Increase weight where the target field has local spatial texture."""

    texture = torch.zeros_like(target)
    count = torch.zeros_like(target)
    for dim in (2, 3, 4):
        if target.shape[dim] <= 1:
            continue
        left = target.narrow(dim, 0, target.shape[dim] - 1)
        right = target.narrow(dim, 1, target.shape[dim] - 1)
        mask_left = mask.narrow(dim, 0, mask.shape[dim] - 1)
        mask_right = mask.narrow(dim, 1, mask.shape[dim] - 1)
        pair_mask = mask_left * mask_right
        local_texture = (right - left).abs() * pair_mask
        texture.narrow(dim, 0, target.shape[dim] - 1).add_(local_texture)
        texture.narrow(dim, 1, target.shape[dim] - 1).add_(local_texture)
        count.narrow(dim, 0, target.shape[dim] - 1).add_(pair_mask)
        count.narrow(dim, 1, target.shape[dim] - 1).add_(pair_mask)
    weight = 1.0 + float(alpha) * (texture / count.clamp_min(1.0)) / max(float(scale), 1.0e-6)
    return weight.clamp(1.0, float(max_weight))


# Compute masked L1 loss on vertical/spatial gradients.
def masked_gradient_l1(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Match spatial gradients of prediction and target."""

    terms = []
    for dim in (2, 3, 4):
        if pred.shape[dim] <= 1:
            continue
        pred_grad = pred.narrow(dim, 1, pred.shape[dim] - 1) - pred.narrow(dim, 0, pred.shape[dim] - 1)
        target_grad = target.narrow(dim, 1, target.shape[dim] - 1) - target.narrow(dim, 0, target.shape[dim] - 1)
        pair_mask = mask.narrow(dim, 1, mask.shape[dim] - 1) * mask.narrow(dim, 0, mask.shape[dim] - 1)
        terms.append(masked_mean((pred_grad - target_grad).abs(), pair_mask))
    if not terms:
        return pred.new_tensor(0.0)
    return torch.stack(terms).mean()


# Compute masked correlation over valid cells.
def masked_correlation(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, eps: float = 1.0e-6) -> torch.Tensor:
    """Return 1 - Pearson R over all valid cells."""

    valid = mask > 0
    if valid.sum() <= 1:
        return pred.new_tensor(0.0)
    p = pred[valid]
    y = target[valid]
    p = p - p.mean()
    y = y - y.mean()
    p_var = p.square().mean()
    y_var = y.square().mean()
    if p_var <= eps or y_var <= eps:
        return pred.new_tensor(0.0)
    corr = (p * y).mean() / torch.sqrt(p_var * y_var + eps)
    return 1.0 - corr.clamp(-1.0, 1.0)


# Compute local masked correlation over neighbourhoods.
def masked_local_correlation(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    pool: int = 32,
    min_valid_fraction: float = 0.40,
    min_target_std: float = 0.45,
    eps: float = 1.0e-6,
) -> torch.Tensor:
    """Average correlation loss over independent horizontal windows."""

    height = (pred.shape[-2] // pool) * pool
    width = (pred.shape[-1] // pool) * pool
    if height == 0 or width == 0:
        return pred.new_tensor(0.0)
    shape = (*pred.shape[:-2], height // pool, pool, width // pool, pool)
    p = pred[..., :height, :width].reshape(shape).permute(0, 1, 2, 3, 5, 4, 6).flatten(-2)
    y = target[..., :height, :width].reshape(shape).permute(0, 1, 2, 3, 5, 4, 6).flatten(-2)
    m = mask[..., :height, :width].reshape(shape).permute(0, 1, 2, 3, 5, 4, 6).flatten(-2)
    count = m.sum(-1)
    usable = count >= float(pool * pool) * min_valid_fraction
    count = count.clamp_min(1.0)
    p_mean = (p * m).sum(-1) / count
    y_mean = (y * m).sum(-1) / count
    pc = (p - p_mean.unsqueeze(-1)) * m
    yc = (y - y_mean.unsqueeze(-1)) * m
    p_var = pc.square().sum(-1) / count
    y_var = yc.square().sum(-1) / count
    usable &= y_var.sqrt() >= min_target_std
    if not bool(usable.any()):
        return pred.new_tensor(0.0)
    corr = (pc * yc).sum(-1) / count / torch.sqrt(p_var * y_var + eps)
    return (1.0 - corr.clamp(-1.0, 1.0))[usable].mean()


# Pool a field for low-frequency loss terms.
def pooled_field(x: torch.Tensor, mask: torch.Tensor, pool: int, min_valid_fraction: float = 0.4) -> tuple[torch.Tensor, torch.Tensor]:
    """Build low-frequency field/mask by horizontal average pooling."""

    kernel = (1, pool, pool)
    pooled_mask = F.avg_pool3d(mask, kernel_size=kernel, stride=kernel)
    pooled_valid = (pooled_mask >= float(min_valid_fraction)).to(mask.dtype)
    pooled_x = F.avg_pool3d(x * mask, kernel_size=kernel, stride=kernel) / pooled_mask.clamp_min(1.0e-6)
    full_x = F.interpolate(pooled_x, size=x.shape[-3:], mode="trilinear", align_corners=False)
    full_mask = F.interpolate(pooled_valid, size=x.shape[-3:], mode="nearest") * mask
    return full_x, full_mask


# Compute the amplitude penalty for active plume regions.
def amplitude_loss(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, eps: float = 1.0e-6) -> torch.Tensor:
    """Compare per-level horizontal standard deviation."""

    count = mask.sum(dim=(3, 4), keepdim=True).clamp_min(1.0)
    pred_mean = (pred * mask).sum(dim=(3, 4), keepdim=True) / count
    target_mean = (target * mask).sum(dim=(3, 4), keepdim=True) / count
    pred_std = torch.sqrt((((pred - pred_mean) * mask).square()).sum(dim=(3, 4), keepdim=True) / count + eps)
    target_std = torch.sqrt((((target - target_mean) * mask).square()).sum(dim=(3, 4), keepdim=True) / count + eps)
    usable = (mask.sum(dim=(3, 4), keepdim=True) > 1.0) & (target_std >= 0.45)
    if usable.sum() <= 0:
        return pred.new_tensor(0.0)
    return (torch.log(pred_std[usable] + eps) - torch.log(target_std[usable] + eps)).abs().mean()


# Compute loss on active concentration-change regions.
def active_delta_loss(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, threshold: float, delta: float) -> torch.Tensor:
    """Huber loss only where the true increment is meaningfully active."""

    active_mask = mask * (target.abs() >= float(threshold)).to(mask.dtype)
    if active_mask.sum() <= 0:
        return pred.new_tensor(0.0)
    return masked_huber(pred, target, active_mask, delta=delta)


# Penalize wrong-signed concentration changes.
def sign_margin_loss(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, min_abs: float, scale: float) -> torch.Tensor:
    """Encourage correct positive/negative sign for larger increments."""

    active_mask = mask * (target.abs() >= float(min_abs)).to(mask.dtype)
    if active_mask.sum() <= 0:
        return pred.new_tensor(0.0)
    signed_margin = pred * target / max(float(scale), 1.0e-6) ** 2
    return masked_mean(F.softplus(-signed_margin), active_mask)


# Compute auxiliary binary-cross-entropy loss.
def auxiliary_bce_loss(logit: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, pos_weight: float = 1.0) -> torch.Tensor:
    """Binary auxiliary loss for active-event and sign heads."""

    if mask.sum() <= 0:
        return logit.new_tensor(0.0)
    loss = F.binary_cross_entropy_with_logits(logit, target, reduction="none")
    class_weight = torch.where(target > 0.5, target.new_tensor(float(pos_weight)), target.new_tensor(1.0))
    return (loss * mask * class_weight).sum() / (mask * class_weight).sum().clamp_min(1.0)


# Combine all V40 loss terms into the final objective.
def v40_composite_loss(
    output: torch.Tensor | dict[str, torch.Tensor],
    target: torch.Tensor,
    mask: torch.Tensor,
    weights: V40LossWeights = V40LossWeights(),
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Readable V40 objective.

    This is the training objective used by the standalone final package.
    """

    components = output if isinstance(output, dict) else {"final": output}
    pred = components["final"]
    point_weight = residual_weight(target) * gradient_weight(target, mask)

    losses: dict[str, torch.Tensor] = {}
    losses["huber"] = masked_huber(pred, target, mask, delta=weights.huber_delta, weight=point_weight)
    losses["gradient"] = masked_gradient_l1(pred, target, mask)
    losses["correlation"] = masked_correlation(pred, target, mask)
    losses["local_correlation"] = masked_local_correlation(pred, target, mask)
    losses["amplitude"] = amplitude_loss(pred, target, mask)
    losses["active_delta"] = active_delta_loss(pred, target, mask, weights.active_threshold, weights.huber_delta)
    losses["sign"] = sign_margin_loss(pred, target, mask, weights.sign_min_abs, weights.sign_scale)

    low_pred, low_mask = pooled_field(pred, mask, pool=16)
    low_target, _ = pooled_field(target, mask, pool=16)
    losses["low_frequency"] = masked_huber(low_pred, low_target, low_mask, delta=weights.huber_delta)
    losses["low_frequency_correlation"] = masked_correlation(low_pred, low_target, low_mask)

    high_pred = pred - low_pred
    high_target = target - low_target
    losses["high_frequency"] = masked_huber(high_pred, high_target, mask, delta=0.85)

    if "active_logit" in components:
        active_target = (target.abs() >= weights.active_threshold).to(target.dtype)
        losses["active_aux"] = auxiliary_bce_loss(components["active_logit"], active_target, mask, pos_weight=2.0)
    else:
        losses["active_aux"] = pred.new_tensor(0.0)

    if "sign_logit" in components:
        sign_mask = mask * (target.abs() >= weights.sign_min_abs).to(target.dtype)
        sign_target = (target > 0).to(target.dtype)
        losses["sign_aux"] = auxiliary_bce_loss(components["sign_logit"], sign_target, sign_mask)
    else:
        losses["sign_aux"] = pred.new_tensor(0.0)

    total = (
        losses["huber"]
        + weights.gradient * losses["gradient"]
        + weights.correlation * losses["correlation"]
        + weights.local_correlation * losses["local_correlation"]
        + weights.amplitude * losses["amplitude"]
        + weights.active_delta * losses["active_delta"]
        + weights.sign * losses["sign"]
        + weights.low_frequency * losses["low_frequency"]
        + weights.low_frequency_correlation * losses["low_frequency_correlation"]
        + weights.high_frequency * losses["high_frequency"]
        + weights.active_aux * losses["active_aux"]
        + weights.sign_aux * losses["sign_aux"]
    )
    losses["total"] = total
    return total, losses

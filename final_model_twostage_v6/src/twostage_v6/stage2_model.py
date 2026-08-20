from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from .unet_blocks import (
    GlobalContextEncoder3D,
    ResidualBlock3D,
    SameConv3d,
    V7ConvBlock3D,
    V7DownBlock3D,
    V7UpBlock3D,
)


# Final V40 model combining local, event-texture, and global-context branches.
class V40EventTextureContextUNet3D(nn.Module):
    """Final V40 model used for Camden autoregressive CO2 increment prediction.

    Local branch:
        3D U-Net over a 256 x 256 local patch.

    Context branch:
        full-domain low-resolution context is encoded and sampled back onto the
        local patch. A context-correction head learns broad low-frequency
        increment corrections.

    Event-texture branch:
        decoder features and aligned context predict a high-frequency residual,
        an active-event logit and a sign logit. The final increment is

            pred_delta = local_delta + context_delta
                         + gate(active_logit) * high_delta_raw

        where gate = min_high_gate + (1 - min_high_gate) * sigmoid(active_logit).
    """

    # Build the neural-network layers for this module.
    def __init__(
        self,
        in_channels: int = 18,
        out_channels: int = 1,
        base_channels: int = 32,
        global_channels: int = 9,
        global_feature_channels: int = 8,
        context_correction_scale: float = 1.0,
        high_delta_scale: float = 1.0,
        min_high_gate: float = 0.20,
    ) -> None:
        super().__init__()
        if out_channels != 1:
            raise ValueError("V40EventTextureContextUNet3D supports out_channels=1")

        self.in_channels = int(in_channels)
        self.global_channels = int(global_channels)
        self.global_feature_channels = int(global_feature_channels) if self.global_channels > 0 else 0
        self.context_correction_scale = float(context_correction_scale)
        self.high_delta_scale = float(high_delta_scale)
        self.min_high_gate = float(min_high_gate)

        c1 = base_channels
        c2 = base_channels * 2
        c3 = base_channels * 4
        c4 = base_channels * 8

        self.enc1 = V7ConvBlock3D(in_channels, c1)
        self.enc2 = V7DownBlock3D(c1, c2)
        self.enc3 = V7DownBlock3D(c2, c3)
        self.bottleneck = V7DownBlock3D(c3, c4)
        self.dec3 = V7UpBlock3D(c4, c3, c3)
        self.dec2 = V7UpBlock3D(c3, c2, c2)
        self.dec1 = V7UpBlock3D(c2, c1, c1)
        self.head = nn.Conv3d(c1, out_channels, kernel_size=1)

        self.global_encoder = (
            GlobalContextEncoder3D(self.global_channels, self.global_feature_channels)
            if self.global_channels > 0
            else None
        )

        if self.global_encoder is not None:
            hidden = max(8, base_channels // 2)
            self.context_head = nn.Sequential(
                SameConv3d(in_channels + self.global_feature_channels, hidden, kernel_size=3),
                nn.SiLU(inplace=True),
                ResidualBlock3D(hidden, dilation=2),
                SameConv3d(hidden, out_channels, kernel_size=1),
            )
        else:
            self.context_head = None

        aux_in = base_channels + (self.global_feature_channels if self.global_channels > 0 else 0)
        hidden = max(8, base_channels // 2)
        self.active_head = nn.Sequential(
            SameConv3d(aux_in, hidden, kernel_size=3),
            nn.SiLU(inplace=True),
            ResidualBlock3D(hidden, dilation=2),
            SameConv3d(hidden, 1, kernel_size=1),
        )
        self.sign_head = nn.Sequential(
            SameConv3d(aux_in, hidden, kernel_size=3),
            nn.SiLU(inplace=True),
            ResidualBlock3D(hidden, dilation=2),
            SameConv3d(hidden, 1, kernel_size=1),
        )
        self.high_head = nn.Sequential(
            SameConv3d(aux_in, hidden, kernel_size=3),
            nn.SiLU(inplace=True),
            ResidualBlock3D(hidden, dilation=1),
            ResidualBlock3D(hidden, dilation=3),
            SameConv3d(hidden, out_channels, kernel_size=1),
        )

    # Sample full-domain context features at local patch coordinates.
    def _global_local_features(
        self,
        x: torch.Tensor,
        global_context: torch.Tensor | None,
        global_grid: torch.Tensor | None,
    ) -> torch.Tensor | None:
        if self.global_encoder is None:
            return None
        if global_context is None:
            raise ValueError("global_context is required when global_channels > 0")

        features = self.global_encoder(global_context)
        if global_grid is not None:
            features = F.grid_sample(
                features,
                global_grid,
                mode="bilinear",
                padding_mode="border",
                align_corners=True,
            )
        else:
            pooled = F.adaptive_avg_pool3d(features, output_size=1)
            features = pooled.expand(-1, -1, *x.shape[-3:])

        if features.shape[-3:] != x.shape[-3:]:
            features = F.interpolate(features, size=x.shape[-3:], mode="trilinear", align_corners=False)
        return features

    # Run the forward pass for this module.
    def forward(
        self,
        x: torch.Tensor,
        global_context: torch.Tensor | None = None,
        global_grid: torch.Tensor | None = None,
        return_components: bool = False,
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        s1 = self.enc1(x)
        s2 = self.enc2(s1)
        s3 = self.enc3(s2)
        bottleneck = self.bottleneck(s3)
        dec = self.dec3(bottleneck, s3)
        dec = self.dec2(dec, s2)
        dec = self.dec1(dec, s1)

        local_delta = self.head(dec)
        global_features = self._global_local_features(x, global_context, global_grid)

        if global_features is None or self.context_head is None:
            context_delta = torch.zeros_like(local_delta)
            aux_input = dec
        else:
            context_input = torch.cat((x, global_features), dim=1)
            context_delta = self.context_head(context_input) * self.context_correction_scale
            aux_input = torch.cat((dec, global_features), dim=1)

        low_delta = local_delta + context_delta
        active_logit = self.active_head(aux_input)
        sign_logit = self.sign_head(aux_input)
        learned_gate = torch.sigmoid(active_logit)
        active_gate = self.min_high_gate + (1.0 - self.min_high_gate) * learned_gate
        high_raw = self.high_head(aux_input) * self.high_delta_scale
        high_delta = active_gate * high_raw
        final_delta = low_delta + high_delta

        if return_components:
            return {
                "final": final_delta,
                "local": local_delta,
                "context": context_delta,
                "low": low_delta,
                "high": high_delta,
                "high_raw": high_raw,
                "active_gate": active_gate,
                "learned_active_gate": learned_gate,
                "active_logit": active_logit,
                "sign_logit": sign_logit,
            }
        return final_delta


# Backward-compatible alias for the historical checkpoint name.
V38EventTextureContextV7UNet3D = V40EventTextureContextUNet3D


# Create the final V40 model from a configuration dictionary.
def build_final_model(**overrides: Any) -> V40EventTextureContextUNet3D:
    config = {
        "in_channels": 18,
        "out_channels": 1,
        "base_channels": 32,
        "global_channels": 9,
        "global_feature_channels": 8,
        "context_correction_scale": 1.0,
        "high_delta_scale": 1.0,
        "min_high_gate": 0.20,
    }
    config.update(overrides)
    return V40EventTextureContextUNet3D(**config)


# Count trainable parameters for logging and checkpoint checks.
def count_trainable_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)

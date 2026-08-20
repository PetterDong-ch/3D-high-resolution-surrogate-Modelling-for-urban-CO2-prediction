from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


# Basic two-convolution block for the local 3D U-Net path.
class LocalConvBlock3D(nn.Module):
    """Two 3D convolutions used by the local U-Net backbone."""

    # Build the neural-network layers for this module.
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(out_channels),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(out_channels),
            nn.LeakyReLU(0.1, inplace=True),
        )

    # Run the forward pass for this module.
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


# Downsamples the local feature volume in the U-Net encoder.
class LocalDownBlock3D(nn.Module):
    """Downsample one U-Net level, then extract features at the coarser scale."""

    # Build the neural-network layers for this module.
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)
        self.conv = LocalConvBlock3D(in_channels, out_channels)

    # Run the forward pass for this module.
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(self.pool(x))


# Upsamples decoder features and merges them with the matching skip connection.
class LocalUpBlock3D(nn.Module):
    """Upsample one decoder level and fuse it with the matching skip feature."""

    # Build the neural-network layers for this module.
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int) -> None:
        super().__init__()
        self.up = nn.ConvTranspose3d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = LocalConvBlock3D(out_channels + skip_channels, out_channels)

    # Run the forward pass for this module.
    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        dz = skip.size(2) - x.size(2)
        dy = skip.size(3) - x.size(3)
        dx = skip.size(4) - x.size(4)
        if dz != 0 or dy != 0 or dx != 0:
            skip = skip[
                :,
                :,
                dz // 2 : skip.size(2) - (dz - dz // 2),
                dy // 2 : skip.size(3) - (dy - dy // 2),
                dx // 2 : skip.size(4) - (dx - dx // 2),
            ]
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


# Replication-padded 3D convolution that preserves spatial size.
class SameConv3d(nn.Module):
    """Replication-padded 3D convolution with preserved spatial shape."""

    # Build the neural-network layers for this module.
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        dilation: int = 1,
        bias: bool = True,
    ) -> None:
        super().__init__()
        padding = dilation * (kernel_size // 2)
        self.pad = nn.ReplicationPad3d(padding)
        self.conv = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            bias=bias,
        )

    # Run the forward pass for this module.
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(self.pad(x))


# Residual 3D block used where normalization would remove low-frequency signal.
class ResidualBlock3D(nn.Module):
    """Small residual block without normalization.

    The context and auxiliary heads use this block because per-sample
    normalization can erase low-frequency concentration gradients.
    """

    # Build the neural-network layers for this module.
    def __init__(self, channels: int, dilation: int = 1) -> None:
        super().__init__()
        self.conv1 = SameConv3d(channels, channels, kernel_size=3, dilation=dilation, bias=True)
        self.conv2 = SameConv3d(channels, channels, kernel_size=3, dilation=dilation, bias=True)
        self.act = nn.SiLU(inplace=True)

    # Run the forward pass for this module.
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.act(self.conv1(x))
        x = self.conv2(x)
        return self.act(x + residual)


# Small residual convolution block used by context and auxiliary heads.
class ConvBlock3D(nn.Module):
    """No-normalization block used by the full-domain context encoder."""

    # Build the neural-network layers for this module.
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.proj = SameConv3d(in_channels, out_channels, kernel_size=3, bias=True)
        self.block1 = ResidualBlock3D(out_channels)
        self.block2 = ResidualBlock3D(out_channels)
        self.act = nn.SiLU(inplace=True)

    # Run the forward pass for this module.
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act(self.proj(x))
        x = self.block1(x)
        return self.block2(x)


# Encodes coarse full-domain context into local-resolution features.
class GlobalContextEncoder3D(nn.Module):
    """Encode low-resolution full-domain fields into local context features."""

    # Build the neural-network layers for this module.
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            ConvBlock3D(in_channels, out_channels),
            ResidualBlock3D(out_channels, dilation=2),
            ResidualBlock3D(out_channels, dilation=4),
        )

    # Run the forward pass for this module.
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


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

        self.enc1 = LocalConvBlock3D(in_channels, c1)
        self.enc2 = LocalDownBlock3D(c1, c2)
        self.enc3 = LocalDownBlock3D(c2, c3)
        self.bottleneck = LocalDownBlock3D(c3, c4)
        self.dec3 = LocalUpBlock3D(c4, c3, c3)
        self.dec2 = LocalUpBlock3D(c3, c2, c2)
        self.dec1 = LocalUpBlock3D(c2, c1, c1)
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


# Create the final V40 model from a configuration dictionary.
def build_final_model(**overrides: Any) -> V40EventTextureContextUNet3D:
    """Build the final V40 model, with optional overrides for experiments."""

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
    """Count trainable parameters for README/report sanity checks."""

    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)

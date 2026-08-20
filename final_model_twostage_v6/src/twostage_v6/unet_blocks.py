from __future__ import annotations

import torch
import torch.nn as nn


# Model component used by the V7conv block3d architecture.
class V7ConvBlock3D(nn.Module):
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


# Model component used by the V7down block3d architecture.
class V7DownBlock3D(nn.Module):
    # Build the neural-network layers for this module.
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)
        self.conv = V7ConvBlock3D(in_channels, out_channels)

    # Run the forward pass for this module.
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(self.pool(x))


# Model component used by the V7up block3d architecture.
class V7UpBlock3D(nn.Module):
    # Build the neural-network layers for this module.
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int) -> None:
        super().__init__()
        self.up = nn.ConvTranspose3d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = V7ConvBlock3D(out_channels + skip_channels, out_channels)

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

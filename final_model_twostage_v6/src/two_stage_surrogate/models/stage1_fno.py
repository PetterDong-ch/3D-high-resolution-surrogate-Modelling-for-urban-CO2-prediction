from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn
import torch.nn.functional as F


# Stores Stage 1 FNO model dimensions and hyperparameters.
@dataclass(frozen=True)
class Stage1ModelConfig:
    geometry_channels: int
    surface_channels: int
    profile_channels: int
    scalar_channels: int
    width: int = 32
    depth: int = 4
    modes_z: int = 8
    modes_y: int = 16
    modes_x: int = 16
    predict_w: bool = True
    output_pressure: bool = False


# Small feed-forward network used for conditioning vectors.
class MLP(nn.Module):
    # Store constructor arguments and initialize object state.
    def __init__(self, in_features: int, hidden_features: int, out_features: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_features),
            nn.GELU(),
            nn.Linear(hidden_features, out_features),
        )

    # Run the forward pass for this module.
    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


# Encodes vertical forcing profiles for Stage 1 conditioning.
class ProfileEncoder(nn.Module):
    # Build the neural-network layers for this module.
    def __init__(self, in_channels: int, width: int) -> None:
        super().__init__()
        if in_channels <= 0:
            self.encoder = None
            self.out_features = width
        else:
            self.encoder = nn.Sequential(
                nn.Conv1d(in_channels, width, kernel_size=3, padding=1),
                nn.GELU(),
                nn.Conv1d(width, width, kernel_size=3, padding=1),
                nn.GELU(),
                nn.AdaptiveAvgPool1d(1),
            )
            self.out_features = width

    # Run the forward pass for this module.
    def forward(self, x: Tensor | None, batch_size: int, device: torch.device) -> Tensor:
        if self.encoder is None:
            return torch.zeros(batch_size, self.out_features, device=device)
        if x is None:
            raise ValueError("profile input is required by this model")
        return self.encoder(x).squeeze(-1)


# Encodes static 2D surface fields for Stage 1 conditioning.
class SurfaceEncoder2D(nn.Module):
    # Build the neural-network layers for this module.
    def __init__(self, in_channels: int, width: int) -> None:
        super().__init__()
        self.in_channels = in_channels
        if in_channels <= 0:
            self.net = None
            self.out_channels = 0
        else:
            self.net = nn.Sequential(
                nn.Conv2d(in_channels, width, kernel_size=3, padding=1),
                nn.GELU(),
                nn.Conv2d(width, width, kernel_size=3, padding=1),
                nn.GELU(),
            )
            self.out_channels = width

    # Run the forward pass for this module.
    def forward(self, x: Tensor | None, z_size: int) -> Tensor | None:
        if self.net is None:
            return None
        if x is None:
            raise ValueError("surface_2d input is required by this model")
        encoded = self.net(x)
        return encoded.unsqueeze(2).expand(-1, -1, z_size, -1, -1)


# 3D Fourier convolution layer used inside the Stage 1 FNO blocks.
class SpectralConv3d(nn.Module):
    # Build the neural-network layers for this module.
    def __init__(self, channels: int, modes_z: int, modes_y: int, modes_x: int) -> None:
        super().__init__()
        self.channels = channels
        self.modes_z = modes_z
        self.modes_y = modes_y
        self.modes_x = modes_x
        scale = 1.0 / max(1, channels * channels)
        self.weight = nn.Parameter(
            scale
            * torch.randn(
                channels,
                channels,
                modes_z,
                modes_y,
                modes_x,
                dtype=torch.cfloat,
            )
        )

    # Run the forward pass for this module.
    def forward(self, x: Tensor) -> Tensor:
        batch, channels, depth, height, width = x.shape
        x_ft = torch.fft.rfftn(x, dim=(-3, -2, -1))
        out_ft = torch.zeros(
            batch,
            channels,
            depth,
            height,
            width // 2 + 1,
            dtype=torch.cfloat,
            device=x.device,
        )

        mz = min(self.modes_z, depth)
        my = min(self.modes_y, height)
        mx = min(self.modes_x, width // 2 + 1)
        out_ft[:, :, :mz, :my, :mx] = torch.einsum(
            "bixyz,ioxyz->boxyz",
            x_ft[:, :, :mz, :my, :mx],
            self.weight[:, :, :mz, :my, :mx],
        )
        return torch.fft.irfftn(out_ft, s=(depth, height, width), dim=(-3, -2, -1))


# Model component used by the Fno block3d architecture.
class FNOBlock3d(nn.Module):
    # Build the neural-network layers for this module.
    def __init__(self, width: int, modes_z: int, modes_y: int, modes_x: int, cond_width: int) -> None:
        super().__init__()
        self.spectral = SpectralConv3d(width, modes_z, modes_y, modes_x)
        self.pointwise = nn.Conv3d(width, width, kernel_size=1)
        self.norm = nn.InstanceNorm3d(width, affine=True)
        self.film = nn.Linear(cond_width, 2 * width)

    # Run the forward pass for this module.
    def forward(self, x: Tensor, cond: Tensor) -> Tensor:
        scale, shift = self.film(cond).chunk(2, dim=1)
        scale = scale[..., None, None, None]
        shift = shift[..., None, None, None]
        y = self.spectral(x) + self.pointwise(x)
        y = self.norm(y)
        y = y * (1.0 + scale) + shift
        return F.gelu(y)


# Model component used by the Local FNO stage1 architecture.
class LocalFNOStage1(nn.Module):
    """Patch-based Stage 1 surrogate for PALM high-resolution microclimate fields.

    Inputs are intentionally separated by information structure:
    - geometry_3d: true voxel fields such as fluid/building masks and coordinates.
    - surface_2d: surface descriptors encoded in 2D, then lifted to the target height.
    - profile: vertical forcing profiles encoded by a 1D network.
    - scalar: sample-level descriptors encoded by an MLP.
    """

    # Build the neural-network layers for this module.
    def __init__(self, config: Stage1ModelConfig | dict[str, Any]) -> None:
        super().__init__()
        if isinstance(config, dict):
            config = Stage1ModelConfig(**config)
        self.config = config

        self.surface_encoder = SurfaceEncoder2D(config.surface_channels, config.width)
        local_in_channels = config.geometry_channels + self.surface_encoder.out_channels
        if local_in_channels <= 0:
            raise ValueError("Stage 1 needs at least one geometry or surface channel")

        self.input_projection = nn.Conv3d(local_in_channels, config.width, kernel_size=1)
        self.profile_encoder = ProfileEncoder(config.profile_channels, config.width)
        self.scalar_encoder = (
            MLP(config.scalar_channels, config.width, config.width)
            if config.scalar_channels > 0
            else None
        )
        self.condition_projection = nn.Sequential(
            nn.Linear(2 * config.width, config.width),
            nn.GELU(),
            nn.Linear(config.width, config.width),
        )
        self.blocks = nn.ModuleList(
            [
                FNOBlock3d(
                    config.width,
                    config.modes_z,
                    config.modes_y,
                    config.modes_x,
                    cond_width=config.width,
                )
                for _ in range(config.depth)
            ]
        )

        self.uv_head = nn.Conv3d(config.width, 2, kernel_size=1)
        self.w_head = nn.Conv3d(config.width, 1, kernel_size=1) if config.predict_w else None
        self.theta_prime_head = nn.Conv3d(config.width, 1, kernel_size=1)
        self.pressure_head = nn.Conv3d(config.width, 1, kernel_size=1) if config.output_pressure else None

    # Run the forward pass for this module.
    def forward(
        self,
        *,
        geometry_3d: Tensor,
        surface_2d: Tensor | None = None,
        profile: Tensor | None = None,
        scalar: Tensor | None = None,
        theta_reference: Tensor | None = None,
    ) -> dict[str, Tensor]:
        batch, _, z_size, _, _ = geometry_3d.shape
        local_parts = [geometry_3d]
        surface_features = self.surface_encoder(surface_2d, z_size)
        if surface_features is not None:
            local_parts.append(surface_features)
        x = torch.cat(local_parts, dim=1)
        x = self.input_projection(x)

        profile_features = self.profile_encoder(profile, batch, geometry_3d.device)
        if self.scalar_encoder is None:
            scalar_features = torch.zeros(batch, self.config.width, device=geometry_3d.device)
        else:
            if scalar is None:
                raise ValueError("scalar input is required by this model")
            scalar_features = self.scalar_encoder(scalar)
        cond = self.condition_projection(torch.cat([profile_features, scalar_features], dim=1))

        for block in self.blocks:
            x = block(x, cond)

        uv = self.uv_head(x)
        theta_prime = self.theta_prime_head(x)
        if theta_reference is None:
            theta = theta_prime
        else:
            theta = theta_reference + theta_prime

        out: dict[str, Tensor] = {
            "uv": uv,
            "u": uv[:, 0:1],
            "v": uv[:, 1:2],
            "theta_prime": theta_prime,
            "theta": theta,
        }
        if self.w_head is not None:
            out["w"] = self.w_head(x)
        if self.pressure_head is not None:
            out["p"] = self.pressure_head(x)
        return out

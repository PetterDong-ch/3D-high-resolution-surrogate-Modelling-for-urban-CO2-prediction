from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# Model component used by the V7conv block3d architecture.
class V7ConvBlock3D(nn.Module):
    # Build the neural-network layers for this module.
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(out_ch),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv3d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(out_ch),
            nn.LeakyReLU(0.1, inplace=True),
        )

    # Run the forward pass for this module.
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


# Model component used by the V7down block3d architecture.
class V7DownBlock3D(nn.Module):
    # Build the neural-network layers for this module.
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)
        self.conv = V7ConvBlock3D(in_ch, out_ch)

    # Run the forward pass for this module.
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(self.pool(x))


# Model component used by the V7up block3d architecture.
class V7UpBlock3D(nn.Module):
    # Build the neural-network layers for this module.
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int) -> None:
        super().__init__()
        self.up = nn.ConvTranspose3d(in_ch, out_ch, kernel_size=2, stride=2)
        self.conv = V7ConvBlock3D(out_ch + skip_ch, out_ch)

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


# Model component used by the V7style u-net 3D architecture.
class V7StyleUNet3D(nn.Module):
    """Direct V7-style 3D U-Net with no coarse branch or gates."""

    # Build the neural-network layers for this module.
    def __init__(
        self,
        in_channels: int,
        out_channels: int = 1,
        base_channels: int = 32,
    ) -> None:
        super().__init__()
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

    # Run the forward pass for this module.
    def forward(
        self,
        x: torch.Tensor,
        global_context: torch.Tensor | None = None,
        global_grid: torch.Tensor | None = None,
        return_components: bool = False,
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        del global_context, global_grid
        s1 = self.enc1(x)
        s2 = self.enc2(s1)
        s3 = self.enc3(s2)
        b = self.bottleneck(s3)
        x = self.dec3(b, s3)
        x = self.dec2(x, s2)
        x = self.dec1(x, s1)
        out = self.head(x)
        if return_components:
            return {"final": out}
        return out


# Replication-padded 3D convolution that preserves spatial size.
class SameConv3d(nn.Module):
    # Build the neural-network layers for this module.
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3, dilation: int = 1, bias: bool = True) -> None:
        super().__init__()
        pad = dilation * (kernel_size // 2)
        self.pad = nn.ReplicationPad3d(pad)
        self.conv = nn.Conv3d(in_ch, out_ch, kernel_size=kernel_size, dilation=dilation, bias=bias)

    # Run the forward pass for this module.
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(self.pad(x))


# Residual 3D block used where normalization would remove low-frequency signal.
class ResidualBlock3D(nn.Module):
    """Small no-normalization residual block.

    InstanceNorm was removed because this is an absolute CO2 regression task
    and per-sample spatial normalization can erase low-frequency concentration
    gradients.
    """

    # Build the neural-network layers for this module.
    def __init__(self, ch: int, dilation: int = 1) -> None:
        super().__init__()
        self.conv1 = SameConv3d(ch, ch, kernel_size=3, dilation=dilation, bias=True)
        self.conv2 = SameConv3d(ch, ch, kernel_size=3, dilation=dilation, bias=True)
        self.act = nn.SiLU(inplace=True)

    # Run the forward pass for this module.
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.act(self.conv1(x))
        x = self.conv2(x)
        return self.act(x + residual)


# Small residual convolution block used by context and auxiliary heads.
class ConvBlock3D(nn.Module):
    # Build the neural-network layers for this module.
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.proj = SameConv3d(in_ch, out_ch, kernel_size=3, bias=True)
        self.block1 = ResidualBlock3D(out_ch)
        self.block2 = ResidualBlock3D(out_ch)
        self.act = nn.SiLU(inplace=True)

    # Run the forward pass for this module.
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act(self.proj(x))
        x = self.block1(x)
        return self.block2(x)


# Model component used by the Down block3d architecture.
class DownBlock3D(nn.Module):
    # Build the neural-network layers for this module.
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.pool = nn.AvgPool3d(kernel_size=2, stride=2)
        self.conv = ConvBlock3D(in_ch, out_ch)

    # Run the forward pass for this module.
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(self.pool(x))


# Model component used by the Up block3d architecture.
class UpBlock3D(nn.Module):
    # Build the neural-network layers for this module.
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int) -> None:
        super().__init__()
        self.reduce = SameConv3d(in_ch, out_ch, kernel_size=1)
        self.conv = ConvBlock3D(out_ch + skip_ch, out_ch)

    # Run the forward pass for this module.
    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[-3:], mode="trilinear", align_corners=False)
        x = self.reduce(x)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


# Model component used by the Local u-net 3D architecture.
class LocalUNet3D(nn.Module):
    # Build the neural-network layers for this module.
    def __init__(self, in_channels: int, base_channels: int) -> None:
        super().__init__()
        c1 = base_channels
        c2 = base_channels * 2
        c3 = base_channels * 4
        c4 = base_channels * 8

        self.enc1 = ConvBlock3D(in_channels, c1)
        self.enc2 = DownBlock3D(c1, c2)
        self.enc3 = DownBlock3D(c2, c3)
        self.bottleneck = DownBlock3D(c3, c4)

        self.dec3 = UpBlock3D(c4, c3, c3)
        self.dec2 = UpBlock3D(c3, c2, c2)
        self.dec1 = UpBlock3D(c2, c1, c1)
        self.head = SameConv3d(c1, 1, kernel_size=1)
        nn.init.normal_(self.head.conv.weight, mean=0.0, std=1e-4)
        nn.init.zeros_(self.head.conv.bias)

    # Run the forward pass for this module.
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s1 = self.enc1(x)
        s2 = self.enc2(s1)
        s3 = self.enc3(s2)
        b = self.bottleneck(s3)
        x = self.dec3(b, s3)
        x = self.dec2(x, s2)
        x = self.dec1(x, s1)
        return self.head(x)


# Coarse 3D branch used to capture low-resolution context.
class CoarseBranch3D(nn.Module):
    """Low-resolution context branch for broad transport patterns."""

    # Store constructor arguments and initialize object state.
    def __init__(self, in_channels: int, base_channels: int, spatial_pool: int = 4) -> None:
        super().__init__()
        self.spatial_pool = spatial_pool
        self.in_proj = ConvBlock3D(in_channels, base_channels)
        self.context = nn.Sequential(
            ResidualBlock3D(base_channels, dilation=1),
            ResidualBlock3D(base_channels, dilation=2),
            ResidualBlock3D(base_channels, dilation=4),
            ResidualBlock3D(base_channels, dilation=8),
        )
        self.head = SameConv3d(base_channels, 1, kernel_size=1)
        nn.init.normal_(self.head.conv.weight, mean=0.0, std=1e-4)
        nn.init.zeros_(self.head.conv.bias)

    # Run the forward pass for this module.
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        target_size = x.shape[-3:]
        if self.spatial_pool > 1:
            x = F.avg_pool3d(x, kernel_size=(1, self.spatial_pool, self.spatial_pool), stride=(1, self.spatial_pool, self.spatial_pool))
        x = self.in_proj(x)
        x = self.context(x)
        x = self.head(x)
        if x.shape[-3:] != target_size:
            x = F.interpolate(x, size=target_size, mode="trilinear", align_corners=False)
        return x


# Encodes coarse full-domain context into local-resolution features.
class GlobalContextEncoder3D(nn.Module):
    """Encode full-domain low-resolution context into local features."""

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


# Model component used by the V20context v7 u-net 3D architecture.
class V20ContextV7UNet3D(V7StyleUNet3D):
    """V19 direct U-Net backbone plus a current-time global context correction.

    The V7 backbone keeps the same parameter names as V19, so a V19 checkpoint
    can initialize the local model. The context head is zero-initialized, making
    the model start as the pretrained V19 prediction and learn only a correction
    from the full-domain low-resolution fields.
    """

    # Build the neural-network layers for this module.
    def __init__(
        self,
        in_channels: int,
        out_channels: int = 1,
        base_channels: int = 32,
        global_channels: int = 0,
        global_feature_channels: int = 8,
        context_correction_scale: float = 1.0,
    ) -> None:
        super().__init__(in_channels=in_channels, out_channels=out_channels, base_channels=base_channels)
        self.global_channels = int(global_channels)
        self.global_feature_channels = int(global_feature_channels) if self.global_channels > 0 else 0
        self.context_correction_scale = float(context_correction_scale)
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
            nn.init.zeros_(self.context_head[-1].conv.weight)
            nn.init.zeros_(self.context_head[-1].conv.bias)
        else:
            self.context_head = None

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
        gfeat = self.global_encoder(global_context)
        if global_grid is not None:
            gfeat = F.grid_sample(
                gfeat,
                global_grid,
                mode="bilinear",
                padding_mode="border",
                align_corners=True,
            )
        else:
            pooled = F.adaptive_avg_pool3d(gfeat, output_size=1)
            gfeat = pooled.expand(-1, -1, *x.shape[-3:])
        if gfeat.shape[-3:] != x.shape[-3:]:
            gfeat = F.interpolate(gfeat, size=x.shape[-3:], mode="trilinear", align_corners=False)
        return gfeat

    # Run the forward pass for this module.
    def forward(
        self,
        x: torch.Tensor,
        global_context: torch.Tensor | None = None,
        global_grid: torch.Tensor | None = None,
        return_components: bool = False,
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        local_pred = super().forward(x, return_components=False)
        global_features = self._global_local_features(x, global_context, global_grid)
        if global_features is None or self.context_head is None:
            context_correction = torch.zeros_like(local_pred)
        else:
            context_input = torch.cat((x, global_features), dim=1)
            context_correction = self.context_head(context_input) * self.context_correction_scale
        final = local_pred + context_correction
        if return_components:
            return {
                "final": final,
                "local": local_pred,
                "context": context_correction,
            }
        return final


# Model component used by the V3 5multi task context v7 u-net 3D architecture.
class V35MultiTaskContextV7UNet3D(V20ContextV7UNet3D):
    """Context V7 model with auxiliary active/sign delta heads.

    The main `final` output remains the raw delta regression target. The
    auxiliary heads are used only during training/evaluation diagnostics:

    - `active_logit`: whether |delta| is meaningfully non-zero.
    - `sign_logit`: whether delta is positive on active cells.
    """

    # Build the neural-network layers for this module.
    def __init__(
        self,
        in_channels: int,
        out_channels: int = 1,
        base_channels: int = 32,
        global_channels: int = 0,
        global_feature_channels: int = 8,
        context_correction_scale: float = 1.0,
    ) -> None:
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            base_channels=base_channels,
            global_channels=global_channels,
            global_feature_channels=global_feature_channels,
            context_correction_scale=context_correction_scale,
        )
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
        for head in (self.active_head, self.sign_head):
            nn.init.zeros_(head[-1].conv.weight)
            nn.init.zeros_(head[-1].conv.bias)

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
        b = self.bottleneck(s3)
        dec = self.dec3(b, s3)
        dec = self.dec2(dec, s2)
        dec = self.dec1(dec, s1)
        local_pred = self.head(dec)

        global_features = self._global_local_features(x, global_context, global_grid)
        if global_features is None or self.context_head is None:
            context_correction = torch.zeros_like(local_pred)
            aux_input = dec
        else:
            context_input = torch.cat((x, global_features), dim=1)
            context_correction = self.context_head(context_input) * self.context_correction_scale
            aux_input = torch.cat((dec, global_features), dim=1)

        final = local_pred + context_correction
        active_logit = self.active_head(aux_input)
        sign_logit = self.sign_head(aux_input)

        if return_components:
            return {
                "final": final,
                "local": local_pred,
                "context": context_correction,
                "active_logit": active_logit,
                "sign_logit": sign_logit,
            }
        return final


# Model component used by the V3 7hard pattern context v7 u-net 3D architecture.
class V37HardPatternContextV7UNet3D(V35MultiTaskContextV7UNet3D):
    """V35 context model with an explicit low/high delta decomposition.

    V35 used active/sign heads only as auxiliary classifiers. V37 lets the
    active head participate in the final regression:

        final_delta = low_delta + sigmoid(active_logit) * high_delta

    The low branch is initialized exactly like V35 (`local + context`), while
    the high branch is zero-initialized. Loading a V35 checkpoint with
    `strict=False` therefore starts from the V35 prediction and only learns a
    hard-case texture correction.
    """

    # Build the neural-network layers for this module.
    def __init__(
        self,
        in_channels: int,
        out_channels: int = 1,
        base_channels: int = 32,
        global_channels: int = 0,
        global_feature_channels: int = 8,
        context_correction_scale: float = 1.0,
        high_delta_scale: float = 1.0,
    ) -> None:
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            base_channels=base_channels,
            global_channels=global_channels,
            global_feature_channels=global_feature_channels,
            context_correction_scale=context_correction_scale,
        )
        aux_in = base_channels + (self.global_feature_channels if self.global_channels > 0 else 0)
        hidden = max(8, base_channels // 2)
        self.high_delta_scale = float(high_delta_scale)
        self.high_head = nn.Sequential(
            SameConv3d(aux_in, hidden, kernel_size=3),
            nn.SiLU(inplace=True),
            ResidualBlock3D(hidden, dilation=2),
            SameConv3d(hidden, out_channels, kernel_size=1),
        )
        nn.init.zeros_(self.high_head[-1].conv.weight)
        nn.init.zeros_(self.high_head[-1].conv.bias)

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
        b = self.bottleneck(s3)
        dec = self.dec3(b, s3)
        dec = self.dec2(dec, s2)
        dec = self.dec1(dec, s1)
        local_pred = self.head(dec)

        global_features = self._global_local_features(x, global_context, global_grid)
        if global_features is None or self.context_head is None:
            context_correction = torch.zeros_like(local_pred)
            aux_input = dec
        else:
            context_input = torch.cat((x, global_features), dim=1)
            context_correction = self.context_head(context_input) * self.context_correction_scale
            aux_input = torch.cat((dec, global_features), dim=1)

        low_delta = local_pred + context_correction
        active_logit = self.active_head(aux_input)
        sign_logit = self.sign_head(aux_input)
        gate = torch.sigmoid(active_logit)
        high_raw = self.high_head(aux_input) * self.high_delta_scale
        high_delta = gate * high_raw
        final = low_delta + high_delta

        if return_components:
            return {
                "final": final,
                "local": local_pred,
                "context": context_correction,
                "low": low_delta,
                "high": high_delta,
                "high_raw": high_raw,
                "active_gate": gate,
                "active_logit": active_logit,
                "sign_logit": sign_logit,
            }
        return final


# Model component used by the V3 8event texture context v7 u-net 3D architecture.
class V38EventTextureContextV7UNet3D(V37HardPatternContextV7UNet3D):
    """V37 with a less suppressive event-texture correction branch.

    V37 improved hard delta cases, but the high-frequency branch can still be
    muted when the learned active gate stays small. V38 keeps the same low/high
    decomposition while making two targeted changes:

    - the high branch has a slightly larger local/dilated receptive field;
    - the active gate has a configurable floor, so hard texture corrections are
      not forced all the way to zero early in fine-tuning.
    """

    # Build the neural-network layers for this module.
    def __init__(
        self,
        in_channels: int,
        out_channels: int = 1,
        base_channels: int = 32,
        global_channels: int = 0,
        global_feature_channels: int = 8,
        context_correction_scale: float = 1.0,
        high_delta_scale: float = 1.0,
        min_high_gate: float = 0.20,
    ) -> None:
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            base_channels=base_channels,
            global_channels=global_channels,
            global_feature_channels=global_feature_channels,
            context_correction_scale=context_correction_scale,
            high_delta_scale=high_delta_scale,
        )
        aux_in = base_channels + (self.global_feature_channels if self.global_channels > 0 else 0)
        hidden = max(8, base_channels // 2)
        self.min_high_gate = float(min_high_gate)
        self.high_head = nn.Sequential(
            SameConv3d(aux_in, hidden, kernel_size=3),
            nn.SiLU(inplace=True),
            ResidualBlock3D(hidden, dilation=1),
            ResidualBlock3D(hidden, dilation=3),
            SameConv3d(hidden, out_channels, kernel_size=1),
        )
        nn.init.zeros_(self.high_head[-1].conv.weight)
        nn.init.zeros_(self.high_head[-1].conv.bias)

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
        b = self.bottleneck(s3)
        dec = self.dec3(b, s3)
        dec = self.dec2(dec, s2)
        dec = self.dec1(dec, s1)
        local_pred = self.head(dec)

        global_features = self._global_local_features(x, global_context, global_grid)
        if global_features is None or self.context_head is None:
            context_correction = torch.zeros_like(local_pred)
            aux_input = dec
        else:
            context_input = torch.cat((x, global_features), dim=1)
            context_correction = self.context_head(context_input) * self.context_correction_scale
            aux_input = torch.cat((dec, global_features), dim=1)

        low_delta = local_pred + context_correction
        active_logit = self.active_head(aux_input)
        sign_logit = self.sign_head(aux_input)
        learned_gate = torch.sigmoid(active_logit)
        gate = self.min_high_gate + (1.0 - self.min_high_gate) * learned_gate
        high_raw = self.high_head(aux_input) * self.high_delta_scale
        high_delta = gate * high_raw
        final = low_delta + high_delta

        if return_components:
            return {
                "final": final,
                "local": local_pred,
                "context": context_correction,
                "low": low_delta,
                "high": high_delta,
                "high_raw": high_raw,
                "active_gate": gate,
                "learned_active_gate": learned_gate,
                "active_logit": active_logit,
                "sign_logit": sign_logit,
            }
        return final


# Learns a gate for texture/detail corrections.
class TextureGate3D(nn.Module):
    """Predict a soft gate for local residual corrections."""

    # Store constructor arguments and initialize object state.
    def __init__(self, in_channels: int, hidden_channels: int) -> None:
        super().__init__()
        hidden = max(8, hidden_channels)
        self.net = nn.Sequential(
            SameConv3d(in_channels, hidden, kernel_size=3),
            nn.SiLU(inplace=True),
            SameConv3d(hidden, hidden, kernel_size=3),
            nn.SiLU(inplace=True),
            SameConv3d(hidden, 1, kernel_size=1),
        )
        nn.init.zeros_(self.net[-1].conv.weight)
        nn.init.zeros_(self.net[-1].conv.bias)

    # Run the forward pass for this module.
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.net(x))


# Model component used by the U-net 3D architecture.
class UNet3D(nn.Module):
    """V13 coarse-to-fine residual CO2 model.

    The model predicts corrected residual CO2 relative to the time-aligned
    ls_forcing_right_CO2 background. It splits the task into two explicitly
    supervised pieces:

        final_residual = low_residual + height_gate * local_high_residual

    If full-domain low-resolution context is provided, it is encoded and sampled
    onto the local tile before both branches. The coarse branch is then allowed
    to learn broad plume/transport trends that are invisible inside a single
    256x256 tile.
    """

    # Build the neural-network layers for this module.
    def __init__(
        self,
        in_channels: int,
        out_channels: int = 1,
        base_channels: int = 24,
        bg_channel_idx: int | None = 1,
        gate_channel_idx: int | None = None,
        coarse_pool: int = 8,
        min_texture_gate: float = 0.05,
        learned_texture_gate: bool = True,
        high_residual_scale: float = 1.0,
        global_channels: int = 0,
        global_feature_channels: int = 8,
    ) -> None:
        super().__init__()
        if out_channels != 1:
            raise ValueError("V13 UNet3D currently supports out_channels=1")
        self.in_channels = in_channels
        self.bg_channel_idx = bg_channel_idx
        self.gate_channel_idx = gate_channel_idx
        self.min_texture_gate = float(min_texture_gate)
        self.learned_texture_gate = bool(learned_texture_gate)
        self.high_residual_scale = float(high_residual_scale)
        self.global_channels = int(global_channels)
        self.global_feature_channels = int(global_feature_channels) if self.global_channels > 0 else 0
        self.global_encoder = (
            GlobalContextEncoder3D(self.global_channels, self.global_feature_channels)
            if self.global_channels > 0
            else None
        )
        model_in_channels = in_channels + self.global_feature_channels
        self.coarse = CoarseBranch3D(model_in_channels, base_channels=base_channels, spatial_pool=coarse_pool)
        self.local = LocalUNet3D(model_in_channels + 1, base_channels=base_channels)
        self.texture_gate = TextureGate3D(model_in_channels, hidden_channels=max(8, base_channels // 2))

    # Internal helper for background.
    def _background(self, x: torch.Tensor) -> torch.Tensor:
        if self.bg_channel_idx is None or self.bg_channel_idx < 0 or self.bg_channel_idx >= x.shape[1]:
            return torch.zeros((x.shape[0], 1, *x.shape[-3:]), dtype=x.dtype, device=x.device)
        return x[:, self.bg_channel_idx : self.bg_channel_idx + 1]

    # Internal helper for height gate.
    def _height_gate(self, x: torch.Tensor) -> torch.Tensor:
        if self.gate_channel_idx is not None and 0 <= self.gate_channel_idx < x.shape[1]:
            return x[:, self.gate_channel_idx : self.gate_channel_idx + 1].clamp(0.0, 1.0)
        return torch.ones((x.shape[0], 1, *x.shape[-3:]), dtype=x.dtype, device=x.device)

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
        gfeat = self.global_encoder(global_context)
        if global_grid is not None:
            gfeat = F.grid_sample(
                gfeat,
                global_grid,
                mode="bilinear",
                padding_mode="border",
                align_corners=True,
            )
        else:
            pooled = F.adaptive_avg_pool3d(gfeat, output_size=1)
            gfeat = pooled.expand(-1, -1, *x.shape[-3:])
        if gfeat.shape[-3:] != x.shape[-3:]:
            gfeat = F.interpolate(gfeat, size=x.shape[-3:], mode="trilinear", align_corners=False)
        return gfeat

    # Run the forward pass for this module.
    def forward(
        self,
        x: torch.Tensor,
        global_context: torch.Tensor | None = None,
        global_grid: torch.Tensor | None = None,
        return_components: bool = False,
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        height_gate = self._height_gate(x)
        global_features = self._global_local_features(x, global_context, global_grid)
        model_x = torch.cat((x, global_features), dim=1) if global_features is not None else x
        if self.learned_texture_gate:
            learned_gate = self.texture_gate(model_x)
            gate = height_gate * learned_gate
        else:
            gate = height_gate
        gate = self.min_texture_gate + (1.0 - self.min_texture_gate) * gate
        low_residual = self.coarse(model_x)
        local_input = torch.cat((model_x, low_residual), dim=1)
        high_raw = self.local(local_input) * self.high_residual_scale
        high_residual = gate * high_raw
        final = low_residual + high_residual
        if return_components:
            return {
                "final": final,
                "low": low_residual,
                "high": high_residual,
                "high_raw": high_raw,
                "gate": gate,
            }
        return final

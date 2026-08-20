from __future__ import annotations

import torch
import torch.nn as nn


# Compact 1D network for vertical CO2 enhancement profiles.
class CompactProfileNet(nn.Module):
    # Build the neural-network layers for this module.
    def __init__(self, local_channels: int, global_channels: int, base_channels: int = 64) -> None:
        super().__init__()
        b = int(base_channels)
        in_channels = int(local_channels) + int(global_channels)
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, b, kernel_size=1),
            nn.GroupNorm(max(1, min(8, b)), b),
            nn.SiLU(inplace=True),
            nn.Conv1d(b, b, kernel_size=3, padding=1),
            nn.GroupNorm(max(1, min(8, b)), b),
            nn.SiLU(inplace=True),
            nn.Conv1d(b, b, kernel_size=3, padding=1),
            nn.GroupNorm(max(1, min(8, b)), b),
            nn.SiLU(inplace=True),
            nn.Conv1d(b, max(8, b // 2), kernel_size=3, padding=1),
            nn.SiLU(inplace=True),
            nn.Conv1d(max(8, b // 2), 1, kernel_size=1),
        )

    # Set the final output bias before training starts.
    def initialize_output_bias(self, value: float) -> None:
        final = self.net[-1]
        if isinstance(final, nn.Conv1d) and final.bias is not None:
            nn.init.constant_(final.bias, float(value))

    # Run the forward pass for this module.
    def forward(self, local: torch.Tensor, global_context: torch.Tensor) -> torch.Tensor:
        x = torch.cat((local, global_context), dim=1)
        return self.net(x).squeeze(1)

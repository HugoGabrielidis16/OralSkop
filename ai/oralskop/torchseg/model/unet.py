"""A small, from-scratch U-Net for semantic segmentation.

Placeholder architecture for the custom torchseg path. It is deliberately
self-contained (no pretrained backbone) so it can train on CPU for smoke tests
and serve as a starting point before swapping in a heavier encoder.

Interface matches the torchvision segmentation models consumed by
``torchseg/train.py``: ``forward`` returns a ``dict`` with key ``"out"`` holding
per-pixel logits of shape ``[B, num_classes, H, W]`` at the input resolution
(``num_classes`` includes background as index 0). There is no ``"aux"`` head.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class _DoubleConv(nn.Module):
    """(conv -> BN -> ReLU) x2, preserving spatial size (padding=1)."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class _Down(nn.Module):
    """Downscale by 2 (maxpool) then double conv."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.net = nn.Sequential(nn.MaxPool2d(2), _DoubleConv(in_ch, out_ch))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class _Up(nn.Module):
    """Upscale by 2, concat the skip connection, then double conv.

    Uses bilinear upsampling and pads to the skip's size so the net accepts
    inputs whose dimensions are not exact multiples of 16.
    """

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.conv = _DoubleConv(in_ch + skip_ch, out_ch)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        dy = skip.shape[-2] - x.shape[-2]
        dx = skip.shape[-1] - x.shape[-1]
        x = F.pad(x, [dx // 2, dx - dx // 2, dy // 2, dy - dy // 2])
        return self.conv(torch.cat([skip, x], dim=1))


class UNet(nn.Module):
    """Classic 4-level U-Net.

    Args:
        num_classes: number of output channels (incl. background at index 0).
        in_channels: input channels (3 for RGB).
        base_ch: channel width of the first encoder block; doubles each level.
    """

    def __init__(self, num_classes: int, in_channels: int = 3, base_ch: int = 64):
        super().__init__()
        c1, c2, c3, c4, c5 = (base_ch * m for m in (1, 2, 4, 8, 16))

        self.inc = _DoubleConv(in_channels, c1)
        self.down1 = _Down(c1, c2)
        self.down2 = _Down(c2, c3)
        self.down3 = _Down(c3, c4)
        self.down4 = _Down(c4, c5)

        self.up1 = _Up(c5, c4, c4)
        self.up2 = _Up(c4, c3, c3)
        self.up3 = _Up(c3, c2, c2)
        self.up4 = _Up(c2, c1, c1)
        self.head = nn.Conv2d(c1, num_classes, 1)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        return {"out": self.head(x)}

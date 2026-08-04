"""U-Net Lite — small residual corrector for Track C reffree reconstruction.

Design choices:

* **Single channel input**: ``phi_classical`` (the polynomial-fit
  reference-free output that still contains the structured stripe
  residual). Single channel output: predicted residual to subtract.
* **Residual output convention**: clean = input - model(input). Init the
  final conv near zero so an untrained network is a no-op (graceful
  degradation: pipeline falls back to Track A poly5 floor).
* **GroupNorm + GELU**: GroupNorm is batch-size independent (we train
  with batch=2-4 on M-series MPS). GELU is the modern default; ReLU was
  fine but GELU usually picks up a fraction of a dB without cost.
* **Base channels = 32**: 4 downs / 4 ups → ~1.5M params. Fits MPS easily,
  inference well under 100 ms on M-series CPU at 1024².
* **No skip-connection halving**: standard U-Net skips concatenate full
  feature maps. We crop spatially to handle non-power-of-2 inputs (1200x1600
  cropped/padded to 1024x1024 by the dataset transform).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _gn(c: int) -> nn.GroupNorm:
    """GroupNorm with 8 groups (or fewer if c<8). Works at any batch size."""
    return nn.GroupNorm(min(8, c), c)


class ConvBlock(nn.Module):
    """3x3 conv → GN → GELU, twice. Standard U-Net building block."""

    def __init__(self, in_c: int, out_c: int) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, padding=1, bias=False),
            _gn(out_c),
            nn.GELU(),
            nn.Conv2d(out_c, out_c, kernel_size=3, padding=1, bias=False),
            _gn(out_c),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x)


class Down(nn.Module):
    """MaxPool 2x then ConvBlock."""

    def __init__(self, in_c: int, out_c: int) -> None:
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = ConvBlock(in_c, out_c)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(self.pool(x))


class Up(nn.Module):
    """Bilinear upsample 2x, concat skip, then ConvBlock.

    Bilinear (vs transposed conv) avoids checkerboard artifacts and has
    no learnable parameters of its own — keeps the model small.
    """

    def __init__(self, in_c: int, skip_c: int, out_c: int) -> None:
        super().__init__()
        self.conv = ConvBlock(in_c + skip_c, out_c)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[-2:],
                          mode="bilinear", align_corners=False)
        return self.conv(torch.cat([x, skip], dim=1))


class UNetLite(nn.Module):
    """Residual U-Net for stripe-pattern removal.

    Forward returns ``(corrected, predicted_residual)``. Production code
    can pick whichever it prefers; training uses the predicted residual
    against the dataset's pre-computed residual target.
    """

    def __init__(self, base_c: int = 32, in_c: int = 1, out_c: int = 1) -> None:
        super().__init__()
        self.in_block = ConvBlock(in_c, base_c)
        self.d1 = Down(base_c, base_c * 2)
        self.d2 = Down(base_c * 2, base_c * 4)
        self.d3 = Down(base_c * 4, base_c * 8)
        self.d4 = Down(base_c * 8, base_c * 8)

        self.u1 = Up(base_c * 8, base_c * 8, base_c * 4)
        self.u2 = Up(base_c * 4, base_c * 4, base_c * 2)
        self.u3 = Up(base_c * 2, base_c * 2, base_c)
        self.u4 = Up(base_c, base_c, base_c)

        self.out_conv = nn.Conv2d(base_c, out_c, kernel_size=1)
        # Near-zero init so an untrained net predicts ~0 residual (no-op).
        nn.init.zeros_(self.out_conv.bias)
        nn.init.normal_(self.out_conv.weight, std=1e-3)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        s0 = self.in_block(x)
        s1 = self.d1(s0)
        s2 = self.d2(s1)
        s3 = self.d3(s2)
        b = self.d4(s3)

        u1 = self.u1(b, s3)
        u2 = self.u2(u1, s2)
        u3 = self.u3(u2, s1)
        u4 = self.u4(u3, s0)

        residual = self.out_conv(u4)
        corrected = x - residual
        return corrected, residual


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Sanity check: param count + forward pass shapes.
    net = UNetLite(base_c=32)
    print(f"UNetLite base=32 params={count_parameters(net):,}")
    x = torch.randn(2, 1, 512, 512)
    corrected, residual = net(x)
    print(f"input  {tuple(x.shape)}")
    print(f"resid  {tuple(residual.shape)}")
    print(f"corrct {tuple(corrected.shape)}")

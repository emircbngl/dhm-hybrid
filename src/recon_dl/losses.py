"""Loss functions for Track C training.

Combine L1 (robust to outliers, doesn't blow up on extreme phase
discontinuities) with a Total Variation regularizer (encourages
smoothness of the predicted correction since the stripe pattern is
low-mid frequency, not high-frequency edges) and a charbonnier term
for stability near zero.

We deliberately avoid SSIM here — SSIM on phase residuals is dominated
by the central object structures, not by the stripe pattern we want to
remove. L1 + TV is a better fit for residual learning.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class TVLoss(nn.Module):
    """Anisotropic total variation. Encourages the predicted residual to
    be smooth (the stripe pattern is low-mid frequency)."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dx = (x[..., 1:] - x[..., :-1]).abs().mean()
        dy = (x[..., 1:, :] - x[..., :-1, :]).abs().mean()
        return dx + dy


class CombinedLoss(nn.Module):
    def __init__(self,
                 l1_weight: float = 1.0,
                 charbonnier_weight: float = 0.0,
                 tv_weight: float = 0.05) -> None:
        super().__init__()
        self.l1_weight = l1_weight
        self.charbonnier_weight = charbonnier_weight
        self.tv_weight = tv_weight
        self.tv = TVLoss()

    @staticmethod
    def _charbonnier(diff: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
        return torch.sqrt(diff * diff + eps * eps).mean()

    def forward(self, pred: torch.Tensor,
                target: torch.Tensor) -> tuple[torch.Tensor, dict]:
        diff = pred - target
        l1 = diff.abs().mean()
        # Disabled terms must be zeros on *pred's* device — a bare
        # torch.tensor(0.0) lives on CPU and adding it to an MPS/CUDA
        # l1 raises "Expected all tensors to be on the same device".
        zero = pred.new_zeros(())
        char = (self._charbonnier(diff)
                if self.charbonnier_weight > 0 else zero)
        tv = self.tv(pred) if self.tv_weight > 0 else zero
        total = (self.l1_weight * l1
                 + self.charbonnier_weight * char
                 + self.tv_weight * tv)
        return total, {
            "loss": float(total),
            "l1": float(l1),
            "charbonnier": float(char),
            "tv": float(tv),
        }

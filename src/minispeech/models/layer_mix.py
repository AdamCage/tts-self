"""Learned softmax mix over cached Qwen layers."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class LayerMix(nn.Module):
    """Mix [B, n_layers, L, H] -> [B, L, H] with learned softmax weights."""

    def __init__(self, n_layers: int = 3):
        super().__init__()
        self.logits = nn.Parameter(torch.zeros(n_layers))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 3:
            return x
        w = F.softmax(self.logits, dim=0)  # [n]
        # x: [B, n, L, H]
        return torch.einsum("n,bnlh->blh", w, x)

    def weights(self) -> torch.Tensor:
        return F.softmax(self.logits.detach(), dim=0)

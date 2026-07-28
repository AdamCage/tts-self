"""Linear projectors."""

from __future__ import annotations

import torch.nn as nn


class MLPProjector(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden: int | None = None, dropout: float = 0.0):
        super().__init__()
        hidden = hidden or max(in_dim, out_dim)
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x):
        return self.net(x)

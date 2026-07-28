"""Regression acoustic head (v0.1) with optional residual prediction."""

from __future__ import annotations

import torch
import torch.nn as nn


class RegressionHead(nn.Module):
    def __init__(
        self,
        d_model: int = 512,
        latent_dim: int = 64,
        hidden: int = 1024,
        layers: int = 3,
        residual: bool = True,
    ):
        super().__init__()
        self.residual = residual
        mods: list[nn.Module] = [nn.Linear(d_model, hidden), nn.GELU()]
        for _ in range(max(0, layers - 2)):
            mods += [nn.Linear(hidden, hidden), nn.GELU()]
        mods.append(nn.Linear(hidden, latent_dim))
        self.net = nn.Sequential(*mods)

    def forward(self, h: torch.Tensor, z_prev: torch.Tensor | None = None) -> torch.Tensor:
        delta = self.net(h)
        if self.residual:
            if z_prev is None:
                raise ValueError("residual head requires z_prev")
            return z_prev + delta
        return delta

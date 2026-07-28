"""Stop and duration prediction heads."""

from __future__ import annotations

import torch
import torch.nn as nn


class StopHead(nn.Module):
    def __init__(self, d_model: int = 512, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.net(h).squeeze(-1)  # [B, T]


class UtteranceDurationHead(nn.Module):
    def __init__(self, d_model: int = 512, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, text_memory: torch.Tensor, text_mask: torch.Tensor | None = None) -> torch.Tensor:
        # Pool text memory
        if text_mask is not None:
            m = text_mask.unsqueeze(-1).float()
            pooled = (text_memory * m).sum(dim=1) / m.sum(dim=1).clamp_min(1.0)
        else:
            pooled = text_memory.mean(dim=1)
        return self.net(pooled).squeeze(-1)  # [B]


class TokenDurationHead(nn.Module):
    """Phase 2b: per-token duration in frames."""

    def __init__(self, d_model: int = 512, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
            nn.Softplus(),
        )

    def forward(self, text_memory: torch.Tensor) -> torch.Tensor:
        return self.net(text_memory).squeeze(-1)  # [B, L]


def stop_decision(
    stop_probs: torch.Tensor,
    *,
    threshold: float = 0.45,
    consecutive: int = 3,
    min_frames: int = 1,
    max_frames: int = 10_000,
    t: int,
) -> bool:
    """Triple-stop rule with min/max duration caps (AGENTS.md)."""
    if t + 1 >= max_frames:
        return True
    if t + 1 < min_frames:
        return False
    if t + 1 < consecutive:
        return False
    window = stop_probs[t + 1 - consecutive : t + 1]
    return bool((window > threshold).all().item())

"""Minimal fake DiscreteSpeechLM for trainer unit tests.

Avoids depending on in-flight model refactors (RoPE / speaker registry).
"""

from __future__ import annotations

from types import SimpleNamespace

import torch
import torch.nn as nn


class FakeRussianCond(nn.Module):
    def __init__(self, d: int = 8):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(d))

    def forward(self, x, **kwargs):
        return x


class FakeHeads(nn.Module):
    def __init__(self, n_residual: int = 15):
        super().__init__()
        self.n_residual = n_residual
        self.bias = nn.Parameter(torch.zeros(1))


class FakeDiscreteSpeechLM(nn.Module):
    """Trainer-compatible stub: CE-like loss from a tiny linear over codes."""

    def __init__(self, *, russian_conditioning: bool = False, d: int = 16):
        super().__init__()
        self.cfg = SimpleNamespace(russian_conditioning=russian_conditioning)
        self.ru_cond = FakeRussianCond(d)
        self.heads = FakeHeads(15)
        self.proj = nn.Linear(1, 1)
        # Keep a couple always-used params so the graph is non-empty.
        self.scale = nn.Parameter(torch.ones(1))

    def count_parameters(self, *, trainable_only: bool = False) -> dict[str, int]:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"total": total, "trainable": trainable}

    def forward(self, input_ids, codes, *, compute_loss: bool = True, **kwargs):
        # Deterministic scalar loss depending on params + batch content
        x = codes.float().mean().view(1, 1)
        pred = self.proj(x) * self.scale
        # Touch ru_cond only when enabled so disabled+frozen stays unused-safe.
        if self.cfg.russian_conditioning:
            pred = pred + self.ru_cond.weight.mean()
        else:
            # Still reference heads.bias so residual weight path is meaningful
            pred = pred + 0.0 * self.heads.bias
        ce_cb0 = (pred.squeeze() - x.squeeze()).pow(2)
        ce_residual = ce_cb0 * 0.5
        stop = ce_cb0 * 0.1
        total = ce_cb0 + ce_residual + stop
        loss = {
            "total": total,
            "ce_cb0": ce_cb0,
            "ce_residual": ce_residual,
            "stop": stop,
        }
        return SimpleNamespace(loss=loss if compute_loss else None)

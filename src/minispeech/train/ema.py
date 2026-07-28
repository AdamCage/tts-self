"""EMA of model parameters."""

from __future__ import annotations

import copy
from typing import Iterable

import torch.nn as nn


class EMA:
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = copy.deepcopy(model)
        self.shadow.eval()
        for p in self.shadow.parameters():
            p.requires_grad_(False)

    def update(self, model: nn.Module) -> None:
        with torch_no_grad():
            for s, p in zip(self.shadow.parameters(), model.parameters()):
                s.data.mul_(self.decay).add_(p.data, alpha=1.0 - self.decay)
            for s, b in zip(self.shadow.buffers(), model.buffers()):
                s.copy_(b)

    def state_dict(self):
        return self.shadow.state_dict()

    def load_state_dict(self, sd):
        self.shadow.load_state_dict(sd)


def torch_no_grad():
    import torch

    return torch.no_grad()

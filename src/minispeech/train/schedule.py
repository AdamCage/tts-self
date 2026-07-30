"""LR warmup + decay schedules shared by Talker and mel trainers."""

from __future__ import annotations

import math
from typing import Literal

import torch


ScheduleName = Literal["cosine", "linear", "constant"]


def lr_multiplier(
    step: int,
    total_steps: int,
    *,
    warmup_steps: int = 0,
    schedule: ScheduleName = "cosine",
    min_lr_ratio: float = 0.05,
) -> float:
    """Return multiplicative factor in ``[min_lr_ratio, 1]`` (warmup → 1.0)."""
    warm = max(int(warmup_steps), 0)
    total = max(int(total_steps), 1)
    s = max(int(step), 0)
    if warm > 0 and s < warm:
        return float(s + 1) / float(warm)
    if schedule == "constant":
        return 1.0
    progress = (s - warm) / max(total - warm, 1)
    progress = min(max(progress, 0.0), 1.0)
    if schedule == "linear":
        return max(min_lr_ratio, 1.0 - (1.0 - min_lr_ratio) * progress)
    # cosine
    cos = 0.5 * (1.0 + math.cos(math.pi * progress))
    return max(min_lr_ratio, min_lr_ratio + (1.0 - min_lr_ratio) * cos)


def set_optimizer_lr(opt: torch.optim.Optimizer, base_lr: float, mult: float) -> float:
    lr = float(base_lr) * float(mult)
    for group in opt.param_groups:
        group["lr"] = lr
    return lr

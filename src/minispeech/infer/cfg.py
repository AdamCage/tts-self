"""CFG helpers."""

from __future__ import annotations


def effective_cfg(scale: float, enabled: bool = True) -> float:
    return scale if enabled else 1.0

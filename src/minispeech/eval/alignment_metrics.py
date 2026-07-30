"""Alignment / cross-attention audit metrics (E5)."""

from __future__ import annotations

from typing import Any

import torch

from minispeech.train.guided_attn import attention_diagonality

__all__ = [
    "attention_diagonality",
    "chance_diagonality",
    "diagonality_vs_chance",
    "uniform_attention",
]


def uniform_attention(t_frames: int, l_tokens: int, *, device=None, dtype=torch.float32) -> torch.Tensor:
    """Uniform cross-attn [1, T, L] (chance baseline)."""
    attn = torch.full((1, t_frames, l_tokens), 1.0 / max(l_tokens, 1), device=device, dtype=dtype)
    return attn


def chance_diagonality(t_frames: int, l_tokens: int) -> float:
    """Diagonality of uniform attention for the same (T, L)."""
    if t_frames <= 0 or l_tokens <= 0:
        return float("nan")
    attn = uniform_attention(t_frames, l_tokens)
    return attention_diagonality(attn)


def diagonality_vs_chance(
    cross_attn: torch.Tensor,
    frame_mask: torch.Tensor | None = None,
    text_mask: torch.Tensor | None = None,
) -> dict[str, Any]:
    """Compare model diagonality to uniform-chance for each batch item.

    cross_attn: [B, H, T, L] or [B, T, L]
    """
    if cross_attn.ndim == 4:
        attn = cross_attn.mean(dim=1)
    else:
        attn = cross_attn
    B, T, L = attn.shape
    rows: list[dict[str, float]] = []
    for b in range(B):
        t_eff = int(frame_mask[b].sum().item()) if frame_mask is not None else T
        l_eff = int(text_mask[b].sum().item()) if text_mask is not None else L
        t_eff = max(1, min(t_eff, T))
        l_eff = max(1, min(l_eff, L))
        a = attn[b : b + 1, :t_eff, :l_eff]
        fm = torch.ones(1, t_eff, dtype=torch.bool, device=attn.device)
        diag = attention_diagonality(a, fm)
        chance = chance_diagonality(t_eff, l_eff)
        rows.append(
            {
                "diagonality": diag,
                "chance": chance,
                "vs_chance": diag - chance,
                "t_frames": float(t_eff),
                "l_tokens": float(l_eff),
            }
        )
    if not rows:
        return {"n": 0, "mean_diagonality": float("nan"), "mean_chance": float("nan"), "rows": []}
    return {
        "n": len(rows),
        "mean_diagonality": sum(r["diagonality"] for r in rows) / len(rows),
        "mean_chance": sum(r["chance"] for r in rows) / len(rows),
        "mean_vs_chance": sum(r["vs_chance"] for r in rows) / len(rows),
        "rows": rows,
    }

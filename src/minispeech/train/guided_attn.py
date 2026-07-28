"""Guided attention loss against alignment prior."""

from __future__ import annotations

import torch


def guided_attention_loss(
    cross_attn: torch.Tensor,
    align_prior: torch.Tensor,
    frame_mask: torch.Tensor | None = None,
    text_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    cross_attn: [B, H, T, L] or [B, T, L]
    align_prior: [B, T, L]
    Encourage attention mass to stay near prior.
    """
    if cross_attn.ndim == 4:
        attn = cross_attn.mean(dim=1)
    else:
        attn = cross_attn
    # KL(attn || prior) approximated as -sum attn * log prior
    loss_map = -(attn * torch.log(align_prior.clamp_min(1e-6))).sum(dim=-1)  # [B,T]
    if frame_mask is not None:
        loss_map = loss_map[frame_mask]
    return loss_map.mean() if loss_map.numel() else attn.new_zeros(())


def guided_weight_schedule(step: int, total_steps: int, start: float = 1.0, end_frac: float = 0.35) -> float:
    end_step = max(1, int(total_steps * end_frac))
    if step >= end_step:
        return 0.0
    return start * (1.0 - step / end_step)


def attention_diagonality(cross_attn: torch.Tensor, frame_mask=None, text_mask=None) -> float:
    """Fraction of mass within ±window of diagonal. Higher = more monotonic."""
    if cross_attn.ndim == 4:
        attn = cross_attn.mean(dim=1)
    else:
        attn = cross_attn
    B, T, L = attn.shape
    t = torch.arange(T, device=attn.device).float().unsqueeze(1) / max(T - 1, 1)
    l = torch.arange(L, device=attn.device).float().unsqueeze(0) / max(L - 1, 1)
    near = (t - l).abs() < (1.5 / max(L, 1))
    mass_near = (attn * near.unsqueeze(0).float()).sum(dim=-1)
    total = attn.sum(dim=-1).clamp_min(1e-8)
    ratio = mass_near / total
    if frame_mask is not None:
        ratio = ratio[frame_mask]
    return float(ratio.mean().item()) if ratio.numel() else 0.0

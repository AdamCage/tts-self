"""Training losses."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def masked_l1(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    m = mask.unsqueeze(-1).float()
    return ((pred - target).abs() * m).sum() / m.sum().clamp_min(1.0) / pred.shape[-1]


def masked_cosine(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    # 1 - cosine similarity, averaged over valid frames
    pred_n = F.normalize(pred, dim=-1)
    tgt_n = F.normalize(target, dim=-1)
    cos = (pred_n * tgt_n).sum(dim=-1)
    m = mask.float()
    return (1 - cos)[mask].mean() if m.any() else pred.new_zeros(())


def stop_bce(
    logits: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    positive_weight: float = 10.0,
) -> torch.Tensor:
    w = torch.where(target > 0.5, positive_weight, 1.0)
    loss = F.binary_cross_entropy_with_logits(logits, target, weight=w, reduction="none")
    return (loss * mask.float()).sum() / mask.float().sum().clamp_min(1.0)


def focal_stop_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    gamma: float = 2.0,
    alpha: float = 0.75,
) -> torch.Tensor:
    p = torch.sigmoid(logits)
    ce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    p_t = p * target + (1 - p) * (1 - target)
    alpha_t = alpha * target + (1 - alpha) * (1 - target)
    loss = alpha_t * (1 - p_t).pow(gamma) * ce
    return (loss * mask.float()).sum() / mask.float().sum().clamp_min(1.0)


def duration_l1(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return (pred - target).abs().mean()


def flow_mse(v_pred: torch.Tensor, v_target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    m = mask.unsqueeze(-1).float()
    return (((v_pred - v_target) ** 2) * m).sum() / m.sum().clamp_min(1.0) / v_pred.shape[-1]


def total_regression_loss(
    z_pred,
    z_target,
    stop_logits,
    stop_target,
    dur_pred,
    durations,
    frame_mask,
    *,
    w_l1=1.0,
    w_cos=0.2,
    w_stop=0.1,
    w_dur=0.05,
    stop_pos_weight=10.0,
    stop_loss: str = "bce",
    stop_focal_gamma: float = 2.0,
    stop_focal_alpha: float = 0.75,
    guided=None,
    w_guided=0.0,
) -> dict[str, torch.Tensor]:
    if stop_loss == "focal":
        stop = focal_stop_loss(
            stop_logits, stop_target, frame_mask, gamma=stop_focal_gamma, alpha=stop_focal_alpha
        )
    else:
        stop = stop_bce(stop_logits, stop_target, frame_mask, stop_pos_weight)
    losses = {
        "l1": masked_l1(z_pred, z_target, frame_mask),
        "cosine": masked_cosine(z_pred, z_target, frame_mask),
        "stop": stop,
        "duration": duration_l1(dur_pred, durations),
    }
    total = (
        w_l1 * losses["l1"]
        + w_cos * losses["cosine"]
        + w_stop * losses["stop"]
        + w_dur * losses["duration"]
    )
    if guided is not None and w_guided > 0:
        losses["guided"] = guided
        total = total + w_guided * guided
    losses["total"] = total
    return losses

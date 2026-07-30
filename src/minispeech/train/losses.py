"""Training losses."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def _weighted_mean(per_item: torch.Tensor, sample_weight: torch.Tensor | None) -> torch.Tensor:
    if sample_weight is None:
        return per_item.mean()
    w = sample_weight.float()
    return (per_item * w).sum() / w.sum().clamp_min(1.0)


def masked_l1(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    sample_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    m = mask.unsqueeze(-1).float()
    per = ((pred - target).abs() * m).sum(dim=(1, 2)) / m.sum(dim=(1, 2)).clamp_min(1.0) / pred.shape[-1]
    return _weighted_mean(per, sample_weight)


def masked_cosine(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    sample_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    # 1 - cosine similarity, averaged over valid frames (per item, then weight)
    pred_n = F.normalize(pred, dim=-1)
    tgt_n = F.normalize(target, dim=-1)
    cos = (pred_n * tgt_n).sum(dim=-1)
    m = mask.float()
    per = ((1 - cos) * m).sum(dim=-1) / m.sum(dim=-1).clamp_min(1.0)
    if not m.any():
        return pred.new_zeros(())
    return _weighted_mean(per, sample_weight)


def stop_bce(
    logits: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    positive_weight: float = 10.0,
    sample_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    # Soft-label aware: scale pos weight by target strength in [0, 1].
    strength = target.clamp(0.0, 1.0)
    w = 1.0 + (float(positive_weight) - 1.0) * strength
    loss = F.binary_cross_entropy_with_logits(logits, target, weight=w, reduction="none")
    m = mask.float()
    per = (loss * m).sum(dim=-1) / m.sum(dim=-1).clamp_min(1.0)
    return _weighted_mean(per, sample_weight)


def focal_stop_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    gamma: float = 2.0,
    alpha: float = 0.75,
    sample_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    p = torch.sigmoid(logits)
    ce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    p_t = p * target + (1 - p) * (1 - target)
    alpha_t = alpha * target + (1 - alpha) * (1 - target)
    loss = alpha_t * (1 - p_t).pow(gamma) * ce
    m = mask.float()
    per = (loss * m).sum(dim=-1) / m.sum(dim=-1).clamp_min(1.0)
    return _weighted_mean(per, sample_weight)


def duration_l1(
    pred: torch.Tensor,
    target: torch.Tensor,
    sample_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    per = (pred - target).abs()
    if per.ndim > 1:
        per = per.mean(dim=-1)
    return _weighted_mean(per, sample_weight)


def token_duration_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    text_mask: torch.Tensor,
    *,
    sample_weight: torch.Tensor | None = None,
    log_domain: bool = False,
) -> torch.Tensor:
    """Optional per-token duration head loss (``TokenDurationHead`` hook).

    ``pred`` / ``target``: ``[B, L]``; ``text_mask``: ``[B, L]`` bool.
    """
    if log_domain:
        p = torch.log1p(pred.clamp_min(0))
        t = torch.log1p(target.clamp_min(0))
    else:
        p, t = pred, target
    err = (p - t).abs()
    m = text_mask.float()
    per = (err * m).sum(dim=-1) / m.sum(dim=-1).clamp_min(1.0)
    return _weighted_mean(per, sample_weight)


def flow_mse(
    v_pred: torch.Tensor,
    v_target: torch.Tensor,
    mask: torch.Tensor,
    sample_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    m = mask.unsqueeze(-1).float()
    per = (((v_pred - v_target) ** 2) * m).sum(dim=(1, 2)) / m.sum(dim=(1, 2)).clamp_min(1.0) / v_pred.shape[-1]
    return _weighted_mean(per, sample_weight)


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
    sample_weight: torch.Tensor | None = None,
    tok_dur_pred: torch.Tensor | None = None,
    tok_dur_target: torch.Tensor | None = None,
    text_mask: torch.Tensor | None = None,
    w_tok_dur: float = 0.0,
) -> dict[str, torch.Tensor]:
    if stop_loss == "focal":
        stop = focal_stop_loss(
            stop_logits,
            stop_target,
            frame_mask,
            gamma=stop_focal_gamma,
            alpha=stop_focal_alpha,
            sample_weight=sample_weight,
        )
    else:
        stop = stop_bce(
            stop_logits, stop_target, frame_mask, stop_pos_weight, sample_weight=sample_weight
        )
    losses = {
        "l1": masked_l1(z_pred, z_target, frame_mask, sample_weight=sample_weight),
        "cosine": masked_cosine(z_pred, z_target, frame_mask, sample_weight=sample_weight),
        "stop": stop,
        "duration": duration_l1(dur_pred, durations, sample_weight=sample_weight),
    }
    total = (
        w_l1 * losses["l1"]
        + w_cos * losses["cosine"]
        + w_stop * losses["stop"]
        + w_dur * losses["duration"]
    )
    if (
        w_tok_dur > 0
        and tok_dur_pred is not None
        and tok_dur_target is not None
        and text_mask is not None
    ):
        losses["tok_duration"] = token_duration_loss(
            tok_dur_pred, tok_dur_target, text_mask, sample_weight=sample_weight
        )
        total = total + w_tok_dur * losses["tok_duration"]
    if guided is not None and w_guided > 0:
        losses["guided"] = guided
        total = total + w_guided * guided
    losses["total"] = total
    return losses

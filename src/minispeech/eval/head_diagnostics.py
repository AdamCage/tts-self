"""Teacher-forced head diagnostics: R², variance ratio, lag corr, spectrum (E3)."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch


def _flat_valid(
    pred: torch.Tensor,
    gt: torch.Tensor,
    mask: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return [N, D] valid frames from [B, T, D] or [T, D]."""
    if pred.ndim == 2:
        pred = pred.unsqueeze(0)
        gt = gt.unsqueeze(0)
        if mask is not None and mask.ndim == 1:
            mask = mask.unsqueeze(0)
    if mask is None:
        mask = torch.ones(pred.shape[:2], dtype=torch.bool, device=pred.device)
    pred = pred.float()
    gt = gt.float()
    mask = mask.bool()
    return pred[mask], gt[mask]


def r2_score(
    pred: torch.Tensor,
    gt: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> float:
    """1 - SS_res / SS_tot over all valid elements (overall R²)."""
    p, g = _flat_valid(pred, gt, mask)
    if p.numel() == 0:
        return float("nan")
    ss_res = ((p - g) ** 2).sum()
    ss_tot = ((g - g.mean()) ** 2).sum().clamp_min(1e-12)
    return float((1.0 - ss_res / ss_tot).item())


def per_dim_r2(
    pred: torch.Tensor,
    gt: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Per-dimension R² → [D]."""
    p, g = _flat_valid(pred, gt, mask)
    if p.numel() == 0:
        d = int(pred.shape[-1])
        return torch.full((d,), float("nan"))
    ss_res = ((p - g) ** 2).sum(dim=0)
    ss_tot = ((g - g.mean(dim=0)) ** 2).sum(dim=0).clamp_min(1e-12)
    return (1.0 - ss_res / ss_tot).detach().cpu()


def variance_ratio(
    pred: torch.Tensor,
    gt: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> float:
    """Var(z_pred) / Var(z_gt) over all valid elements (collapse check)."""
    p, g = _flat_valid(pred, gt, mask)
    if p.numel() < 2:
        return float("nan")
    vp = float(p.var(unbiased=False).item())
    vg = float(g.var(unbiased=False).item())
    return vp / max(vg, 1e-12)


def lag_correlations(
    pred: torch.Tensor,
    gt: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> dict[str, float]:
    """Mean Pearson corr of z_pred[t] with z_gt[t] vs z_gt[t-1] (per dim, then mean)."""
    if pred.ndim == 2:
        pred = pred.unsqueeze(0)
        gt = gt.unsqueeze(0)
        if mask is not None and mask.ndim == 1:
            mask = mask.unsqueeze(0)
    if mask is None:
        mask = torch.ones(pred.shape[:2], dtype=torch.bool, device=pred.device)
    pred = pred.float()
    gt = gt.float()
    # Use frames where both t and t-1 are valid
    m_t = mask[:, 1:]
    m_tm1 = mask[:, :-1]
    both = m_t & m_tm1
    if not both.any():
        return {"corr_gt_t": float("nan"), "corr_gt_tm1": float("nan")}

    p = pred[:, 1:][both]  # [N, D]
    g = gt[:, 1:][both]
    g_prev = gt[:, :-1][both]

    def _mean_pearson(a: torch.Tensor, b: torch.Tensor) -> float:
        a = a - a.mean(dim=0)
        b = b - b.mean(dim=0)
        num = (a * b).mean(dim=0)
        den = a.std(dim=0).clamp_min(1e-8) * b.std(dim=0).clamp_min(1e-8)
        return float((num / den).mean().item())

    return {
        "corr_gt_t": _mean_pearson(p, g),
        "corr_gt_tm1": _mean_pearson(p, g_prev),
    }


def temporal_highfreq_attenuation(
    pred: torch.Tensor,
    gt: torch.Tensor,
    mask: torch.Tensor | None = None,
    *,
    high_frac: float = 0.25,
) -> dict[str, float]:
    """Compare high-frequency temporal energy (FFT) of pred vs gt.

    Returns mean ratio of high-band power (pred/gt). Values << 1 mean oversmoothing.
    """
    if pred.ndim == 2:
        pred = pred.unsqueeze(0)
        gt = gt.unsqueeze(0)
        if mask is not None and mask.ndim == 1:
            mask = mask.unsqueeze(0)
    if mask is None:
        mask = torch.ones(pred.shape[:2], dtype=torch.bool, device=pred.device)

    ratios: list[float] = []
    hf_pred: list[float] = []
    hf_gt: list[float] = []
    for b in range(pred.shape[0]):
        n = int(mask[b].sum().item())
        if n < 8:
            continue
        p = pred[b, :n].float().cpu().numpy()  # [T, D]
        g = gt[b, :n].float().cpu().numpy()
        # Mean spectrum over dims
        fp = np.abs(np.fft.rfft(p, axis=0)) ** 2
        fg = np.abs(np.fft.rfft(g, axis=0)) ** 2
        freqs = np.fft.rfftfreq(n)
        cutoff = float(high_frac) * 0.5  # Nyquist = 0.5 cycles/sample
        hi = freqs >= cutoff
        if not hi.any():
            continue
        ep = float(fp[hi].mean())
        eg = float(fg[hi].mean())
        hf_pred.append(ep)
        hf_gt.append(eg)
        ratios.append(ep / max(eg, 1e-12))
    if not ratios:
        return {
            "hf_power_ratio": float("nan"),
            "hf_power_pred": float("nan"),
            "hf_power_gt": float("nan"),
        }
    return {
        "hf_power_ratio": float(np.mean(ratios)),
        "hf_power_pred": float(np.mean(hf_pred)),
        "hf_power_gt": float(np.mean(hf_gt)),
    }


def summarize_head_diagnostics(
    pred: torch.Tensor,
    gt: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> dict[str, Any]:
    """Aggregate E3 metrics for one batch (or concatenated batches)."""
    dim_r2 = per_dim_r2(pred, gt, mask)
    lags = lag_correlations(pred, gt, mask)
    spec = temporal_highfreq_attenuation(pred, gt, mask)
    return {
        "r2_overall": r2_score(pred, gt, mask),
        "r2_per_dim_mean": float(dim_r2.nanmean().item()) if dim_r2.numel() else float("nan"),
        "r2_per_dim": [float(x) for x in dim_r2.tolist()],
        "var_ratio": variance_ratio(pred, gt, mask),
        **lags,
        **spec,
    }

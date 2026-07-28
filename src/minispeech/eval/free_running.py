"""Free-running evaluation metrics (DTW L1, duration, stability)."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import torch

from minispeech.eval.dtw import dtw_align


@dataclass
class FreeRunningMetrics:
    dtw_l1: float
    path_slope: float
    duration_ratio: float
    raw_l1_diag: float | None
    stop_success: bool
    runaway: bool
    has_nan: bool
    silence_collapse: bool
    n_frames_pred: int
    n_frames_gt: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _raw_l1(pred: torch.Tensor, gt: torch.Tensor) -> float | None:
    t = min(pred.shape[0], gt.shape[0])
    if t == 0:
        return None
    return float((pred[:t] - gt[:t]).abs().mean().item())


def evaluate_free_running_latents(
    z_pred: torch.Tensor,
    z_gt: torch.Tensor,
    *,
    stop_triggered: bool = True,
    hit_max_frames: bool = False,
    silence_rms: float | None = None,
    silence_thresh: float = 1e-3,
) -> FreeRunningMetrics:
    """
    z_pred / z_gt: [T, 64] or [1, T, 64] normalized.
    """
    if z_pred.ndim == 3:
        z_pred = z_pred[0]
    if z_gt.ndim == 3:
        z_gt = z_gt[0]
    has_nan = bool(torch.isnan(z_pred).any() or torch.isinf(z_pred).any())
    dtw = dtw_align(z_pred, z_gt, metric="l1")
    n_p, n_g = int(z_pred.shape[0]), int(z_gt.shape[0])
    ratio = n_p / max(n_g, 1)
    silence = False
    if silence_rms is not None:
        silence = silence_rms < silence_thresh
    return FreeRunningMetrics(
        dtw_l1=dtw.mean_cost,
        path_slope=dtw.path_slope,
        duration_ratio=float(ratio),
        raw_l1_diag=_raw_l1(z_pred, z_gt),
        stop_success=bool(stop_triggered and not hit_max_frames),
        runaway=bool(hit_max_frames),
        has_nan=has_nan,
        silence_collapse=silence,
        n_frames_pred=n_p,
        n_frames_gt=n_g,
    )


def duration_bucket(dur_s: float) -> str:
    if dur_s < 2:
        return "0-2"
    if dur_s < 4:
        return "2-4"
    if dur_s < 6:
        return "4-6"
    return "6-10"


def aggregate_stability(rows: list[FreeRunningMetrics]) -> dict[str, float]:
    if not rows:
        return {}
    n = len(rows)
    return {
        "stop_success_rate": sum(r.stop_success for r in rows) / n,
        "runaway_rate": sum(r.runaway for r in rows) / n,
        "nan_rate": sum(r.has_nan for r in rows) / n,
        "silence_rate": sum(r.silence_collapse for r in rows) / n,
        "mean_dtw_l1": float(np.mean([r.dtw_l1 for r in rows])),
        "duration_in_band_rate": sum(0.8 <= r.duration_ratio <= 1.2 for r in rows) / n,
    }

"""Band-limited DTW for free-running latent alignment."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class DtwResult:
    mean_cost: float
    path_slope: float
    slope_deviation: float
    path_len: int


def _frame_cost(a: np.ndarray, b: np.ndarray, metric: str) -> float:
    if metric == "l1":
        return float(np.mean(np.abs(a - b)))
    # 1 - cosine
    na = np.linalg.norm(a) + 1e-8
    nb = np.linalg.norm(b) + 1e-8
    return float(1.0 - np.dot(a, b) / (na * nb))


def dtw_align(
    pred: torch.Tensor | np.ndarray,
    target: torch.Tensor | np.ndarray,
    *,
    band_frac: float = 0.20,
    metric: str = "l1",
) -> DtwResult:
    """
    Sakoe-Chiba band DTW (±band_frac of max length).
    pred/target: [T, D] normalized latents.
    """
    p = pred.detach().cpu().numpy() if torch.is_tensor(pred) else np.asarray(pred)
    t = target.detach().cpu().numpy() if torch.is_tensor(target) else np.asarray(target)
    n, m = p.shape[0], t.shape[0]
    if n == 0 or m == 0:
        return DtwResult(float("inf"), 0.0, float("inf"), 0)

    band = max(1, int(band_frac * max(n, m)))
    inf = 1e18
    cost = np.full((n + 1, m + 1), inf, dtype=np.float64)
    cost[0, 0] = 0.0
    ptr = np.zeros((n + 1, m + 1), dtype=np.int8)  # 1=diag, 2=up, 3=left

    for i in range(1, n + 1):
        j0 = max(1, i - band)
        j1 = min(m, i + band)
        for j in range(j0, j1 + 1):
            c = _frame_cost(p[i - 1], t[j - 1], metric)
            cands = (
                (cost[i - 1, j - 1] + c, 1),
                (cost[i - 1, j] + c, 2),
                (cost[i, j - 1] + c, 3),
            )
            best, which = min(cands, key=lambda x: x[0])
            cost[i, j] = best
            ptr[i, j] = which

    # backtrack
    i, j = n, m
    path: list[tuple[int, int]] = []
    while i > 0 and j > 0:
        path.append((i - 1, j - 1))
        w = ptr[i, j]
        if w == 1:
            i, j = i - 1, j - 1
        elif w == 2:
            i -= 1
        elif w == 3:
            j -= 1
        else:
            break
    path.reverse()
    if not path:
        return DtwResult(float(cost[n, m] / max(n, m)), float(n) / max(m, 1), abs(float(n) / max(m, 1) - 1.0), 0)

    costs = [_frame_cost(p[i], t[j], metric) for i, j in path]
    mean_cost = float(np.mean(costs))
    # slope ≈ Δi/Δj along path
    di = path[-1][0] - path[0][0] + 1e-8
    dj = path[-1][1] - path[0][1] + 1e-8
    slope = float(di / dj)
    return DtwResult(
        mean_cost=mean_cost,
        path_slope=slope,
        slope_deviation=abs(slope - 1.0),
        path_len=len(path),
    )

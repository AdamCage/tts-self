"""Cached vs full decode equivalence helpers for Phase 0 / streaming."""

from __future__ import annotations

from typing import Any

import torch

from minispeech.constants import SAMPLES_PER_FRAME


def decode_chunked(
    codec: Any,
    latents: torch.Tensor,
    *,
    chunk_frames: int = 4,
) -> torch.Tensor:
    """Causal cached decode in latent chunks; returns audio [B, 1, N]."""
    cache = None
    chunks: list[torch.Tensor] = []
    t = latents.shape[1]
    for i in range(0, t, chunk_frames):
        out = codec.decode(
            latents[:, i : i + chunk_frames],
            use_cache=True,
            padding_cache=cache,
        )
        cache = out.padding_cache
        chunks.append(out.audio)
    return torch.cat(chunks, dim=-1)


def cached_vs_full_stats(
    full: torch.Tensor,
    cached: torch.Tensor,
    *,
    chunk_frames: int = 4,
    edge: int = 64,
) -> dict[str, float | int | bool]:
    """Compare full vs chunked audio; localize errors near chunk seams.

    A true streaming seam bug spikes near boundaries; bf16 noise is global
    (interior max ≈ global max, boundary max often smaller).
    """
    nmin = min(int(full.shape[-1]), int(cached.shape[-1]))
    diff = (full[..., :nmin].float() - cached[..., :nmin].float()).abs()
    flat = diff.reshape(-1)
    if flat.numel() == 0:
        return {
            "n_samples": 0,
            "len_full": int(full.shape[-1]),
            "len_cached": int(cached.shape[-1]),
            "len_delta": int(cached.shape[-1] - full.shape[-1]),
            "max_abs": 0.0,
            "mean_abs": 0.0,
            "p95_abs": 0.0,
            "p99_abs": 0.0,
            "rms": 0.0,
            "interior_max_abs": 0.0,
            "interior_p95_abs": 0.0,
            "boundary_max_abs": 0.0,
            "boundary_ratio": 0.0,
        }

    step = chunk_frames * SAMPLES_PER_FRAME
    interior = torch.ones(nmin, dtype=torch.bool, device=diff.device)
    interior[:edge] = False
    interior[max(0, nmin - edge) :] = False
    for b in range(step, nmin, step):
        interior[max(0, b - edge) : min(nmin, b + edge)] = False

    boundary_max = 0.0
    for b in range(step, nmin, step):
        lo, hi = max(0, b - edge), min(nmin, b + edge)
        if hi > lo:
            boundary_max = max(boundary_max, float(diff[..., lo:hi].max().cpu()))

    if interior.any():
        interior_vals = diff[..., interior].reshape(-1)
        interior_max = float(interior_vals.max().cpu())
        interior_p95 = float(torch.quantile(interior_vals, 0.95).cpu())
    else:
        interior_max = float(flat.max().cpu())
        interior_p95 = float(torch.quantile(flat, 0.95).cpu())

    ratio = boundary_max / max(interior_p95, 1e-8)

    return {
        "n_samples": nmin,
        "len_full": int(full.shape[-1]),
        "len_cached": int(cached.shape[-1]),
        "len_delta": int(cached.shape[-1] - full.shape[-1]),
        "max_abs": float(flat.max().cpu()),
        "mean_abs": float(flat.mean().cpu()),
        "p95_abs": float(torch.quantile(flat, 0.95).cpu()),
        "p99_abs": float(torch.quantile(flat, 0.99).cpu()),
        "rms": float(diff.pow(2).mean().sqrt().cpu()),
        "interior_max_abs": interior_max,
        "interior_p95_abs": interior_p95,
        "boundary_max_abs": boundary_max,
        "boundary_ratio": float(ratio),
    }


def evaluate_cache_gate(
    stats: dict[str, float | int | bool],
    *,
    max_abs_limit: float,
    boundary_ratio_max: float = 10.0,
    boundary_abs_floor: float = 1e-3,
) -> tuple[bool, str]:
    """Hard-fail on large max-abs or localized boundary spikes."""
    max_abs = float(stats["max_abs"])
    boundary_max = float(stats["boundary_max_abs"])
    interior_p95 = float(stats["interior_p95_abs"])
    if max_abs >= max_abs_limit:
        return False, f"max_abs={max_abs:.6g} >= limit={max_abs_limit:g}"
    # Seam bug: error concentrates at chunk boundaries. Use a tiny floor so a
    # near-perfect interior (p95≈0) with a large boundary peak still fails.
    denom = max(interior_p95, 1e-8)
    if boundary_max >= boundary_abs_floor and boundary_max > boundary_ratio_max * denom:
        return (
            False,
            f"boundary_spike boundary_max={boundary_max:.6g} > "
            f"{boundary_ratio_max:g}×interior_p95={interior_p95:.6g}",
        )
    if abs(int(stats["len_delta"])) > SAMPLES_PER_FRAME:
        return False, f"len_delta={stats['len_delta']} exceeds one frame"
    return True, "ok"

"""Latent-space perturbations in *normalized* units (E1 calibration)."""

from __future__ import annotations

import torch


def ar1_noise(
    shape: tuple[int, ...] | torch.Size,
    rho: float,
    sigma: float,
    *,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Temporally correlated noise; time is axis 1 for [B, T, D]."""
    if len(shape) < 2:
        raise ValueError("ar1_noise expects at least [B, T, ...]")
    eps = torch.randn(shape, device=device, dtype=dtype, generator=generator)
    out = torch.zeros_like(eps)
    out[:, 0] = eps[:, 0]
    scale = (1.0 - rho**2) ** 0.5
    for t in range(1, shape[1]):
        out[:, t] = rho * out[:, t - 1] + scale * eps[:, t]
    return out * sigma


def add_white_noise(z_norm: torch.Tensor, sigma: float, *, generator=None) -> torch.Tensor:
    """z' = z + σ·ε in normalized latent space."""
    if sigma <= 0:
        return z_norm.clone()
    noise = torch.randn(z_norm.shape, device=z_norm.device, dtype=z_norm.dtype, generator=generator)
    return z_norm + sigma * noise


def add_ar1_noise(
    z_norm: torch.Tensor,
    sigma: float,
    rho: float = 0.8,
    *,
    generator=None,
) -> torch.Tensor:
    """Add AR(1) noise in normalized space. Accepts [T,D] or [B,T,D]."""
    if sigma <= 0:
        return z_norm.clone()
    squeezed = False
    z = z_norm
    if z.ndim == 2:
        z = z.unsqueeze(0)
        squeezed = True
    noise = ar1_noise(
        z.shape,
        rho,
        sigma,
        device=z.device,
        dtype=z.dtype,
        generator=generator,
    )
    out = z + noise
    return out[0] if squeezed else out


def variance_collapse(z_norm: torch.Tensor, alpha: float) -> torch.Tensor:
    """ẑ' = α·ẑ — models regression variance collapse toward the mean (0 in norm space)."""
    return z_norm * float(alpha)


def temporal_smooth(z_norm: torch.Tensor, kernel_size: int = 5) -> torch.Tensor:
    """Moving-average along time (reflect pad). Accepts [T,D] or [B,T,D]."""
    k = max(1, int(kernel_size))
    if k == 1:
        return z_norm.clone()
    squeezed = False
    z = z_norm
    if z.ndim == 2:
        z = z.unsqueeze(0)
        squeezed = True
    # [B, T, D] → [B, D, T] for conv1d
    x = z.transpose(1, 2)
    pad = k // 2
    x = torch.nn.functional.pad(x, (pad, pad), mode="reflect")
    weight = torch.ones(x.shape[1], 1, k, device=z.device, dtype=z.dtype) / float(k)
    y = torch.nn.functional.conv1d(x, weight, groups=x.shape[1])
    out = y.transpose(1, 2)
    return out[0] if squeezed else out

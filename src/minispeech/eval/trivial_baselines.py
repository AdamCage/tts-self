"""Trivial latent predictors for free-running gate audits (E0)."""

from __future__ import annotations

from typing import Literal

import torch

BaselineName = Literal[
    "constant_mean",
    "copy_prev",
    "random_corpus",
    "random_other_utt",
]


def constant_mean_latents(
    n_frames: int,
    mean: torch.Tensor,
    *,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Repeat corpus mean for ``n_frames`` → [T, D].

    In *normalized* space ``mean`` is typically zeros (unit-variance latents).
    """
    mean = mean.detach().float().reshape(-1)
    d = mean.numel()
    out = mean.unsqueeze(0).expand(n_frames, d).contiguous()
    if device is not None or dtype is not None:
        out = out.to(device=device or out.device, dtype=dtype or out.dtype)
    return out


def copy_prev_frame_latents(z_gt: torch.Tensor) -> torch.Tensor:
    """Copy previous GT frame (frame 0 repeated). z_gt: [T, D] → [T, D]."""
    if z_gt.ndim != 2:
        raise ValueError(f"expected [T,D], got {tuple(z_gt.shape)}")
    if z_gt.shape[0] == 0:
        return z_gt.clone()
    out = torch.empty_like(z_gt)
    out[0] = z_gt[0]
    if z_gt.shape[0] > 1:
        out[1:] = z_gt[:-1]
    return out


def random_corpus_latents(
    n_frames: int,
    pool: torch.Tensor,
    *,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Sample ``n_frames`` rows with replacement from pool [N, D]."""
    if pool.ndim != 2 or pool.shape[0] == 0:
        raise ValueError("pool must be non-empty [N, D]")
    idx = torch.randint(
        0,
        pool.shape[0],
        (n_frames,),
        generator=generator,
        device=pool.device,
    )
    return pool[idx].clone()


def random_other_utt_latents(
    n_frames: int,
    candidates: list[torch.Tensor],
    *,
    generator: torch.Generator | None = None,
    exclude: torch.Tensor | None = None,
) -> torch.Tensor:
    """Pick a random other utterance and crop/pad to ``n_frames``.

    Prefer candidates with length ≥ n_frames; otherwise tile the shortest usable.
    """
    usable = [z for z in candidates if z.ndim == 2 and z.shape[0] > 0]
    if exclude is not None:
        usable = [z for z in usable if z.data_ptr() != exclude.data_ptr()]
    if not usable:
        raise ValueError("no candidate latents for random_other_utt")
    # Prefer long enough sequences
    long_enough = [z for z in usable if z.shape[0] >= n_frames]
    pool = long_enough or usable
    if generator is None:
        i = int(torch.randint(0, len(pool), (1,)).item())
    else:
        i = int(torch.randint(0, len(pool), (1,), generator=generator).item())
    src = pool[i]
    if src.shape[0] >= n_frames:
        if generator is None:
            start = int(torch.randint(0, src.shape[0] - n_frames + 1, (1,)).item())
        else:
            start = int(
                torch.randint(0, src.shape[0] - n_frames + 1, (1,), generator=generator).item()
            )
        return src[start : start + n_frames].clone()
    # Tile short utterance
    reps = (n_frames + src.shape[0] - 1) // src.shape[0]
    tiled = src.repeat(reps, 1)
    return tiled[:n_frames].clone()


def predict_baseline(
    name: BaselineName,
    z_gt: torch.Tensor,
    *,
    mean: torch.Tensor | None = None,
    pool: torch.Tensor | None = None,
    candidates: list[torch.Tensor] | None = None,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Dispatch trivial predictor. ``z_gt`` is [T, D] (usually normalized)."""
    n = int(z_gt.shape[0])
    if name == "constant_mean":
        if mean is None:
            mean = torch.zeros(z_gt.shape[-1], dtype=z_gt.dtype, device=z_gt.device)
        return constant_mean_latents(n, mean, device=z_gt.device, dtype=z_gt.dtype)
    if name == "copy_prev":
        return copy_prev_frame_latents(z_gt)
    if name == "random_corpus":
        if pool is None:
            raise ValueError("random_corpus requires pool")
        return random_corpus_latents(n, pool.to(z_gt.device), generator=generator)
    if name == "random_other_utt":
        if not candidates:
            raise ValueError("random_other_utt requires candidates")
        return random_other_utt_latents(
            n, [c.to(z_gt.device) for c in candidates], generator=generator, exclude=z_gt
        )
    raise ValueError(f"unknown baseline: {name}")

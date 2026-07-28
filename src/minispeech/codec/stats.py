"""Latent mean/std and temporal autocorrelation."""

from __future__ import annotations

from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

from minispeech.constants import LATENT_DIM
from minispeech.utils import expand


def compute_latent_stats(latents_list: list[torch.Tensor]) -> dict[str, torch.Tensor]:
    """latents_list: each [T, 64] in raw (unnormalized) space."""
    cat = torch.cat([z.reshape(-1, LATENT_DIM).float() for z in latents_list], dim=0)
    mean = cat.mean(dim=0)
    std = cat.std(dim=0).clamp_min(1e-5)
    # Lag-1 autocorrelation per dim
    ac1 = []
    for z in latents_list:
        if z.shape[0] < 2:
            continue
        a, b = z[:-1].float(), z[1:].float()
        a = a - a.mean(0)
        b = b - b.mean(0)
        num = (a * b).mean(0)
        den = a.std(0).clamp_min(1e-5) * b.std(0).clamp_min(1e-5)
        ac1.append(num / den)
    ac = torch.stack(ac1).mean(0) if ac1 else torch.zeros(LATENT_DIM)
    return {
        "latent_mean": mean.half(),
        "latent_std": std.half(),
        "latent_ac1": ac.half(),
    }


def save_stats(stats: dict[str, torch.Tensor], path: str | Path) -> None:
    path = expand(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    save_file(stats, str(path))


def load_stats(path: str | Path) -> dict[str, torch.Tensor]:
    return load_file(str(expand(path)))


class LatentNormalizer:
    def __init__(self, mean: torch.Tensor, std: torch.Tensor):
        self.mean = mean.float().view(1, 1, -1)
        self.std = std.float().view(1, 1, -1).clamp_min(1e-5)

    @classmethod
    def from_file(cls, path: str | Path) -> "LatentNormalizer":
        s = load_stats(path)
        return cls(s["latent_mean"], s["latent_std"])

    def normalize(self, z: torch.Tensor) -> torch.Tensor:
        mean = self.mean.to(device=z.device, dtype=z.dtype)
        std = self.std.to(device=z.device, dtype=z.dtype)
        return (z - mean) / std

    def denormalize(self, z: torch.Tensor) -> torch.Tensor:
        mean = self.mean.to(device=z.device, dtype=z.dtype)
        std = self.std.to(device=z.device, dtype=z.dtype)
        return z * std + mean

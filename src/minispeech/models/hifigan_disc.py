"""HiFi-GAN multi-period and multi-scale waveform discriminators."""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm, weight_norm

LRELU_SLOPE = 0.1


def _norm(layer: nn.Module, *, use_spectral_norm: bool) -> nn.Module:
    return spectral_norm(layer) if use_spectral_norm else weight_norm(layer)


class PeriodDiscriminator(nn.Module):
    """2-D discriminator operating on a waveform reshaped by one period."""

    def __init__(self, period: int, *, use_spectral_norm: bool = False):
        super().__init__()
        self.period = int(period)
        channels = (1, 32, 128, 512, 1024, 1024)
        strides = (3, 3, 3, 3, 1)
        self.convs = nn.ModuleList(
            _norm(
                nn.Conv2d(
                    channels[i],
                    channels[i + 1],
                    (5, 1),
                    (strides[i], 1),
                    padding=(2, 0),
                ),
                use_spectral_norm=use_spectral_norm,
            )
            for i in range(len(strides))
        )
        self.conv_post = _norm(
            nn.Conv2d(1024, 1, (3, 1), 1, padding=(1, 0)),
            use_spectral_norm=use_spectral_norm,
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """Return flattened scores and intermediate feature maps for ``x [B,1,T]``."""
        if x.ndim != 3 or x.shape[1] != 1:
            raise ValueError(f"expected waveform [B, 1, T], got {tuple(x.shape)}")
        b, _, t = x.shape
        if t % self.period:
            pad = self.period - (t % self.period)
            x = F.pad(x, (0, pad), mode="reflect" if t > 1 else "constant")
            t += pad
        x = x.view(b, 1, t // self.period, self.period)
        fmap: list[torch.Tensor] = []
        for conv in self.convs:
            x = F.leaky_relu(conv(x), LRELU_SLOPE)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        return torch.flatten(x, 1), fmap


class MultiPeriodDiscriminator(nn.Module):
    """HiFi-GAN MPD using periods 2, 3, 5, 7, and 11."""

    def __init__(self, periods: Sequence[int] = (2, 3, 5, 7, 11)):
        super().__init__()
        self.discriminators = nn.ModuleList(PeriodDiscriminator(p) for p in periods)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[list[torch.Tensor], list[list[torch.Tensor]]]:
        scores, features = [], []
        for discriminator in self.discriminators:
            score, fmap = discriminator(x)
            scores.append(score)
            features.append(fmap)
        return scores, features


class ScaleDiscriminator(nn.Module):
    """1-D discriminator used at a single waveform scale."""

    def __init__(self, *, use_spectral_norm: bool = False):
        super().__init__()
        channels = (1, 128, 128, 256, 512, 1024, 1024, 1024)
        kernels = (15, 41, 41, 41, 41, 41, 5)
        strides = (1, 4, 4, 4, 4, 1, 1)
        groups = (1, 4, 16, 16, 16, 1, 1)
        self.convs = nn.ModuleList(
            _norm(
                nn.Conv1d(
                    channels[i],
                    channels[i + 1],
                    kernels[i],
                    strides[i],
                    padding=(kernels[i] - 1) // 2,
                    groups=groups[i],
                ),
                use_spectral_norm=use_spectral_norm,
            )
            for i in range(len(kernels))
        )
        self.conv_post = _norm(
            nn.Conv1d(1024, 1, 3, 1, padding=1), use_spectral_norm=use_spectral_norm
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        if x.ndim != 3 or x.shape[1] != 1:
            raise ValueError(f"expected waveform [B, 1, T], got {tuple(x.shape)}")
        fmap: list[torch.Tensor] = []
        for conv in self.convs:
            x = F.leaky_relu(conv(x), LRELU_SLOPE)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        return torch.flatten(x, 1), fmap


class MultiScaleDiscriminator(nn.Module):
    """HiFi-GAN MSD at the native, ×2, and ×4 waveform scales."""

    def __init__(self):
        super().__init__()
        self.discriminators = nn.ModuleList(
            [
                ScaleDiscriminator(use_spectral_norm=True),
                ScaleDiscriminator(),
                ScaleDiscriminator(),
            ]
        )
        self.meanpools = nn.ModuleList(
            [
                nn.Identity(),
                nn.AvgPool1d(4, 2, padding=2),
                nn.AvgPool1d(4, 2, padding=2),
            ]
        )

    def forward(
        self, x: torch.Tensor
    ) -> tuple[list[torch.Tensor], list[list[torch.Tensor]]]:
        scores, features = [], []
        for discriminator, pool in zip(self.discriminators, self.meanpools):
            x = pool(x)
            score, fmap = discriminator(x)
            scores.append(score)
            features.append(fmap)
        return scores, features


class HiFiGanDiscriminators(nn.Module):
    """Combined MPD and MSD discriminator bank."""

    def __init__(self):
        super().__init__()
        self.mpd = MultiPeriodDiscriminator()
        self.msd = MultiScaleDiscriminator()

    def forward(
        self, x: torch.Tensor
    ) -> tuple[list[torch.Tensor], list[list[torch.Tensor]]]:
        mpd_scores, mpd_features = self.mpd(x)
        msd_scores, msd_features = self.msd(x)
        return mpd_scores + msd_scores, mpd_features + msd_features

"""sigma-VAE acoustic encoder (causal depth-wise conv stages)."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from minispeech.models.tokenizer.blocks import (
    CausalConv1d,
    CausalDownsample,
    build_stage_blocks,
)


# 6 downsampling factors: 2*4*5*5*4*4 = 3200 -> 24 kHz / 3200 = 7.5 Hz
DEFAULT_DOWN_FACTORS: tuple[int, ...] = (2, 4, 5, 5, 4, 4)


@dataclass(frozen=True)
class EncoderConfig:
    """T0 default is about 1/4 of the 340M reference encoder."""

    in_channels: int = 1
    latent_dim: int = 64
    # 7 stage widths (pre-down0 ... post-down5); T0 narrow channels.
    channels: tuple[int, ...] = (128, 256, 384, 512, 640, 768, 896)
    blocks_per_stage: tuple[int, ...] = (2, 2, 2, 3, 3, 4, 4)
    down_factors: tuple[int, ...] = DEFAULT_DOWN_FACTORS
    kernel_size: int = 7
    ffn_mult: int = 4
    dropout: float = 0.0
    # Fixed posterior std (sigma-VAE); encode(sample=False) returns mean.
    vae_std: float = 0.625

    def __post_init__(self) -> None:
        if len(self.channels) != len(self.blocks_per_stage):
            raise ValueError("channels and blocks_per_stage must match in length")
        if len(self.down_factors) != len(self.channels) - 1:
            raise ValueError("need len(channels) == len(down_factors) + 1")
        hop = 1
        for f in self.down_factors:
            hop *= int(f)
        if hop != 3200:
            raise ValueError(f"down_factors must multiply to 3200 for 7.5 Hz, got {hop}")


class SigmaVAEEncoder(nn.Module):
    """Waveform -> continuous latents [B, T', D] at 7.5 Hz."""

    def __init__(self, cfg: EncoderConfig | None = None):
        super().__init__()
        self.cfg = cfg or EncoderConfig()
        c0 = self.cfg.channels[0]
        self.stem = CausalConv1d(self.cfg.in_channels, c0, kernel_size=7)

        stages: list[nn.Module] = []
        downs: list[nn.Module] = []
        for i, ch in enumerate(self.cfg.channels):
            stages.append(
                build_stage_blocks(
                    ch,
                    self.cfg.blocks_per_stage[i],
                    kernel_size=self.cfg.kernel_size,
                    ffn_mult=self.cfg.ffn_mult,
                    dropout=self.cfg.dropout,
                )
            )
            if i < len(self.cfg.down_factors):
                downs.append(
                    CausalDownsample(ch, self.cfg.channels[i + 1], self.cfg.down_factors[i])
                )
        self.stages = nn.ModuleList(stages)
        self.downs = nn.ModuleList(downs)
        last = self.cfg.channels[-1]
        self.to_mean = nn.Conv1d(last, self.cfg.latent_dim, kernel_size=1)
        self.register_buffer("vae_std", torch.tensor(float(self.cfg.vae_std)), persistent=False)

    @property
    def hop_length(self) -> int:
        h = 1
        for f in self.cfg.down_factors:
            h *= int(f)
        return h

    def forward(
        self,
        wav: torch.Tensor,
        *,
        sample: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode waveform to latents.

        wav: [B, 1, T] or [B, T] mono at 24 kHz.
        sample: if True, z = mean + vae_std * eps; if False, z = mean.

        Returns z, mean both [B, T', D].
        """
        if wav.dim() == 2:
            wav = wav.unsqueeze(1)
        x = self.stem(wav)
        for i, stage in enumerate(self.stages):
            x = stage(x)
            if i < len(self.downs):
                x = self.downs[i](x)
        mean = self.to_mean(x)
        if sample and self.training:
            eps = torch.randn_like(mean)
            z = mean + self.vae_std * eps
        else:
            z = mean
        return z.transpose(1, 2).contiguous(), mean.transpose(1, 2).contiguous()

"""sigma-VAE acoustic decoder (mirror of the encoder, causal)."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from minispeech.models.tokenizer.blocks import (
    CausalConv1d,
    CausalUpsample,
    build_stage_blocks,
)
from minispeech.models.tokenizer.encoder import DEFAULT_DOWN_FACTORS, EncoderConfig


@dataclass(frozen=True)
class DecoderConfig:
    out_channels: int = 1
    latent_dim: int = 64
    channels: tuple[int, ...] = (896, 768, 640, 512, 384, 256, 128)
    blocks_per_stage: tuple[int, ...] = (4, 4, 3, 3, 2, 2, 2)
    up_factors: tuple[int, ...] = tuple(reversed(DEFAULT_DOWN_FACTORS))
    kernel_size: int = 7
    ffn_mult: int = 4
    dropout: float = 0.0

    def __post_init__(self) -> None:
        if len(self.channels) != len(self.blocks_per_stage):
            raise ValueError("channels and blocks_per_stage must match")
        if len(self.up_factors) != len(self.channels) - 1:
            raise ValueError("need len(channels) == len(up_factors) + 1")
        hop = 1
        for f in self.up_factors:
            hop *= int(f)
        if hop != 3200:
            raise ValueError(f"up_factors must multiply to 3200, got {hop}")

    @classmethod
    def from_encoder(cls, enc: EncoderConfig) -> "DecoderConfig":
        return cls(
            out_channels=enc.in_channels,
            latent_dim=enc.latent_dim,
            channels=tuple(reversed(enc.channels)),
            blocks_per_stage=tuple(reversed(enc.blocks_per_stage)),
            up_factors=tuple(reversed(enc.down_factors)),
            kernel_size=enc.kernel_size,
            ffn_mult=enc.ffn_mult,
            dropout=enc.dropout,
        )


class SigmaVAEDecoder(nn.Module):
    """Latents [B, T', D] -> waveform [B, 1, T] (causal, streamable)."""

    def __init__(self, cfg: DecoderConfig | None = None):
        super().__init__()
        self.cfg = cfg or DecoderConfig()
        c0 = self.cfg.channels[0]
        self.from_latent = nn.Conv1d(self.cfg.latent_dim, c0, kernel_size=1)

        stages: list[nn.Module] = []
        ups: list[nn.Module] = []
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
            if i < len(self.cfg.up_factors):
                ups.append(
                    CausalUpsample(ch, self.cfg.channels[i + 1], self.cfg.up_factors[i])
                )
        self.stages = nn.ModuleList(stages)
        self.ups = nn.ModuleList(ups)
        self.head = CausalConv1d(self.cfg.channels[-1], self.cfg.out_channels, kernel_size=7)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latents [B, T', D] to waveform [B, 1, T]."""
        x = self.from_latent(z.transpose(1, 2).contiguous())
        for i, stage in enumerate(self.stages):
            x = stage(x)
            if i < len(self.ups):
                x = self.ups[i](x)
        return self.head(x)  # no tanh: was saturating / killing corr

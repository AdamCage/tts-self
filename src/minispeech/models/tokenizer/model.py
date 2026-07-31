"""Acoustic sigma-VAE tokenizer: encoder + decoder wrapper."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn

from minispeech.models.tokenizer.decoder import DecoderConfig, SigmaVAEDecoder
from minispeech.models.tokenizer.encoder import EncoderConfig, SigmaVAEEncoder


@dataclass
class TokenizerOutput:
    recon: torch.Tensor
    z: torch.Tensor
    mean: torch.Tensor


class AcousticTokenizer(nn.Module):
    """Mirror-symmetric sigma-VAE acoustic tokenizer (Track N).

    Encode TTS targets with sample=False (deterministic mean). Training may
    sample z = mean + vae_std * eps.
    """

    def __init__(
        self,
        enc_cfg: EncoderConfig | None = None,
        dec_cfg: DecoderConfig | None = None,
    ):
        super().__init__()
        self.enc_cfg = enc_cfg or EncoderConfig()
        self.dec_cfg = dec_cfg or DecoderConfig.from_encoder(self.enc_cfg)
        self.encoder = SigmaVAEEncoder(self.enc_cfg)
        self.decoder = SigmaVAEDecoder(self.dec_cfg)

    @property
    def hop_length(self) -> int:
        return self.encoder.hop_length

    @property
    def latent_dim(self) -> int:
        return int(self.enc_cfg.latent_dim)

    @property
    def vae_std(self) -> float:
        return float(self.enc_cfg.vae_std)

    def encode(self, wav: torch.Tensor, *, sample: bool = False) -> torch.Tensor:
        z, _ = self.encoder(wav, sample=sample)
        return z

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, wav: torch.Tensor, *, sample: bool | None = None) -> TokenizerOutput:
        if sample is None:
            sample = self.training
        z, mean = self.encoder(wav, sample=sample)
        recon = self.decoder(z)
        if wav.dim() == 2:
            t = wav.shape[-1]
        else:
            t = wav.shape[-1]
        if recon.shape[-1] > t:
            recon = recon[..., :t]
        elif recon.shape[-1] < t:
            pad = t - recon.shape[-1]
            recon = torch.nn.functional.pad(recon, (0, pad))
        return TokenizerOutput(recon=recon, z=z, mean=mean)

    def count_parameters(self) -> dict[str, int]:
        def _n(m: nn.Module) -> int:
            return sum(p.numel() for p in m.parameters())

        return {
            "encoder": _n(self.encoder),
            "decoder": _n(self.decoder),
            "total": _n(self),
        }

    @classmethod
    def from_config_dict(cls, model_cfg: dict[str, Any]) -> "AcousticTokenizer":
        """Build from configs/tokenizer_t0.yaml model: section."""
        tok = model_cfg.get("tokenizer", model_cfg)
        channels = tuple(tok.get("channels", EncoderConfig().channels))
        blocks = tuple(tok.get("blocks_per_stage", EncoderConfig().blocks_per_stage))
        downs = tuple(tok.get("down_factors", EncoderConfig().down_factors))
        enc = EncoderConfig(
            in_channels=int(tok.get("in_channels", 1)),
            latent_dim=int(tok.get("latent_dim", 64)),
            channels=channels,
            blocks_per_stage=blocks,
            down_factors=downs,
            kernel_size=int(tok.get("kernel_size", 7)),
            ffn_mult=int(tok.get("ffn_mult", 4)),
            dropout=float(tok.get("dropout", 0.0)),
            vae_std=float(tok.get("vae_std", 0.625)),
        )
        return cls(enc_cfg=enc, dec_cfg=DecoderConfig.from_encoder(enc))

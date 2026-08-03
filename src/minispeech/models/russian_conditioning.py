"""Parallel Russian feature channels for text-prefix tokens.

Channels are **off by default** until a text-only base checkpoint passes
content gates. When enabled, embeddings are summed into text token embeddings
(not substituted for stressed natural Russian BPE).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class RussianConditioningConfig:
    enabled: bool = False
    phoneme_vocab: int = 256
    stress_vocab: int = 4  # pad / none / stressed / other
    span_vocab: int = 16
    word_bound_vocab: int = 2
    phrase_bound_vocab: int = 2


class RussianConditioning(nn.Module):
    """Token-aligned additive channels: phoneme, stress, bounds, span type."""

    def __init__(self, d_model: int, cfg: RussianConditioningConfig | None = None):
        super().__init__()
        self.cfg = cfg or RussianConditioningConfig()
        self.enabled = bool(self.cfg.enabled)
        c = self.cfg
        self.phoneme_emb = nn.Embedding(c.phoneme_vocab, d_model, padding_idx=0)
        self.stress_emb = nn.Embedding(c.stress_vocab, d_model, padding_idx=0)
        self.word_bound_emb = nn.Embedding(c.word_bound_vocab, d_model, padding_idx=0)
        self.phrase_bound_emb = nn.Embedding(c.phrase_bound_vocab, d_model, padding_idx=0)
        self.span_emb = nn.Embedding(c.span_vocab, d_model, padding_idx=0)
        self.proj = nn.Linear(d_model, d_model, bias=False)
        self.norm = nn.LayerNorm(d_model)
        # Keep channels near-zero at init so enabling mid-run is soft.
        nn.init.zeros_(self.proj.weight)

    def forward(
        self,
        text_emb: torch.Tensor,
        *,
        phoneme_ids: torch.Tensor | None = None,
        stress: torch.Tensor | None = None,
        word_bound: torch.Tensor | None = None,
        phrase_bound: torch.Tensor | None = None,
        span_type: torch.Tensor | None = None,
        text_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Add RU features to ``text_emb`` ``[B, L, D]`` when enabled."""
        if not self.enabled:
            return text_emb

        B, L, D = text_emb.shape
        device = text_emb.device
        zero = torch.zeros(B, L, dtype=torch.long, device=device)

        def _ch(x: torch.Tensor | None) -> torch.Tensor:
            if x is None:
                return zero
            if x.shape[:2] != (B, L):
                raise ValueError(
                    f"RU channel shape {tuple(x.shape)} incompatible with text [B,L]=[{B},{L}]"
                )
            return x.long()

        additive = (
            self.phoneme_emb(_ch(phoneme_ids))
            + self.stress_emb(_ch(stress))
            + self.word_bound_emb(_ch(word_bound))
            + self.phrase_bound_emb(_ch(phrase_bound))
            + self.span_emb(_ch(span_type))
        )
        out = text_emb + self.proj(additive)
        out = self.norm(out)
        if text_mask is not None:
            out = out * text_mask.to(dtype=out.dtype).unsqueeze(-1)
        return out

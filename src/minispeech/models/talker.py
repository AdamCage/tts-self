"""Causal Talker with cross-attention to text memory (+ optional non-AR fallback)."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class TalkerOutput:
    hidden: torch.Tensor  # [B, T, D]
    cross_attn: torch.Tensor | None  # [B, H, T, L] averaged over layers or last


class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, max_seq: int = 4096, base: float = 10000.0):
        super().__init__()
        inv = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv, persistent=False)
        self.max_seq = max_seq

    def forward(self, seq_len: int, device, dtype):
        t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        return emb.cos().to(dtype), emb.sin().to(dtype)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    return torch.cat([-x2, x1], dim=-1)


def apply_rope(q, k, cos, sin):
    # q,k: [B, H, T, D]
    cos = cos[None, None, :, :]
    sin = sin[None, None, :, :]
    q = q * cos + _rotate_half(q) * sin
    k = k * cos + _rotate_half(k) * sin
    return q, k


class TalkerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        ffn_dim: int,
        dropout: float = 0.1,
        causal: bool = True,
    ):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.causal = causal
        self.norm1 = nn.LayerNorm(d_model)
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out = nn.Linear(d_model, d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.q_cross = nn.Linear(d_model, d_model)
        self.kv_cross = nn.Linear(d_model, 2 * d_model)
        self.out_cross = nn.Linear(d_model, d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, d_model),
            nn.Dropout(dropout),
        )
        self.dropout = nn.Dropout(dropout)

    def _shape(self, x: torch.Tensor, B: int, T: int) -> torch.Tensor:
        return x.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

    def forward(
        self,
        x: torch.Tensor,
        text_memory: torch.Tensor,
        self_mask: torch.Tensor | None,
        text_mask: torch.Tensor | None,
        rope_cos_sin: tuple[torch.Tensor, torch.Tensor] | None,
        attn_bias: torch.Tensor | None = None,
        past_kv: tuple[torch.Tensor, torch.Tensor] | None = None,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, tuple[torch.Tensor, torch.Tensor] | None]:
        B, T, D = x.shape
        L = text_memory.shape[1]

        # Self-attention
        h = self.norm1(x)
        qkv = self.qkv(h)
        q, k, v = qkv.chunk(3, dim=-1)
        q, k, v = self._shape(q, B, T), self._shape(k, B, T), self._shape(v, B, T)
        if rope_cos_sin is not None:
            q, k = apply_rope(q, k, rope_cos_sin[0], rope_cos_sin[1])
        if past_kv is not None:
            pk, pv = past_kv
            k = torch.cat([pk, k], dim=2)
            v = torch.cat([pv, v], dim=2)
        present = (k, v) if use_cache else None
        attn_mask = None
        if self.causal and past_kv is None:
            causal = torch.triu(torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1)
            attn_mask = causal
        # key padding from frame mask: True = keep
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if attn_mask is not None:
            scores = scores.masked_fill(attn_mask[None, None], float("-inf"))
        if self_mask is not None and past_kv is None:
            # self_mask [B,T] True=valid
            km = ~self_mask[:, None, None, :]
            scores = scores.masked_fill(km, float("-inf"))
        a = torch.softmax(scores, dim=-1)
        a = self.dropout(a)
        self_out = torch.matmul(a, v).transpose(1, 2).contiguous().view(B, T, D)
        x = x + self.out(self_out)

        # Cross-attention
        h = self.norm2(x)
        q = self._shape(self.q_cross(h), B, T)
        kv = self.kv_cross(text_memory)
        k, v = kv.chunk(2, dim=-1)
        k, v = self._shape(k, B, L), self._shape(v, B, L)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if attn_bias is not None:
            # attn_bias [B, T, L] log-prior
            scores = scores + attn_bias[:, None, :, :]
        if text_mask is not None:
            scores = scores.masked_fill(~text_mask[:, None, None, :], float("-inf"))
        ca = torch.softmax(scores, dim=-1)
        ca = self.dropout(ca)
        cross_out = torch.matmul(ca, v).transpose(1, 2).contiguous().view(B, T, D)
        x = x + self.out_cross(cross_out)

        x = x + self.ff(self.norm3(x))
        return x, ca.detach(), present


class Talker(nn.Module):
    def __init__(
        self,
        d_model: int = 512,
        num_layers: int = 4,
        num_heads: int = 8,
        ffn_dim: int = 2048,
        latent_dim: int = 64,
        dropout: float = 0.1,
        rope: bool = True,
        causal: bool = True,
    ):
        super().__init__()
        self.d_model = d_model
        self.causal = causal
        self.latent_in = nn.Linear(latent_dim, d_model)
        self.bos = nn.Parameter(torch.randn(latent_dim) * 0.02)
        self.layers = nn.ModuleList(
            [
                TalkerBlock(d_model, num_heads, ffn_dim, dropout=dropout, causal=causal)
                for _ in range(num_layers)
            ]
        )
        self.norm = nn.LayerNorm(d_model)
        self.rope = RotaryEmbedding(d_model // num_heads) if rope else None
        # Non-AR length regulator projection (used when causal=False with expanded memory)
        self.token_expand = nn.Linear(d_model, d_model)

    def apply_bos(self, z_input: torch.Tensor) -> torch.Tensor:
        """Replace placeholder zeros at position 0 with learned BOS when needed."""
        out = z_input.clone()
        out[:, 0] = self.bos.to(dtype=out.dtype)
        return out

    def forward(
        self,
        z_input: torch.Tensor,
        text_memory: torch.Tensor,
        frame_mask: torch.Tensor | None = None,
        text_mask: torch.Tensor | None = None,
        align_prior: torch.Tensor | None = None,
        guided_weight: float = 0.0,
    ) -> TalkerOutput:
        """
        z_input: [B, T, 64] already shifted (pos0 = BOS)
        text_memory: [B, L, D]
        """
        z_input = self.apply_bos(z_input)
        x = self.latent_in(z_input)
        B, T, _ = x.shape
        rope = None
        if self.rope is not None:
            rope = self.rope(T, x.device, x.dtype)

        attn_bias = None
        if align_prior is not None and guided_weight > 0:
            # soft bias toward prior
            attn_bias = guided_weight * torch.log(align_prior.clamp_min(1e-6))

        last_ca = None
        for layer in self.layers:
            x, ca, _ = layer(x, text_memory, frame_mask, text_mask, rope, attn_bias)
            last_ca = ca
        return TalkerOutput(hidden=self.norm(x), cross_attn=last_ca)

    def forward_step(
        self,
        z_t: torch.Tensor,
        text_memory: torch.Tensor,
        past_kvs: list[tuple[torch.Tensor, torch.Tensor]] | None,
        text_mask: torch.Tensor | None = None,
        position: int = 0,
    ) -> tuple[TalkerOutput, list[tuple[torch.Tensor, torch.Tensor]]]:
        """
        Incremental step with self-attn KV cache.
        z_t: [B, 1, 64] — BOS at position 0, else previous latent.
        """
        x = self.latent_in(z_t)
        B, T, _ = x.shape
        assert T == 1
        rope = None
        if self.rope is not None:
            # RoPE for absolute positions [0..pos]; take last row for current token
            cos, sin = self.rope(position + 1, x.device, x.dtype)
            rope = (cos[position : position + 1], sin[position : position + 1])

        new_past: list[tuple[torch.Tensor, torch.Tensor]] = []
        last_ca = None
        for i, layer in enumerate(self.layers):
            pk = past_kvs[i] if past_kvs is not None else None
            x, ca, present = layer(
                x,
                text_memory,
                None,
                text_mask,
                rope,
                None,
                past_kv=pk,
                use_cache=True,
            )
            assert present is not None
            new_past.append(present)
            last_ca = ca
        return TalkerOutput(hidden=self.norm(x), cross_attn=last_ca), new_past


class NonARTalker(nn.Module):
    """Fallback: duration expand + bidirectional Talker blocks (FastSpeech-style)."""

    def __init__(self, talker: Talker):
        super().__init__()
        self.talker = talker
        # force non-causal
        for blk in self.talker.layers:
            blk.causal = False
        self.talker.causal = False

    def length_regulate(self, text_memory: torch.Tensor, token_durations: torch.Tensor) -> torch.Tensor:
        """text_memory [B,L,D], token_durations [B,L] int/float -> [B,T,D]."""
        B, L, D = text_memory.shape
        outs = []
        max_t = int(token_durations.sum(dim=-1).max().item())
        for b in range(B):
            frames = []
            for l in range(L):
                d = int(token_durations[b, l].item())
                if d <= 0:
                    continue
                frames.append(text_memory[b, l : l + 1].expand(d, -1))
            if frames:
                seq = torch.cat(frames, dim=0)
            else:
                seq = text_memory.new_zeros(1, D)
            if seq.shape[0] < max_t:
                pad = text_memory.new_zeros(max_t - seq.shape[0], D)
                seq = torch.cat([seq, pad], dim=0)
            outs.append(seq[:max_t])
        return torch.stack(outs, dim=0)

    def forward(
        self,
        text_memory: torch.Tensor,
        token_durations: torch.Tensor,
        text_mask: torch.Tensor | None = None,
        frame_mask: torch.Tensor | None = None,
    ) -> TalkerOutput:
        expanded = self.length_regulate(text_memory, token_durations)
        # Use zeros as z_input (content comes from expanded memory via cross-attn / residual add)
        B, T, D = expanded.shape
        z_input = expanded.new_zeros(B, T, self.talker.latent_in.in_features)
        # Inject expanded text as starting hidden via latent_in bypass
        x = expanded
        rope = None
        if self.talker.rope is not None:
            rope = self.talker.rope(T, x.device, x.dtype)
        last_ca = None
        for layer in self.talker.layers:
            x, ca, _ = layer(x, text_memory, frame_mask, text_mask, rope, None)
            last_ca = ca
        return TalkerOutput(hidden=self.talker.norm(x), cross_attn=last_ca)

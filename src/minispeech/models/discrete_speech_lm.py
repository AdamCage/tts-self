"""Own causal discrete Speech LM (~300M default / ~443M D1b; tiny for CPU tests).

Architecture contract (D1 / D1b)
--------------------------------
- Full text as **prefix** before the acoustic stream (no dual-track streaming).
- CB0 predicted autoregressively; CB1–CB15 via MTP conditioned on ``h`` and CB0.
- Speaker-ID embeddings only (no zero-shot cloning).
- No VibeVoice continuous / DDPM / mel losses.
- Termination is the frame-level ``stop_head`` only (last valid ``frame_mask``
  frame). ``SPEECH_EOS_ID=2048`` is cache-schema only, not a trainable CB0 EOS.
  Shared infer knobs: ``minispeech.infer.discrete_config``.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from minispeech.constants import QWEN3TTS_12HZ, get_target
from minispeech.models.multi_codebook_head import MultiCodebookHead, MultiCodebookOutput
from minispeech.models.russian_conditioning import (
    RussianConditioning,
    RussianConditioningConfig,
)


@dataclass
class DiscreteSpeechLMConfig:
    """Config-sized Speech LM. Use :meth:`tiny` for CPU unit tests."""

    d_model: int = 1024
    n_layers: int = 12
    n_heads: int = 16
    ffn_dim: int = 2816
    vocab_size: int = 151936
    max_speakers: int = 256
    dropout: float = 0.0
    max_seq: int = 8192
    rope_theta: float = 10_000.0
    n_codebooks: int = 16
    codebook_size: int = 2048
    termination_mode: str = "frame_stop_v1"
    eos_id: int | None = None
    semantic_vocab_size: int | None = None
    codec_vocab_size: int | None = None
    mtp_hidden: int | None = None
    russian_conditioning: bool = False
    phoneme_vocab: int = 256
    stress_vocab: int = 4
    span_vocab: int = 16
    pad_token_id: int = 0
    target: str = QWEN3TTS_12HZ.name

    @classmethod
    def tiny(cls) -> DiscreteSpeechLMConfig:
        """Small config for CPU causality / overfit tests."""
        return cls(
            d_model=64,
            n_layers=2,
            n_heads=4,
            ffn_dim=128,
            vocab_size=128,
            max_speakers=8,
            dropout=0.0,
            max_seq=512,
            n_codebooks=16,
            codebook_size=32,
            mtp_hidden=64,
            russian_conditioning=False,
            pad_token_id=0,
        )

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> DiscreteSpeechLMConfig:
        m = d.get("model", d)
        known = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        kwargs = {k: v for k, v in m.items() if k in known}
        # Nested RU flag
        if "russian_conditioning" in m and isinstance(m["russian_conditioning"], dict):
            kwargs["russian_conditioning"] = bool(
                m["russian_conditioning"].get("enabled", False)
            )
            for kk in ("phoneme_vocab", "stress_vocab", "span_vocab"):
                if kk in m["russian_conditioning"]:
                    kwargs[kk] = int(m["russian_conditioning"][kk])
        termination = m.get("termination")
        if isinstance(termination, dict):
            kwargs["termination_mode"] = termination.get("mode", "frame_stop_v1")
            for key in ("eos_id", "semantic_vocab_size", "codec_vocab_size"):
                if key in termination:
                    kwargs[key] = int(termination[key])
        return cls(**kwargs)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class DiscreteLMOutput:
    hidden: torch.Tensor  # [B, S, D] full sequence
    acoustic_hidden: torch.Tensor  # [B, T, D]
    heads: MultiCodebookOutput
    text_len: torch.Tensor  # [B]
    loss: dict[str, torch.Tensor] | None = None
    stop_logits: torch.Tensor | None = None  # [B, T]
    extras: dict[str, Any] = field(default_factory=dict)


@dataclass
class DiscreteGenerationOutput:
    """Canonical semantic-EOS generation result; EOS is excluded from ``codes``."""

    codes: torch.Tensor
    frame_mask: torch.Tensor
    lengths: torch.Tensor
    rows: list[dict[str, Any]]


class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, max_seq: int = 8192, base: float = 10_000.0):
        super().__init__()
        inv = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv, persistent=False)
        self.max_seq = max_seq

    def forward(
        self,
        position_ids: torch.Tensor | int,
        device=None,
        dtype=None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Cos/sin for positions.

        ``position_ids`` may be:
        - ``int`` seq length → legacy ``arange(S)`` → cos/sin ``[S, Dh]``
        - 1-D ``[S]`` → cos/sin ``[S, Dh]``
        - 2-D ``[B, S]`` → cos/sin ``[B, S, Dh]`` (per-sample RoPE)
        """
        if isinstance(position_ids, int):
            if device is None:
                device = self.inv_freq.device
            t = torch.arange(position_ids, device=device, dtype=self.inv_freq.dtype)
            freqs = torch.outer(t, self.inv_freq.to(device=device))
            emb = torch.cat([freqs, freqs], dim=-1)
            if dtype is None:
                return emb.cos(), emb.sin()
            return emb.cos().to(dtype), emb.sin().to(dtype)

        if device is None:
            device = position_ids.device
        pos = position_ids.to(device=device, dtype=self.inv_freq.dtype)
        # freqs: [..., Dh/2]
        freqs = pos.unsqueeze(-1) * self.inv_freq.to(device=device)
        emb = torch.cat([freqs, freqs], dim=-1)
        if dtype is None:
            return emb.cos(), emb.sin()
        return emb.cos().to(dtype), emb.sin().to(dtype)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    return torch.cat([-x2, x1], dim=-1)


def apply_rope(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    # q,k: [B, H, S, Dh]; cos/sin: [S, Dh] or [B, S, Dh]
    if cos.ndim == 2:
        cos = cos[None, None, :, :]
        sin = sin[None, None, :, :]
    elif cos.ndim == 3:
        cos = cos[:, None, :, :]
        sin = sin[:, None, :, :]
    else:
        raise ValueError(f"cos/sin expected 2-D or 3-D, got {tuple(cos.shape)}")
    q = q * cos + _rotate_half(q) * sin
    k = k * cos + _rotate_half(k) * sin
    return q, k


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out = nn.Linear(d_model, d_model, bias=False)
        self.dropout = float(dropout)

    def forward(
        self,
        x: torch.Tensor,
        *,
        rope: tuple[torch.Tensor, torch.Tensor] | None,
        key_padding_mask: torch.Tensor | None,
        attn_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        B, S, D = x.shape
        qkv = self.qkv(x).view(B, S, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q = q.transpose(1, 2)  # [B,H,S,Dh]
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        if rope is not None:
            q, k = apply_rope(q, k, rope[0], rope[1])

        # Build additive mask: True/bool keep → float mask for SDPA
        # key_padding_mask: [B, S] True=valid
        float_mask = None
        if attn_mask is not None or key_padding_mask is not None:
            float_mask = torch.zeros(B, 1, S, S, device=x.device, dtype=q.dtype)
            if attn_mask is not None:
                # attn_mask True = block
                float_mask = float_mask.masked_fill(
                    attn_mask[None, None, :, :], torch.finfo(q.dtype).min
                )
            if key_padding_mask is not None:
                # block keys that are padding
                km = ~key_padding_mask[:, None, None, :]
                float_mask = float_mask.masked_fill(km, torch.finfo(q.dtype).min)

        if hasattr(F, "scaled_dot_product_attention"):
            drop = self.dropout if self.training else 0.0
            # When float_mask is set, is_causal must be False (mask already causal).
            y = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=float_mask,
                dropout_p=drop,
                is_causal=float_mask is None,
            )
        else:  # pragma: no cover
            scale = 1.0 / math.sqrt(self.head_dim)
            scores = torch.matmul(q, k.transpose(-2, -1)) * scale
            if float_mask is not None:
                scores = scores + float_mask
            else:
                causal = torch.triu(
                    torch.ones(S, S, device=x.device, dtype=torch.bool), diagonal=1
                )
                scores = scores.masked_fill(causal[None, None], float("-inf"))
            a = torch.softmax(scores, dim=-1)
            if self.training and self.dropout > 0:
                a = F.dropout(a, p=self.dropout)
            y = torch.matmul(a, v)

        y = y.transpose(1, 2).contiguous().view(B, S, D)
        return self.out(y)


class DecoderBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, ffn_dim: int, dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads, dropout=dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, d_model),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        x: torch.Tensor,
        *,
        rope: tuple[torch.Tensor, torch.Tensor] | None,
        key_padding_mask: torch.Tensor | None,
        attn_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        h = self.norm1(x)
        x = x + self.attn(
            h, rope=rope, key_padding_mask=key_padding_mask, attn_mask=attn_mask
        )
        x = x + self.ff(self.norm2(x))
        return x


class DiscreteSpeechLM(nn.Module):
    """Text-prefix causal LM over discrete multi-codebook speech tokens."""

    def __init__(self, cfg: DiscreteSpeechLMConfig | None = None):
        super().__init__()
        self.cfg = cfg or DiscreteSpeechLMConfig()
        c = self.cfg
        spec = get_target(c.target)
        self.n_codebooks = int(c.n_codebooks or spec.n_codebooks or 16)
        self.codebook_size = int(c.codebook_size or spec.codebook_size or 2048)
        self.termination_mode = str(c.termination_mode)
        self.codec_vocab_size = int(c.codec_vocab_size or self.codebook_size)
        self.eos_id = int(c.eos_id if c.eos_id is not None else self.codebook_size)
        self.semantic_vocab_size = int(
            c.semantic_vocab_size
            if c.semantic_vocab_size is not None
            else (self.codebook_size + 1 if self.termination_mode == "semantic_eos_v1" else self.codebook_size)
        )
        if self.termination_mode not in {"frame_stop_v1", "semantic_eos_v1"}:
            raise ValueError(f"unknown termination mode: {self.termination_mode}")
        if self.codec_vocab_size != self.codebook_size:
            raise ValueError("codec_vocab_size must equal codebook_size")
        if self.eos_id != self.codebook_size:
            raise ValueError("eos_id must equal codebook_size")
        expected_semantic = self.codebook_size + 1 if self.termination_mode == "semantic_eos_v1" else self.codebook_size
        if self.semantic_vocab_size != expected_semantic:
            raise ValueError(
                f"semantic_vocab_size={self.semantic_vocab_size} incompatible with "
                f"termination_mode={self.termination_mode}"
            )
        self.d_model = c.d_model
        self.pad_token_id = int(c.pad_token_id)

        self.text_emb = nn.Embedding(c.vocab_size, c.d_model, padding_idx=c.pad_token_id)
        self.speech_emb = nn.Embedding(self.codebook_size, c.d_model)
        self.speech_start = nn.Parameter(torch.randn(c.d_model) * 0.02)
        self.speaker_emb = nn.Embedding(c.max_speakers, c.d_model)
        self.modality_speech = nn.Parameter(torch.zeros(c.d_model))

        ru_cfg = RussianConditioningConfig(
            enabled=bool(c.russian_conditioning),
            phoneme_vocab=c.phoneme_vocab,
            stress_vocab=c.stress_vocab,
            span_vocab=c.span_vocab,
        )
        self.ru_cond = RussianConditioning(c.d_model, ru_cfg)

        self.blocks = nn.ModuleList(
            [
                DecoderBlock(c.d_model, c.n_heads, c.ffn_dim, dropout=c.dropout)
                for _ in range(c.n_layers)
            ]
        )
        self.norm = nn.LayerNorm(c.d_model)
        self.rope = RotaryEmbedding(
            c.d_model // c.n_heads, max_seq=c.max_seq, base=c.rope_theta
        )
        self.heads = MultiCodebookHead(
            c.d_model,
            n_codebooks=self.n_codebooks,
            codebook_size=self.codebook_size,
            semantic_vocab_size=self.semantic_vocab_size,
            mtp_hidden=c.mtp_hidden,
            dropout=c.dropout,
        )
        # Frame-level stop head — sole D1b termination (1 on last valid frame).
        # Not a CB0 id: SPEECH_EOS_ID=2048 sits outside codebook_size=2048 and
        # must not be treated as trainable EOS. Backbone = text + acoustic frames
        # only (no EOS token in the LM stream).
        # Input = [acoustic_hidden || normalized frame index] to reduce FR exposure bias.
        self.stop_head: nn.Module | None
        self.stop_head = (
            nn.Sequential(
                nn.Linear(c.d_model + 1, max(32, c.d_model // 4)),
                nn.GELU(),
                nn.Linear(max(32, c.d_model // 4), 1),
            )
            if self.termination_mode == "frame_stop_v1"
            else None
        )
        self.stop_pos_scale = 250.0  # ~20s @ 12.5 Hz
        self.drop = nn.Dropout(c.dropout)

    def _stop_logits(self, acoustic: torch.Tensor, *, frame_index: torch.Tensor) -> torch.Tensor:
        """``acoustic`` [B,T,D], ``frame_index`` [B,T] float → logits [B,T]."""
        if self.stop_head is None:
            raise RuntimeError("semantic_eos_v1 has no stop_head")
        pos = (frame_index / float(self.stop_pos_scale)).unsqueeze(-1)
        return self.stop_head(torch.cat([acoustic, pos.to(dtype=acoustic.dtype)], dim=-1)).squeeze(-1)

    def _validate_speaker_id(self, speaker_id: torch.Tensor) -> torch.Tensor:
        """Fail-closed range check (no silent clamp)."""
        if speaker_id.dtype != torch.long:
            speaker_id = speaker_id.long()
        n = int(self.speaker_emb.num_embeddings)
        if speaker_id.numel() == 0:
            return speaker_id
        lo = int(speaker_id.min().item())
        hi = int(speaker_id.max().item())
        if lo < 0 or hi >= n:
            raise ValueError(
                f"speaker_id out of range for embedding table size {n}: "
                f"min={lo} max={hi} (no clamp)"
            )
        return speaker_id

    @staticmethod
    def build_position_ids(
        *,
        text_lens: torch.Tensor,
        acoustic_lens: torch.Tensor,
        L: int,
        T: int,
    ) -> torch.Tensor:
        """Per-sample RoPE positions for physical layout ``[text_pad || speech]``.

        - real text tokens: ``0 .. text_len-1``
        - text / acoustic padding: dummy ``0`` (blocked by key padding mask)
        - speech frame ``t``: ``text_len + t``
        """
        B = int(text_lens.shape[0])
        device = text_lens.device
        pos = torch.zeros(B, L + T, dtype=torch.long, device=device)
        text_arange = torch.arange(L, device=device).unsqueeze(0).expand(B, -1)
        speech_arange = torch.arange(T, device=device).unsqueeze(0).expand(B, -1)
        tl = text_lens.long().view(B, 1)
        al = acoustic_lens.long().view(B, 1)
        text_valid = text_arange < tl
        speech_valid = speech_arange < al
        pos[:, :L] = torch.where(text_valid, text_arange, torch.zeros_like(text_arange))
        pos[:, L:] = torch.where(
            speech_valid, tl + speech_arange, torch.zeros_like(speech_arange)
        )
        return pos

    @classmethod
    def from_config_dict(cls, d: dict[str, Any]) -> DiscreteSpeechLM:
        return cls(DiscreteSpeechLMConfig.from_dict(d))

    def count_parameters(self, *, trainable_only: bool = False) -> dict[str, int]:
        def _n(params) -> int:
            return int(sum(p.numel() for p in params))

        total = _n(self.parameters())
        trainable = _n(p for p in self.parameters() if p.requires_grad)
        textish = _n(
            list(self.text_emb.parameters())
            + list(self.blocks.parameters())
            + list(self.norm.parameters())
        )
        speechish = _n(
            [self.speech_start, self.modality_speech]
            + list(self.speech_emb.parameters())
            + list(self.speaker_emb.parameters())
            + list(self.heads.parameters())
            + (list(self.stop_head.parameters()) if self.stop_head is not None else [])
            + list(self.ru_cond.parameters())
        )
        return {
            "total": total,
            "trainable": trainable if trainable_only else trainable,
            "text_blocks_and_emb": textish,
            "speech_heads_adapters": speechish,
        }

    def build_inputs(
        self,
        input_ids: torch.Tensor,
        codes: torch.Tensor,
        *,
        text_mask: torch.Tensor | None = None,
        frame_mask: torch.Tensor | None = None,
        semantic_mask: torch.Tensor | None = None,
        speaker_id: torch.Tensor | None = None,
        phoneme_ids: torch.Tensor | None = None,
        stress: torch.Tensor | None = None,
        word_bound: torch.Tensor | None = None,
        phrase_bound: torch.Tensor | None = None,
        span_type: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Pack text prefix + AR-shifted speech stream.

        Acoustic teacher-forced input at frame ``t`` is:
        - ``t == 0``: learned ``speech_start`` (+ speaker)
        - ``t > 0``: embedding of ``codes[:, t-1, 0]``  (shift ``c_t → c_{t+1}``)

        Targets at acoustic positions are ``codes[:, t, :]``.
        """
        B, L = input_ids.shape
        frame_T = codes.shape[1]
        T = frame_T + 1 if self.termination_mode == "semantic_eos_v1" else frame_T
        if codes.shape[-1] != self.n_codebooks:
            raise ValueError(
                f"codes last dim {codes.shape[-1]} != n_codebooks {self.n_codebooks}"
            )
        if text_mask is None:
            text_mask = input_ids != self.pad_token_id
        if frame_mask is None:
            frame_mask = torch.ones(B, frame_T, dtype=torch.bool, device=codes.device)
        if self.termination_mode == "semantic_eos_v1":
            if semantic_mask is None:
                semantic_mask = torch.zeros(B, T, dtype=torch.bool, device=codes.device)
                acoustic_lens = frame_mask.long().sum(dim=1)
                for i, n in enumerate(acoustic_lens.tolist()):
                    semantic_mask[i, : n + 1] = True
            if tuple(semantic_mask.shape) != (B, T):
                raise ValueError(f"semantic_mask shape {tuple(semantic_mask.shape)} != {(B, T)}")
        if speaker_id is None:
            speaker_id = torch.zeros(B, dtype=torch.long, device=codes.device)
        speaker_id = self._validate_speaker_id(speaker_id)

        text_e = self.text_emb(input_ids)
        text_e = self.ru_cond(
            text_e,
            phoneme_ids=phoneme_ids,
            stress=stress,
            word_bound=word_bound,
            phrase_bound=phrase_bound,
            span_type=span_type,
            text_mask=text_mask,
        )

        # Speech inputs: [speech_start, c0_0, ..., c0_{T-2}]
        speech_in = torch.zeros(B, T, self.d_model, device=codes.device, dtype=text_e.dtype)
        spk = self.speaker_emb(speaker_id)
        speech_in[:, 0] = self.speech_start.to(dtype=text_e.dtype) + spk
        if frame_T > 1:
            prev = codes[:, :-1, 0].clamp(0, self.codebook_size - 1)
            speech_in[:, 1:frame_T] = self.speech_emb(prev)
        if self.termination_mode == "semantic_eos_v1" and frame_T:
            # The EOS prediction slot consumes the last *real* CB0 token.
            last_indices = frame_mask.long().sum(dim=1).clamp_min(1) - 1
            last_cb0 = codes[torch.arange(B, device=codes.device), last_indices, 0]
            speech_in[torch.arange(B, device=codes.device), last_indices + 1] = self.speech_emb(
                last_cb0.clamp(0, self.codebook_size - 1)
            )
        speech_in = speech_in + self.modality_speech.to(dtype=text_e.dtype)

        embeds = torch.cat([text_e, speech_in], dim=1)  # [B, L+T, D]
        embeds = self.drop(embeds)

        acoustic_mask = semantic_mask if semantic_mask is not None else frame_mask
        key_padding = torch.cat([text_mask, acoustic_mask], dim=1)
        # Causal mask over full sequence length
        S = L + T
        causal = torch.triu(
            torch.ones(S, S, device=codes.device, dtype=torch.bool), diagonal=1
        )
        text_lens = text_mask.long().sum(dim=1)
        acoustic_lens = acoustic_mask.long().sum(dim=1)
        position_ids = self.build_position_ids(
            text_lens=text_lens, acoustic_lens=acoustic_lens, L=L, T=T
        )
        return {
            "embeds": embeds,
            "key_padding_mask": key_padding,
            "attn_mask": causal,
            "position_ids": position_ids,
            "text_lens": text_lens,
            "acoustic_lens": acoustic_lens,
            "frame_mask": frame_mask,
            "text_mask": text_mask,
            "L": torch.tensor(L, device=codes.device),
            "T": torch.tensor(T, device=codes.device),
        }

    def forward_backbone(
        self,
        embeds: torch.Tensor,
        *,
        key_padding_mask: torch.Tensor | None,
        attn_mask: torch.Tensor | None,
        position_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        B, S, _ = embeds.shape
        if position_ids is None:
            position_ids = torch.arange(S, device=embeds.device).unsqueeze(0).expand(B, -1)
        elif position_ids.ndim == 1:
            position_ids = position_ids.unsqueeze(0).expand(B, -1)
        if tuple(position_ids.shape) != (B, S):
            raise ValueError(
                f"position_ids shape {tuple(position_ids.shape)} != {(B, S)}"
            )
        rope = self.rope(position_ids, embeds.device, embeds.dtype)
        x = embeds
        for block in self.blocks:
            x = block(
                x, rope=rope, key_padding_mask=key_padding_mask, attn_mask=attn_mask
            )
        return self.norm(x)

    @staticmethod
    def stop_targets_from_mask(frame_mask: torch.Tensor) -> torch.Tensor:
        """Binary stop labels: 1.0 on the last valid frame of each utterance."""
        B, T = frame_mask.shape
        lengths = frame_mask.long().sum(dim=1).clamp_min(0)
        targets = torch.zeros(B, T, device=frame_mask.device, dtype=torch.float32)
        for i in range(B):
            L_i = int(lengths[i].item())
            if L_i > 0:
                targets[i, L_i - 1] = 1.0
        return targets

    def forward(
        self,
        input_ids: torch.Tensor,
        codes: torch.Tensor,
        *,
        text_mask: torch.Tensor | None = None,
        frame_mask: torch.Tensor | None = None,
        semantic_targets: torch.Tensor | None = None,
        semantic_mask: torch.Tensor | None = None,
        speaker_id: torch.Tensor | None = None,
        phoneme_ids: torch.Tensor | None = None,
        stress: torch.Tensor | None = None,
        word_bound: torch.Tensor | None = None,
        phrase_bound: torch.Tensor | None = None,
        span_type: torch.Tensor | None = None,
        compute_loss: bool = True,
        cb0_weight: float = 1.0,
        residual_weights: torch.Tensor | None = None,
        stop_weight: float = 1.0,
        stop_pos_weight: float = 20.0,
        eos_loss_weight: float = 1.0,
    ) -> DiscreteLMOutput:
        packed = self.build_inputs(
            input_ids,
            codes,
            text_mask=text_mask,
            frame_mask=frame_mask,
            semantic_mask=semantic_mask,
            speaker_id=speaker_id,
            phoneme_ids=phoneme_ids,
            stress=stress,
            word_bound=word_bound,
            phrase_bound=phrase_bound,
            span_type=span_type,
        )
        hidden = self.forward_backbone(
            packed["embeds"],
            key_padding_mask=packed["key_padding_mask"],
            attn_mask=packed["attn_mask"],
            position_ids=packed["position_ids"],
        )
        L = int(packed["L"])
        T = int(packed["T"])
        acoustic = hidden[:, L : L + T, :]
        frame_T = codes.shape[1]
        # MTP conditioned on teacher CB0 for the *target* frame
        cb0_ids = codes[:, :, 0].clamp(0, self.codebook_size - 1)
        cb0_emb = self.speech_emb(cb0_ids)
        if self.termination_mode == "semantic_eos_v1":
            # EOS has a semantic logit but never a residual/MTP target.
            semantic_logits = self.heads.semantic(acoustic)
            residual_out = self.heads(acoustic[:, :frame_T], cb0_emb)
            head_out = MultiCodebookOutput(
                cb0_logits=semantic_logits, residual_logits=residual_out.residual_logits
            )
            stop_logits = None
        else:
            head_out = self.heads(acoustic, cb0_emb)
            B_a = acoustic.shape[0]
            frame_index = (
                torch.arange(T, device=acoustic.device, dtype=torch.float32)
                .unsqueeze(0)
                .expand(B_a, -1)
            )
            stop_logits = self._stop_logits(acoustic, frame_index=frame_index)

        loss_dict = None
        if compute_loss:
            if self.termination_mode == "semantic_eos_v1":
                if semantic_targets is None:
                    raise ValueError("semantic_eos_v1 requires semantic_targets")
                if semantic_mask is None:
                    raise ValueError("semantic_eos_v1 requires semantic_mask")
                if tuple(semantic_targets.shape) != tuple(head_out.cb0_logits.shape[:2]):
                    raise ValueError("semantic_targets must match semantic slots")
                fmask = packed["frame_mask"]
                ignore = -100
                real_target = semantic_targets[:, :frame_T].masked_fill(~fmask, ignore)
                cb0_frame_ce = F.cross_entropy(
                    head_out.cb0_logits[:, :frame_T].reshape(-1, self.semantic_vocab_size),
                    real_target.reshape(-1),
                    ignore_index=ignore,
                )
                residual_only = MultiCodebookOutput(
                    cb0_logits=head_out.cb0_logits[:, :frame_T],
                    residual_logits=head_out.residual_logits,
                )
                residual_ce = self.heads.loss(
                    residual_only, codes, fmask, cb0_weight=0.0, residual_weights=residual_weights
                )["ce_residual"]
                eos_positions = fmask.long().sum(dim=1)
                eos_logits = head_out.cb0_logits[
                    torch.arange(codes.shape[0], device=codes.device), eos_positions
                ]
                eos_targets = semantic_targets[
                    torch.arange(codes.shape[0], device=codes.device), eos_positions
                ]
                semantic_eos_ce = F.cross_entropy(eos_logits, eos_targets)
                eos_prob = torch.softmax(eos_logits.float(), dim=-1)[:, self.eos_id]
                eos_top = eos_logits.argmax(dim=-1)
                early_top = head_out.cb0_logits[:, :frame_T].argmax(dim=-1).eq(self.eos_id) & fmask
                total = float(cb0_weight) * cb0_frame_ce + residual_ce + float(eos_loss_weight) * semantic_eos_ce
                loss_dict = {
                    "total": total,
                    "loss_total": total.detach(),
                    "ce_cb0": cb0_frame_ce.detach(),
                    "ce_cb0_frame": cb0_frame_ce.detach(),
                    "ce_residual": residual_ce.detach(),
                    "ce_semantic_eos": semantic_eos_ce.detach(),
                    "tf_eos_final_top1_accuracy": eos_top.eq(self.eos_id).float().mean().detach(),
                    "tf_eos_early_top1_false_positive_count": early_top.sum().detach(),
                    "tf_eos_final_probability_mean": eos_prob.mean().detach(),
                    "tf_eos_final_probability_min": eos_prob.min().detach(),
                }
            else:
                loss_dict = self.heads.loss(
                    head_out, codes, packed["frame_mask"], cb0_weight=cb0_weight,
                    residual_weights=residual_weights,
                )
                fmask = packed["frame_mask"]
                stop_tgt = self.stop_targets_from_mask(fmask)
                assert stop_logits is not None
                pw = torch.tensor([float(stop_pos_weight)], device=stop_logits.device)
                bce = F.binary_cross_entropy_with_logits(
                    stop_logits.float(), stop_tgt, pos_weight=pw, reduction="none",
                )
                denom = fmask.float().sum().clamp_min(1.0)
                stop_loss = (bce * fmask.float()).sum() / denom
                loss_dict["stop"] = stop_loss
                loss_dict["total"] = loss_dict["total"] + float(stop_weight) * stop_loss
        return DiscreteLMOutput(
            hidden=hidden,
            acoustic_hidden=acoustic,
            heads=head_out,
            text_len=packed["text_lens"],
            loss=loss_dict,
            stop_logits=stop_logits,
        )

    @torch.no_grad()
    def generate_semantic_eos(
        self,
        input_ids: torch.Tensor,
        *,
        text_mask: torch.Tensor | None = None,
        speaker_id: torch.Tensor | None = None,
        max_frames: int = 750,
        min_frames: int = 4,
        max_cb0_repeat: int = 12,
        repetition_penalty: float = 2.0,
        temperature: float = 0.0,
        top_k: int = 0,
        decode_seed: int | None = None,
        **features: torch.Tensor | None,
    ) -> DiscreteGenerationOutput:
        """Canonical EOS decode. EOS terminates before residual prediction/embedding."""
        if self.termination_mode != "semantic_eos_v1":
            raise RuntimeError("generate_semantic_eos requires semantic_eos_v1")
        self.eval()
        B, L = input_ids.shape
        device = input_ids.device
        text_mask = text_mask if text_mask is not None else input_ids.ne(self.pad_token_id)
        speaker_id = self._validate_speaker_id(
            speaker_id if speaker_id is not None else torch.zeros(B, dtype=torch.long, device=device)
        )
        text_e = self.ru_cond(
            self.text_emb(input_ids),
            phoneme_ids=features.get("phoneme_ids"), stress=features.get("stress"),
            word_bound=features.get("word_bound"), phrase_bound=features.get("phrase_bound"),
            span_type=features.get("span_type"), text_mask=text_mask,
        )
        text_lens = text_mask.long().sum(dim=1)
        spk = self.speaker_emb(speaker_id)
        seq = (self.speech_start.to(text_e.dtype) + spk + self.modality_speech.to(text_e.dtype)).unsqueeze(1)
        seq_mask = torch.ones(B, 1, dtype=torch.bool, device=device)
        alive = torch.ones(B, dtype=torch.bool, device=device)
        lengths = torch.zeros(B, dtype=torch.long, device=device)
        last = torch.full((B,), -1, dtype=torch.long, device=device)
        runs = torch.zeros(B, dtype=torch.long, device=device)
        frames: list[torch.Tensor] = []
        rows: list[dict[str, Any]] = [
            {
                "termination_mode": self.termination_mode, "termination_reason": None,
                "termination_step": None, "n_decoded_frames": 0, "eos_selected": False,
                "eos_logit_at_stop": None, "eos_probability_at_stop": None,
                "eos_margin_to_best_real_cb0": None, "max_eos_probability_before_stop": 0.0,
                "early_eos_suppressed_count": 0, "max_cb0_run": 0,
                "safety_max_frames": int(max_frames), "safety_max_cb0_repeat": int(max_cb0_repeat),
                "decode_seed": decode_seed,
            }
            for _ in range(B)
        ]
        for t in range(int(max_frames)):
            if not bool(alive.any()):
                break
            embeds = torch.cat([text_e, seq], dim=1)
            key_mask = torch.cat([text_mask, seq_mask], dim=1)
            S = embeds.shape[1]
            causal = torch.triu(torch.ones(S, S, device=device, dtype=torch.bool), diagonal=1)
            live_slots = lengths + 1
            pos = self.build_position_ids(text_lens=text_lens, acoustic_lens=live_slots, L=L, T=seq.shape[1])
            h = self.forward_backbone(embeds, key_padding_mask=key_mask, attn_mask=causal, position_ids=pos)[:, -1:]
            raw = self.heads.semantic(h).squeeze(1)
            raw_prob = torch.softmax(raw.float(), dim=-1)[:, self.eos_id]
            logits = raw.clone()
            raw_best = raw.argmax(dim=-1)
            for b in range(B):
                if not bool(alive[b]):
                    continue
                rows[b]["max_eos_probability_before_stop"] = max(
                    float(rows[b]["max_eos_probability_before_stop"]), float(raw_prob[b])
                )
                if int(lengths[b]) < int(min_frames):
                    if int(raw_best[b]) == self.eos_id:
                        rows[b]["early_eos_suppressed_count"] += 1
                    logits[b, self.eos_id] = torch.finfo(logits.dtype).min
                if repetition_penalty > 0 and int(runs[b]) >= 2 and int(last[b]) >= 0:
                    logits[b, int(last[b])] -= float(repetition_penalty) * float(runs[b])
            token = _sample_logits(logits, temperature=temperature, top_k=top_k)
            next_embeds = torch.zeros(B, self.d_model, dtype=text_e.dtype, device=device)
            next_valid = torch.zeros(B, dtype=torch.bool, device=device)
            out_frame = torch.zeros(B, self.n_codebooks, dtype=torch.long, device=device)
            for b in range(B):
                if not bool(alive[b]):
                    continue
                if int(token[b]) == self.eos_id:
                    real_best = raw[b, : self.codebook_size].max()
                    rows[b].update(
                        termination_reason="learned_eos", termination_step=t,
                        n_decoded_frames=int(lengths[b]), eos_selected=True,
                        eos_logit_at_stop=float(raw[b, self.eos_id]),
                        eos_probability_at_stop=float(raw_prob[b]),
                        eos_margin_to_best_real_cb0=float(raw[b, self.eos_id] - real_best),
                    )
                    alive[b] = False
                    continue
                cb0 = token[b:b + 1]
                residual = self.heads(h[b:b + 1], self.speech_emb(cb0).unsqueeze(1)).residual_logits
                res = [_sample_logits(residual[:, 0, i], temperature=temperature, top_k=top_k)[0]
                       for i in range(self.heads.n_residual)]
                out_frame[b] = torch.stack([cb0[0], *res]) if res else cb0[0].unsqueeze(0)
                lengths[b] += 1
                runs[b] = runs[b] + 1 if int(cb0[0]) == int(last[b]) else 1
                last[b] = cb0[0]
                rows[b]["max_cb0_run"] = max(int(rows[b]["max_cb0_run"]), int(runs[b]))
                rows[b]["n_decoded_frames"] = int(lengths[b])
                if int(runs[b]) >= int(max_cb0_repeat):
                    rows[b].update(termination_reason="safety_cb0_repeat", termination_step=t)
                    alive[b] = False
                elif int(lengths[b]) >= int(max_frames):
                    rows[b].update(termination_reason="safety_max_frames", termination_step=t)
                    alive[b] = False
                else:
                    next_embeds[b] = self.speech_emb(cb0)[0] + self.modality_speech.to(text_e.dtype)
                    next_valid[b] = True
            frames.append(out_frame)
            seq = torch.cat([seq, next_embeds.unsqueeze(1)], dim=1)
            seq_mask = torch.cat([seq_mask, next_valid.unsqueeze(1)], dim=1)
        for b, row in enumerate(rows):
            if row["termination_reason"] is None:
                row.update(termination_reason="safety_max_frames", termination_step=int(max_frames) - 1)
            if frames and int(lengths[b]) > 0:
                code_bytes = b"".join(
                    f"{int(tok)},".encode()
                    for tok in torch.stack(frames, dim=1)[b, : lengths[b]].flatten()
                )
            else:
                code_bytes = b""
            import hashlib
            row["generated_codes_sha256"] = hashlib.sha256(code_bytes).hexdigest()
        max_len = int(lengths.max().item()) if B else 0
        if frames and max_len > 0:
            codes = torch.stack(frames, dim=1)[:, :max_len]
        else:
            # Immediate EOS / empty decode: no phantom codec frames.
            codes = torch.zeros(B, 0, self.n_codebooks, dtype=torch.long, device=device)
        frame_mask = torch.arange(codes.shape[1], device=device).unsqueeze(0) < lengths.unsqueeze(1)
        return DiscreteGenerationOutput(codes=codes, frame_mask=frame_mask, lengths=lengths, rows=rows)

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        *,
        text_mask: torch.Tensor | None = None,
        speaker_id: torch.Tensor | None = None,
        max_frames: int = 500,
        temperature: float = 1.0,
        top_k: int = 0,
        eos_cb0: int | None = None,
        stop_threshold: float = 0.5,
        stop_consecutive: int = 1,
        min_frames: int = 4,
        max_cb0_repeat: int = 12,
        repetition_penalty: float = 2.0,
        frames_per_text_token: float = 2.5,
        stop_overrun_bias: float = 0.0,
        stop_earliest_frac: float = 0.7,
        phoneme_ids: torch.Tensor | None = None,
        stress: torch.Tensor | None = None,
        word_bound: torch.Tensor | None = None,
        phrase_bound: torch.Tensor | None = None,
        span_type: torch.Tensor | None = None,
        return_metadata: bool = False,
    ) -> torch.Tensor | DiscreteGenerationOutput:
        """Greedy / sampled free-running decode → codes ``[B, T, n_codebooks]``.

        Stops when the stop head fires ``stop_consecutive`` times above
        ``stop_threshold`` (after ``min_frames``), when the same CB0 repeats
        ``max_cb0_repeat`` times, optional ``eos_cb0``, or a hard overrun cap
        based on ``frames_per_text_token * text_len``.
        """
        if self.termination_mode == "semantic_eos_v1":
            return self.generate_semantic_eos(
                input_ids, text_mask=text_mask, speaker_id=speaker_id, max_frames=max_frames,
                min_frames=min_frames, max_cb0_repeat=max_cb0_repeat,
                repetition_penalty=repetition_penalty, temperature=temperature, top_k=top_k,
                phoneme_ids=phoneme_ids, stress=stress, word_bound=word_bound,
                phrase_bound=phrase_bound, span_type=span_type,
            )
        self.eval()
        B, L = input_ids.shape
        device = input_ids.device
        if text_mask is None:
            text_mask = input_ids != self.pad_token_id
        if speaker_id is None:
            speaker_id = torch.zeros(B, dtype=torch.long, device=device)
        speaker_id = self._validate_speaker_id(speaker_id)

        text_e = self.text_emb(input_ids)
        text_e = self.ru_cond(
            text_e,
            phoneme_ids=phoneme_ids,
            stress=stress,
            word_bound=word_bound,
            phrase_bound=phrase_bound,
            span_type=span_type,
            text_mask=text_mask,
        )
        spk = self.speaker_emb(speaker_id)
        text_lens_raw = text_mask.long().sum(dim=1)
        text_lens = text_lens_raw.clamp_min(1)
        expected = (
            text_lens.float() * float(frames_per_text_token)
        ).long().clamp_min(int(min_frames))
        hard_cap = (expected.float() * 1.8 + 8).long().clamp_max(int(max_frames))

        frames: list[torch.Tensor] = []
        # Growing acoustic embeddings
        speech_e = self.speech_start.to(dtype=text_e.dtype) + spk  # [B, D]
        speech_e = speech_e + self.modality_speech.to(dtype=text_e.dtype)
        speech_seq = speech_e.unsqueeze(1)  # [B, 1, D]
        speech_pad = torch.ones(B, 1, dtype=torch.bool, device=device)
        # Speech start occupies acoustic slot t=0 → position text_len + 0.
        acoustic_live = torch.ones(B, dtype=torch.long, device=device)

        stop_run = torch.zeros(B, dtype=torch.long, device=device)
        done = torch.zeros(B, dtype=torch.bool, device=device)
        last_cb0 = torch.full((B,), -1, dtype=torch.long, device=device)
        cb0_run = torch.zeros(B, dtype=torch.long, device=device)
        live_len = torch.zeros(B, dtype=torch.long, device=device)

        for t in range(max_frames):
            if bool(done.all()):
                break
            embeds = torch.cat([text_e, speech_seq], dim=1)
            key_pad = torch.cat([text_mask, speech_pad], dim=1)
            S = embeds.shape[1]
            causal = torch.triu(
                torch.ones(S, S, device=device, dtype=torch.bool), diagonal=1
            )
            position_ids = self.build_position_ids(
                text_lens=text_lens_raw,
                acoustic_lens=acoustic_live,
                L=L,
                T=int(speech_seq.shape[1]),
            )
            hidden = self.forward_backbone(
                embeds,
                key_padding_mask=key_pad,
                attn_mask=causal,
                position_ids=position_ids,
            )
            h_t = hidden[:, -1:, :]  # [B, 1, D]
            logits_cb0 = self.heads.semantic(h_t).squeeze(1)  # [B, V]

            # Soft-ban the currently looping CB0 token.
            if repetition_penalty and repetition_penalty > 0:
                for b in range(B):
                    if int(cb0_run[b]) >= 2 and int(last_cb0[b]) >= 0:
                        logits_cb0[b, int(last_cb0[b])] -= float(repetition_penalty) * float(
                            cb0_run[b]
                        )

            cb0 = _sample_logits(logits_cb0, temperature=temperature, top_k=top_k)
            if t > 0:
                cb0 = torch.where(done, last_cb0.clamp_min(0), cb0)

            cb0_emb = self.speech_emb(cb0).unsqueeze(1)
            head_out = self.heads(h_t, cb0_emb)
            idx = torch.full((B, 1), float(t), device=device)
            stop_logit = self._stop_logits(h_t, frame_index=idx).squeeze(-1)  # [B]
            # Encourage stop after the text-length prior (FR exposure-bias mitigation).
            overrun = (t + 1) - expected.float()
            stop_logit = stop_logit + float(stop_overrun_bias) * overrun.clamp_min(0.0)
            stop_prob = torch.sigmoid(stop_logit.float())

            res = []
            for i in range(self.heads.n_residual):
                li = head_out.residual_logits[:, 0, i, :]
                res.append(_sample_logits(li, temperature=temperature, top_k=top_k))
            if res:
                frame = torch.stack([cb0, *res], dim=-1)  # [B, n_cb]
            else:
                frame = cb0.unsqueeze(-1)
            frames.append(frame)
            live_len = torch.where(done, live_len, torch.full_like(live_len, t + 1))

            same = cb0 == last_cb0
            cb0_run = torch.where(same, cb0_run + 1, torch.ones_like(cb0_run))
            last_cb0 = cb0

            # Don't trust stop head before ``stop_earliest_frac`` of the length prior.
            # WAVE3b used 0.7 and truncated long CV utts (short dur → CER spike).
            frac = float(stop_earliest_frac)
            earliest = torch.maximum(
                torch.full_like(expected, int(min_frames)),
                (expected.float() * frac).long(),
            )
            fire = (stop_prob >= float(stop_threshold)) & ((t + 1) >= earliest)
            stop_run = torch.where(fire, stop_run + 1, torch.zeros_like(stop_run))
            stop_now = stop_run >= int(max(1, stop_consecutive))
            if eos_cb0 is not None:
                stop_now = stop_now | (cb0 == int(eos_cb0))
            if max_cb0_repeat and max_cb0_repeat > 0:
                stop_now = stop_now | (cb0_run >= int(max_cb0_repeat))
            stop_now = stop_now | ((t + 1) >= hard_cap)
            done = done | stop_now

            next_e = self.speech_emb(cb0) + self.modality_speech.to(dtype=text_e.dtype)
            speech_seq = torch.cat([speech_seq, next_e.unsqueeze(1)], dim=1)
            speech_pad = torch.cat(
                [speech_pad, torch.ones(B, 1, dtype=torch.bool, device=device)], dim=1
            )
            acoustic_live = acoustic_live + 1

        if not frames:
            return torch.zeros(B, 0, self.n_codebooks, dtype=torch.long, device=device)
        codes = torch.stack(frames, dim=1)
        # Trim to first finished length when B==1 (usual path).
        if B == 1:
            n_keep = int(live_len[0].item()) if int(live_len[0].item()) > 0 else codes.shape[1]
            return codes[:, :n_keep]
        return codes  # [B, T_stopped, C]


def _sample_logits(
    logits: torch.Tensor, *, temperature: float = 1.0, top_k: int = 0
) -> torch.Tensor:
    if temperature <= 0:
        return torch.argmax(logits, dim=-1)
    logits = logits / max(temperature, 1e-6)
    if top_k and top_k > 0:
        v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
        cutoff = v[:, -1].unsqueeze(-1)
        logits = logits.masked_fill(logits < cutoff, torch.finfo(logits.dtype).min)
    if temperature == 1.0 and top_k == 0:
        # deterministic greedy when defaults used
        return torch.argmax(logits, dim=-1)
    probs = torch.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1).squeeze(-1)


def build_discrete_speech_lm(cfg: dict[str, Any]) -> DiscreteSpeechLM:
    return DiscreteSpeechLM.from_config_dict(cfg)

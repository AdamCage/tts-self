"""Duration-aware non-AR mel acoustic model (Track A / A4).

Phone (or grapheme) embeddings → optional encoder → hard/soft duration
upsample → frame decoder → log-mel ``[B, T, n_mels]``.

No VibeVoice / Qwen dependency. Designed for S1-scale overfit then ladder.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn as nn

from minispeech.constants import MEL80


def hard_upsample(phone_h: torch.Tensor, durations: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Repeat phone states by integer durations.

    Parameters
    ----------
    phone_h : [B, P, D]
    durations : [B, P] non-negative ints (float ok; truncated)

    Returns
    -------
    frames : [B, T, D]
    frame_mask : [B, T] bool
    """
    B, P, D = phone_h.shape
    durs = durations.long().clamp_min(0)
    lengths = durs.sum(dim=-1)
    T = int(lengths.max().item()) if B else 0
    T = max(T, 1)
    out = phone_h.new_zeros(B, T, D)
    mask = torch.zeros(B, T, dtype=torch.bool, device=phone_h.device)
    for b in range(B):
        frames: list[torch.Tensor] = []
        for p in range(P):
            n = int(durs[b, p].item())
            if n <= 0:
                continue
            frames.append(phone_h[b, p : p + 1].expand(n, -1))
        if not frames:
            mask[b, 0] = True
            continue
        seq = torch.cat(frames, dim=0)
        t = seq.shape[0]
        out[b, :t] = seq
        mask[b, :t] = True
    return out, mask


def soft_upsample(phone_h: torch.Tensor, W: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Soft overlap upsample: frames = W^T @ phones (column-stochastic preferred).

    Parameters
    ----------
    phone_h : [B, P, D]
    W : [B, P, T] soft/hard overlap weights
    """
    # [B, T, P] @ [B, P, D] -> [B, T, D]
    frames = torch.matmul(W.transpose(1, 2), phone_h)
    frame_mask = W.sum(dim=1) > 1e-6
    return frames, frame_mask


class _FFN(nn.Module):
    def __init__(self, d_model: int, ffn_dim: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class _EncoderBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, ffn_dim: int, dropout: float):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.ffn = _FFN(d_model, ffn_dim, dropout)
        self.n1 = nn.LayerNorm(d_model)
        self.n2 = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, key_padding_mask: torch.Tensor | None) -> torch.Tensor:
        # key_padding_mask: True = ignore (PyTorch convention)
        h, _ = self.attn(x, x, x, key_padding_mask=key_padding_mask, need_weights=False)
        x = self.n1(x + self.drop(h))
        x = self.n2(x + self.ffn(x))
        return x


class _ConvDecoder(nn.Module):
    """Light Conv1d stack over time (channels = d_model)."""

    def __init__(self, d_model: int, layers: int = 4, kernel: int = 5, dropout: float = 0.1):
        super().__init__()
        blocks: list[nn.Module] = []
        pad = kernel // 2
        for _ in range(layers):
            blocks.append(
                nn.Sequential(
                    nn.Conv1d(d_model, d_model, kernel, padding=pad),
                    nn.GELU(),
                    nn.Dropout(dropout),
                )
            )
        self.blocks = nn.ModuleList(blocks)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x [B, T, D] → conv over T
        h = x.transpose(1, 2)
        for blk in self.blocks:
            h = h + blk(h)
        return self.norm(h.transpose(1, 2))


@dataclass
class MelBaselineOutput:
    mel_pred: torch.Tensor
    frame_mask: torch.Tensor
    phone_mask: torch.Tensor
    dur_pred: torch.Tensor | None
    extras: dict


class _PostNet(nn.Module):
    """Tacotron2-style 5-layer Conv residual post-net over mel bins."""

    def __init__(self, n_mels: int, hidden: int = 512, layers: int = 5, kernel: int = 5, dropout: float = 0.1):
        super().__init__()
        pad = kernel // 2
        blocks: list[nn.Module] = []
        for i in range(layers):
            in_ch = n_mels if i == 0 else hidden
            out_ch = n_mels if i == layers - 1 else hidden
            blocks.append(
                nn.Sequential(
                    nn.Conv1d(in_ch, out_ch, kernel, padding=pad),
                    nn.BatchNorm1d(out_ch),
                    nn.Tanh() if i < layers - 1 else nn.Identity(),
                    nn.Dropout(dropout),
                )
            )
        self.blocks = nn.ModuleList(blocks)

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        # mel [B, T, n_mels]
        h = mel.transpose(1, 2)
        for blk in self.blocks:
            h = blk(h)
        return mel + h.transpose(1, 2)


class MelBaseline(nn.Module):
    """Small FastSpeech-style non-AR mel model."""

    def __init__(
        self,
        vocab_size: int = 512,
        d_model: int = 256,
        n_mels: int = MEL80.dim,
        enc_layers: int = 3,
        dec_layers: int = 4,
        n_heads: int = 4,
        ffn_dim: int = 1024,
        dropout: float = 0.1,
        decoder: Literal["conv", "transformer"] = "conv",
        upsample: Literal["hard", "soft"] = "hard",
        predict_duration: bool = True,
        predict_pitch: bool = False,
        predict_energy: bool = False,
        use_postnet: bool = False,
        pad_id: int = 0,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_mels = n_mels
        self.upsample_mode = upsample
        self.pad_id = pad_id
        self.emb = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.pos_phone = nn.Embedding(4096, d_model)
        self.enc_layers = nn.ModuleList(
            [_EncoderBlock(d_model, n_heads, ffn_dim, dropout) for _ in range(enc_layers)]
        )
        self.dur_head = (
            nn.Sequential(
                nn.Linear(d_model, d_model // 2),
                nn.GELU(),
                nn.Linear(d_model // 2, 1),
                nn.Softplus(),
            )
            if predict_duration
            else None
        )
        # FastSpeech2-style variance adaptor (phone-level, before length regulator).
        self.pitch_head = (
            nn.Sequential(nn.Linear(d_model, d_model // 2), nn.GELU(), nn.Linear(d_model // 2, 1))
            if predict_pitch
            else None
        )
        self.energy_head = (
            nn.Sequential(nn.Linear(d_model, d_model // 2), nn.GELU(), nn.Linear(d_model // 2, 1))
            if predict_energy
            else None
        )
        self.pitch_emb = nn.Linear(1, d_model) if predict_pitch else None
        self.energy_emb = nn.Linear(1, d_model) if predict_energy else None
        if decoder == "transformer":
            self.decoder: nn.Module = nn.ModuleList(
                [_EncoderBlock(d_model, n_heads, ffn_dim, dropout) for _ in range(dec_layers)]
            )
            self._dec_kind = "transformer"
        else:
            self.decoder = _ConvDecoder(d_model, layers=dec_layers, dropout=dropout)
            self._dec_kind = "conv"
        self.mel_proj = nn.Linear(d_model, n_mels)
        self.postnet = _PostNet(n_mels, dropout=dropout) if use_postnet else None
        self.drop = nn.Dropout(dropout)

    def encode_phones(
        self,
        phone_ids: torch.Tensor,
        phone_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """phone_ids [B, P] → [B, P, D]."""
        B, P = phone_ids.shape
        pos = torch.arange(P, device=phone_ids.device).clamp(max=4095)
        x = self.emb(phone_ids) + self.pos_phone(pos)[None, :, :]
        x = self.drop(x)
        kpm = None
        if phone_mask is not None:
            kpm = ~phone_mask  # True = pad
        for blk in self.enc_layers:
            x = blk(x, kpm)
        return x

    def forward(
        self,
        phone_ids: torch.Tensor,
        *,
        phone_mask: torch.Tensor | None = None,
        durations: torch.Tensor | None = None,
        proj_W: torch.Tensor | None = None,
        use_pred_duration: bool = False,
        pitch_target: torch.Tensor | None = None,
        energy_target: torch.Tensor | None = None,
    ) -> MelBaselineOutput:
        """
        Parameters
        ----------
        phone_ids : [B, P]
        phone_mask : [B, P] bool (True = valid)
        durations : [B, P] teacher frame counts (hard upsample)
        proj_W : [B, P, T] soft/hard overlap (soft upsample)
        use_pred_duration : if True and dur_head exists, use predicted durs at infer
        pitch_target / energy_target : [B, P] teacher values for adaptor emb (train)
        """
        if phone_mask is None:
            phone_mask = phone_ids != self.pad_id
        phone_h = self.encode_phones(phone_ids, phone_mask)
        dur_pred = None
        if self.dur_head is not None:
            dur_pred = self.dur_head(phone_h).squeeze(-1)  # [B, P]

        pitch_pred = energy_pred = None
        if self.pitch_head is not None:
            pitch_pred = self.pitch_head(phone_h)  # [B, P, 1]
            pitch_for_emb = (
                pitch_target.unsqueeze(-1)
                if pitch_target is not None and self.training
                else pitch_pred
            )
            phone_h = phone_h + self.pitch_emb(pitch_for_emb)  # type: ignore[misc]
        if self.energy_head is not None:
            energy_pred = self.energy_head(phone_h)  # [B, P, 1]
            energy_for_emb = (
                energy_target.unsqueeze(-1)
                if energy_target is not None and self.training
                else energy_pred
            )
            phone_h = phone_h + self.energy_emb(energy_for_emb)  # type: ignore[misc]

        if self.upsample_mode == "soft":
            if proj_W is None:
                raise ValueError("soft upsample requires proj_W [B,P,T]")
            frames, frame_mask = soft_upsample(phone_h, proj_W)
        else:
            if use_pred_duration and dur_pred is not None:
                durs = dur_pred.detach().clamp_min(1.0).round()
                durs = durs * phone_mask.float()
            elif durations is not None:
                durs = durations.float() * phone_mask.float()
            elif dur_pred is not None:
                durs = dur_pred.clamp_min(1.0).round() * phone_mask.float()
            else:
                raise ValueError("hard upsample needs durations or dur_head")
            frames, frame_mask = hard_upsample(phone_h, durs)

        if self._dec_kind == "conv":
            h = self.decoder(frames)
        else:
            kpm = ~frame_mask
            h = frames
            for blk in self.decoder:  # type: ignore[union-attr]
                h = blk(h, kpm)
        mel_pre = self.mel_proj(h)
        mel_pred = self.postnet(mel_pre) if self.postnet is not None else mel_pre
        # When post-net is on, supervise both coarse and refined mel (Tacotron-style).
        return MelBaselineOutput(
            mel_pred=mel_pred,
            frame_mask=frame_mask,
            phone_mask=phone_mask,
            dur_pred=dur_pred,
            extras={
                "mel_pre_postnet": mel_pre,
                "pitch_pred": pitch_pred.squeeze(-1) if pitch_pred is not None else None,
                "energy_pred": energy_pred.squeeze(-1) if energy_pred is not None else None,
            },
        )

    @torch.no_grad()
    def infer(
        self,
        phone_ids: torch.Tensor,
        *,
        phone_mask: torch.Tensor | None = None,
        durations: torch.Tensor | None = None,
        proj_W: torch.Tensor | None = None,
    ) -> MelBaselineOutput:
        self.eval()
        return self.forward(
            phone_ids,
            phone_mask=phone_mask,
            durations=durations,
            proj_W=proj_W,
            use_pred_duration=durations is None and proj_W is None,
        )


def phone_scalar_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    phone_mask: torch.Tensor,
    *,
    sample_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    """Masked L1 over phone-level scalars (pitch / energy)."""
    err = (pred - target).abs()
    m = phone_mask.float()
    per = (err * m).sum(dim=-1) / m.sum(dim=-1).clamp_min(1.0)
    if sample_weight is not None:
        w = sample_weight.float()
        return (per * w).sum() / w.sum().clamp_min(1.0)
    return per.mean()


def mel_l1_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    frame_mask: torch.Tensor,
    *,
    sample_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    """Masked L1 over mel bins; optional per-utterance ``sample_weight`` [B]."""
    t = min(pred.shape[1], target.shape[1])
    pred = pred[:, :t]
    target = target[:, :t]
    mask = frame_mask[:, :t]
    m = mask.unsqueeze(-1).float()
    per = ((pred - target).abs() * m).sum(dim=(1, 2)) / m.sum(dim=(1, 2)).clamp_min(1.0) / pred.shape[-1]
    if sample_weight is not None:
        w = sample_weight.float()
        return (per * w).sum() / w.sum().clamp_min(1.0)
    return per.mean()


def phone_duration_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    phone_mask: torch.Tensor,
    *,
    sample_weight: torch.Tensor | None = None,
    log_domain: bool = True,
) -> torch.Tensor:
    """Per-phone duration L1 (optional log1p); ``pred``/``target`` [B, P]."""
    if log_domain:
        p = torch.log1p(pred.clamp_min(0))
        t = torch.log1p(target.clamp_min(0))
    else:
        p, t = pred, target
    err = (p - t).abs()
    m = phone_mask.float()
    per = (err * m).sum(dim=-1) / m.sum(dim=-1).clamp_min(1.0)
    if sample_weight is not None:
        w = sample_weight.float()
        return (per * w).sum() / w.sum().clamp_min(1.0)
    return per.mean()

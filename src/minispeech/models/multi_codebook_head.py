"""Multi-codebook prediction heads: AR CB0 + parallel MTP for CB1–CB15."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class MultiCodebookOutput:
    """Logits for semantic (CB0) and residual (CB1..CB_{n-1}) codebooks."""

    cb0_logits: torch.Tensor  # [B, T, V]
    residual_logits: torch.Tensor  # [B, T, n_residual, V]


class SemanticCodebookHead(nn.Module):
    """Linear AR head over the semantic codebook (CB0)."""

    def __init__(self, d_model: int, codebook_size: int = 2048):
        super().__init__()
        self.proj = nn.Linear(d_model, codebook_size, bias=False)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        """``hidden``: ``[B, T, D]`` → logits ``[B, T, V]``."""
        return self.proj(hidden)


class ResidualMTPHead(nn.Module):
    """Per-codebook MLP conditioned on backbone hidden state and CB0 embedding."""

    def __init__(
        self,
        d_model: int,
        codebook_size: int = 2048,
        *,
        hidden: int | None = None,
        dropout: float = 0.0,
    ):
        super().__init__()
        h = int(hidden or d_model)
        self.net = nn.Sequential(
            nn.Linear(d_model * 2, h),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(h, codebook_size, bias=False),
        )

    def forward(self, hidden: torch.Tensor, cb0_emb: torch.Tensor) -> torch.Tensor:
        """Both args ``[B, T, D]`` → logits ``[B, T, V]``."""
        return self.net(torch.cat([hidden, cb0_emb], dim=-1))


class MultiCodebookHead(nn.Module):
    """CB0 AR head + MTP residuals CB1…CB15.

    Residual heads are conditioned on the backbone state **and** the CB0
    embedding for the *same* frame (teacher CB0 at train, sampled at infer).
    """

    def __init__(
        self,
        d_model: int,
        *,
        n_codebooks: int = 16,
        codebook_size: int = 2048,
        mtp_hidden: int | None = None,
        dropout: float = 0.0,
    ):
        super().__init__()
        if n_codebooks < 1:
            raise ValueError("n_codebooks must be >= 1")
        self.n_codebooks = int(n_codebooks)
        self.codebook_size = int(codebook_size)
        self.n_residual = self.n_codebooks - 1
        self.semantic = SemanticCodebookHead(d_model, codebook_size)
        self.residuals = nn.ModuleList(
            [
                ResidualMTPHead(
                    d_model,
                    codebook_size,
                    hidden=mtp_hidden,
                    dropout=dropout,
                )
                for _ in range(self.n_residual)
            ]
        )

    def forward(
        self,
        hidden: torch.Tensor,
        cb0_emb: torch.Tensor,
    ) -> MultiCodebookOutput:
        """
        Parameters
        ----------
        hidden:
            Acoustic-position backbone states ``[B, T, D]``.
        cb0_emb:
            Embeddings of CB0 for the same frames ``[B, T, D]``.
        """
        cb0_logits = self.semantic(hidden)
        if self.n_residual == 0:
            residual = hidden.new_zeros(
                hidden.shape[0], hidden.shape[1], 0, self.codebook_size
            )
        else:
            parts = [head(hidden, cb0_emb) for head in self.residuals]
            residual = torch.stack(parts, dim=2)  # [B, T, R, V]
        return MultiCodebookOutput(cb0_logits=cb0_logits, residual_logits=residual)

    def loss(
        self,
        out: MultiCodebookOutput,
        codes: torch.Tensor,
        frame_mask: torch.Tensor,
        *,
        cb0_weight: float = 1.0,
        residual_weights: torch.Tensor | None = None,
        ignore_index: int = -100,
    ) -> dict[str, torch.Tensor]:
        """Masked CE over CB0 + residuals.

        ``codes``: ``[B, T, n_codebooks]``. Positions with ``frame_mask=False``
        are ignored.
        """
        B, T, n_cb = codes.shape
        if n_cb != self.n_codebooks:
            raise ValueError(
                f"codes n_codebooks={n_cb} != head n_codebooks={self.n_codebooks}"
            )
        # Flatten valid frames
        target_cb0 = codes[:, :, 0].clone()
        target_cb0 = target_cb0.masked_fill(~frame_mask, ignore_index)
        loss_cb0 = F.cross_entropy(
            out.cb0_logits.reshape(B * T, -1),
            target_cb0.reshape(B * T),
            ignore_index=ignore_index,
        )

        loss_res = codes.new_zeros(())
        if self.n_residual > 0:
            if residual_weights is None:
                w = torch.ones(self.n_residual, device=codes.device, dtype=torch.float32)
            else:
                w = residual_weights.to(device=codes.device, dtype=torch.float32)
                if w.numel() != self.n_residual:
                    raise ValueError(
                        f"residual_weights length {w.numel()} != {self.n_residual}"
                    )
            w = w / w.sum().clamp_min(1e-8)
            res_losses = []
            for i in range(self.n_residual):
                tgt = codes[:, :, i + 1].clone()
                tgt = tgt.masked_fill(~frame_mask, ignore_index)
                li = F.cross_entropy(
                    out.residual_logits[:, :, i, :].reshape(B * T, -1),
                    tgt.reshape(B * T),
                    ignore_index=ignore_index,
                )
                res_losses.append(li * w[i])
            loss_res = torch.stack(res_losses).sum()

        total = float(cb0_weight) * loss_cb0 + loss_res
        return {
            "total": total,
            "ce_cb0": loss_cb0.detach(),
            "ce_residual": loss_res.detach()
            if torch.is_tensor(loss_res)
            else torch.tensor(0.0),
        }

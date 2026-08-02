"""Latent ↔ Qwen-hidden projections and text/acoustic sequence masks (N2 / A1).

Geometry (latent dim, frame rate) comes from ``TARGETS`` via ``get_target``;
hidden size matches the frozen Qwen backbone (default ``QWEN_HIDDEN_SIZE``).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from minispeech.constants import QWEN_HIDDEN_SIZE, get_target


class LatentInProjector(nn.Module):
    """Project continuous latents into the LLM hidden space (64 → H)."""

    def __init__(
        self,
        latent_dim: int | None = None,
        hidden_size: int = QWEN_HIDDEN_SIZE,
        *,
        mlp_hidden: int | None = None,
        dropout: float = 0.0,
    ):
        super().__init__()
        latent_dim = int(latent_dim if latent_dim is not None else get_target().dim)
        self.latent_dim = latent_dim
        self.hidden_size = int(hidden_size)
        if mlp_hidden is None:
            self.net = nn.Linear(latent_dim, self.hidden_size)
        else:
            self.net = nn.Sequential(
                nn.Linear(latent_dim, mlp_hidden),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(mlp_hidden, self.hidden_size),
            )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """z: [..., latent_dim] → [..., hidden_size]."""
        return self.net(z)


class LatentOutProjector(nn.Module):
    """Project LLM hidden states back to latent space (H → 64)."""

    def __init__(
        self,
        latent_dim: int | None = None,
        hidden_size: int = QWEN_HIDDEN_SIZE,
        *,
        mlp_hidden: int | None = None,
        dropout: float = 0.0,
    ):
        super().__init__()
        latent_dim = int(latent_dim if latent_dim is not None else get_target().dim)
        self.latent_dim = latent_dim
        self.hidden_size = int(hidden_size)
        if mlp_hidden is None:
            self.net = nn.Linear(self.hidden_size, latent_dim)
        else:
            self.net = nn.Sequential(
                nn.Linear(self.hidden_size, mlp_hidden),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(mlp_hidden, latent_dim),
            )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """h: [..., hidden_size] → [..., latent_dim]."""
        return self.net(h)


class AcousticMarker(nn.Module):
    """Learned additive marker distinguishing acoustic frames from text tokens."""

    def __init__(self, hidden_size: int = QWEN_HIDDEN_SIZE):
        super().__init__()
        self.marker = nn.Parameter(torch.zeros(hidden_size))
        nn.init.normal_(self.marker, std=0.02)

    def forward(self, lat_embeds: torch.Tensor) -> torch.Tensor:
        return lat_embeds + self.marker.to(dtype=lat_embeds.dtype)


@dataclass(slots=True)
class ModalityMasks:
    """Masks for a padded ``[text ∥ latents]`` batch.

    All bool masks use True = valid / keep.
    """

    text_mask: torch.Tensor  # [B, L]
    frame_mask: torch.Tensor  # [B, T]
    seq_mask: torch.Tensor  # [B, L+T]
    is_acoustic: torch.Tensor  # [B, L+T] True on acoustic positions
    text_lens: torch.Tensor  # [B]
    acoustic_lens: torch.Tensor  # [B]
    max_text_len: int
    max_acoustic_len: int


def build_modality_masks(
    text_lens: torch.Tensor | list[int],
    acoustic_lens: torch.Tensor | list[int],
    *,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.bool,
    max_text_len: int | None = None,
    max_acoustic_len: int | None = None,
) -> ModalityMasks:
    """Build text / acoustic / joint masks for a mixed sequence batch.

    Optional ``max_text_len`` / ``max_acoustic_len`` keep mask width aligned with
    already-padded ``input_ids`` / ``z`` tensors (pad width may exceed max lens).
    """
    if not torch.is_tensor(text_lens):
        text_lens_t = torch.tensor(text_lens, dtype=torch.long)
    else:
        text_lens_t = text_lens.to(dtype=torch.long)
    if not torch.is_tensor(acoustic_lens):
        acoustic_lens_t = torch.tensor(acoustic_lens, dtype=torch.long)
    else:
        acoustic_lens_t = acoustic_lens.to(dtype=torch.long)

    if device is not None:
        text_lens_t = text_lens_t.to(device)
        acoustic_lens_t = acoustic_lens_t.to(device)

    B = int(text_lens_t.shape[0])
    L = int(text_lens_t.max().item()) if B else 0
    T = int(acoustic_lens_t.max().item()) if B else 0
    if max_text_len is not None:
        L = max(L, int(max_text_len))
    if max_acoustic_len is not None:
        T = max(T, int(max_acoustic_len))
    S = L + T

    text_mask = torch.zeros(B, L, dtype=dtype, device=text_lens_t.device)
    frame_mask = torch.zeros(B, T, dtype=dtype, device=text_lens_t.device)
    seq_mask = torch.zeros(B, S, dtype=dtype, device=text_lens_t.device)
    is_acoustic = torch.zeros(B, S, dtype=dtype, device=text_lens_t.device)

    for b in range(B):
        tl = int(text_lens_t[b].item())
        al = int(acoustic_lens_t[b].item())
        if L > 0 and tl > 0:
            text_mask[b, :tl] = True
            seq_mask[b, :tl] = True
        if T > 0 and al > 0:
            frame_mask[b, :al] = True
            seq_mask[b, L : L + al] = True
            is_acoustic[b, L : L + al] = True

    return ModalityMasks(
        text_mask=text_mask,
        frame_mask=frame_mask,
        seq_mask=seq_mask,
        is_acoustic=is_acoustic,
        text_lens=text_lens_t,
        acoustic_lens=acoustic_lens_t,
        max_text_len=L,
        max_acoustic_len=T,
    )


def acoustic_positions(L: int, T: int, device: torch.device | None = None) -> torch.Tensor:
    """Absolute indices of acoustic frames inside a ``[text ∥ latents]`` sequence."""
    return torch.arange(L, L + T, device=device, dtype=torch.long)


def shift_latents_for_ar(
    z_target: torch.Tensor,
    frame_mask: torch.Tensor,
    *,
    bos: torch.Tensor | None = None,
) -> torch.Tensor:
    """Teacher-forced AR input: ``[BOS, z_0, …, z_{T-2}]`` with padding zeros.

    ``z_target`` / return: ``[B, T, D]``. ``bos`` defaults to zeros.
    """
    B, T, D = z_target.shape
    z_input = z_target.new_zeros(B, T, D)
    if bos is None:
        bos_v = z_target.new_zeros(D)
    else:
        bos_v = bos.to(device=z_target.device, dtype=z_target.dtype)
    z_input[:, 0] = bos_v
    if T > 1:
        z_input[:, 1:] = z_target[:, :-1]
    # Clear padded frames (keep BOS only on valid utterances).
    z_input = z_input * frame_mask.unsqueeze(-1).to(dtype=z_input.dtype)
    # Restore BOS on first frame for every row that has at least one frame.
    has = frame_mask[:, 0] if T > 0 else frame_mask.new_zeros(B, dtype=torch.bool)
    if T > 0:
        z_input[:, 0] = torch.where(
            has.unsqueeze(-1),
            bos_v.expand(B, D),
            z_input[:, 0],
        )
    return z_input

"""Qwen3.5 as a causal backbone over concatenated ``[text ∥ latents]`` (N2 / A1).

At start the pretrained transformer is frozen; only ``LatentInProjector``,
optional acoustic marker, and (externally) the diffusion head are trained.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import torch
import torch.nn as nn

from minispeech.constants import QWEN_HIDDEN_SIZE, QWEN_MODEL_ID, get_target
from minispeech.models.latent_io import (
    AcousticMarker,
    LatentInProjector,
    ModalityMasks,
    build_modality_masks,
)


@dataclass
class BackboneLMOutput:
    """Hidden states at acoustic positions plus masks."""

    acoustic_hidden: torch.Tensor  # [B, T, H]
    full_hidden: torch.Tensor  # [B, L+T, H]
    text_mask: torch.Tensor  # [B, L]
    frame_mask: torch.Tensor  # [B, T]
    seq_mask: torch.Tensor  # [B, L+T]
    is_acoustic: torch.Tensor  # [B, L+T]
    text_len: int
    acoustic_len: int
    extras: dict[str, Any] = field(default_factory=dict)


class _TinyCausalLM(nn.Module):
    """Minimal causal stack for unit tests (no HuggingFace weights)."""

    def __init__(self, vocab_size: int = 128, hidden_size: int = 64, n_layers: int = 2):
        super().__init__()
        self.hidden_size = hidden_size
        self.embed = nn.Embedding(vocab_size, hidden_size)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=4,
            dim_feedforward=hidden_size * 4,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            layer, num_layers=n_layers, enable_nested_tensor=False
        )
        self.config = type("cfg", (), {"hidden_size": hidden_size, "vocab_size": vocab_size})()

    def get_input_embeddings(self) -> nn.Embedding:
        return self.embed

    def forward(
        self,
        *,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        output_hidden_states: bool = False,
        **_: Any,
    ) -> Any:
        B, S, H = inputs_embeds.shape
        # Causal mask: True = blocked (nn.Transformer convention with src_mask).
        causal = torch.triu(
            torch.ones(S, S, device=inputs_embeds.device, dtype=torch.bool),
            diagonal=1,
        )
        key_padding = None
        if attention_mask is not None:
            # attention_mask: 1/True = keep → key_padding_mask True = pad
            key_padding = ~attention_mask.bool()
        h = self.encoder(inputs_embeds, mask=causal, src_key_padding_mask=key_padding)
        hs = (h,) if output_hidden_states else None

        class _Out:
            pass

        out = _Out()
        out.last_hidden_state = h
        out.hidden_states = hs
        return out


def load_qwen_backbone(
    model_id: str = QWEN_MODEL_ID,
    *,
    dtype: torch.dtype | str = torch.bfloat16,
    device: str | torch.device | None = None,
    trust_remote_code: bool = True,
    local_files_only: bool = False,
) -> nn.Module:
    """Load pretrained Qwen text model (no LM head required)."""
    if isinstance(dtype, str):
        dtype = getattr(torch, dtype)
    try:
        from transformers import Qwen3_5TextModel

        model = Qwen3_5TextModel.from_pretrained(
            model_id,
            torch_dtype=dtype,
            trust_remote_code=trust_remote_code,
            local_files_only=local_files_only,
        )
    except Exception:
        from transformers import AutoModel

        model = AutoModel.from_pretrained(
            model_id,
            torch_dtype=dtype,
            trust_remote_code=trust_remote_code,
            local_files_only=local_files_only,
        )
    if device is not None:
        model = model.to(device)
    return model


class QwenLatentBackbone(nn.Module):
    """Frozen Qwen + trainable latent→hidden projection over ``[text ∥ z]``."""

    def __init__(
        self,
        *,
        target: str = "vibevoice64",
        hidden_size: int | None = None,
        model_id: str = QWEN_MODEL_ID,
        backbone: nn.Module | None = None,
        freeze_backbone: bool = True,
        use_acoustic_marker: bool = True,
        projector_mlp_hidden: int | None = None,
        dropout: float = 0.0,
        dtype: torch.dtype | str = torch.float32,
        device: str | torch.device | None = None,
        load_pretrained: bool = False,
        local_files_only: bool = False,
    ):
        super().__init__()
        spec = get_target(target)
        if spec.kind != "latent":
            raise ValueError(f"N2 backbone requires a latent target, got {spec.name} kind={spec.kind}")
        self.target_name = spec.name
        self.latent_dim = spec.dim
        self.frame_rate = spec.frame_rate
        self.model_id = model_id

        if backbone is not None:
            self.backbone = backbone
        elif load_pretrained:
            self.backbone = load_qwen_backbone(
                model_id,
                dtype=dtype,
                device=device,
                local_files_only=local_files_only,
            )
        else:
            # Deferred: caller must inject backbone or set load_pretrained=True.
            self.backbone = None  # type: ignore[assignment]

        if hidden_size is None:
            if self.backbone is not None and hasattr(self.backbone, "config"):
                hidden_size = int(getattr(self.backbone.config, "hidden_size", QWEN_HIDDEN_SIZE))
            else:
                hidden_size = QWEN_HIDDEN_SIZE
        self.hidden_size = int(hidden_size)

        self.latent_in = LatentInProjector(
            latent_dim=self.latent_dim,
            hidden_size=self.hidden_size,
            mlp_hidden=projector_mlp_hidden,
            dropout=dropout,
        )
        self.acoustic_marker: AcousticMarker | None
        if use_acoustic_marker:
            self.acoustic_marker = AcousticMarker(self.hidden_size)
        else:
            self.acoustic_marker = None
        self.bos_latent = nn.Parameter(torch.zeros(self.latent_dim))
        nn.init.normal_(self.bos_latent, std=0.02)

        self.freeze_backbone_flag = freeze_backbone
        if freeze_backbone and self.backbone is not None:
            self.freeze_backbone()

    def attach_backbone(self, backbone: nn.Module, *, freeze: bool | None = None) -> None:
        """Inject / replace the transformer stack (tests or late load)."""
        self.backbone = backbone
        if hasattr(backbone, "config") and hasattr(backbone.config, "hidden_size"):
            h = int(backbone.config.hidden_size)
            if h != self.hidden_size:
                raise ValueError(
                    f"backbone hidden_size {h} != projector hidden_size {self.hidden_size}"
                )
        do_freeze = self.freeze_backbone_flag if freeze is None else freeze
        if do_freeze:
            self.freeze_backbone()

    def freeze_backbone(self) -> None:
        if self.backbone is None:
            return
        self.backbone.eval()
        for p in self.backbone.parameters():
            p.requires_grad_(False)

    def backbone_trainable_parameter_count(self) -> int:
        if self.backbone is None:
            return 0
        return sum(p.numel() for p in self.backbone.parameters() if p.requires_grad)

    def trainable_parameters(self) -> list[nn.Parameter]:
        """Projection / marker / BOS only (excludes frozen backbone)."""
        params: list[nn.Parameter] = []
        params.extend(self.latent_in.parameters())
        if self.acoustic_marker is not None:
            params.extend(self.acoustic_marker.parameters())
        params.append(self.bos_latent)
        return [p for p in params if p.requires_grad]

    def trainable_parameter_count(self) -> int:
        return sum(p.numel() for p in self.trainable_parameters())

    def _text_embeds(self, input_ids: torch.Tensor) -> torch.Tensor:
        assert self.backbone is not None, "backbone not attached"
        emb = self.backbone.get_input_embeddings()
        return emb(input_ids)

    def build_inputs_embeds(
        self,
        input_ids: torch.Tensor,
        z_input: torch.Tensor,
        *,
        text_mask: torch.Tensor | None = None,
        frame_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, ModalityMasks]:
        """Concatenate text token embeds and projected latents.

        Parameters
        ----------
        input_ids:
            ``[B, L]`` padded text token ids.
        z_input:
            ``[B, T, 64]`` teacher-forced latents (pos0 = BOS).
        """
        B, L = input_ids.shape
        T = z_input.shape[1]
        if text_mask is None:
            text_mask = input_ids.ne(0)
            # Prefer non-pad: if pad id is not 0, caller should pass mask.
        if frame_mask is None:
            frame_mask = z_input.abs().sum(dim=-1) > 0
            frame_mask[:, 0] = True

        text_lens = text_mask.sum(dim=-1)
        acoustic_lens = frame_mask.sum(dim=-1)
        masks = build_modality_masks(
            text_lens,
            acoustic_lens,
            device=input_ids.device,
            max_text_len=L,
            max_acoustic_len=T,
        )

        text_emb = self._text_embeds(input_ids)  # [B, L, H]
        # Zero padded text positions so they do not leak into sum-based checks.
        text_emb = text_emb * masks.text_mask.unsqueeze(-1).to(dtype=text_emb.dtype)

        # Learned BOS on acoustic position 0 (matches Talker.apply_bos).
        z_use = z_input
        if z_input.shape[1] > 0:
            z_use = z_input.clone()
            bos = self.bos_latent.to(dtype=z_use.dtype, device=z_use.device)
            has = masks.frame_mask[:, 0]
            z_use[:, 0] = torch.where(has.unsqueeze(-1), bos.expand_as(z_use[:, 0]), z_use[:, 0])

        lat_emb = self.latent_in(z_use)
        if self.acoustic_marker is not None:
            lat_emb = self.acoustic_marker(lat_emb)
        lat_emb = lat_emb * masks.frame_mask.unsqueeze(-1).to(dtype=lat_emb.dtype)

        # Pad text/latent lengths to batch maxima used in masks.
        if text_emb.shape[1] != masks.max_text_len:
            raise ValueError("input_ids width must equal max text length in batch")
        if lat_emb.shape[1] != masks.max_acoustic_len:
            raise ValueError("z_input width must equal max acoustic length in batch")

        embeds = torch.cat([text_emb, lat_emb], dim=1)  # [B, L+T, H]
        return embeds, masks

    def forward(
        self,
        input_ids: torch.Tensor,
        z_input: torch.Tensor,
        *,
        text_mask: torch.Tensor | None = None,
        frame_mask: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
    ) -> BackboneLMOutput:
        """Causal forward; returns hidden states at acoustic positions.

        The transformer sees ``[text ∥ latents]`` with a causal mask. Acoustic
        hidden ``h_t`` at sequence index ``L+t`` conditions prediction of the
        next latent (head is external).
        """
        if self.backbone is None:
            raise RuntimeError("QwenLatentBackbone.backbone is None; call attach_backbone or load_pretrained")

        embeds, masks = self.build_inputs_embeds(
            input_ids, z_input, text_mask=text_mask, frame_mask=frame_mask
        )
        # Match frozen Qwen dtype (often bf16) — projectors may be fp32.
        try:
            bb_dtype = next(self.backbone.parameters()).dtype
            embeds = embeds.to(dtype=bb_dtype)
        except StopIteration:
            pass
        attn = attention_mask if attention_mask is not None else masks.seq_mask
        # HF models expect long/int attention masks with 1 = keep.
        if attn.dtype == torch.bool:
            attn_hf = attn.long()
        else:
            attn_hf = attn

        was_training = self.backbone.training
        if self.freeze_backbone_flag:
            self.backbone.eval()

        out = self.backbone(
            inputs_embeds=embeds,
            attention_mask=attn_hf,
            output_hidden_states=False,
            use_cache=False,
            return_dict=True,
        )
        if was_training and not self.freeze_backbone_flag:
            self.backbone.train()

        full = out.last_hidden_state  # [B, L+T, H]
        L = masks.max_text_len
        T = masks.max_acoustic_len
        acoustic = full[:, L : L + T, :]
        return BackboneLMOutput(
            acoustic_hidden=acoustic,
            full_hidden=full,
            text_mask=masks.text_mask,
            frame_mask=masks.frame_mask,
            seq_mask=masks.seq_mask,
            is_acoustic=masks.is_acoustic,
            text_len=L,
            acoustic_len=T,
        )


def make_tiny_backbone_for_tests(
    *,
    vocab_size: int = 128,
    hidden_size: int = 64,
    n_layers: int = 2,
    target: str = "vibevoice64",
    freeze_backbone: bool = True,
) -> QwenLatentBackbone:
    """Factory used by unit tests (no pretrained weights)."""
    tiny = _TinyCausalLM(vocab_size=vocab_size, hidden_size=hidden_size, n_layers=n_layers)
    return QwenLatentBackbone(
        target=target,
        hidden_size=hidden_size,
        backbone=tiny,
        freeze_backbone=freeze_backbone,
        use_acoustic_marker=True,
        load_pretrained=False,
    )


TrainablePolicy = Literal["projections_only"]
"""Documented start policy: backbone frozen; projections (+ head) trainable."""

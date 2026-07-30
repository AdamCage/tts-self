"""Duration-conditioned CFM / rectified-flow mel decoder (Track A / a5).

Phone/char embeddings → encoder → hard/soft duration upsample → FlowHead
velocity field over log-mel. Reuses MelBaseline encode/upsample primitives.
No VibeVoice / Qwen / Talker dependency.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn as nn

from minispeech.constants import MEL80
from minispeech.models.heads_flow import FlowHead, rectified_flow_train_step, sample_rectified_flow
from minispeech.models.mel_baseline import (
    MelBaseline,
    hard_upsample,
    phone_duration_loss,
    soft_upsample,
)


@dataclass
class MelCFMOutput:
    mel_pred: torch.Tensor | None
    v_pred: torch.Tensor | None
    v_target: torch.Tensor | None
    frame_mask: torch.Tensor
    phone_mask: torch.Tensor
    dur_pred: torch.Tensor | None
    cond: torch.Tensor
    extras: dict


class MelCFM(nn.Module):
    """Non-AR CFM mel decoder conditioned on duration-upsampled phone states."""

    def __init__(
        self,
        vocab_size: int = 512,
        d_model: int = 256,
        n_mels: int = MEL80.dim,
        enc_layers: int = 3,
        n_heads: int = 4,
        ffn_dim: int = 1024,
        dropout: float = 0.1,
        upsample: Literal["hard", "soft"] = "hard",
        predict_duration: bool = True,
        pad_id: int = 0,
        flow_hidden: int = 512,
        flow_layers: int = 4,
        time_embed_dim: int = 256,
        cfg_dropout: float = 0.1,
        time_sampling: str = "logit_normal",
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_mels = n_mels
        self.upsample_mode = upsample
        self.pad_id = pad_id
        self.cfg_dropout = float(cfg_dropout)
        self.time_sampling = time_sampling

        # Reuse MelBaseline encoder + duration head (no regression mel decoder).
        self.backbone = MelBaseline(
            vocab_size=vocab_size,
            d_model=d_model,
            n_mels=n_mels,
            enc_layers=enc_layers,
            dec_layers=1,  # unused; flow head replaces mel decoder
            n_heads=n_heads,
            ffn_dim=ffn_dim,
            dropout=dropout,
            decoder="conv",
            upsample=upsample,
            predict_duration=predict_duration,
            pad_id=pad_id,
        )
        # Drop unused regression projection / decoder params from state footprint
        # by keeping them but never using them in the CFM path (harmless; tiny).
        self.flow = FlowHead(
            d_model=d_model,
            latent_dim=n_mels,
            hidden=flow_hidden,
            layers=flow_layers,
            time_embed_dim=time_embed_dim,
        )

    @property
    def emb(self) -> nn.Embedding:
        return self.backbone.emb

    @property
    def dur_head(self) -> nn.Module | None:
        return self.backbone.dur_head

    def encode_condition(
        self,
        phone_ids: torch.Tensor,
        *,
        phone_mask: torch.Tensor | None = None,
        durations: torch.Tensor | None = None,
        proj_W: torch.Tensor | None = None,
        use_pred_duration: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """Return (cond [B,T,D], frame_mask [B,T], phone_mask [B,P], dur_pred)."""
        if phone_mask is None:
            phone_mask = phone_ids != self.pad_id
        phone_h = self.backbone.encode_phones(phone_ids, phone_mask)
        dur_pred = None
        if self.backbone.dur_head is not None:
            dur_pred = self.backbone.dur_head(phone_h).squeeze(-1)

        if self.upsample_mode == "soft":
            if proj_W is None:
                raise ValueError("soft upsample requires proj_W [B,P,T]")
            frames, frame_mask = soft_upsample(phone_h, proj_W)
        else:
            if use_pred_duration and dur_pred is not None:
                durs = dur_pred.detach().clamp_min(1.0).round() * phone_mask.float()
            elif durations is not None:
                durs = durations.float() * phone_mask.float()
            elif dur_pred is not None:
                durs = dur_pred.clamp_min(1.0).round() * phone_mask.float()
            else:
                raise ValueError("hard upsample needs durations or dur_head")
            frames, frame_mask = hard_upsample(phone_h, durs)
        return frames, frame_mask, phone_mask, dur_pred

    def forward(
        self,
        phone_ids: torch.Tensor,
        *,
        mel_target: torch.Tensor | None = None,
        phone_mask: torch.Tensor | None = None,
        durations: torch.Tensor | None = None,
        proj_W: torch.Tensor | None = None,
        use_pred_duration: bool = False,
    ) -> MelCFMOutput:
        """Train step when ``mel_target`` is set; otherwise returns condition only."""
        cond, frame_mask, phone_mask, dur_pred = self.encode_condition(
            phone_ids,
            phone_mask=phone_mask,
            durations=durations,
            proj_W=proj_W,
            use_pred_duration=use_pred_duration,
        )
        if mel_target is None:
            return MelCFMOutput(
                mel_pred=None,
                v_pred=None,
                v_target=None,
                frame_mask=frame_mask,
                phone_mask=phone_mask,
                dur_pred=dur_pred,
                cond=cond,
                extras={},
            )

        t = min(cond.shape[1], mel_target.shape[1])
        cond_t = cond[:, :t]
        mel_t = mel_target[:, :t]
        mask_t = frame_mask[:, :t]
        # Zero-pad mel beyond mask is fine; loss masks frames.
        v_pred, v_target = rectified_flow_train_step(
            self.flow,
            mel_t,
            cond_t,
            cfg_dropout=self.cfg_dropout if self.training else 0.0,
            time_sampling=self.time_sampling,
        )
        return MelCFMOutput(
            mel_pred=None,
            v_pred=v_pred,
            v_target=v_target,
            frame_mask=mask_t,
            phone_mask=phone_mask,
            dur_pred=dur_pred,
            cond=cond_t,
            extras={},
        )

    @torch.inference_mode()
    def sample(
        self,
        phone_ids: torch.Tensor,
        *,
        phone_mask: torch.Tensor | None = None,
        durations: torch.Tensor | None = None,
        proj_W: torch.Tensor | None = None,
        steps: int = 16,
        method: str = "heun",
        cfg_scale: float = 1.0,
        generator: torch.Generator | None = None,
    ) -> MelCFMOutput:
        """Integrate noise → mel (t:1→0) for N steps."""
        self.eval()
        cond, frame_mask, phone_mask, dur_pred = self.encode_condition(
            phone_ids,
            phone_mask=phone_mask,
            durations=durations,
            proj_W=proj_W,
            use_pred_duration=durations is None and proj_W is None,
        )
        mel = sample_rectified_flow(
            self.flow,
            cond,
            steps=steps,
            method=method,
            cfg_scale=cfg_scale,
            generator=generator,
        )
        return MelCFMOutput(
            mel_pred=mel,
            v_pred=None,
            v_target=None,
            frame_mask=frame_mask,
            phone_mask=phone_mask,
            dur_pred=dur_pred,
            cond=cond,
            extras={"flow_steps": steps, "method": method, "cfg_scale": cfg_scale},
        )

    @torch.no_grad()
    def infer(
        self,
        phone_ids: torch.Tensor,
        *,
        phone_mask: torch.Tensor | None = None,
        durations: torch.Tensor | None = None,
        proj_W: torch.Tensor | None = None,
        steps: int = 16,
        method: str = "heun",
        cfg_scale: float = 1.0,
    ) -> MelCFMOutput:
        return self.sample(
            phone_ids,
            phone_mask=phone_mask,
            durations=durations,
            proj_W=proj_W,
            steps=steps,
            method=method,
            cfg_scale=cfg_scale,
        )

    def load_backbone_from_regression(self, state_dict: dict, *, strict: bool = False) -> tuple:
        """Warm-start encoder/emb/dur from a MelBaseline ckpt state_dict."""
        bb_sd = self.backbone.state_dict()
        filtered = {
            k: v
            for k, v in state_dict.items()
            if k in bb_sd and tuple(bb_sd[k].shape) == tuple(v.shape)
        }
        return self.backbone.load_state_dict(filtered, strict=strict)


def mel_flow_mse(
    v_pred: torch.Tensor,
    v_target: torch.Tensor,
    frame_mask: torch.Tensor,
    *,
    sample_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    """Masked MSE over mel velocity; optional per-utt weight [B]."""
    from minispeech.train.losses import flow_mse

    return flow_mse(v_pred, v_target, frame_mask, sample_weight=sample_weight)


__all__ = [
    "MelCFM",
    "MelCFMOutput",
    "mel_flow_mse",
    "phone_duration_loss",
]

"""Full MiniSpeech module (regression or flow)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import torch
import torch.nn as nn

from minispeech.models.heads_flow import FlowHead, rectified_flow_train_step, sample_rectified_flow
from minispeech.models.heads_regression import RegressionHead
from minispeech.models.layer_mix import LayerMix
from minispeech.models.projector import MLPProjector
from minispeech.models.stop_duration import StopHead, TokenDurationHead, UtteranceDurationHead
from minispeech.models.talker import NonARTalker, Talker


@dataclass
class MiniSpeechOutput:
    z_pred: torch.Tensor | None
    stop_logits: torch.Tensor
    dur_pred: torch.Tensor
    tok_dur_pred: torch.Tensor | None
    cross_attn: torch.Tensor | None
    h: torch.Tensor
    extras: dict[str, Any]


class MiniSpeech(nn.Module):
    def __init__(
        self,
        mode: Literal["regression", "flow"] = "regression",
        d_model: int = 512,
        num_layers: int = 4,
        num_heads: int = 8,
        ffn_dim: int = 2048,
        latent_dim: int = 64,
        text_dim: int = 1024,
        n_cached_layers: int = 3,
        dropout: float = 0.1,
        residual_head: bool = True,
        causal: bool = True,
        use_token_duration: bool = False,
        flow_hidden: int = 512,
        flow_layers: int = 4,
    ):
        super().__init__()
        self.mode = mode
        self.latent_dim = latent_dim
        self.use_token_duration = use_token_duration
        self.layer_mix = LayerMix(n_cached_layers)
        self.text_proj = MLPProjector(text_dim, d_model, dropout=dropout)
        self.talker = Talker(
            d_model=d_model,
            num_layers=num_layers,
            num_heads=num_heads,
            ffn_dim=ffn_dim,
            latent_dim=latent_dim,
            dropout=dropout,
            causal=causal,
        )
        self.nonar: NonARTalker | None = None
        if mode == "regression":
            self.head = RegressionHead(
                d_model=d_model,
                latent_dim=latent_dim,
                residual=residual_head,
            )
        else:
            self.head = FlowHead(
                d_model=d_model,
                latent_dim=latent_dim,
                hidden=flow_hidden,
                layers=flow_layers,
            )
        self.stop_head = StopHead(d_model)
        self.dur_head = UtteranceDurationHead(d_model)
        self.tok_dur_head = TokenDurationHead(d_model) if use_token_duration else None

    def enable_nonar_fallback(self) -> None:
        self.nonar = NonARTalker(self.talker)

    def encode_text(self, text_hidden: torch.Tensor) -> torch.Tensor:
        mixed = self.layer_mix(text_hidden)
        return self.text_proj(mixed)

    def forward(
        self,
        z_input: torch.Tensor,
        text_hidden: torch.Tensor,
        frame_mask: torch.Tensor | None = None,
        text_mask: torch.Tensor | None = None,
        align_prior: torch.Tensor | None = None,
        guided_weight: float = 0.0,
        z_target: torch.Tensor | None = None,
        cfg_dropout: float = 0.1,
        time_sampling: str = "logit_normal",
        mask_text: bool = False,
        acoustic_dropout: float = 0.0,
    ) -> MiniSpeechOutput:
        if mask_text:
            text_hidden = torch.zeros_like(text_hidden)
        # Drop previous-latent conditioning so the model cannot ignore text.
        if self.training and acoustic_dropout > 0:
            if torch.rand((), device=z_input.device) < acoustic_dropout:
                z_input = torch.zeros_like(z_input)
        text_memory = self.encode_text(text_hidden)
        tout = self.talker(
            z_input,
            text_memory,
            frame_mask=frame_mask,
            text_mask=text_mask,
            align_prior=align_prior,
            guided_weight=guided_weight,
        )
        h = tout.hidden
        stop_logits = self.stop_head(h)
        dur_pred = self.dur_head(text_memory, text_mask)
        tok_dur = self.tok_dur_head(text_memory) if self.tok_dur_head is not None else None

        extras: dict[str, Any] = {}
        z_pred = None
        if self.mode == "regression":
            assert isinstance(self.head, RegressionHead)
            z_pred = self.head(h, z_prev=z_input)
        else:
            assert isinstance(self.head, FlowHead)
            if z_target is None:
                raise ValueError("flow training requires z_target")
            v_pred, v_target = rectified_flow_train_step(
                self.head,
                z_target,
                h,
                cfg_dropout=cfg_dropout,
                time_sampling=time_sampling,
            )
            extras["v_pred"] = v_pred
            extras["v_target"] = v_target
            z_pred = None  # not used in flow train loss directly

        return MiniSpeechOutput(
            z_pred=z_pred,
            stop_logits=stop_logits,
            dur_pred=dur_pred,
            tok_dur_pred=tok_dur,
            cross_attn=tout.cross_attn,
            h=h,
            extras=extras,
        )

    @torch.inference_mode()
    def generate(
        self,
        text_hidden: torch.Tensor,
        text_mask: torch.Tensor | None = None,
        *,
        max_frames: int | None = None,
        cfg_scale: float = 1.3,
        flow_steps: int = 4,
        sampler: str = "heun",
        stop_threshold: float = 0.5,
        stop_consecutive: int = 2,
        min_frac: float = 0.5,
        max_frac: float = 1.5,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Autoregressive generation. Returns z [1,T,64] normalized."""
        self.eval()
        text_memory = self.encode_text(text_hidden)
        dur = self.dur_head(text_memory, text_mask)
        dur_frames = int(dur.item()) if dur.ndim == 0 or dur.numel() == 1 else int(dur[0].item())
        dur_frames = max(1, dur_frames)
        min_f = max(1, int(dur_frames * min_frac))
        max_f = int(dur_frames * max_frac) if max_frames is None else min(int(dur_frames * max_frac), max_frames)

        device = text_memory.device
        dtype = text_memory.dtype
        z_frames: list[torch.Tensor] = []
        z_prev = self.talker.bos.to(device=device, dtype=dtype).view(1, 1, -1)
        stop_probs: list[float] = []

        for t in range(max_f):
            z_in = z_prev if t == 0 else torch.cat([self.talker.bos.view(1, 1, -1).to(z_prev), torch.cat(z_frames, dim=1)[:, -t:]], dim=1)
            # rebuild full shifted input
            if t == 0:
                z_input = self.talker.bos.view(1, 1, -1).to(device=device, dtype=dtype)
            else:
                prev = torch.cat(z_frames, dim=1)
                z_input = torch.cat(
                    [self.talker.bos.view(1, 1, -1).to(device=device, dtype=dtype), prev],
                    dim=1,
                )
            tout = self.talker(z_input, text_memory, text_mask=text_mask)
            h_t = tout.hidden[:, -1:, :]
            if self.mode == "regression":
                assert isinstance(self.head, RegressionHead)
                z_t = self.head(h_t, z_prev=z_input[:, -1:, :])
            else:
                assert isinstance(self.head, FlowHead)
                z_t = sample_rectified_flow(
                    self.head, h_t, steps=flow_steps, method=sampler, cfg_scale=cfg_scale
                )
            z_frames.append(z_t)
            sp = torch.sigmoid(self.stop_head(h_t))[0, 0].item()
            stop_probs.append(sp)
            if t + 1 >= min_f:
                if len(stop_probs) >= stop_consecutive and all(
                    p > stop_threshold for p in stop_probs[-stop_consecutive:]
                ):
                    break

        z = torch.cat(z_frames, dim=1)
        return z, {"dur_pred": dur_frames, "stop_probs": stop_probs, "n_frames": z.shape[1]}

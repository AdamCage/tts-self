"""Full MiniSpeech module (regression or flow)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterator, Literal

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
    def iter_generate(
        self,
        text_hidden: torch.Tensor,
        text_mask: torch.Tensor | None = None,
        *,
        max_frames: int | None = None,
        cfg_scale: float = 1.3,
        flow_steps: int = 4,
        sampler: str = "heun",
        stop_threshold: float = 0.45,
        stop_consecutive: int = 3,
        min_frac: float = 0.94,
        max_frac: float = 1.6,
        duration_scale: float = 1.04,
        use_kv_cache: bool = True,
        duration_stop_boost: bool = True,
    ) -> Iterator[tuple[torch.Tensor, dict[str, Any]]]:
        """Yield ``(z_t [1,1,64], step_info)`` frames for streaming pipelines."""
        self.eval()
        import time

        t_text0 = time.perf_counter()
        text_memory = self.encode_text(text_hidden)
        if text_memory.device.type == "cuda":
            torch.cuda.synchronize()
        talker_text_ms = (time.perf_counter() - t_text0) * 1000.0

        dur = self.dur_head(text_memory, text_mask)
        dur_raw = float(dur.item()) if dur.ndim == 0 or dur.numel() == 1 else float(dur[0].item())
        dur_frames = max(1, int(round(dur_raw * float(duration_scale))))
        min_f = max(1, int(dur_frames * min_frac))
        max_f = int(dur_frames * max_frac) if max_frames is None else min(int(dur_frames * max_frac), max_frames)
        max_f = max(max_f, min_f)

        device = text_memory.device
        dtype = text_memory.dtype
        z_frames: list[torch.Tensor] = []
        stop_probs: list[float] = []
        past_kvs = None
        first_latent_ms = 0.0
        first_sampler_ms = 0.0
        t_loop0 = time.perf_counter()
        hit_max = True

        for t in range(max_f):
            if t == 0:
                z_step = self.talker.bos.view(1, 1, -1).to(device=device, dtype=dtype)
            else:
                z_step = z_frames[-1]

            if use_kv_cache:
                tout, past_kvs = self.talker.forward_step(
                    z_step, text_memory, past_kvs, text_mask=text_mask, position=t
                )
                h_t = tout.hidden
                z_prev_for_head = z_step
            else:
                # Full-prefix fallback (tests / debug)
                if t == 0:
                    z_input = z_step
                else:
                    prev = torch.cat(z_frames, dim=1)
                    z_input = torch.cat(
                        [self.talker.bos.view(1, 1, -1).to(device=device, dtype=dtype), prev],
                        dim=1,
                    )
                tout = self.talker(z_input, text_memory, text_mask=text_mask)
                h_t = tout.hidden[:, -1:, :]
                z_prev_for_head = z_input[:, -1:, :]

            if self.mode == "regression":
                assert isinstance(self.head, RegressionHead)
                z_t = self.head(h_t, z_prev=z_prev_for_head)
                sampler_ms_t = 0.0
            else:
                assert isinstance(self.head, FlowHead)
                t_samp0 = time.perf_counter()
                z_t = sample_rectified_flow(
                    self.head, h_t, steps=flow_steps, method=sampler, cfg_scale=cfg_scale
                )
                if device.type == "cuda":
                    torch.cuda.synchronize()
                sampler_ms_t = (time.perf_counter() - t_samp0) * 1000.0
            z_frames.append(z_t)
            if t == 0:
                if device.type == "cuda":
                    torch.cuda.synchronize()
                first_latent_ms = (time.perf_counter() - t_loop0) * 1000.0
                first_sampler_ms = sampler_ms_t
            sp = torch.sigmoid(self.stop_head(h_t))[0, 0].item()
            # After predicted duration, blend toward stop so weak stop heads
            # still terminate before max_frac (reduces runaway without early cuts).
            if duration_stop_boost and (t + 1) >= dur_frames and max_f > dur_frames:
                overshoot = ((t + 1) - dur_frames) / float(max_f - dur_frames)
                sp = max(sp, stop_threshold + (1.0 - stop_threshold) * min(overshoot, 1.0))
            stop_probs.append(sp)
            should_stop = False
            if t + 1 >= min_f:
                if len(stop_probs) >= stop_consecutive and all(
                    p > stop_threshold for p in stop_probs[-stop_consecutive:]
                ):
                    hit_max = False
                    should_stop = True
            step_info = {
                "t": t,
                "dur_pred": dur_frames,
                "dur_pred_raw": dur_raw,
                "duration_scale": float(duration_scale),
                "stop_prob": sp,
                "stop_probs": list(stop_probs),
                "should_stop": should_stop,
                "hit_max_frames": hit_max and not should_stop and (t + 1) >= max_f,
                "talker_text_ms": talker_text_ms,
                # Legacy alias: Talker text projector encode (not Qwen).
                "text_encode_ms": talker_text_ms,
                "first_latent_ms": first_latent_ms,
                "sampler_ms": first_sampler_ms,
                "n_frames": t + 1,
            }
            yield z_t, step_info
            if should_stop:
                break

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
        stop_threshold: float = 0.45,
        stop_consecutive: int = 3,
        min_frac: float = 0.94,
        max_frac: float = 1.6,
        duration_scale: float = 1.04,
        use_kv_cache: bool = True,
        duration_stop_boost: bool = True,
        on_frame: Callable[[int, torch.Tensor, dict[str, Any]], None] | None = None,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Autoregressive generation. Returns z [1,T,64] normalized.

        Uses incremental Talker KV-cache by default (O(T) vs full-prefix O(T²)).
        Stop rule: ``stop_consecutive`` frames above ``stop_threshold`` after
        ``min_frac * dur_pred`` (AGENTS.md triple-stop), capped at
        ``max_frac * dur_pred``. Optional duration boost after ``dur_pred``
        raises stop probability so late endings do not always hit max.
        ``duration_scale`` multiplies the head's frame prediction (infer calibration).
        """
        z_frames: list[torch.Tensor] = []
        last: dict[str, Any] = {}
        for z_t, step in self.iter_generate(
            text_hidden,
            text_mask,
            max_frames=max_frames,
            cfg_scale=cfg_scale,
            flow_steps=flow_steps,
            sampler=sampler,
            stop_threshold=stop_threshold,
            stop_consecutive=stop_consecutive,
            min_frac=min_frac,
            max_frac=max_frac,
            duration_scale=duration_scale,
            use_kv_cache=use_kv_cache,
            duration_stop_boost=duration_stop_boost,
        ):
            z_frames.append(z_t)
            last = step
            if on_frame is not None:
                on_frame(step["t"], z_t, step)

        z = torch.cat(z_frames, dim=1)
        return z, {
            "dur_pred": last.get("dur_pred", z.shape[1]),
            "dur_pred_raw": last.get("dur_pred_raw", float(z.shape[1])),
            "duration_scale": last.get("duration_scale", float(duration_scale)),
            "stop_probs": last.get("stop_probs", []),
            "n_frames": z.shape[1],
            "stop_triggered": bool(last.get("should_stop", False)),
            "hit_max_frames": bool(last.get("hit_max_frames", True)),
            "talker_text_ms": float(last.get("talker_text_ms", 0.0)),
            "text_encode_ms": float(last.get("text_encode_ms", 0.0)),
            "first_latent_ms": float(last.get("first_latent_ms", 0.0)),
            "sampler_ms": float(last.get("sampler_ms", 0.0)),
        }

"""Streaming helpers and TTFB measurement."""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from typing import Any, Callable

import numpy as np

from minispeech.constants import FRAME_MS, SR


@dataclass
class LatencyBreakdown:
    text_encode_ms: float  # Qwen (post text-normalization), protocol name
    first_latent_ms: float  # wall time from TTFB clock to first latent
    first_pcm_ms: float  # wall time to first PCM ≥ min_pcm_ms
    ttfb_ms: float
    rtf: float
    audio_s: float
    total_ms: float = 0.0
    estimated: bool = False
    qwen_ms: float = 0.0
    talker_ms: float = 0.0
    sampler_ms: float = 0.0
    codec_first_ms: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def measure_ttfb(
    generate_fn: Callable[[], tuple[np.ndarray, dict]],
    *,
    min_pcm_ms: float = 40.0,
) -> LatencyBreakdown:
    """
    Measure TTFB from generate_fn timing hooks when present.

    generate_fn should return (audio, info) where info may include:
      text_encode_ms / qwen_ms, first_latent_ms, first_pcm_ms, sampler_ms, codec_first_ms
    If first_pcm_ms is missing, fall back to wall-clock until audio returns
    (streaming callers should set first_pcm_ms when the first ≥min_pcm_ms buffer is ready).
    """
    t0 = time.perf_counter()
    audio, info = generate_fn()
    t1 = time.perf_counter()
    audio_s = len(audio) / float(SR)
    total_ms = (t1 - t0) * 1000.0

    qwen_ms = float(info.get("qwen_ms", info.get("text_encode_ms", 0.0)))
    text_encode_ms = float(info.get("text_encode_ms", qwen_ms))
    first_latent_ms = float(info.get("first_latent_ms", 0.0))
    first_pcm_ms = info.get("first_pcm_ms")
    estimated = bool(info.get("estimated", False))
    if first_pcm_ms is None:
        # Prefer latent timing + one frame decode; else crude total/n_frames estimate
        if first_latent_ms > 0:
            first_pcm_ms = first_latent_ms + FRAME_MS
            estimated = True
        else:
            n_frames = int(info.get("n_frames", max(1, int(round(audio_s * 7.5)))))
            first_latent_ms = total_ms / max(n_frames, 1)
            first_pcm_ms = first_latent_ms + FRAME_MS
            estimated = True
    first_pcm_ms = float(first_pcm_ms)
    # TTFB: first PCM buffer ≥ min_pcm_ms (caller responsibility for stream path)
    ttfb = max(first_pcm_ms, min_pcm_ms) if estimated else first_pcm_ms
    talker_ms = float(info.get("talker_ms", max(0.0, first_latent_ms - qwen_ms)))
    sampler_ms = float(info.get("sampler_ms", 0.0))
    codec_first_ms = float(info.get("codec_first_ms", max(0.0, first_pcm_ms - first_latent_ms)))
    return LatencyBreakdown(
        text_encode_ms=text_encode_ms,
        first_latent_ms=first_latent_ms,
        first_pcm_ms=first_pcm_ms,
        ttfb_ms=ttfb,
        rtf=total_ms / 1000.0 / max(audio_s, 1e-6),
        audio_s=audio_s,
        total_ms=total_ms,
        estimated=estimated,
        qwen_ms=qwen_ms,
        talker_ms=talker_ms,
        sampler_ms=sampler_ms,
        codec_first_ms=codec_first_ms,
    )


def fill_stream_timing(
    info: dict[str, Any],
    *,
    t_start: float,
    t_text_done: float | None = None,
    t_first_latent: float | None = None,
    t_first_pcm: float | None = None,
) -> dict[str, Any]:
    """Helper for streaming pipelines to stamp real wall-clock marks into info."""
    if t_text_done is not None:
        info["text_encode_ms"] = (t_text_done - t_start) * 1000.0
        info.setdefault("qwen_ms", info["text_encode_ms"])
    if t_first_latent is not None:
        info["first_latent_ms"] = (t_first_latent - t_start) * 1000.0
    if t_first_pcm is not None:
        info["first_pcm_ms"] = (t_first_pcm - t_start) * 1000.0
    return info

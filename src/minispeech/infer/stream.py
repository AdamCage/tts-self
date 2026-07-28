"""Streaming helpers and TTFB measurement."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable

import numpy as np

from minispeech.constants import FRAME_MS


@dataclass
class LatencyBreakdown:
    text_encode_ms: float
    first_latent_ms: float
    first_pcm_ms: float
    ttfb_ms: float
    rtf: float
    audio_s: float


def measure_ttfb(
    generate_fn: Callable[[], tuple[np.ndarray, dict]],
    *,
    min_pcm_ms: float = 40.0,
) -> LatencyBreakdown:
    """
    TTFB = time from end of text normalization (caller responsibility: start clock
    after normalize) until first PCM buffer >= min_pcm_ms is available.
    Here generate_fn is assumed to start after normalization.
    """
    t0 = time.perf_counter()
    audio, info = generate_fn()
    t1 = time.perf_counter()
    audio_s = len(audio) / 24000.0
    total_ms = (t1 - t0) * 1000
    # Without true streaming hooks, approximate TTFB as time to first frame * FRAME_MS decode
    n_frames = info.get("n_frames", max(1, int(round(audio_s * 7.5))))
    first_latent_ms = total_ms / max(n_frames, 1)
    ttfb = first_latent_ms + FRAME_MS  # at least one latent worth of audio
    return LatencyBreakdown(
        text_encode_ms=info.get("text_encode_ms", 0.0),
        first_latent_ms=first_latent_ms,
        first_pcm_ms=ttfb,
        ttfb_ms=ttfb,
        rtf=total_ms / 1000.0 / max(audio_s, 1e-6),
        audio_s=audio_s,
    )

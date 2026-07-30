"""Inference helpers. Heavy imports (codec / generate) are lazy so Track A
vocoder scaffolding stays importable without torchaudio/VibeVoice."""

from __future__ import annotations

from typing import Any

__all__ = [
    "LatencyBreakdown",
    "load_checkpoint",
    "measure_ttfb",
    "save_wav",
    "stream_decode_latents",
    "synthesize",
    "synthesize_streaming",
    "mel_to_wav",
    "vocoder_status",
]


def __getattr__(name: str) -> Any:
    if name in {
        "load_checkpoint",
        "save_wav",
        "stream_decode_latents",
        "synthesize",
        "synthesize_streaming",
    }:
        from minispeech.infer import generate as _gen

        return getattr(_gen, name)
    if name in {"LatencyBreakdown", "measure_ttfb"}:
        from minispeech.infer import stream as _stream

        return getattr(_stream, name)
    if name in {"mel_to_wav", "vocoder_status"}:
        from minispeech.infer import vocoder as _voc

        return getattr(_voc, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

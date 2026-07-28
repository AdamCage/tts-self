"""Codec / audio constants for MiniSpeech-RU.

All VibeVoice Acoustic Tokenizer geometry lives here so a codec fallback
is a one-module change.
"""

from __future__ import annotations

# VibeVoice Acoustic Tokenizer (microsoft/VibeVoice-AcousticTokenizer)
SAMPLES_PER_FRAME: int = 3200
SR: int = 24_000
FRAME_RATE: float = 7.5  # Hz
FRAME_MS: float = 1000.0 / FRAME_RATE  # ~133.333 ms
LATENT_DIM: int = 64
VAE_STD: float = 0.625  # posterior std; NOT an error budget for the head

CODEC_MODEL_ID: str = "microsoft/VibeVoice-AcousticTokenizer"
QWEN_MODEL_ID: str = "Qwen/Qwen3.5-0.8B-Base"

# Cached Qwen hidden layers (0-based transformer block indices; last = 23)
# Plan uses layers 12, 18, 24 where 24 means final hidden (after last block).
# We store after-block indices 11, 17, 23 plus we also allow "final".
CACHED_LAYER_INDICES: tuple[int, ...] = (11, 17, 23)
QWEN_HIDDEN_SIZE: int = 1024
TALKER_D_MODEL: int = 512

# Unicode combining acute accent used by RUSLAN Stressed
STRESS_CHAR: str = "\u0301"


def n_frames_from_samples(n_samples: int) -> int:
    """True frame count before pad_to_multiple_of padding."""
    return (n_samples + SAMPLES_PER_FRAME - 1) // SAMPLES_PER_FRAME


def pad_samples_to_multiple(n_samples: int, multiple: int = SAMPLES_PER_FRAME) -> int:
    rem = n_samples % multiple
    if rem == 0:
        return n_samples
    return n_samples + (multiple - rem)

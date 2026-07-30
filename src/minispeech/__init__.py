"""MiniSpeech-RU: Russian TTS research prototype."""

from minispeech.constants import (
    FRAME_MS,
    FRAME_RATE,
    LATENT_DIM,
    MEL80,
    SAMPLES_PER_FRAME,
    SR,
    VIBEVOICE64,
    get_target,
)

__all__ = [
    "FRAME_MS",
    "FRAME_RATE",
    "LATENT_DIM",
    "MEL80",
    "SAMPLES_PER_FRAME",
    "SR",
    "VIBEVOICE64",
    "get_target",
    "__version__",
]

__version__ = "0.1.0"

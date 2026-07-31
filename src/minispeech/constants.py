"""Codec / audio / target geometry for MiniSpeech-RU.

VibeVoice Acoustic Tokenizer geometry and Track A mel targets live in the
``TARGETS`` registry below. Module-level aliases
(``LATENT_DIM``, ``FRAME_RATE``, ``SAMPLES_PER_FRAME``, ``SR``, …) remain the
VibeVoice-64 defaults so existing imports keep working.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class TargetSpec:
    """Registered synthesis / encode target.

    Attributes
    ----------
    name:
        Registry key (e.g. ``\"vibevoice64\"``, ``\"mel80\"``).
    kind:
        ``\"latent\"`` (codec latents) or ``\"mel\"`` (log-mel spectrogram).
    dim:
        Feature dimension (64 latents / 80 mel bins).
    sample_rate:
        Waveform sample rate in Hz for this target's STFT / codec path.
    frame_rate:
        Frames per second (exact float; see hop for mel).
    samples_per_frame:
        Wave samples per frame (= hop for mel; codec hop for latents).
    n_fft / hop_length / win_length / n_mels:
        Mel STFT parameters (None for latent targets).
    fmin / fmax / mel_scale / mel_norm:
        Mel filterbank domain (None → backend defaults; mel targets should set
        these explicitly). ``mel_scale`` is ``\"slaney\"`` or ``\"htk\"``;
        ``mel_norm`` is ``\"slaney\"`` or ``None``.
    """

    name: str
    kind: Literal["latent", "mel"]
    dim: int
    sample_rate: int
    frame_rate: float
    samples_per_frame: int
    n_fft: int | None = None
    hop_length: int | None = None
    win_length: int | None = None
    n_mels: int | None = None
    fmin: float | None = None
    fmax: float | None = None
    mel_scale: str | None = None
    mel_norm: str | None = None
    notes: str = ""

    @property
    def frame_ms(self) -> float:
        return 1000.0 / self.frame_rate

    def n_frames_from_samples(self, n_samples: int) -> int:
        """True frame count before pad-to-multiple (ceil division by hop)."""
        hop = self.samples_per_frame
        return (n_samples + hop - 1) // hop

    def pad_samples_to_multiple(self, n_samples: int) -> int:
        hop = self.samples_per_frame
        rem = n_samples % hop
        if rem == 0:
            return n_samples
        return n_samples + (hop - rem)


# ---------------------------------------------------------------------------
# Registered targets
# ---------------------------------------------------------------------------

# VibeVoice Acoustic Tokenizer (microsoft/VibeVoice-AcousticTokenizer)
# 24 kHz, 3200 samples/frame → 7.5 Hz, 64-D latents.
VIBEVOICE64 = TargetSpec(
    name="vibevoice64",
    kind="latent",
    dim=64,
    sample_rate=24_000,
    frame_rate=7.5,
    samples_per_frame=3200,
    notes="Frozen VibeVoice Acoustic Tokenizer; encode with sample=False.",
)

# Track A mel target @ ~86.1 Hz.
# Exact: frame_rate = 22050 / 256 = 86.1328125 Hz.
# hop=256 is the standard 22.05 kHz setting that lands on ~86.1 Hz.
# fmin/fmax/mel_scale/mel_norm match historical librosa defaults used to build
# the production mel cache (do not change — paper2 freeze / S4 numbers).
MEL80 = TargetSpec(
    name="mel80",
    kind="mel",
    dim=80,
    sample_rate=22_050,
    frame_rate=22_050 / 256,  # 86.1328125 Hz
    samples_per_frame=256,  # = hop_length
    n_fft=1024,
    hop_length=256,
    win_length=1024,
    n_mels=80,
    fmin=0.0,
    fmax=11_025.0,  # sr/2 — historical Track A domain
    mel_scale="slaney",
    mel_norm="slaney",
    notes=(
        "Log-mel (Track A). STFT: sr=22050, n_fft=1024, win=1024, hop=256, "
        "n_mels=80, fmax=11025 (nyquist) → frame_rate=22050/256≈86.1328125 Hz."
    ),
)

# HiFi-GAN-matched mel domain (SpeechBrain LibriTTS 22050 Hz vocoder train).
# Same hop/n_fft/n_mels as mel80; fmax=8000 + slaney filters per model card.
MEL80_HIFIGAN = TargetSpec(
    name="mel80_hifigan",
    kind="mel",
    dim=80,
    sample_rate=22_050,
    frame_rate=22_050 / 256,
    samples_per_frame=256,
    n_fft=1024,
    hop_length=256,
    win_length=1024,
    n_mels=80,
    fmin=0.0,
    fmax=8_000.0,
    mel_scale="slaney",
    mel_norm="slaney",
    notes=(
        "Log-mel matched to speechbrain/tts-hifigan-libritts-22050Hz "
        "(fmax=8000, slaney/slaney). Cache separately from mel80."
    ),
)

# Track N σ-VAE acoustic tokenizer T0 (own weights; 7.5 Hz, P0 gate decision).
# Same hop geometry as vibevoice64; not the frozen VibeVoice codec.
SIGMA_VAE64_T0 = TargetSpec(
    name="sigma_vae64_t0",
    kind="latent",
    dim=64,
    sample_rate=24_000,
    frame_rate=7.5,
    samples_per_frame=3200,
    notes=(
        "Track N acoustic σ-VAE T0 (÷4 size). Encode with sample=False for TTS "
        "targets. Downsample factors 2·4·5·5·4·4 = 3200 @ 24 kHz → 7.5 Hz."
    ),
)

TARGETS: dict[str, TargetSpec] = {
    VIBEVOICE64.name: VIBEVOICE64,
    MEL80.name: MEL80,
    MEL80_HIFIGAN.name: MEL80_HIFIGAN,
    SIGMA_VAE64_T0.name: SIGMA_VAE64_T0,
}

DEFAULT_TARGET: str = VIBEVOICE64.name


def get_target(name: str | TargetSpec | None = None) -> TargetSpec:
    """Resolve a target by name, pass-through ``TargetSpec``, or default."""
    if name is None:
        return TARGETS[DEFAULT_TARGET]
    if isinstance(name, TargetSpec):
        return name
    try:
        return TARGETS[name]
    except KeyError as exc:
        known = ", ".join(sorted(TARGETS))
        raise KeyError(f"Unknown target {name!r}; registered: {known}") from exc


# ---------------------------------------------------------------------------
# Backward-compatible VibeVoice module aliases (do not remove)
# ---------------------------------------------------------------------------

SAMPLES_PER_FRAME: int = VIBEVOICE64.samples_per_frame
SR: int = VIBEVOICE64.sample_rate
FRAME_RATE: float = VIBEVOICE64.frame_rate
FRAME_MS: float = VIBEVOICE64.frame_ms
LATENT_DIM: int = VIBEVOICE64.dim
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
    """True frame count before pad_to_multiple_of padding (VibeVoice hop)."""
    return VIBEVOICE64.n_frames_from_samples(n_samples)


def pad_samples_to_multiple(n_samples: int, multiple: int = SAMPLES_PER_FRAME) -> int:
    rem = n_samples % multiple
    if rem == 0:
        return n_samples
    return n_samples + (multiple - rem)

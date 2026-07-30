"""Mel target registry + filter kwargs (Track A domain guard)."""

from __future__ import annotations

import numpy as np
import pytest

from minispeech.constants import MEL80, MEL80_HIFIGAN, get_target
from minispeech.data.mel import mel_filter_kwargs


def test_mel80_preserves_historical_domain() -> None:
    assert MEL80.fmin == 0.0
    assert MEL80.fmax == 11025.0
    assert MEL80.mel_scale == "slaney"
    assert MEL80.mel_norm == "slaney"
    assert MEL80.n_mels == 80
    assert MEL80.hop_length == 256


def test_mel80_hifigan_registered() -> None:
    spec = get_target("mel80_hifigan")
    assert spec is MEL80_HIFIGAN
    assert spec.fmax == 8000.0
    assert spec.mel_scale == "slaney"
    assert spec.mel_norm == "slaney"
    assert spec.sample_rate == MEL80.sample_rate
    assert spec.hop_length == MEL80.hop_length


def test_mel_filter_kwargs_defaults() -> None:
    fk = mel_filter_kwargs(MEL80)
    assert fk["fmax"] == 11025.0
    fk2 = mel_filter_kwargs(MEL80_HIFIGAN)
    assert fk2["fmax"] == 8000.0


def test_librosa_torchaudio_close_for_same_spec() -> None:
    """Both backends should agree closely once scale/norm are explicit."""
    librosa = pytest.importorskip("librosa")
    torchaudio = pytest.importorskip("torchaudio")
    import torch

    from minispeech.data.mel import mel_backend_name, waveform_to_logmel

    sr = 22050
    t = np.linspace(0, 0.5, int(sr * 0.5), endpoint=False, dtype=np.float32)
    wav = 0.2 * np.sin(2 * np.pi * 220 * t).astype(np.float32)

    # Force each backend by temporarily patching mel_backend_name.
    import minispeech.data.mel as mel_mod

    orig = mel_mod.mel_backend_name
    try:
        mel_mod.mel_backend_name = lambda: "librosa"  # type: ignore[assignment]
        a = waveform_to_logmel(wav, target="mel80", sample_rate=sr)
        mel_mod.mel_backend_name = lambda: "torchaudio"  # type: ignore[assignment]
        b = waveform_to_logmel(wav, target="mel80", sample_rate=sr)
    finally:
        mel_mod.mel_backend_name = orig  # type: ignore[assignment]

    assert a.shape == b.shape
    mae = float(torch.mean(torch.abs(a - b)).item())
    # Filters should match closely; librosa vs torchaudio still have small
    # numeric / padding differences even with identical fmin/fmax/scale/norm.
    assert mae < 0.1, mae
    _ = librosa, torchaudio  # silence unused if importorskip kept them

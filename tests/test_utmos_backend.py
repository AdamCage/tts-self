"""Tests for optional UTMOS backend (graceful skip when unavailable)."""

from __future__ import annotations

from pathlib import Path
from unittest import mock

import numpy as np
import pytest


def test_utmos_score_missing_file_returns_none() -> None:
    from minispeech.eval.utmos import reset_utmos_cache, utmos_score

    reset_utmos_cache()
    assert utmos_score("/nonexistent/path/does_not_exist.wav") is None


def test_utmos_score_none_when_hub_unavailable(tmp_path: Path) -> None:
    from minispeech.eval import utmos as utmos_mod

    utmos_mod.reset_utmos_cache()
    wav = tmp_path / "tone.wav"
    # Minimal PCM via numpy + soundfile if present; else skip write path.
    sr = 16000
    t = np.linspace(0, 0.5, int(sr * 0.5), endpoint=False, dtype=np.float32)
    x = 0.1 * np.sin(2 * np.pi * 440 * t)
    try:
        import soundfile as sf

        sf.write(str(wav), x, sr)
    except ImportError:
        pytest.skip("soundfile required")

    with mock.patch.object(utmos_mod, "_load_predictor", return_value=None):
        assert utmos_mod.utmos_score(str(wav)) is None


def test_utmos_score_range_when_backend_present() -> None:
    from minispeech.eval.utmos import reset_utmos_cache, utmos_score, utmos_load_error

    reset_utmos_cache()
    # Prefer a real corpus wav if present (WSL); otherwise skip.
    candidates = [
        Path.home() / "datasets/tts/ruslan/wavs",
        Path("out/mel_s4_demo_wavs"),
    ]
    sample: Path | None = None
    for root in candidates:
        if root.is_dir():
            wavs = sorted(root.glob("**/*.wav"))
            if wavs:
                sample = wavs[0]
                break
    if sample is None:
        pytest.skip("no wav available for live UTMOS probe")

    score = utmos_score(str(sample))
    if score is None:
        pytest.skip(f"UTMOS backend unavailable: {utmos_load_error()}")
    assert 1.0 <= score <= 5.0, score

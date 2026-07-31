"""Tests for telephone channel simulation (Track N / N12).

Loads ``channel.py`` by file path so collection does not depend on
``minispeech.data`` package ``__init__`` (sibling N11 may leave
``manifest.py`` temporarily unimportable).
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

_CHANNEL_PATH = Path(__file__).resolve().parents[1] / "src" / "minispeech" / "data" / "channel.py"


def _load_channel():
    name = "minispeech_data_channel_n12"
    spec = importlib.util.spec_from_file_location(name, _CHANNEL_PATH)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod  # required before @dataclass executes
    spec.loader.exec_module(mod)
    return mod


channel = _load_channel()
TELEPHONY_SR = channel.TELEPHONY_SR
OpusUnavailableError = channel.OpusUnavailableError
apply_telephony_channel = channel.apply_telephony_channel
available_codecs = channel.available_codecs
opus_available = channel.opus_available


def _tone(sr: int = 22050, dur_s: float = 0.5, freq: float = 440.0) -> np.ndarray:
    t = np.arange(int(sr * dur_s), dtype=np.float32) / float(sr)
    # Mix low + mid + high so band-limit and codec visibly change the signal.
    wav = (
        0.4 * np.sin(2 * np.pi * freq * t)
        + 0.3 * np.sin(2 * np.pi * 1200.0 * t)
        + 0.2 * np.sin(2 * np.pi * 5000.0 * t)
    )
    return wav.astype(np.float32)


def test_available_codecs_include_g711() -> None:
    codecs = available_codecs()
    assert "mulaw" in codecs
    assert "alaw" in codecs
    assert "none" in codecs


def test_length_preserved_restore_sr() -> None:
    sr = 22050
    wav = _tone(sr)
    out, out_sr = apply_telephony_channel(wav, sr, codec="mulaw", restore_sr=True)
    assert out_sr == sr
    assert out.shape == wav.shape
    assert out.dtype == np.float32


def test_telephony_sr_when_not_restored() -> None:
    sr = 22050
    wav = _tone(sr)
    out, out_sr = apply_telephony_channel(wav, sr, codec="mulaw", restore_sr=False)
    assert out_sr == TELEPHONY_SR
    expected = int(round(wav.shape[0] * TELEPHONY_SR / sr))
    assert abs(out.shape[0] - expected) <= 2


def test_result_differs_from_input() -> None:
    sr = 22050
    wav = _tone(sr)
    out, _ = apply_telephony_channel(wav, sr, codec="mulaw", restore_sr=True)
    assert not np.allclose(out, wav, atol=1e-3)
    mae = float(np.mean(np.abs(out - wav)))
    assert mae > 0.01, mae


def test_determinism_fixed_seed() -> None:
    sr = 22050
    wav = _tone(sr)
    a, _ = apply_telephony_channel(
        wav,
        sr,
        codec="mulaw",
        noise_snr_db=30.0,
        level_jitter_db=3.0,
        seed=42,
    )
    b, _ = apply_telephony_channel(
        wav,
        sr,
        codec="mulaw",
        noise_snr_db=30.0,
        level_jitter_db=3.0,
        seed=42,
    )
    c, _ = apply_telephony_channel(
        wav,
        sr,
        codec="mulaw",
        noise_snr_db=30.0,
        level_jitter_db=3.0,
        seed=7,
    )
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)


def test_distortions_off_by_default_deterministic() -> None:
    sr = 22050
    wav = _tone(sr)
    a, _ = apply_telephony_channel(wav, sr, codec="mulaw")
    b, _ = apply_telephony_channel(wav, sr, codec="mulaw")
    assert np.array_equal(a, b)


def test_alaw_roundtrip_changes_signal() -> None:
    sr = 16000
    wav = _tone(sr)
    out, out_sr = apply_telephony_channel(wav, sr, codec="alaw", restore_sr=True)
    assert out_sr == sr
    assert out.shape == wav.shape
    assert not np.allclose(out, wav, atol=1e-3)


def test_codec_none_is_simplified_bandlimit() -> None:
    """codec='none': band-limit + resample only (no G.711 / Opus)."""
    sr = 22050
    wav = _tone(sr)
    out, out_sr = apply_telephony_channel(wav, sr, codec="none", restore_sr=True)
    assert out_sr == sr
    assert out.shape == wav.shape
    assert not np.allclose(out, wav, atol=1e-3)


def test_opus_path() -> None:
    sr = 22050
    wav = _tone(sr)
    if not opus_available():
        with pytest.raises(OpusUnavailableError):
            apply_telephony_channel(wav, sr, codec="opus")
        return
    out, out_sr = apply_telephony_channel(wav, sr, codec="opus", restore_sr=True)
    assert out_sr == sr
    assert out.shape == wav.shape
    assert not np.allclose(out, wav, atol=1e-3)

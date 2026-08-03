"""Unit tests for Qwen3-TTS discrete target geometry (no GPU / no weights)."""

from __future__ import annotations

import pytest

from minispeech.constants import (
    QWEN3_TTS_TOKENIZER_ID,
    QWEN3TTS_12HZ,
    TARGETS,
    get_target,
)


def test_qwen3tts_12hz_registered() -> None:
    spec = get_target("qwen3tts_12hz")
    assert spec is QWEN3TTS_12HZ
    assert TARGETS["qwen3tts_12hz"] is QWEN3TTS_12HZ


def test_qwen3tts_12hz_geometry() -> None:
    assert QWEN3TTS_12HZ.kind == "discrete"
    assert QWEN3TTS_12HZ.sample_rate == 24_000
    assert QWEN3TTS_12HZ.samples_per_frame == 1920
    assert QWEN3TTS_12HZ.frame_rate == pytest.approx(12.5)
    assert QWEN3TTS_12HZ.n_codebooks == 16
    assert QWEN3TTS_12HZ.codebook_size == 2048
    assert QWEN3TTS_12HZ.dim == 16
    assert QWEN3TTS_12HZ.code_dtype == "torch.int64"
    assert QWEN3_TTS_TOKENIZER_ID == "Qwen/Qwen3-TTS-Tokenizer-12Hz"


def test_qwen3tts_frame_helpers() -> None:
    # 2.0 s → exactly 25 frames at 12.5 Hz
    n = QWEN3TTS_12HZ.n_frames_from_samples(24_000 * 2)
    assert n == 25
    assert QWEN3TTS_12HZ.pad_samples_to_multiple(100) == 1920
    assert QWEN3TTS_12HZ.pad_samples_to_multiple(1920) == 1920


def test_sample_true_rejected_without_weights() -> None:
    """encode(sample=True) must fail closed without loading weights."""
    import numpy as np

    from minispeech.codec.qwen3_tts_wrapper import Qwen3TTSCodec

    obj = object.__new__(Qwen3TTSCodec)
    obj.spec = QWEN3TTS_12HZ
    with pytest.raises(ValueError, match="sample=False"):
        Qwen3TTSCodec.encode(obj, np.zeros(100, dtype=np.float32), 24_000, sample=True)


def test_check_codes_shape() -> None:
    import torch

    from minispeech.codec.qwen3_tts_wrapper import Qwen3TTSCodec

    obj = object.__new__(Qwen3TTSCodec)
    obj.spec = QWEN3TTS_12HZ
    bad = torch.zeros(10, 8, dtype=torch.long)
    with pytest.raises(ValueError, match="Expected codes"):
        obj._check_codes(bad)
    good = torch.zeros(10, 16, dtype=torch.long)
    obj._check_codes(good)

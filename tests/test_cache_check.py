"""Unit tests for cached-vs-full decode metrics (no codec weights required)."""

from __future__ import annotations

import torch

from minispeech.codec.cache_check import cached_vs_full_stats, evaluate_cache_gate
from minispeech.constants import SAMPLES_PER_FRAME


def _tone(n: int) -> torch.Tensor:
    t = torch.linspace(0, 1, n)
    return (0.1 * torch.sin(2 * 3.14159 * 40 * t)).view(1, 1, -1)


def test_identical_audio_passes_gate():
    audio = _tone(SAMPLES_PER_FRAME * 12)
    stats = cached_vs_full_stats(audio, audio, chunk_frames=4, edge=64)
    ok, reason = evaluate_cache_gate(stats, max_abs_limit=1e-3, boundary_ratio_max=10.0)
    assert ok, reason
    assert stats["max_abs"] == 0.0
    assert stats["boundary_ratio"] == 0.0


def test_global_bf16_like_noise_no_boundary_spike():
    """Uniform-ish noise should not look like a seam bug."""
    n = SAMPLES_PER_FRAME * 12
    full = _tone(n)
    # ~1.5e-2 peak noise everywhere (matches observed bf16 max-abs scale)
    g = torch.Generator().manual_seed(0)
    noise = (torch.rand(full.shape, generator=g) - 0.5) * 0.03
    cached = full + noise
    stats = cached_vs_full_stats(full, cached, chunk_frames=4, edge=64)
    # Soft bf16 limit 2e-2; hard fp32 limit 1e-3 would fail — expected.
    ok_soft, _ = evaluate_cache_gate(stats, max_abs_limit=2e-2, boundary_ratio_max=10.0)
    assert ok_soft
    assert stats["boundary_ratio"] < 10.0


def test_boundary_spike_fails_gate():
    n = SAMPLES_PER_FRAME * 12
    full = _tone(n)
    cached = full.clone()
    step = 4 * SAMPLES_PER_FRAME
    # Inject a large discontinuity only at chunk seams.
    for b in range(step, n, step):
        cached[..., b : b + 8] += 0.5
    stats = cached_vs_full_stats(full, cached, chunk_frames=4, edge=64)
    ok, reason = evaluate_cache_gate(stats, max_abs_limit=1.0, boundary_ratio_max=10.0)
    assert not ok
    assert "boundary_spike" in reason
    assert float(stats["boundary_max_abs"]) >= 0.5
    assert float(stats["boundary_ratio"]) > 10.0


def test_length_mismatch_fails_gate():
    full = _tone(SAMPLES_PER_FRAME * 8)
    cached = _tone(SAMPLES_PER_FRAME * 8 + SAMPLES_PER_FRAME + 10)
    # Align comparison on prefix; len_delta still recorded.
    stats = cached_vs_full_stats(full, cached, chunk_frames=4, edge=64)
    ok, reason = evaluate_cache_gate(stats, max_abs_limit=1.0, boundary_ratio_max=10.0)
    assert not ok
    assert "len_delta" in reason

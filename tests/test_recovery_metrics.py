"""Pure-math tests for Part 1 recovery helpers (no GPU / ASR)."""

from __future__ import annotations

import torch

from minispeech.eval.alignment_metrics import chance_diagonality, diagonality_vs_chance
from minispeech.eval.free_running import aggregate_stability, evaluate_free_running_latents
from minispeech.eval.head_diagnostics import (
    lag_correlations,
    r2_score,
    variance_ratio,
)
from minispeech.eval.latent_perturb import (
    add_white_noise,
    temporal_smooth,
    variance_collapse,
)
from minispeech.eval.trivial_baselines import (
    constant_mean_latents,
    copy_prev_frame_latents,
    predict_baseline,
    random_corpus_latents,
)
from minispeech.frontend.verbalize import verbalize_int, verbalize_text


def test_constant_mean_mae_near_sqrt_2_over_pi():
    torch.manual_seed(0)
    z = torch.randn(5000, 64)
    mean = torch.zeros(64)
    pred = constant_mean_latents(z.shape[0], mean)
    mae = (pred - z).abs().mean().item()
    assert abs(mae - 0.7979) < 0.05


def test_copy_prev_shifts():
    z = torch.arange(12, dtype=torch.float32).view(3, 4)
    out = copy_prev_frame_latents(z)
    assert torch.equal(out[0], z[0])
    assert torch.equal(out[1], z[0])
    assert torch.equal(out[2], z[1])


def test_trivial_baselines_pass_duration_gates_with_oracle_length():
    torch.manual_seed(1)
    gt_list = [torch.randn(20 + i, 64) for i in range(8)]
    pool = torch.cat(gt_list, dim=0)
    mean = torch.zeros(64)
    rows = []
    for z in gt_list:
        pred = predict_baseline("constant_mean", z, mean=mean, pool=pool, candidates=gt_list)
        rows.append(
            evaluate_free_running_latents(pred, z, stop_triggered=True, hit_max_frames=False)
        )
    stab = aggregate_stability(rows)
    assert stab["stop_success_rate"] == 1.0
    assert stab["runaway_rate"] == 0.0
    assert stab["duration_in_band_rate"] == 1.0


def test_random_corpus_shape():
    pool = torch.randn(100, 64)
    out = random_corpus_latents(17, pool)
    assert out.shape == (17, 64)


def test_r2_perfect_and_constant():
    torch.manual_seed(2)
    gt = torch.randn(4, 30, 64)
    mask = torch.ones(4, 30, dtype=torch.bool)
    assert r2_score(gt, gt, mask) > 0.99
    const = torch.zeros_like(gt)
    # For unit-ish random, constant predictor → R² near 0
    assert r2_score(const, gt, mask) < 0.05


def test_variance_ratio_collapse():
    torch.manual_seed(3)
    gt = torch.randn(2, 40, 64)
    mask = torch.ones(2, 40, dtype=torch.bool)
    pred = gt * 0.1
    vr = variance_ratio(pred, gt, mask)
    assert 0.005 < vr < 0.03


def test_lag_corr_copy_prev():
    torch.manual_seed(4)
    gt = torch.randn(1, 50, 16)
    pred = torch.zeros_like(gt)
    pred[:, 0] = gt[:, 0]
    pred[:, 1:] = gt[:, :-1]
    mask = torch.ones(1, 50, dtype=torch.bool)
    lags = lag_correlations(pred, gt, mask)
    assert lags["corr_gt_tm1"] > lags["corr_gt_t"]


def test_variance_collapse_and_smooth():
    z = torch.randn(1, 32, 8)
    assert torch.allclose(variance_collapse(z, 0.5), z * 0.5)
    sm = temporal_smooth(z[0], kernel_size=5)
    assert sm.shape == z[0].shape
    # Smoothing reduces frame-to-frame L1
    raw_diff = (z[0, 1:] - z[0, :-1]).abs().mean()
    sm_diff = (sm[1:] - sm[:-1]).abs().mean()
    assert sm_diff < raw_diff


def test_add_white_noise_zero():
    z = torch.randn(10, 4)
    assert torch.equal(add_white_noise(z, 0.0), z)


def test_chance_diagonality_positive():
    c = chance_diagonality(40, 20)
    assert 0.0 < c < 1.0
    attn = torch.rand(1, 40, 20)
    attn = attn / attn.sum(dim=-1, keepdim=True)
    stats = diagonality_vs_chance(attn)
    assert stats["n"] == 1
    assert "vs_chance" in stats["rows"][0]


def test_verbalize_numbers():
    assert "двадцать" in verbalize_int(21)
    assert "сто" in verbalize_int(100)
    t = verbalize_text("В 2020 г. было 15 человек")
    assert "год" in t.lower() or "две" in t or "пятнадцать" in t
    assert "2020" not in t or "два" in t  # year verbalized

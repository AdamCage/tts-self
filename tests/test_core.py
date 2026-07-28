"""Unit tests that do not require GPU models."""

from __future__ import annotations

import torch

from minispeech.constants import LATENT_DIM, n_frames_from_samples
from minispeech.data.alignment import monotonic_gaussian_prior, validate_token_projection
from minispeech.data.dataset import collate_minispeech
from minispeech.frontend.stress import stress_agreement, strip_stress, to_canonical_stress
from minispeech.models.minispeech import MiniSpeech
from minispeech.train.losses import stop_bce, total_regression_loss


def test_n_frames():
    assert n_frames_from_samples(3200) == 1
    assert n_frames_from_samples(3201) == 2


def test_stress_canonical():
    s = to_canonical_stress("+Привет")
    assert "\u0301" in s or "Привет" in strip_stress(s)
    assert stress_agreement("замо́к", "замо́к") == 1.0


def test_collate_shift():
    batch = [
        {
            "utt_id": "a",
            "z_target": torch.randn(5, 64),
            "text_hidden": torch.randn(3, 4, 1024),
            "n_frames": 5,
            "text_len": 4,
        },
        {
            "utt_id": "b",
            "z_target": torch.randn(3, 64),
            "text_hidden": torch.randn(3, 2, 1024),
            "n_frames": 3,
            "text_len": 2,
        },
    ]
    out = collate_minispeech(batch)
    assert out["z_input"].shape == (2, 5, 64)
    assert torch.allclose(out["z_input"][0, 1:5], out["z_target"][0, 0:4])
    assert out["stop_target"][0, 4] == 1.0
    assert out["stop_target"][0, :4].sum() == 0


def test_model_forward_regression():
    m = MiniSpeech(mode="regression")
    B, T, L = 2, 10, 6
    out = m(
        torch.zeros(B, T, LATENT_DIM),
        torch.randn(B, 3, L, 1024),
        frame_mask=torch.ones(B, T, dtype=torch.bool),
        text_mask=torch.ones(B, L, dtype=torch.bool),
        z_target=torch.randn(B, T, LATENT_DIM),
    )
    assert out.z_pred.shape == (B, T, LATENT_DIM)
    losses = total_regression_loss(
        out.z_pred,
        torch.randn(B, T, LATENT_DIM),
        out.stop_logits,
        torch.zeros(B, T),
        out.dur_pred,
        torch.tensor([10.0, 10.0]),
        torch.ones(B, T, dtype=torch.bool),
    )
    assert torch.isfinite(losses["total"])


def test_model_forward_flow():
    m = MiniSpeech(mode="flow")
    B, T, L = 2, 8, 5
    z = torch.randn(B, T, LATENT_DIM)
    out = m(
        torch.zeros(B, T, LATENT_DIM),
        torch.randn(B, 3, L, 1024),
        frame_mask=torch.ones(B, T, dtype=torch.bool),
        text_mask=torch.ones(B, L, dtype=torch.bool),
        z_target=z,
    )
    assert out.extras["v_pred"].shape == z.shape


def test_stop_pos_weight():
    logits = torch.zeros(2, 10)
    target = torch.zeros(2, 10)
    target[:, -1] = 1
    mask = torch.ones(2, 10, dtype=torch.bool)
    loss = stop_bce(logits, target, mask, positive_weight=10.0)
    assert torch.isfinite(loss)


def test_gaussian_prior():
    p = monotonic_gaussian_prior(10, 8)
    assert p.shape == (10, 8)
    assert torch.allclose(p.sum(-1), torch.ones(10), atol=1e-5)


def test_validate_projection():
    assert validate_token_projection(0.05)
    assert not validate_token_projection(0.2)

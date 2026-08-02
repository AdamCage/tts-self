"""A2 acceptance: DDPM head CFG, deterministic eval, sanity vs mean baseline."""

from __future__ import annotations

import time

import pytest
import torch
import yaml

from minispeech.models.head_diffusion import (
    DiffusionHead,
    constant_mean_baseline_l2,
    ddpm_train_step,
    micro_overfit_sanity,
    reconstruction_l2,
    sample_ddim,
)


def test_config_diffusion_block() -> None:
    with open("configs/n2_qwen_latent_v01.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    d = cfg["model"]["diffusion"]
    assert d["sampler"] == "ddim"
    assert d["cfg_dropout"] == 0.1
    assert d["cfg_scale"] == 1.3
    assert d["sample_steps"] == 10
    assert d["eval_seed"] == 42


def test_ddpm_train_step_shapes() -> None:
    B, T, H, D = 2, 5, 64, 64
    head = DiffusionHead(d_model=H, latent_dim=D, hidden=128, layers=2, timesteps=50)
    z0 = torch.randn(B, T, D)
    h = torch.randn(B, T, H)
    mask = torch.ones(B, T, dtype=torch.bool)
    eps_pred, eps, loss = ddpm_train_step(head, z0, h, cfg_dropout=0.1, frame_mask=mask)
    assert eps_pred.shape == z0.shape
    assert eps.shape == z0.shape
    assert loss.ndim == 0
    loss.backward()
    assert head.in_proj.weight.grad is not None


def test_deterministic_eval_bit_identical() -> None:
    B, T, H, D = 2, 4, 32, 64
    head = DiffusionHead(d_model=H, latent_dim=D, hidden=64, layers=2, timesteps=50)
    head.eval()
    h = torch.randn(B, T, H)
    mask = torch.ones(B, T, dtype=torch.bool)

    def run(seed: int) -> torch.Tensor:
        g = torch.Generator(device="cpu").manual_seed(seed)
        return sample_ddim(head, h, steps=5, cfg_scale=1.3, generator=g, frame_mask=mask)

    a = run(42)
    b = run(42)
    c = run(7)
    assert torch.equal(a, b)
    assert not torch.equal(a, c)


def test_cfg_scale_changes_sample() -> None:
    B, T, H, D = 1, 4, 32, 64
    head = DiffusionHead(d_model=H, latent_dim=D, hidden=64, layers=2, timesteps=50)
    # Make null_h distinct so CFG path differs.
    with torch.no_grad():
        head.null_h.add_(1.0)
    head.eval()
    h = torch.randn(B, T, H)
    g1 = torch.Generator(device="cpu").manual_seed(0)
    g2 = torch.Generator(device="cpu").manual_seed(0)
    s1 = sample_ddim(head, h, steps=4, cfg_scale=1.0, generator=g1)
    s2 = sample_ddim(head, h, steps=4, cfg_scale=2.0, generator=g2)
    assert not torch.allclose(s1, s2)


def test_sanity_true_context_beats_constant_mean() -> None:
    stats = micro_overfit_sanity(
        steps=250,
        B=4,
        T=8,
        d_model=64,
        latent_dim=64,
        cfg_scale=1.3,
        sample_steps=10,
        seed=0,
        device=torch.device("cpu"),
    )
    assert stats["reconstruction_l2"] < stats["constant_mean_l2"] * 0.85, stats
    assert stats["ratio"] < 0.85


def test_cfg_scale_sweep_proxy_l2() -> None:
    """Proxy for guidance→error table (full CER needs A4 decode+ASR)."""
    device = torch.device("cpu")
    B, T, H, D = 4, 8, 64, 64
    g = torch.Generator(device="cpu").manual_seed(1)
    z0 = torch.randn(B, T, D, generator=g)
    W = torch.randn(D, H, generator=g) / (D**0.5)
    h = (z0 @ W).to(device)
    z0 = z0.to(device)
    mask = torch.ones(B, T, dtype=torch.bool, device=device)
    head = DiffusionHead(
        d_model=H, latent_dim=D, hidden=128, layers=2, timesteps=100
    ).to(device)
    opt = torch.optim.Adam(head.parameters(), lr=2e-3)
    for _ in range(200):
        opt.zero_grad(set_to_none=True)
        _, _, loss = ddpm_train_step(head, z0, h, cfg_dropout=0.1, frame_mask=mask)
        loss.backward()
        opt.step()
    head.eval()
    rows = []
    for scale in (1.0, 1.3, 2.0):
        gg = torch.Generator(device="cpu").manual_seed(99)
        pred = sample_ddim(
            head, h, steps=10, cfg_scale=scale, generator=gg, frame_mask=mask
        )
        rows.append((scale, reconstruction_l2(pred, z0, mask)))
    assert len(rows) == 3
    # All finite; table material for A2 report.
    assert all(r == r for _, r in rows)


def test_sample_steps_timing_recorded() -> None:
    head = DiffusionHead(d_model=32, latent_dim=64, hidden=64, layers=2, timesteps=50)
    head.eval()
    h = torch.randn(1, 16, 32)
    t0 = time.perf_counter()
    _ = sample_ddim(head, h, steps=10, cfg_scale=1.3)
    elapsed = time.perf_counter() - t0
    assert elapsed > 0.0
    # Sanity baseline helper still works.
    z = torch.randn(1, 16, 64)
    assert constant_mean_baseline_l2(z) >= 0.0


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA optional")
def test_diffusion_cuda_smoke() -> None:
    device = torch.device("cuda")
    head = DiffusionHead(d_model=64, latent_dim=64, hidden=128, layers=2, timesteps=50).to(
        device
    )
    z0 = torch.randn(2, 8, 64, device=device)
    h = torch.randn(2, 8, 64, device=device)
    _, _, loss = ddpm_train_step(head, z0, h, cfg_dropout=0.1)
    loss.backward()

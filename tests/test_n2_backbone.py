"""A1 acceptance: Qwen latent backbone shapes, freeze policy, causality."""

from __future__ import annotations

import pytest
import torch
import yaml

from minispeech.constants import QWEN_HIDDEN_SIZE, get_target
from minispeech.models.backbone_lm import (
    QwenLatentBackbone,
    make_tiny_backbone_for_tests,
)
from minispeech.models.latent_io import build_modality_masks, shift_latents_for_ar


def test_target_geometry_from_registry() -> None:
    spec = get_target("vibevoice64")
    assert spec.dim == 64
    assert spec.frame_rate == 7.5
    assert spec.sample_rate == 24_000
    # Config must not invent parallel dims — smoke-read YAML.
    with open("configs/n2_qwen_latent_v01.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    assert cfg["target"] == "vibevoice64"
    assert cfg["qwen"]["hidden_size"] == QWEN_HIDDEN_SIZE
    assert cfg["model"]["backbone"]["freeze"] is True


def test_modality_masks_shapes() -> None:
    m = build_modality_masks([3, 5], [4, 2])
    assert m.text_mask.shape == (2, 5)
    assert m.frame_mask.shape == (2, 4)
    assert m.seq_mask.shape == (2, 9)
    assert m.is_acoustic[0, 5:9].all()
    assert not m.is_acoustic[0, :5].any()
    assert m.seq_mask[1, :5].all()
    assert m.seq_mask[1, 5:7].all()
    assert not m.seq_mask[1, 7:].any()


def test_forward_shapes_batch2() -> None:
    H = 64
    model = make_tiny_backbone_for_tests(hidden_size=H, freeze_backbone=True)
    B, L, T = 2, 5, 7
    input_ids = torch.randint(1, 50, (B, L))
    text_mask = torch.ones(B, L, dtype=torch.bool)
    text_mask[1, 3:] = False
    z_target = torch.randn(B, T, 64)
    frame_mask = torch.ones(B, T, dtype=torch.bool)
    frame_mask[1, 5:] = False
    z_input = shift_latents_for_ar(z_target, frame_mask, bos=model.bos_latent.detach())

    out = model(input_ids, z_input, text_mask=text_mask, frame_mask=frame_mask)
    assert out.acoustic_hidden.shape == (B, T, H)
    assert out.full_hidden.shape == (B, L + T, H)
    assert out.text_mask.shape == (B, L)
    assert out.frame_mask.shape == (B, T)
    assert out.is_acoustic.shape == (B, L + T)


def test_backbone_trainable_zero_projections_nonzero() -> None:
    model = make_tiny_backbone_for_tests(hidden_size=64, freeze_backbone=True)
    assert model.backbone_trainable_parameter_count() == 0
    n_train = model.trainable_parameter_count()
    assert n_train > 0
    # Gradients only in allowed params.
    B, L, T = 2, 4, 5
    input_ids = torch.randint(1, 40, (B, L))
    text_mask = torch.ones(B, L, dtype=torch.bool)
    z_input = torch.randn(B, T, 64, requires_grad=False)
    frame_mask = torch.ones(B, T, dtype=torch.bool)
    out = model(input_ids, z_input, text_mask=text_mask, frame_mask=frame_mask)
    loss = out.acoustic_hidden.float().pow(2).mean()
    loss.backward()

    for name, p in model.backbone.named_parameters():
        assert p.grad is None, f"backbone param {name} unexpectedly has grad"
        assert not p.requires_grad
    assert model.latent_in.net.weight.grad is not None
    assert model.bos_latent.grad is not None
    if model.acoustic_marker is not None:
        assert model.acoustic_marker.marker.grad is not None


def test_causality_latent_t_does_not_affect_earlier() -> None:
    model = make_tiny_backbone_for_tests(hidden_size=64, freeze_backbone=True)
    model.eval()
    B, L, T = 1, 3, 6
    input_ids = torch.randint(1, 40, (B, L))
    text_mask = torch.ones(B, L, dtype=torch.bool)
    z0 = torch.randn(B, T, 64)
    frame_mask = torch.ones(B, T, dtype=torch.bool)
    z_input = shift_latents_for_ar(z0, frame_mask, bos=model.bos_latent.detach())

    with torch.no_grad():
        base = model(input_ids, z_input, text_mask=text_mask, frame_mask=frame_mask)

    t_change = 4
    z_alt = z_input.clone()
    z_alt[:, t_change] = z_alt[:, t_change] + 3.0
    with torch.no_grad():
        alt = model(input_ids, z_alt, text_mask=text_mask, frame_mask=frame_mask)

    # Outputs at acoustic positions < t_change must match.
    delta_before = (base.acoustic_hidden[:, :t_change] - alt.acoustic_hidden[:, :t_change]).abs().max()
    delta_at = (base.acoustic_hidden[:, t_change:] - alt.acoustic_hidden[:, t_change:]).abs().max()
    assert float(delta_before) < 1e-5, f"causality broken: maxΔ before t={delta_before}"
    assert float(delta_at) > 1e-5, "changing z_t should affect outputs at/after t"


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for VRAM smoke")
def test_vram_smoke_batch2() -> None:
    """Optional GPU smoke: real or tiny stack peak ≤ 14.5 GB."""
    device = torch.device("cuda")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    # Prefer tiny on CI; real Qwen load is heavy — try config path, fall back.
    model = make_tiny_backbone_for_tests(hidden_size=64, freeze_backbone=True).to(device)
    B, L, T = 2, 64, 64  # ~8.5 s audio @ 7.5 Hz
    input_ids = torch.randint(1, 50, (B, L), device=device)
    text_mask = torch.ones(B, L, dtype=torch.bool, device=device)
    z_input = torch.randn(B, T, 64, device=device)
    frame_mask = torch.ones(B, T, dtype=torch.bool, device=device)
    out = model(input_ids, z_input, text_mask=text_mask, frame_mask=frame_mask)
    loss = out.acoustic_hidden.float().pow(2).mean()
    loss.backward()
    peak_gb = torch.cuda.max_memory_allocated() / 1e9
    assert peak_gb <= 14.5, f"peak VRAM {peak_gb:.2f} GB > 14.5"


def test_qwen_latent_backbone_rejects_mel_target() -> None:
    with pytest.raises(ValueError, match="latent target"):
        QwenLatentBackbone(target="mel80", backbone=None, hidden_size=64)

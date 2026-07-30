"""Unit / smoke tests for Track A CFM mel decoder (CPU, tiny tensors)."""

from __future__ import annotations

import torch

from minispeech.constants import MEL80
from minispeech.frontend.phone_vocab import PhoneVocab
from minispeech.models.mel_cfm import MelCFM, mel_flow_mse
from minispeech.models.mel_baseline import phone_duration_loss


def test_mel_cfm_train_step_backward_cpu():
    vocab = PhoneVocab()
    vocab.add_phones(["a", "b", "v", "g"])
    model = MelCFM(
        vocab_size=len(vocab),
        d_model=32,
        n_mels=MEL80.dim,
        enc_layers=1,
        n_heads=2,
        ffn_dim=64,
        dropout=0.0,
        upsample="hard",
        pad_id=vocab.pad_id,
        flow_hidden=64,
        flow_layers=2,
        time_embed_dim=32,
        cfg_dropout=0.0,
        time_sampling="uniform",
    )
    phone_ids = torch.tensor([[vocab.stoi["a"], vocab.stoi["b"], vocab.stoi["v"], 0]])
    phone_mask = torch.tensor([[True, True, True, False]])
    durations = torch.tensor([[3.0, 2.0, 4.0, 0.0]])
    T = 9
    mel_target = torch.randn(1, T, MEL80.dim)
    out = model(phone_ids, mel_target=mel_target, phone_mask=phone_mask, durations=durations)
    assert out.v_pred is not None and out.v_target is not None
    assert out.v_pred.shape == (1, T, MEL80.dim)
    loss = mel_flow_mse(out.v_pred, out.v_target, out.frame_mask)
    if out.dur_pred is not None:
        loss = loss + phone_duration_loss(out.dur_pred, durations, phone_mask)
    loss.backward()
    assert torch.isfinite(loss)


def test_mel_cfm_sample_shapes():
    model = MelCFM(
        vocab_size=32,
        d_model=32,
        n_mels=80,
        enc_layers=1,
        n_heads=2,
        ffn_dim=64,
        dropout=0.0,
        flow_hidden=64,
        flow_layers=1,
        time_embed_dim=32,
        cfg_dropout=0.0,
    )
    ids = torch.randint(2, 30, (2, 4))
    mask = torch.ones(2, 4, dtype=torch.bool)
    durs = torch.ones(2, 4) * 3
    out = model.sample(ids, phone_mask=mask, durations=durs, steps=4, method="euler", cfg_scale=1.0)
    assert out.mel_pred is not None
    assert out.mel_pred.shape == (2, 12, 80)
    assert int(out.frame_mask[0].sum()) == 12


def test_mel_cfm_loss_decreases_on_tiny_overfit():
    """Smoke: a few Adam steps on one batch should reduce flow MSE."""
    torch.manual_seed(0)
    model = MelCFM(
        vocab_size=16,
        d_model=32,
        n_mels=16,
        enc_layers=1,
        n_heads=2,
        ffn_dim=64,
        dropout=0.0,
        flow_hidden=32,
        flow_layers=1,
        time_embed_dim=16,
        cfg_dropout=0.0,
        time_sampling="uniform",
    )
    phone_ids = torch.tensor([[2, 3, 4, 0]])
    phone_mask = torch.tensor([[True, True, True, False]])
    durations = torch.tensor([[2.0, 2.0, 2.0, 0.0]])
    mel_target = torch.randn(1, 6, 16)
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    losses = []
    for _ in range(30):
        opt.zero_grad(set_to_none=True)
        out = model(
            phone_ids, mel_target=mel_target, phone_mask=phone_mask, durations=durations
        )
        assert out.v_pred is not None and out.v_target is not None
        loss = mel_flow_mse(out.v_pred, out.v_target, out.frame_mask)
        loss.backward()
        opt.step()
        losses.append(float(loss.detach()))
    assert losses[-1] < losses[0]


def test_mel_cfm_backbone_warmstart_from_regression_keys():
    from minispeech.models.mel_baseline import MelBaseline

    bb = MelBaseline(vocab_size=32, d_model=32, n_mels=80, enc_layers=1, dec_layers=1, n_heads=2, ffn_dim=64)
    cfm = MelCFM(
        vocab_size=32,
        d_model=32,
        n_mels=80,
        enc_layers=1,
        n_heads=2,
        ffn_dim=64,
        flow_hidden=32,
        flow_layers=1,
        time_embed_dim=16,
    )
    missing, unexpected = cfm.load_backbone_from_regression(bb.state_dict(), strict=False)
    # dur_head / emb / enc should load; decoder leftovers may be unexpected on source side only
    assert len(missing) < len(cfm.backbone.state_dict())
    # Embedded weights match
    assert torch.allclose(cfm.backbone.emb.weight, bb.emb.weight)

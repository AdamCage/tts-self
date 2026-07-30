"""Unit / smoke tests for Track A mel baseline (CPU, tiny random tensors)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from minispeech.constants import MEL80
from minispeech.data.mel import mel_backend_name, mel_n_frames, waveform_to_logmel
from minispeech.data.mel_dataset import collate_mel
from minispeech.frontend.phone_vocab import PhoneVocab
from minispeech.infer.vocoder import griffin_lim, mel_to_wav, save_mel, vocoder_status
from minispeech.models.mel_baseline import (
    MelBaseline,
    hard_upsample,
    mel_l1_loss,
    phone_duration_loss,
    soft_upsample,
)
from minispeech.train.losses import token_duration_loss, total_regression_loss
from minispeech.train.mel_trainer import cer_selection_key
from minispeech.train.schedule import lr_multiplier, set_optimizer_lr
from minispeech.train.trainer import cer_aware_selection_key
from minispeech.utils import git_sha, write_run_manifest


def test_hard_and_soft_upsample_shapes():
    B, P, D, T = 2, 4, 8, 10
    phone_h = torch.randn(B, P, D)
    durs = torch.tensor([[2, 3, 1, 0], [1, 1, 1, 1]], dtype=torch.float)
    frames, mask = hard_upsample(phone_h, durs)
    assert frames.shape[0] == B
    assert frames.shape[-1] == D
    assert mask.shape[0] == B
    assert int(mask[0].sum()) == 6
    W = torch.zeros(B, P, T)
    W[0, 0, :3] = 1.0 / 3
    W[0, 1, 3:6] = 1.0 / 3
    frames_s, mask_s = soft_upsample(phone_h, W)
    assert frames_s.shape == (B, T, D)
    assert mask_s.shape == (B, T)


def test_mel_baseline_forward_backward_cpu():
    vocab = PhoneVocab()
    vocab.add_phones(["a", "b", "v", "g"])
    model = MelBaseline(
        vocab_size=len(vocab),
        d_model=32,
        n_mels=MEL80.dim,
        enc_layers=1,
        dec_layers=2,
        n_heads=2,
        ffn_dim=64,
        dropout=0.0,
        decoder="conv",
        upsample="hard",
        pad_id=vocab.pad_id,
    )
    phone_ids = torch.tensor([[vocab.stoi["a"], vocab.stoi["b"], vocab.stoi["v"], 0]])
    phone_mask = torch.tensor([[True, True, True, False]])
    durations = torch.tensor([[3.0, 2.0, 4.0, 0.0]])
    T = 9
    mel_target = torch.randn(1, T, MEL80.dim)
    out = model(phone_ids, phone_mask=phone_mask, durations=durations)
    assert out.mel_pred.shape[-1] == MEL80.dim
    loss = mel_l1_loss(out.mel_pred, mel_target, out.frame_mask)
    if out.dur_pred is not None:
        loss = loss + phone_duration_loss(out.dur_pred, durations, phone_mask)
    loss.backward()
    assert torch.isfinite(loss)


def test_mel_baseline_transformer_decoder():
    model = MelBaseline(
        vocab_size=32,
        d_model=32,
        n_mels=80,
        enc_layers=1,
        dec_layers=1,
        n_heads=2,
        ffn_dim=64,
        decoder="transformer",
        upsample="hard",
    )
    ids = torch.randint(2, 30, (2, 5))
    mask = torch.ones(2, 5, dtype=torch.bool)
    durs = torch.ones(2, 5) * 2
    out = model(ids, phone_mask=mask, durations=durs)
    assert out.mel_pred.shape[1] == 10


def test_collate_mel_and_sample_weight():
    items = [
        {
            "utt_id": "u0",
            "phone_ids": torch.tensor([2, 3]),
            "durations": torch.tensor([2.0, 3.0]),
            "mel_target": torch.randn(5, 80),
            "n_phones": 2,
            "n_frames": 5,
            "sample_weight": 0.5,
        },
        {
            "utt_id": "u1",
            "phone_ids": torch.tensor([4]),
            "durations": torch.tensor([4.0]),
            "mel_target": torch.randn(4, 80),
            "n_phones": 1,
            "n_frames": 4,
            "sample_weight": 1.0,
        },
    ]
    batch = collate_mel(items, pad_id=0)
    assert batch["phone_ids"].shape == (2, 2)
    assert batch["mel_target"].shape == (2, 5, 80)
    assert batch["sample_weight"].tolist() == [0.5, 1.0]


def test_lr_warmup_and_cosine():
    assert lr_multiplier(0, 100, warmup_steps=10) == pytest.approx(0.1)
    assert lr_multiplier(9, 100, warmup_steps=10) == pytest.approx(1.0)
    mid = lr_multiplier(55, 100, warmup_steps=10, schedule="cosine")
    end = lr_multiplier(99, 100, warmup_steps=10, schedule="cosine", min_lr_ratio=0.05)
    assert mid > end
    assert end >= 0.05
    opt = torch.optim.SGD([torch.zeros(1, requires_grad=True)], lr=1.0)
    set_optimizer_lr(opt, 2e-4, 0.5)
    assert opt.param_groups[0]["lr"] == pytest.approx(1e-4)


def test_token_duration_loss_hook_and_sample_weight():
    B, T, L, D = 2, 5, 3, 4
    z = torch.randn(B, T, D)
    stop = torch.zeros(B, T)
    mask = torch.ones(B, T, dtype=torch.bool)
    text_mask = torch.tensor([[True, True, False], [True, True, True]])
    tok_pred = torch.rand(B, L) + 1
    tok_tgt = torch.rand(B, L) + 1
    sw = torch.tensor([1.0, 0.0])
    losses = total_regression_loss(
        z,
        z,
        stop,
        stop,
        torch.ones(B) * T,
        torch.ones(B) * T,
        mask,
        w_l1=1.0,
        w_cos=0.0,
        w_stop=0.0,
        w_dur=0.0,
        sample_weight=sw,
        tok_dur_pred=tok_pred,
        tok_dur_target=tok_tgt,
        text_mask=text_mask,
        w_tok_dur=0.5,
    )
    assert "tok_duration" in losses
    assert torch.isfinite(losses["total"])
    td = token_duration_loss(tok_pred, tok_tgt, text_mask, sample_weight=sw)
    assert torch.isfinite(td)


def test_cer_selection_prefers_lower_cer():
    k_hi = cer_selection_key({"cer": 0.9, "mel_l1": 0.1})
    k_lo = cer_selection_key({"cer": 0.2, "mel_l1": 0.5})
    assert k_lo > k_hi
    k_fr = cer_aware_selection_key(cer=None, fr_stability={"duration_in_band_rate": 0.9})
    k_cer = cer_aware_selection_key(cer=0.1, fr_stability={"duration_in_band_rate": 0.5})
    assert isinstance(k_fr, tuple) and isinstance(k_cer, tuple)


def test_git_sha_and_run_manifest(tmp_path: Path):
    sha = git_sha()
    assert isinstance(sha, str) and len(sha) >= 7
    # Should not be unknown inside this repo
    assert sha != "unknown"
    p = write_run_manifest(tmp_path / "run", {"train": {"lr": 1e-4}})
    data = p.read_text(encoding="utf-8")
    assert "git_sha" in data
    assert sha[:7] in data or sha in data


def test_vocoder_save_mel_and_status(tmp_path: Path):
    mel = torch.randn(20, 80)
    st = vocoder_status()
    assert "fallback" in st
    res = mel_to_wav(mel, prefer="save_mel", save_mel_path=tmp_path / "m")
    assert res.backend == "save_mel"
    assert res.mel_path is not None
    assert Path(res.mel_path).is_file()
    p2 = save_mel(mel.numpy(), tmp_path / "m2")
    assert p2.is_file()


def test_griffin_lim_optional():
    # May use librosa/torchaudio; skip soft if neither can invert
    mel = torch.randn(16, 80) * 0.1 - 3.0
    try:
        wav = griffin_lim(mel, n_iter=4)
    except Exception as exc:
        pytest.skip(f"griffin-lim unavailable: {exc}")
    assert isinstance(wav, np.ndarray)
    assert wav.ndim == 1
    assert wav.shape[0] > 0


def test_waveform_to_logmel_optional():
    if mel_backend_name() == "none":
        pytest.skip("no mel backend")
    sr = MEL80.sample_rate
    wav = np.random.randn(sr // 5).astype(np.float32) * 0.01
    mel = waveform_to_logmel(wav, target="mel80", sample_rate=sr)
    assert mel.ndim == 2 and mel.shape[1] == 80
    assert mel_n_frames(len(wav)) >= 1


def test_phone_vocab_roundtrip(tmp_path: Path):
    v = PhoneVocab.build([["a", "b"], ["b", "v"]])
    ids = v.encode(["a", "z"])  # z → unk
    assert ids[1] == v.unk_id
    path = v.save(tmp_path / "vocab.json")
    v2 = PhoneVocab.load(path)
    assert v2.encode(["a", "b"]) == v.encode(["a", "b"])

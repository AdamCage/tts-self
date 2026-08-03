"""Unit tests for discrete Speech LM (CPU). GPU-marked tests skip without CUDA."""

from __future__ import annotations

import pytest
import torch

from minispeech.data.collate_codec_seq import collate_codec_seq as collate_landed
from minispeech.data.codec_seq_protocol import collate_codec_seq
from minispeech.models.discrete_speech_lm import (
    DiscreteSpeechLM,
    DiscreteSpeechLMConfig,
)
from minispeech.models.multi_codebook_head import MultiCodebookHead
from minispeech.models.text_init import TextInitReport, _is_scratch_key, text_init_manifest_extra


requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA required"
)


def _tiny_batch(B: int = 2, L: int = 5, T: int = 7, n_cb: int = 16, V: int = 32):
    items = []
    for i in range(B):
        t = T - i  # variable lengths
        items.append(
            {
                "utt_id": f"u{i}",
                "input_ids": torch.randint(1, 64, (L - i,)),
                "codes": torch.randint(0, V, (t, n_cb)),
                "speaker_id": i % 3,
            }
        )
    return collate_codec_seq(items, pad_token_id=0)


def test_shapes_forward_and_heads():
    cfg = DiscreteSpeechLMConfig.tiny()
    m = DiscreteSpeechLM(cfg)
    batch = _tiny_batch(V=cfg.codebook_size)
    out = m(
        batch["input_ids"],
        batch["codes"],
        text_mask=batch["text_mask"],
        frame_mask=batch["frame_mask"],
        speaker_id=batch["speaker_id"],
    )
    B, T, _ = batch["codes"].shape
    assert out.acoustic_hidden.shape == (B, T, cfg.d_model)
    assert out.heads.cb0_logits.shape == (B, T, cfg.codebook_size)
    assert out.heads.residual_logits.shape == (B, T, cfg.n_codebooks - 1, cfg.codebook_size)
    assert out.loss is not None
    assert torch.isfinite(out.loss["total"])


def test_ar_shift_c_t_to_c_t1():
    """Teacher-forced speech inputs must be [BOS, c0_0, ..., c0_{T-2}]."""
    cfg = DiscreteSpeechLMConfig.tiny()
    m = DiscreteSpeechLM(cfg)
    B, L, T = 1, 3, 5
    input_ids = torch.randint(1, 50, (B, L))
    codes = torch.randint(0, cfg.codebook_size, (B, T, cfg.n_codebooks))
    packed = m.build_inputs(input_ids, codes)
    # Reconstruct speech slice of embeds and compare to expected emb chain
    speech = packed["embeds"][:, L:, :]
    # pos0 = speech_start + speaker(0) + modality
    spk = m.speaker_emb(torch.zeros(B, dtype=torch.long))
    expect0 = m.speech_start + spk + m.modality_speech
    assert torch.allclose(speech[:, 0], expect0, atol=1e-5)
    for t in range(1, T):
        expect = m.speech_emb(codes[:, t - 1, 0]) + m.modality_speech
        assert torch.allclose(speech[:, t], expect, atol=1e-5)


def test_causality_no_future_leak():
    """Perturbing a future acoustic frame must not change earlier logits."""
    cfg = DiscreteSpeechLMConfig.tiny()
    m = DiscreteSpeechLM(cfg)
    m.eval()
    B, L, T = 1, 4, 6
    torch.manual_seed(0)
    input_ids = torch.randint(1, 40, (B, L))
    codes = torch.randint(0, cfg.codebook_size, (B, T, cfg.n_codebooks))
    with torch.no_grad():
        out_a = m(input_ids, codes, compute_loss=False)
        codes_b = codes.clone()
        codes_b[:, -1, :] = (codes_b[:, -1, :] + 3) % cfg.codebook_size
        out_b = m(input_ids, codes_b, compute_loss=False)
    # AR inputs never read codes[:, T-1], so CB0 logits are fully invariant.
    assert torch.allclose(out_a.heads.cb0_logits, out_b.heads.cb0_logits, atol=1e-5)
    # Residual MTP at the last frame uses teacher CB0 emb → may change there only.
    if out_a.heads.residual_logits.numel():
        assert torch.allclose(
            out_a.heads.residual_logits[:, :-1],
            out_b.heads.residual_logits[:, :-1],
            atol=1e-5,
        )


def test_padding_invariance():
    """Right-padding acoustic frames (same text L) must not change valid logits."""
    cfg = DiscreteSpeechLMConfig.tiny()
    m = DiscreteSpeechLM(cfg)
    m.eval()
    torch.manual_seed(1)
    ids = torch.randint(1, 40, (1, 4))
    codes = torch.randint(0, cfg.codebook_size, (1, 5, cfg.n_codebooks))
    with torch.no_grad():
        out = m(ids, codes, compute_loss=False)
        # Pad frames only — text length unchanged so RoPE acoustic offsets match.
        codes_p = torch.nn.functional.pad(codes, (0, 0, 0, 3), value=0)
        frame_mask = torch.zeros(1, codes_p.shape[1], dtype=torch.bool)
        frame_mask[:, : codes.shape[1]] = True
        out_p = m(ids, codes_p, frame_mask=frame_mask, compute_loss=False)
    assert torch.allclose(
        out.heads.cb0_logits, out_p.heads.cb0_logits[:, : codes.shape[1]], atol=1e-4
    )


def test_multi_codebook_head_loss_shapes():
    head = MultiCodebookHead(32, n_codebooks=16, codebook_size=20, mtp_hidden=32)
    B, T = 2, 5
    h = torch.randn(B, T, 32)
    cb0 = torch.randn(B, T, 32)
    codes = torch.randint(0, 20, (B, T, 16))
    mask = torch.ones(B, T, dtype=torch.bool)
    mask[1, -2:] = False
    out = head(h, cb0)
    losses = head.loss(out, codes, mask)
    assert set(losses) >= {"total", "ce_cb0", "ce_residual"}
    assert torch.isfinite(losses["total"])


def test_tiny_overfit_synthetic_codes():
    """Sanity: tiny model drives CE down on one synthetic batch (CPU)."""
    cfg = DiscreteSpeechLMConfig.tiny()
    m = DiscreteSpeechLM(cfg)
    batch = _tiny_batch(B=2, L=6, T=8, V=cfg.codebook_size)
    opt = torch.optim.AdamW(m.parameters(), lr=3e-3)
    m.train()
    losses = []
    for _ in range(80):
        opt.zero_grad(set_to_none=True)
        out = m(
            batch["input_ids"],
            batch["codes"],
            text_mask=batch["text_mask"],
            frame_mask=batch["frame_mask"],
            speaker_id=batch["speaker_id"],
        )
        assert out.loss is not None
        out.loss["total"].backward()
        opt.step()
        losses.append(float(out.loss["total"].detach()))
    assert losses[-1] < losses[0] * 0.7


def test_param_count_full_config_in_band():
    """Full ~300M config param count in [250M, 400M] (meta/device count)."""
    cfg = DiscreteSpeechLMConfig()  # defaults ≈300M
    # Prefer meta to avoid allocating ~1GB+; fall back to CPU.
    try:
        with torch.device("meta"):
            m = DiscreteSpeechLM(cfg)
        n = sum(p.numel() for p in m.parameters())
    except Exception:
        m = DiscreteSpeechLM(cfg)
        n = m.count_parameters()["total"]
    assert 250_000_000 <= n <= 400_000_000, f"param_count={n}"


def test_scratch_keys_never_text_init():
    assert _is_scratch_key("speech_emb.weight")
    assert _is_scratch_key("heads.semantic.proj.weight")
    assert _is_scratch_key("stop_head.0.weight")
    assert _is_scratch_key("modality_speech")
    assert not _is_scratch_key("text_emb.weight")
    assert not _is_scratch_key("blocks.0.attn.qkv.weight")
    rep = TextInitReport(source="x", n_mapped_params=10, n_total_src_params=100)
    extra = text_init_manifest_extra(rep)
    assert extra["text_init"]["enabled"] is True
    assert abs(extra["text_init"]["coverage"] - 0.1) < 1e-9


def test_russian_conditioning_off_by_default():
    cfg = DiscreteSpeechLMConfig.tiny()
    assert cfg.russian_conditioning is False
    m = DiscreteSpeechLM(cfg)
    assert m.ru_cond.enabled is False
    batch = _tiny_batch(V=cfg.codebook_size)
    out = m(
        batch["input_ids"],
        batch["codes"],
        text_mask=batch["text_mask"],
        frame_mask=batch["frame_mask"],
        speaker_id=batch["speaker_id"],
        phoneme_ids=torch.ones_like(batch["input_ids"]),
    )
    assert torch.isfinite(out.loss["total"])


def test_generate_codes_shape():
    cfg = DiscreteSpeechLMConfig.tiny()
    m = DiscreteSpeechLM(cfg)
    ids = torch.randint(1, 40, (1, 5))
    codes = m.generate(ids, max_frames=4, temperature=1.0, top_k=0, min_frames=1)
    assert codes.shape[0] == 1 and codes.shape[2] == cfg.n_codebooks
    assert 1 <= codes.shape[1] <= 4


def test_stop_target_last_frame():
    mask = torch.tensor([[True, True, True, False], [True, True, False, False]])
    tgt = DiscreteSpeechLM.stop_targets_from_mask(mask)
    assert tgt[0, 2] == 1.0 and tgt[0, :2].sum() == 0
    assert tgt[1, 1] == 1.0 and tgt[1, 0] == 0


def test_stop_loss_and_early_generate():
    """Stop head trains; generate can stop before max_frames via stop or repeat cap."""
    cfg = DiscreteSpeechLMConfig.tiny()
    m = DiscreteSpeechLM(cfg)
    batch = _tiny_batch(B=2, L=4, T=6, V=cfg.codebook_size)
    out = m(
        batch["input_ids"],
        batch["codes"],
        text_mask=batch["text_mask"],
        frame_mask=batch["frame_mask"],
        speaker_id=batch["speaker_id"],
        stop_weight=1.0,
        stop_pos_weight=5.0,
    )
    assert out.stop_logits is not None
    assert out.stop_logits.shape == batch["codes"].shape[:2]
    assert "stop" in out.loss and torch.isfinite(out.loss["stop"])

    # Force stop head to fire immediately → early exit after min_frames.
    with torch.no_grad():
        m.stop_head[-1].weight.zero_()
        m.stop_head[-1].bias.fill_(10.0)
    ids = torch.randint(1, 40, (1, 4))
    codes = m.generate(
        ids,
        max_frames=20,
        temperature=0.0,
        stop_threshold=0.5,
        stop_consecutive=1,
        min_frames=3,
        max_cb0_repeat=100,
        stop_overrun_bias=0.0,
        frames_per_text_token=0.01,  # earliest gate ≈ min_frames
        stop_earliest_frac=0.0,
    )
    assert codes.shape[1] == 3


@requires_cuda
def test_bf16_forward_cuda():
    cfg = DiscreteSpeechLMConfig.tiny()
    m = DiscreteSpeechLM(cfg).cuda()
    batch = _tiny_batch(V=cfg.codebook_size)
    batch = {k: v.cuda() if torch.is_tensor(v) else v for k, v in batch.items()}
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        out = m(
            batch["input_ids"],
            batch["codes"],
            text_mask=batch["text_mask"],
            frame_mask=batch["frame_mask"],
            speaker_id=batch["speaker_id"],
        )
    assert torch.isfinite(out.loss["total"].float())


def test_text_padding_rope_batch_solo_parity():
    """Short sample logits must match B=1 when batched with a longer text peer."""
    cfg = DiscreteSpeechLMConfig.tiny()
    m = DiscreteSpeechLM(cfg)
    m.eval()
    torch.manual_seed(7)
    n_cb = cfg.n_codebooks
    V = cfg.codebook_size

    short_ids = torch.randint(1, 40, (3,))
    long_ids = torch.randint(1, 40, (9,))
    short_codes = torch.randint(0, V, (5, n_cb))
    long_codes = torch.randint(0, V, (6, n_cb))
    spk = 2

    with torch.no_grad():
        solo = m(
            short_ids.unsqueeze(0),
            short_codes.unsqueeze(0),
            speaker_id=torch.tensor([spk]),
            compute_loss=False,
        )

        def _run_batch(order: list[str]):
            items = []
            for name in order:
                if name == "short":
                    items.append(
                        {
                            "utt_id": "short",
                            "input_ids": short_ids,
                            "codes": short_codes,
                            "speaker_id": spk,
                        }
                    )
                else:
                    items.append(
                        {
                            "utt_id": "long",
                            "input_ids": long_ids,
                            "codes": long_codes,
                            "speaker_id": 1,
                        }
                    )
            batch = collate_landed(items, pad_token_id=0)
            out = m(
                batch["input_ids"],
                batch["codes"],
                text_mask=batch["text_mask"],
                frame_mask=batch["frame_mask"],
                speaker_id=batch["speaker_id"],
                compute_loss=False,
            )
            idx = order.index("short")
            t = short_codes.shape[0]
            return (
                out.heads.cb0_logits[idx, :t],
                out.heads.residual_logits[idx, :t],
                out.stop_logits[idx, :t],
            )

        cb0_a, res_a, stop_a = _run_batch(["short", "long"])
        cb0_b, res_b, stop_b = _run_batch(["long", "short"])

    atol = 1e-5
    assert torch.allclose(solo.heads.cb0_logits[0], cb0_a, atol=atol)
    assert torch.allclose(solo.heads.cb0_logits[0], cb0_b, atol=atol)
    assert torch.allclose(solo.heads.residual_logits[0], res_a, atol=atol)
    assert torch.allclose(solo.heads.residual_logits[0], res_b, atol=atol)
    assert torch.allclose(solo.stop_logits[0], stop_a, atol=atol)
    assert torch.allclose(solo.stop_logits[0], stop_b, atol=atol)


def test_position_ids_speech_starts_at_own_text_len():
    cfg = DiscreteSpeechLMConfig.tiny()
    m = DiscreteSpeechLM(cfg)
    # Physical L=5 pad, short text_len=2 → speech frame0 at RoPE pos 2.
    input_ids = torch.tensor([[11, 12, 0, 0, 0], [21, 22, 23, 24, 25]])
    text_mask = torch.tensor(
        [[True, True, False, False, False], [True, True, True, True, True]]
    )
    codes = torch.randint(0, cfg.codebook_size, (2, 3, cfg.n_codebooks))
    frame_mask = torch.ones(2, 3, dtype=torch.bool)
    packed = m.build_inputs(
        input_ids, codes, text_mask=text_mask, frame_mask=frame_mask
    )
    pos = packed["position_ids"]
    assert pos[0, :2].tolist() == [0, 1]
    assert pos[0, 2:5].tolist() == [0, 0, 0]  # text pad dummy
    assert pos[0, 5:8].tolist() == [2, 3, 4]  # text_len + t
    assert pos[1, :5].tolist() == [0, 1, 2, 3, 4]
    assert pos[1, 5:8].tolist() == [5, 6, 7]


def test_no_attention_to_text_padding_keys():
    """Key padding must block text-pad positions (RoPE dummy must not leak)."""
    cfg = DiscreteSpeechLMConfig.tiny()
    m = DiscreteSpeechLM(cfg)
    m.eval()
    torch.manual_seed(3)
    ids = torch.tensor([[5, 6, 0, 0]])
    mask = torch.tensor([[True, True, False, False]])
    codes = torch.randint(0, cfg.codebook_size, (1, 4, cfg.n_codebooks))
    with torch.no_grad():
        out_a = m(ids, codes, text_mask=mask, compute_loss=False)
        ids_b = ids.clone()
        ids_b[0, 2:] = torch.randint(1, 40, (2,))
        out_b = m(ids_b, codes, text_mask=mask, compute_loss=False)
    assert torch.allclose(out_a.heads.cb0_logits, out_b.heads.cb0_logits, atol=1e-5)
    assert torch.allclose(out_a.stop_logits, out_b.stop_logits, atol=1e-5)

"""CPU contract tests for D1c's explicit semantic EOS path."""

from __future__ import annotations

import torch

from minispeech.data.collate_codec_seq import collate_codec_seq
from minispeech.models.discrete_speech_lm import (
    DiscreteGenerationOutput,
    DiscreteSpeechLM,
    DiscreteSpeechLMConfig,
)


def _model() -> DiscreteSpeechLM:
    cfg = DiscreteSpeechLMConfig.tiny()
    cfg.termination_mode = "semantic_eos_v1"
    cfg.eos_id = cfg.codebook_size
    cfg.codec_vocab_size = cfg.codebook_size
    cfg.semantic_vocab_size = cfg.codebook_size + 1
    return DiscreteSpeechLM(cfg)


def _batch(m: DiscreteSpeechLM) -> dict:
    return collate_codec_seq(
        [
            {
                "utt_id": "short",
                "input_ids": torch.tensor([3, 4]),
                "codes": torch.randint(0, m.codebook_size, (2, m.n_codebooks)),
                "speaker_id": 0,
            },
            {
                "utt_id": "long",
                "input_ids": torch.tensor([5, 6, 7]),
                "codes": torch.randint(0, m.codebook_size, (4, m.n_codebooks)),
                "speaker_id": 1,
            },
        ],
        pad_token_id=0,
        speech_eos_id=m.eos_id,
        termination_mode="semantic_eos_v1",
    )


def test_semantic_vocab_and_collate_own_eos_positions():
    m = _model()
    batch = _batch(m)
    assert m.speech_emb.num_embeddings == m.codebook_size
    assert m.heads.semantic.proj.out_features == m.codebook_size + 1
    assert m.heads.residuals[0].net[-1].out_features == m.codebook_size
    assert batch["eos_positions"].tolist() == [2, 4]
    assert batch["semantic_targets"][0, 2].item() == m.eos_id
    assert batch["semantic_targets"][1, 4].item() == m.eos_id
    assert batch["semantic_mask"].sum(dim=1).tolist() == [3, 5]
    assert torch.equal(batch["residual_mask"], batch["codec_frame_mask"])


def test_eos_slot_loss_has_no_residual_or_stop_head():
    m = _model()
    batch = _batch(m)
    out = m(
        batch["input_ids"], batch["codes"], text_mask=batch["text_mask"],
        frame_mask=batch["codec_frame_mask"], semantic_targets=batch["semantic_targets"],
        semantic_mask=batch["semantic_mask"], speaker_id=batch["speaker_id"],
    )
    assert m.stop_head is None
    assert out.stop_logits is None
    assert out.heads.cb0_logits.shape[1] == batch["codes"].shape[1] + 1
    assert out.heads.residual_logits.shape[1] == batch["codes"].shape[1]
    assert out.loss is not None
    assert {"ce_cb0_frame", "ce_residual", "ce_semantic_eos", "loss_total"} <= set(out.loss)


def test_eos_slot_consumes_last_real_cb0():
    m = _model()
    batch = _batch(m)
    packed = m.build_inputs(
        batch["input_ids"], batch["codes"], text_mask=batch["text_mask"],
        frame_mask=batch["codec_frame_mask"], semantic_mask=batch["semantic_mask"],
        speaker_id=batch["speaker_id"],
    )
    L = batch["input_ids"].shape[1]
    for row, length in enumerate(batch["eos_positions"].tolist()):
        expected = m.speech_emb(batch["codes"][row, length - 1, 0]) + m.modality_speech
        assert torch.allclose(packed["embeds"][row, L + length], expected)


def test_generation_returns_metadata_and_excludes_eos():
    m = _model()
    out = m.generate(
        torch.tensor([[3, 4, 5]]), max_frames=4, min_frames=4,
        max_cb0_repeat=100, temperature=0.0, top_k=0,
    )
    assert isinstance(out, DiscreteGenerationOutput)
    assert out.codes.shape[-1] == m.n_codebooks
    assert int(out.lengths[0]) == 4
    assert out.rows[0]["termination_reason"] == "safety_max_frames"
    assert int(out.codes.max()) < m.codebook_size


def test_legacy_mode_still_has_stop_head():
    legacy = DiscreteSpeechLM(DiscreteSpeechLMConfig.tiny())
    assert legacy.stop_head is not None
    assert legacy.heads.semantic.proj.out_features == legacy.codebook_size

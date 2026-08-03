"""Wave 1: codec-token packing, collate, shard roundtrip (no full encode)."""

from __future__ import annotations

import json
from pathlib import Path

import torch

from minispeech.constants import QWEN3TTS_12HZ
from minispeech.data.codec_tokens import (
    SPEECH_EOS_ID,
    CodecTokenExample,
    CodecTokenShardReader,
    CodecTokenShardWriter,
    RuTextFeatures,
    assert_pack_train_infer_equal,
    build_ru_text_features,
    pack_codec_sequence,
    pack_for_infer,
    pack_for_train,
    span_type_id,
    verify_trackn_no_leak,
)
from minispeech.data.collate_codec_seq import (
    assert_batch_pack_matches_items,
    collate_codec_seq,
    masked_ce_placeholder,
)
from minispeech.data.manifest import Utterance, write_manifest
from minispeech.eval.critical_spans import CriticalSpan
from minispeech.frontend.phone_vocab import PhoneVocab


def _feats(L: int, *, seed: int = 0) -> RuTextFeatures:
    g = torch.Generator().manual_seed(seed)
    return RuTextFeatures(
        stress=torch.randint(0, 2, (L,), generator=g),
        phoneme=torch.randint(0, 20, (L,), generator=g),
        span_type=torch.randint(0, 5, (L,), generator=g),
        word_bound=torch.randint(0, 2, (L,), generator=g),
        phrase_bound=torch.randint(0, 2, (L,), generator=g),
    )


def _example(utt: str, L: int, T: int, *, seed: int = 0) -> CodecTokenExample:
    g = torch.Generator().manual_seed(seed)
    C = int(QWEN3TTS_12HZ.n_codebooks or 16)
    return CodecTokenExample(
        utt_id=utt,
        codes=torch.randint(0, 2048, (T, C), generator=g),
        text_ids=torch.randint(1, 1000, (L,), generator=g),
        features=_feats(L, seed=seed + 1),
        corpus="ruslan",
        speaker_id="ruslan",
        license="CC BY-NC-SA 4.0",
        codec_id=QWEN3TTS_12HZ.name,
        codec_hash="deadbeef",
        dur_s=T / QWEN3TTS_12HZ.frame_rate,
    )


def test_pack_shapes_and_eos() -> None:
    ex = _example("a", L=4, T=6, seed=1)
    p = pack_for_train(ex)
    assert p.text_lens == 4
    assert p.acoustic_lens == 6
    assert p.seq_len == 4 + 6 + 1
    assert int(p.is_text.sum()) == 4
    assert int(p.is_acoustic.sum()) == 6
    assert int(p.is_eos.sum()) == 1
    assert bool(p.is_eos[-1])
    assert p.speech_eos_id == SPEECH_EOS_ID
    # AR shift
    assert torch.equal(p.codes_input[0], torch.zeros(16, dtype=torch.long))
    assert torch.equal(p.codes_input[1], p.codes_target[0])


def test_train_infer_pack_equality() -> None:
    ex = _example("eq", L=5, T=3, seed=2)
    assert_pack_train_infer_equal(ex)
    inf = pack_for_infer(ex.text_ids, ex.features, n_codebooks=16)
    assert inf.acoustic_lens == 0
    assert inf.seq_len == 5 + 0 + 1
    assert torch.equal(inf.stress, ex.features.stress)


def test_collate_shapes_and_order_invariance() -> None:
    a = _example("short", L=2, T=3, seed=10)
    b = _example("long", L=9, T=12, seed=11)

    def as_item(ex: CodecTokenExample) -> dict:
        return {
            "utt_id": ex.utt_id,
            "text_ids": ex.text_ids,
            "codes": ex.codes,
            "stress": ex.features.stress,
            "phoneme": ex.features.phoneme,
            "span_type": ex.features.span_type,
            "word_bound": ex.features.word_bound,
            "phrase_bound": ex.features.phrase_bound,
            "dur_s": ex.dur_s,
            "sample_weight": 1.0,
            "corpus": ex.corpus,
            # Collate requires registry-resolved ints (string hash fallback removed).
            "speaker_id": 1 if ex.utt_id == "short" else 2,
        }

    b1 = collate_codec_seq([as_item(a), as_item(b)])
    b2 = collate_codec_seq([as_item(b), as_item(a)])
    assert b1["codes_target"].shape == (2, 12, 16)
    assert b1["input_ids"].shape == (2, 9)
    assert int(b1["frame_mask"][0].sum()) == 3
    assert int(b1["is_eos"].sum()) == 2

    pred1 = torch.zeros_like(b1["codes_target"], dtype=torch.float32)
    pred2 = torch.zeros_like(b2["codes_target"], dtype=torch.float32)
    l1 = masked_ce_placeholder(pred1, b1["codes_target"], b1["frame_mask"])
    l2 = masked_ce_placeholder(pred2, b2["codes_target"], b2["frame_mask"])
    assert torch.allclose(l1, l2, rtol=0, atol=1e-6)

    assert_batch_pack_matches_items([as_item(a), as_item(b)])


def test_shard_roundtrip(tmp_path: Path) -> None:
    ex = _example("utt_rt", L=3, T=5, seed=7)
    w = CodecTokenShardWriter(tmp_path, shard_size=10, resume=False)
    w.add(ex)
    w.close()
    r = CodecTokenShardReader(tmp_path)
    got = r.get("utt_rt")
    assert torch.equal(got.codes, ex.codes)
    assert torch.equal(got.text_ids, ex.text_ids)
    assert torch.equal(got.features.stress, ex.features.stress)
    assert got.codec_hash == "deadbeef"
    assert got.license == "CC BY-NC-SA 4.0"
    assert_pack_train_infer_equal(got)


def test_build_ru_features_stress_and_span() -> None:
    text = "Код 12."
    # Fake BPE offsets covering each char roughly as one token + specials.
    # Use char-level offsets (grapheme BPE stand-in).
    offsets = [(i, i + 1) for i in range(len(text))]
    vocab = PhoneVocab()
    spans = [CriticalSpan(start=4, end=6, type="code", expected_spoken="двенадцать")]
    feats = build_ru_text_features(text, offsets, phone_vocab=vocab, critical_spans=spans)
    assert feats.stress.numel() == len(offsets)
    assert int(feats.span_type[4]) == span_type_id("code")
    assert int(feats.span_type[5]) == span_type_id("code")
    assert int(feats.word_bound.sum()) >= 1


def test_verify_trackn_no_leak(tmp_path: Path) -> None:
    rows = [
        Utterance(
            utt_id="tr1",
            wav="/tmp/a.wav",
            text_raw="а",
            text_stressed="а",
            corpus="ruslan",
            speaker_id="spk_a",
            split="train",
            dur_s=1.0,
            qc_tag="clean",
        ).materialize_defaults(),
        Utterance(
            utt_id="va1",
            wav="/tmp/b.wav",
            text_raw="б",
            text_stressed="б",
            corpus="ruslan",
            speaker_id="spk_b",
            split="val",
            dur_s=1.0,
            qc_tag="clean",
        ).materialize_defaults(),
        Utterance(
            utt_id="te1",
            wav="/tmp/c.wav",
            text_raw="в",
            text_stressed="в",
            corpus="common_voice_ru_17_hf_mirror",
            speaker_id="spk_c",
            split="test",
            dur_s=1.0,
            qc_tag="clean",
        ).materialize_defaults(),
    ]
    man = tmp_path / "manifest.jsonl"
    write_manifest(man, rows)
    sets = {}
    for split, uid in (("train", "tr1"), ("val", "va1"), ("test", "te1")):
        p = tmp_path / f"{split}.json"
        p.write_text(
            json.dumps({"id": split, "kind": "utt_ids", "utt_ids": [uid]}),
            encoding="utf-8",
        )
        sets[split] = str(p)
    report = verify_trackn_no_leak(man, sets)
    assert report["ok"] is True
    assert report["utt_overlaps"]["train∩test"] == 0


def test_pack_codec_sequence_empty_codes() -> None:
    ids = torch.tensor([1, 2, 3])
    codes = torch.zeros(0, 16, dtype=torch.long)
    p = pack_codec_sequence(ids, codes, _feats(3))
    assert p.seq_len == 4
    assert bool(p.is_eos[-1])

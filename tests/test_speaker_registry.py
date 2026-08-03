"""Deterministic speaker registry (D1b) — order / DDP / fail-closed tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from minispeech.data.collate_codec_seq import collate_codec_seq
from minispeech.data.speaker_registry import (
    UNKNOWN_SPEAKER_ID,
    SpeakerRegistry,
    assert_integer_speaker_id,
    build_dense_mapping,
)


def test_mapping_independent_of_input_order():
    a = ["z_spk", "a_spk", "m_spk", "a_spk"]
    b = list(reversed(a)) + ["m_spk"]
    m1 = build_dense_mapping(a)
    m2 = build_dense_mapping(b)
    assert m1 == m2
    assert list(m1.values()) == [1, 2, 3]
    assert m1["a_spk"] == 1
    assert m1["m_spk"] == 2
    assert m1["z_spk"] == 3


def test_registry_sha_stable_and_no_collisions(tmp_path: Path):
    r1 = SpeakerRegistry.from_speakers(
        ["cv:b", "ruslan", "cv:a"],
        source_manifest_sha256="man",
        train_set_sha256="train",
    )
    r2 = SpeakerRegistry.from_speakers(
        ["ruslan", "cv:a", "cv:b", "cv:a"],
        source_manifest_sha256="man",
        train_set_sha256="train",
    )
    assert r1.sha256 == r2.sha256
    assert r1.mapping == r2.mapping
    assert r1.n_speakers == 3
    assert r1.max_speakers == 4
    assert set(r1.mapping.values()) == {1, 2, 3}
    path = r1.save(tmp_path / "map.json")
    loaded = SpeakerRegistry.load(path)
    assert loaded.sha256 == r1.sha256
    assert loaded.to_id("ruslan", mode="train") == r1.to_id("ruslan", mode="train")


def test_train_eval_infer_same_known_id():
    reg = SpeakerRegistry.from_speakers(["spk_a", "spk_b"])
    for mode in ("train", "eval", "infer"):
        assert reg.to_id("spk_a", mode=mode) == 1
        assert reg.to_id("spk_b", mode=mode) == 2


def test_unknown_fail_closed_train_eval_ok_infer():
    reg = SpeakerRegistry.from_speakers(["spk_a"])
    with pytest.raises(KeyError, match="not in registry"):
        reg.to_id("never_seen", mode="train")
    with pytest.raises(KeyError, match="not in registry"):
        reg.to_id("never_seen", mode="eval")
    assert reg.to_id("never_seen", mode="infer") == UNKNOWN_SPEAKER_ID
    with pytest.raises(ValueError, match="unknown speaker id 0"):
        reg.to_id(0, mode="train")
    assert reg.to_id(0, mode="infer") == 0


def test_out_of_range_int_raises():
    reg = SpeakerRegistry.from_speakers(["spk_a"])
    with pytest.raises(ValueError, match="out of range"):
        reg.to_id(99, mode="infer")


def test_simulated_ddp_ranks_same_ids():
    speakers = [f"spk_{i:03d}" for i in range(20)]
    # Two ranks see shuffled shards but share the same frozen map.
    rank0 = speakers[::2]
    rank1 = speakers[1::2]
    global_map = SpeakerRegistry.from_speakers(speakers)
    r0 = SpeakerRegistry.load  # noqa: F841 — keep API reference
    for name in rank0 + rank1:
        assert global_map.to_id(name, mode="train") == global_map.mapping[name]
    # Rebuilding from either rank's local set alone would differ — the point is
    # both ranks must load the same frozen artifact, not build locally.
    local0 = SpeakerRegistry.from_speakers(rank0)
    local1 = SpeakerRegistry.from_speakers(rank1)
    assert local0.mapping != local1.mapping
    assert global_map.to_id("spk_001", mode="train") == global_map.to_id(
        "spk_001", mode="eval"
    )


def test_collate_rejects_string_speaker():
    items = [
        {
            "utt_id": "u0",
            "text_ids": torch.tensor([1, 2, 3]),
            "codes": torch.randint(0, 8, (4, 16)),
            "speaker_id": "ruslan",
        }
    ]
    with pytest.raises(TypeError, match="SpeakerRegistry"):
        collate_codec_seq(items)


def test_assert_integer_speaker_id():
    assert assert_integer_speaker_id(3) == 3
    with pytest.raises(TypeError):
        assert_integer_speaker_id("x")


def test_model_speaker_oob_raises_no_clamp():
    from minispeech.models.discrete_speech_lm import (
        DiscreteSpeechLM,
        DiscreteSpeechLMConfig,
    )

    cfg = DiscreteSpeechLMConfig.tiny()  # max_speakers=8
    m = DiscreteSpeechLM(cfg)
    ids = torch.randint(1, 40, (1, 3))
    codes = torch.randint(0, cfg.codebook_size, (1, 4, cfg.n_codebooks))
    with pytest.raises(ValueError, match="no clamp"):
        m(ids, codes, speaker_id=torch.tensor([99]), compute_loss=False)


def test_roundtrip_json_rejects_tampered_sha(tmp_path: Path):
    reg = SpeakerRegistry.from_speakers(["a", "b"])
    path = reg.save(tmp_path / "m.json")
    obj = json.loads(path.read_text(encoding="utf-8"))
    obj["sha256"] = "0" * 64
    path.write_text(json.dumps(obj), encoding="utf-8")
    with pytest.raises(ValueError, match="sha256 mismatch"):
        SpeakerRegistry.load(path)

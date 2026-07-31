"""N11: multi-corpus manifest — BC defaults, dedup, speaker split, streaming."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from minispeech.data.corpus_registry import CHANNELS, get_corpus
from minispeech.data.manifest import (
    Utterance,
    assert_no_speaker_leak,
    audio_fingerprint,
    audio_fingerprint_bytes,
    container_speaker_split,
    deduplicate_utterances,
    detect_duplicates,
    filter_split,
    filter_split_iter,
    iter_deduped_manifests,
    iter_manifest,
    read_manifest,
    speakers_by_split,
    text_dedup_key,
    write_manifest,
    write_manifest_iter,
)


def _utt(
    uid: str,
    text: str,
    *,
    speaker: str = "ruslan",
    speaker_id: str | None = None,
    wav: str = "/tmp/x.wav",
    dur_s: float = 1.0,
    split: str = "train",
    **kwargs,
) -> Utterance:
    return Utterance(
        utt_id=uid,
        wav=wav,
        text_raw=text,
        text_stressed=text,
        speaker=speaker,
        speaker_id=speaker_id,
        dur_s=dur_s,
        split=split,
        **kwargs,
    )


def test_legacy_ruslan_json_gets_defaults():
    """Existing RUSLAN row (no corpus/channel/...) remains readable."""
    legacy = {
        "utt_id": "ruslan_000000_RUSLAN",
        "wav": "/home/adam/datasets/tts/ruslan/wavs/000000_RUSLAN.wav",
        "text_raw": "С трево́жным чу́вством беру́сь я́ за́ перо́.",
        "text_stressed": "С трево́жным чу́вством беру́сь я́ за́ перо́.",
        "speaker": "ruslan",
        "n_samples_24k": 53874,
        "n_frames": 17,
        "dur_s": 2.24475,
        "split": "train",
        "asr_cer": 0.0,
        "qc_tag": "clean",
        "challenge": False,
        "sample_weight": 1.0,
    }
    u = Utterance.from_json(legacy)
    assert u.utt_id == legacy["utt_id"]
    assert u.speaker == "ruslan"
    assert u.split == "train"
    assert u.corpus == "ruslan"
    assert u.speaker_id == "ruslan"
    assert u.channel == "studio"
    assert u.channel in CHANNELS
    assert u.sample_rate == 22050
    info = get_corpus("ruslan")
    assert info is not None
    assert u.license == info.license
    assert u.source_url == info.source_url


def test_legacy_roundtrip_without_forcing_migration(tmp_path: Path):
    """Read→write must not migrate legacy JSON by injecting new corpus fields."""
    legacy_line = {
        "utt_id": "ruslan_1",
        "wav": "a.wav",
        "text_raw": "привет",
        "text_stressed": "приве́т",
        "speaker": "ruslan",
        "split": "train",
        "dur_s": 1.0,
    }
    p = tmp_path / "manifest.jsonl"
    p.write_text(json.dumps(legacy_line, ensure_ascii=False) + "\n", encoding="utf-8")
    rows = read_manifest(p)
    assert len(rows) == 1
    assert rows[0].corpus == "ruslan"
    assert rows[0].channel == "studio"
    assert rows[0].text_raw == "привет"
    assert rows[0].speaker == "ruslan"

    out = tmp_path / "manifest_out.jsonl"
    write_manifest(out, rows)
    written = json.loads(out.read_text(encoding="utf-8").splitlines()[0])
    for k in ("corpus", "speaker_id", "channel", "sample_rate", "license", "source_url"):
        assert k not in written, f"legacy writeback migrated field {k}"
    assert written["utt_id"] == "ruslan_1"
    assert written["speaker"] == "ruslan"


def test_text_dedup_normalization():
    a = text_dedup_key("За́мок!")
    b = text_dedup_key("замок")
    c = text_dedup_key("ЗАМОК.")
    assert a == b == c


def test_detect_and_deduplicate_by_text():
    rows = [
        _utt("a", "Один и тот же текст"),
        _utt("b", "один и тот же текст!"),
        _utt("c", "Другой текст"),
    ]
    hits = detect_duplicates(rows, by_text=True, by_audio=False)
    assert len(hits) == 1
    assert hits[0].key_type == "text"
    assert set(hits[0].utt_ids) == {"a", "b"}

    kept, stats = deduplicate_utterances(rows, by_text=True, by_audio=False)
    assert [u.utt_id for u in kept] == ["a", "c"]
    assert stats.n_in == 3
    assert stats.n_out == 2
    assert stats.n_dropped_text == 1


def test_audio_fingerprint_and_dedup(tmp_path: Path):
    wav_a = tmp_path / "a.wav"
    wav_b = tmp_path / "b.wav"
    wav_c = tmp_path / "c.wav"
    payload_dup = b"RIFF" + b"\x00" * 100 + b"same-audio-payload"
    payload_other = b"RIFF" + b"\x00" * 100 + b"other-audio-payload"
    wav_a.write_bytes(payload_dup)
    wav_b.write_bytes(payload_dup)
    wav_c.write_bytes(payload_other)

    assert audio_fingerprint(wav_a) == audio_fingerprint(wav_b)
    assert audio_fingerprint(wav_a) != audio_fingerprint(wav_c)
    assert audio_fingerprint(wav_a) == audio_fingerprint_bytes(payload_dup)

    rows = [
        _utt("a", "текст а", wav=str(wav_a)),
        _utt("b", "текст б", wav=str(wav_b)),
        _utt("c", "текст в", wav=str(wav_c)),
    ]
    hits = detect_duplicates(rows, by_text=False, by_audio=True, compute_missing_audio=True)
    assert len(hits) == 1
    assert hits[0].key_type == "audio"
    assert set(hits[0].utt_ids) == {"a", "b"}

    kept, stats = deduplicate_utterances(
        rows, by_text=False, by_audio=True, compute_missing_audio=True
    )
    assert [u.utt_id for u in kept] == ["a", "c"]
    assert stats.n_dropped_audio == 1


def test_speaker_split_no_leak():
    rows = []
    for i, spk in enumerate(["spk_a", "spk_b", "spk_c", "spk_d", "spk_e"]):
        for j in range(3):
            rows.append(
                _utt(
                    f"{spk}_{j}",
                    f"текст {spk} {j}",
                    speaker=spk,
                    speaker_id=spk,
                    dur_s=1.0 + 0.1 * i,
                )
            )
    container_speaker_split(rows, train=0.6, val=0.2, test=0.2)
    assert_no_speaker_leak(rows)
    by = speakers_by_split(rows)
    # Multi-speaker path must not put the same speaker in two splits
    assert by.get("train") or by.get("val") or by.get("test")
    train_spk = by.get("train", set())
    test_spk = by.get("test", set())
    assert train_spk.isdisjoint(test_spk)
    assert "train" in {r.split for r in rows}


def test_single_speaker_falls_back_to_contiguous():
    """RUSLAN-like: one speaker → book-range contiguous split (not speaker hold-out)."""
    rows = [_utt(f"ruslan_{i:04d}", f"фраза {i}", speaker="ruslan", speaker_id="ruslan") for i in range(20)]
    container_speaker_split(rows, train=0.8, val=0.1, test=0.1)
    assert rows[0].split == "train"
    assert rows[15].split == "train"
    assert rows[16].split == "val"
    assert rows[18].split == "test"
    # Same speaker may appear in all splits under contiguous (container) mode.


def test_assert_no_speaker_leak_raises_on_mixed():
    rows = [
        _utt("a1", "x", speaker_id="s1", speaker="s1", split="train"),
        _utt("a2", "y", speaker_id="s1", speaker="s1", split="test"),
    ]
    with pytest.raises(AssertionError, match="speaker leak"):
        assert_no_speaker_leak(rows)


def test_streaming_iter_and_filter(tmp_path: Path):
    rows = [
        _utt("u0", "а", split="train"),
        _utt("u1", "б", split="val"),
        _utt("u2", "в", split="train"),
    ]
    p = tmp_path / "m.jsonl"
    n = write_manifest_iter(p, rows)
    assert n == 3

    streamed = list(iter_manifest(p))
    assert [u.utt_id for u in streamed] == ["u0", "u1", "u2"]

    train_ids = [u.utt_id for u in filter_split_iter(p, "train")]
    assert train_ids == ["u0", "u2"]
    assert [u.utt_id for u in filter_split(streamed, "val")] == ["u1"]


def test_iter_deduped_manifests_streaming(tmp_path: Path):
    p1 = tmp_path / "c1.jsonl"
    p2 = tmp_path / "c2.jsonl"
    write_manifest(
        p1,
        [
            _utt("c1_a", "общий текст", speaker="a", speaker_id="a"),
            _utt("c1_b", "только первый", speaker="a", speaker_id="a"),
        ],
    )
    write_manifest(
        p2,
        [
            _utt("c2_a", "Общий текст!", speaker="b", speaker_id="b"),
            _utt("c2_c", "только второй", speaker="b", speaker_id="b"),
        ],
    )
    out = list(iter_deduped_manifests([p1, p2], by_text=True, by_audio=False))
    ids = [u.utt_id for u in out]
    assert ids == ["c1_a", "c1_b", "c2_c"]


def test_explicit_multi_corpus_fields_preserved():
    u = Utterance.from_json(
        {
            "utt_id": "golos_1",
            "wav": "x.wav",
            "text_raw": "да",
            "text_stressed": "да",
            "speaker": "anon_01",
            "corpus": "golos",
            "speaker_id": "anon_01",
            "channel": "crowd",
            "sample_rate": 16000,
            "license": "custom",
            "source_url": "https://example.com/golos",
        }
    )
    assert u.corpus == "golos"
    assert u.channel == "crowd"
    assert u.sample_rate == 16000
    assert u.license == "custom"
    assert u.source_url == "https://example.com/golos"
    # Unknown corpus → no registry overwrite of explicit fields
    assert u.speaker_id == "anon_01"


def test_invalid_channel_rejected():
    with pytest.raises(ValueError, match="invalid channel"):
        Utterance.from_json(
            {
                "utt_id": "x",
                "wav": "a.wav",
                "text_raw": "t",
                "text_stressed": "t",
                "channel": "satellite",
            }
        )

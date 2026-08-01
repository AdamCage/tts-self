"""D07 cross-corpus merge, priority, marking, and report checks."""

from __future__ import annotations

import json
from pathlib import Path

from minispeech.data.manifest import (
    Utterance,
    assert_no_speaker_leak,
    audio_fingerprint_bytes,
    read_manifest,
    text_fingerprint,
    write_manifest,
)
from minispeech.data.merge import merge_manifests, write_merged_outputs


def _row(
    uid: str,
    *,
    corpus: str,
    speaker: str,
    text: str,
    audio: bytes,
) -> Utterance:
    return Utterance(
        utt_id=uid,
        wav=f"/not-needed/{uid}.wav",
        text_raw=text,
        text_stressed=text,
        speaker=f"{corpus}:{speaker}",
        corpus=corpus,
        speaker_id=f"{corpus}:{speaker}",
        channel="crowd",
        sample_rate=16000,
        license="CC0-1.0",
        source_url=f"https://example.com/{corpus}",
        source_id=uid,
        container_id=f"{corpus}:container",
        audio_sha256=audio_fingerprint_bytes(audio),
        text_sha256=text_fingerprint(text),
        dur_s=2.0,
    )


def _manifests(tmp_path: Path) -> tuple[Path, Path]:
    first = tmp_path / "first.jsonl"
    second = tmp_path / "second.jsonl"
    write_manifest(
        first,
        [
            _row("a1", corpus="a", speaker="s1", text="Общий текст", audio=b"a1"),
            _row("a2", corpus="a", speaker="s2", text="Только A", audio=b"shared-audio"),
            _row("a3", corpus="a", speaker="s3", text="Повтор внутри", audio=b"a3"),
            _row("a4", corpus="a", speaker="s4", text="Повтор внутри", audio=b"a4"),
        ],
    )
    write_manifest(
        second,
        [
            _row("b1", corpus="b", speaker="s1", text="общий текст!", audio=b"b1"),
            _row("b2", corpus="b", speaker="s2", text="Только B", audio=b"shared-audio"),
            _row("b3", corpus="b", speaker="s3", text="Уникальный B", audio=b"b3"),
            _row("b4", corpus="b", speaker="s4", text="Ещё B", audio=b"b4"),
        ],
    )
    return first, second


def test_merge_marks_lower_priority_duplicates_and_has_no_active_audio_dup(
    tmp_path: Path,
) -> None:
    first, second = _manifests(tmp_path)
    rows, duplicates, card = merge_manifests(
        [first, second],
        priority=["a", "b"],
        split_strategy="speaker",
        train=0.5,
        val=0.25,
        test=0.25,
    )
    assert len(rows) == 8
    marked = {row.utt_id for row in rows if row.qc_tag == "duplicate"}
    assert marked == {"b1", "b2"}
    assert all(row.sample_weight == 0.0 for row in rows if row.qc_tag == "duplicate")
    active = [row for row in rows if row.qc_tag != "duplicate"]
    assert {row.utt_id for row in active} == {"a1", "a2", "a3", "a4", "b3", "b4"}
    active_audio = [row.audio_sha256 for row in active if row.audio_sha256]
    assert len(active_audio) == len(set(active_audio))
    assert duplicates["n_marked_duplicate"] == 2
    assert duplicates["marked_by_reason"] == {"audio": 1, "text": 1}
    assert duplicates["active_audio_fingerprints_unique"] is True
    assert_no_speaker_leak(rows)
    assert card["n"] == 6
    assert card["n_manifest"] == 8

    outputs = write_merged_outputs(
        tmp_path / "merged.jsonl",
        rows,
        duplicates,
        card,
        duplicate_report_path=tmp_path / "merge_dedup_report.json",
    )
    written = read_manifest(outputs["manifest"])
    assert len(written) == 8
    report = json.loads(Path(outputs["duplicate_report"]).read_text(encoding="utf-8"))
    assert {row["duplicate_utt_id"] for row in report["duplicates"]} == {"b1", "b2"}
    assert all(row["conflicts"] for row in report["duplicates"])
    dataset_card = json.loads(Path(outputs["dataset_card"]).read_text(encoding="utf-8"))
    assert dataset_card["source_manifests"][0]["sha256"]


def test_explicit_priority_overrides_manifest_order(tmp_path: Path) -> None:
    first, second = _manifests(tmp_path)
    rows, _, _ = merge_manifests(
        [first, second],
        priority=["b", "a"],
        split_strategy="speaker",
    )
    marked = {row.utt_id for row in rows if row.qc_tag == "duplicate"}
    assert marked == {"a1", "a2"}


def test_duplicate_utt_id_is_a_hard_error(tmp_path: Path) -> None:
    first = tmp_path / "first.jsonl"
    second = tmp_path / "second.jsonl"
    write_manifest(
        first,
        [_row("same", corpus="a", speaker="s1", text="A", audio=b"a")],
    )
    write_manifest(
        second,
        [_row("same", corpus="b", speaker="s2", text="B", audio=b"b")],
    )
    try:
        merge_manifests([first, second])
    except ValueError as exc:
        assert "duplicate utt_id" in str(exc)
    else:
        raise AssertionError("duplicate utt_id was accepted")

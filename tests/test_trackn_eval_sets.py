"""D08 / R1 frozen Track N speaker-held-out set tests."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from minispeech.data.manifest import Utterance, read_manifest, write_manifest

ROOT = Path(__file__).resolve().parents[1]


def _row(
    uid: str,
    speaker: str,
    split: str,
    *,
    qc_tag: str = "clean",
) -> Utterance:
    return Utterance(
        utt_id=uid,
        wav=f"{uid}.wav",
        text_raw=uid,
        text_stressed=uid,
        speaker=speaker,
        corpus="common_voice_ru_17_hf_mirror",
        speaker_id=speaker,
        channel="crowd",
        sample_rate=48000,
        license="CC0-1.0",
        source_url="https://example.com",
        source_id=uid,
        container_id=f"container:{speaker}",
        dur_s=2.0,
        split=split,
        qc_tag=qc_tag,
    )


def _run(script: str, *args: str) -> subprocess.CompletedProcess[str]:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT / "src")
    return subprocess.run(
        [sys.executable, str(ROOT / "scripts" / script), *args],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )


def test_freeze_trackn_sets_and_register_hashes(tmp_path: Path) -> None:
    manifest = tmp_path / "merged.jsonl"
    write_manifest(
        manifest,
        [
            _row("train-a-1", "speaker-a", "train"),
            _row("train-a-2", "speaker-a", "train"),
            _row("train-a-ex", "speaker-a", "train", qc_tag="exclude"),
            _row("train-a-rev", "speaker-a", "train", qc_tag="review"),
            _row("train-b-1", "speaker-b", "train"),
            _row("val-e-1", "speaker-e", "val"),
            _row("val-e-2", "speaker-e", "val"),
            _row("test-c-1", "speaker-c", "test"),
            _row("test-c-2", "speaker-c", "test"),
            _row("test-d-1", "speaker-d", "test"),
            _row("dup-x-1", "speaker-x", "train", qc_tag="duplicate"),
        ],
    )
    sets = tmp_path / "sets"
    report = tmp_path / "trackn_eval_hashes.txt"
    frozen = _run(
        "58_freeze_trackn_sets.py",
        "--manifest",
        str(manifest),
        "--out-dir",
        str(sets),
        "--report",
        str(report),
    )
    assert frozen.returncode == 0, frozen.stderr
    result = json.loads(frozen.stdout)
    assert result["speaker_overlap"] == []
    assert result["sets"]["trackn_train"]["speakers"] == 2
    assert result["sets"]["trackn_val"]["speakers"] == 1
    assert result["sets"]["trackn_test"]["speakers"] == 2
    assert result["sets"]["trackn_review"]["n"] == 1
    assert result["accounting"]["n_remainder_exclude"] == 1
    assert (
        result["accounting"]["n_trackn_train"]
        + result["accounting"]["n_trackn_val"]
        + result["accounting"]["n_trackn_test"]
        + result["accounting"]["n_trackn_review"]
        + result["accounting"]["n_remainder_exclude"]
        == result["accounting"]["n_active"]
    )

    for split in ("train", "val", "test", "review"):
        payload = json.loads(
            (sets / f"trackn_{split}.json").read_text(encoding="utf-8")
        )
        assert payload["version"] == "2.0.0"
        assert payload["kind"] == "utt_ids"
        assert payload["n"] == len(payload["utt_ids"])

    train_payload = json.loads(
        (sets / "trackn_train.json").read_text(encoding="utf-8")
    )
    assert "train-a-ex" not in train_payload["utt_ids"]
    assert "train-a-rev" not in train_payload["utt_ids"]
    assert "dup-x-1" not in train_payload["utt_ids"]
    assert "review" in train_payload["source"]["selection"]
    assert "hours=" in report.read_text(encoding="utf-8")

    hashes = tmp_path / "eval_set_hashes.json"
    made = _run(
        "54_eval_sets.py",
        "hash",
        "--sets-dir",
        str(sets),
        "--hash-file",
        str(hashes),
    )
    assert made.returncode == 0, made.stderr
    registry = json.loads(hashes.read_text(encoding="utf-8"))
    assert set(registry["sets"]) == {
        "trackn_train",
        "trackn_val",
        "trackn_test",
        "trackn_review",
    }


def test_trackn_train_has_no_exclude_or_duplicate_qc_tags(tmp_path: Path) -> None:
    """Acceptance: no train utt_id has qc_tag in {exclude, duplicate}."""
    manifest = tmp_path / "merged.jsonl"
    write_manifest(
        manifest,
        [
            _row("train-ok", "speaker-a", "train", qc_tag="clean"),
            _row("train-bad", "speaker-a", "train", qc_tag="exclude"),
            _row("val-ok", "speaker-b", "val", qc_tag="clean"),
            _row("test-ok", "speaker-c", "test", qc_tag="clean"),
        ],
    )
    sets = tmp_path / "sets"
    frozen = _run(
        "58_freeze_trackn_sets.py",
        "--manifest",
        str(manifest),
        "--out-dir",
        str(sets),
        "--report",
        str(tmp_path / "hashes.txt"),
    )
    assert frozen.returncode == 0, frozen.stderr
    train_ids = set(
        json.loads((sets / "trackn_train.json").read_text(encoding="utf-8"))[
            "utt_ids"
        ]
    )
    by_id = {row.utt_id: row for row in read_manifest(manifest)}
    forbidden = {"exclude", "duplicate"}
    for utt_id in train_ids:
        assert by_id[utt_id].qc_tag not in forbidden
    assert "train-bad" not in train_ids

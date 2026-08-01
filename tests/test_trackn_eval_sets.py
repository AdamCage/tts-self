"""D08 frozen Track N speaker-held-out set tests."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from minispeech.data.manifest import Utterance, write_manifest

ROOT = Path(__file__).resolve().parents[1]


def _row(uid: str, speaker: str, split: str) -> Utterance:
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
            _row("train-b-1", "speaker-b", "train"),
            _row("test-c-1", "speaker-c", "test"),
            _row("test-c-2", "speaker-c", "test"),
            _row("test-d-1", "speaker-d", "test"),
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
    assert result["sets"]["trackn_test"]["speakers"] == 2

    for split in ("train", "test"):
        payload = json.loads(
            (sets / f"trackn_{split}.json").read_text(encoding="utf-8")
        )
        assert payload["version"] == "1.0.0"
        assert payload["kind"] == "utt_ids"
        assert payload["n"] == len(payload["utt_ids"])
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
    assert set(registry["sets"]) == {"trackn_train", "trackn_test"}

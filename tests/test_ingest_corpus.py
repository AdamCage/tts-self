"""D06 synthetic tests for generic and source-specific corpus ingestion."""

from __future__ import annotations

import gc
import json
import os
import subprocess
import sys
import tracemalloc
import wave
from pathlib import Path

from minispeech.data.ingest import (
    build_directory_manifest,
    build_manifest_from_config,
    ingest_records,
    stable_utt_id,
)
from minispeech.data.manifest import assert_no_speaker_leak, read_manifest

ROOT = Path(__file__).resolve().parents[1]
CORPUS = "common_voice_ru_17_hf_mirror"


def _write_wav(path: Path, *, frames: int = 16000, sample_rate: int = 16000) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(b"\x00\x00" * frames)


def test_jsonl_handler_builds_stable_materialized_manifest(tmp_path: Path) -> None:
    audio = tmp_path / "audio"
    audio.mkdir()
    _write_wav(audio / "one.wav")
    _write_wav(audio / "two.wav", frames=24000)
    source = tmp_path / "source.jsonl"
    records = [
        {
            "path": "one.wav",
            "sentence": "Первый замо́к.",
            "client_id": "speaker-a",
            "split": "train",
            "up_votes": 3,
        },
        {
            "path": "two.wav",
            "sentence": "Второй текст.",
            "client_id": "speaker-b",
            "split": "test",
            "up_votes": 2,
        },
    ]
    source.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in records) + "\n",
        encoding="utf-8",
    )
    manifest = tmp_path / "manifest.jsonl"
    config = tmp_path / "source.yaml"
    config.write_text(
        f"""
ingest:
  corpus: {CORPUS}
  format: jsonl
  inputs: {source.as_posix()}
  audio_root: {audio.as_posix()}
  audio_cache: {(tmp_path / "cache").as_posix()}
  manifest: {manifest.as_posix()}
  field_map:
    audio: path
    text: sentence
    source_id: path
    speaker_id: client_id
    container_id: client_id
    split: split
  extra_fields: [up_votes]
  split:
    strategy: source
""",
        encoding="utf-8",
    )

    first = build_manifest_from_config(config)
    first_bytes = manifest.read_bytes()
    second = build_manifest_from_config(config)
    assert manifest.read_bytes() == first_bytes
    assert first["n"] == second["n"] == 2

    rows = read_manifest(manifest)
    assert [row.split for row in rows] == ["train", "test"]
    assert rows[0].utt_id == stable_utt_id(CORPUS, "one.wav")
    assert rows[0].source_id == "one.wav"
    assert rows[0].speaker_id == f"{CORPUS}:speaker-a"
    assert rows[0].container_id == f"{CORPUS}:speaker-a"
    assert rows[0].corpus == CORPUS
    assert rows[0].sample_rate == 16000
    assert rows[0].channel == "crowd"
    assert rows[0].license == "CC0-1.0"
    assert rows[0].text_stressed == "Первый замо́к."
    assert len(rows[0].audio_sha256 or "") == 64
    assert len(rows[0].text_sha256 or "") == 64
    assert rows[0].extra["up_votes"] == 3


def test_embedded_audio_bytes_are_cached_and_duplicate_source_is_dropped(
    tmp_path: Path,
) -> None:
    wav = tmp_path / "payload.wav"
    _write_wav(wav)
    payload = wav.read_bytes()
    record = {
        "audio": {"path": "clip.wav", "bytes": payload},
        "sentence": "Тест.",
        "client_id": "speaker",
        "path": "clip.wav",
    }
    cfg = {
        "corpus": CORPUS,
        "format": "parquet",
        "audio_cache": str(tmp_path / "cache"),
        "field_map": {
            "audio": "audio",
            "text": "sentence",
            "source_id": "path",
            "speaker_id": "client_id",
            "container_id": "client_id",
        },
        "split": {"strategy": "source"},
    }
    rows, meta = ingest_records(
        [(record, "train"), (record, "train")],
        cfg,
        base_dir=tmp_path,
    )
    assert len(rows) == 1
    assert Path(rows[0].wav).is_file()
    assert rows[0].split == "train"
    assert meta["filter_counters"]["dropped_duplicate_source_id"] == 1


def _make_directory_config(tmp_path: Path, *, n: int) -> Path:
    source = tmp_path / f"source_{n}"
    for index in range(n):
        speaker = f"speaker-{index % 3}"
        wav = source / speaker / f"clip-{index:04d}.wav"
        _write_wav(wav, frames=160)
        wav.with_suffix(".txt").write_text(f"Фраза номер {index}.", encoding="utf-8")
    config = tmp_path / f"directory_{n}.yaml"
    config.write_text(
        f"""
ingest:
  corpus: {CORPUS}
  format: directory
  root: {source.as_posix()}
  audio_glob: "**/*.wav"
  transcript_template: "{{relative_no_suffix}}.txt"
  speaker_from: topdir
  container_from: relative_parent
  manifest: {(tmp_path / f"manifest_{n}.jsonl").as_posix()}
  min_dur_s: 0.0
  max_dur_s: 15.0
  split:
    strategy: speaker_holdout
    train: 0.6
    val: 0.2
    test: 0.2
""",
        encoding="utf-8",
    )
    return config


def _peak_build(config: Path) -> tuple[int, dict[str, object]]:
    gc.collect()
    tracemalloc.start()
    result = build_directory_manifest(config)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return peak, result


def test_directory_cli_streams_three_speakers_and_memory_is_bounded(
    tmp_path: Path,
) -> None:
    small = _make_directory_config(tmp_path, n=10)
    large = _make_directory_config(tmp_path, n=1000)
    small_peak, small_result = _peak_build(small)
    large_peak, large_result = _peak_build(large)
    assert small_result["n"] == 10
    assert large_result["n"] == 1000
    assert small_result["streaming"] is True
    assert large_result["streaming"] is True
    # File count grows 100x; peak traced Python memory stays effectively flat.
    assert large_peak <= small_peak + 2_000_000

    rows = read_manifest(large_result["manifest"])
    assert len(rows) == 1000
    assert {row.container_id for row in rows} == {
        f"{CORPUS}:speaker-0",
        f"{CORPUS}:speaker-1",
        f"{CORPUS}:speaker-2",
    }
    assert all(row.source_id and row.license and row.source_url for row in rows)
    assert_no_speaker_leak(rows)

    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT / "src")
    verified = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "55_corpus_registry.py"),
            "verify",
            "--data",
            str(large),
        ],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    assert verified.returncode == 0, verified.stderr or verified.stdout

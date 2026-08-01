"""Cross-corpus manifest merge, deduplication, split, and dataset-card reports."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

from minispeech.data.manifest import (
    Utterance,
    assert_no_speaker_leak,
    audio_fingerprint,
    iter_manifest,
    speaker_split,
    text_fingerprint,
    write_manifest,
)
from minispeech.utils import expand


def _file_sha256(path: Path, *, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _corpus_id(row: Utterance) -> str:
    return row.corpus or row.utt_id.split("__", 1)[0] or "unknown"


def merge_manifests(
    manifests: Sequence[str | Path],
    *,
    compute_missing_audio: bool = False,
    split_strategy: str = "speaker",
    priority: Sequence[str] | None = None,
    train: float = 0.9,
    val: float = 0.05,
    test: float = 0.05,
) -> tuple[list[Utterance], dict[str, Any], dict[str, Any]]:
    """Merge in corpus priority order and mark cross-corpus duplicates.

    Repeated text or audio inside one source corpus is retained here; source
    handlers own within-corpus QC. Lower-priority conflicts remain in the
    manifest with ``qc_tag=duplicate`` and zero weight, preserving provenance.
    """
    if not manifests:
        raise ValueError("at least one manifest is required")
    paths = [expand(path) for path in manifests]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"manifest not found: {missing}")

    seen_ids: dict[str, str] = {}
    seen_text: dict[str, tuple[str, str]] = {}
    seen_audio: dict[str, tuple[str, str]] = {}
    merged: list[Utterance] = []
    duplicate_rows: list[dict[str, Any]] = []
    input_counts: dict[str, int] = defaultdict(int)
    marked_by_reason: dict[str, int] = defaultdict(int)
    entries: list[tuple[int, Path, Utterance]] = []
    serial = 0
    for path in paths:
        for row in iter_manifest(path):
            corpus = _corpus_id(row)
            input_counts[corpus] += 1
            entries.append((serial, path, row))
            serial += 1

    priority_order = list(priority or [])
    priority_rank = {corpus: index for index, corpus in enumerate(priority_order)}
    if priority_order:
        entries.sort(
            key=lambda item: (
                priority_rank.get(_corpus_id(item[2]), len(priority_rank)),
                item[0],
            )
        )

    for _, path, row in entries:
        corpus = _corpus_id(row)
        previous_manifest = seen_ids.get(row.utt_id)
        if previous_manifest is not None:
            raise ValueError(
                f"duplicate utt_id {row.utt_id!r} in {previous_manifest} and {path}"
            )
        seen_ids[row.utt_id] = str(path)

        text_hash = row.text_sha256 or text_fingerprint(
            row.text_raw or row.text_stressed or ""
        )
        audio_hash = row.audio_sha256
        if audio_hash is None and compute_missing_audio and row.wav:
            try:
                audio_hash = audio_fingerprint(row.wav)
                row.audio_sha256 = audio_hash
                row._defaulted_fields.discard("audio_sha256")
            except OSError:
                audio_hash = None

        conflicts: list[dict[str, str]] = []
        prior_text = seen_text.get(text_hash) if text_hash else None
        if prior_text is not None and prior_text[0] != corpus:
            conflicts.append(
                {
                    "kind": "text",
                    "sha256": text_hash,
                    "kept_corpus": prior_text[0],
                    "kept_utt_id": prior_text[1],
                }
            )
        prior_audio = seen_audio.get(audio_hash) if audio_hash else None
        if prior_audio is not None and prior_audio[0] != corpus:
            conflicts.append(
                {
                    "kind": "audio",
                    "sha256": audio_hash,
                    "kept_corpus": prior_audio[0],
                    "kept_utt_id": prior_audio[1],
                }
            )
        if conflicts:
            kinds = "+".join(sorted({item["kind"] for item in conflicts}))
            marked_by_reason[kinds] += 1
            row.qc_tag = "duplicate"
            row.sample_weight = 0.0
            duplicate_rows.append(
                {
                    "duplicate_corpus": corpus,
                    "duplicate_utt_id": row.utt_id,
                    "source_manifest": str(path),
                    "conflicts": conflicts,
                }
            )
            merged.append(row.materialize_defaults())
            continue

        if text_hash and text_hash not in seen_text:
            seen_text[text_hash] = (corpus, row.utt_id)
            row.text_sha256 = text_hash
            row._defaulted_fields.discard("text_sha256")
        if audio_hash and audio_hash not in seen_audio:
            seen_audio[audio_hash] = (corpus, row.utt_id)
        merged.append(row.materialize_defaults())

    if split_strategy == "speaker":
        merged = speaker_split(merged, train=train, val=val, test=test)
        assert_no_speaker_leak(merged)
    elif split_strategy == "preserve":
        assert_no_speaker_leak(merged)
    else:
        raise ValueError("split_strategy must be 'speaker' or 'preserve'")

    source_files = [
        {
            "path": str(path),
            "sha256": _file_sha256(path),
        }
        for path in paths
    ]
    duplicate_report = {
        "schema_version": 1,
        "policy": "higher-priority corpus wins; conflicts remain qc_tag=duplicate",
        "priority": priority_order or "manifest_order",
        "compute_missing_audio": compute_missing_audio,
        "n_input": sum(input_counts.values()),
        "n_manifest": len(merged),
        "n_active": sum(row.qc_tag != "duplicate" for row in merged),
        "n_marked_duplicate": len(duplicate_rows),
        "marked_by_reason": dict(sorted(marked_by_reason.items())),
        "active_audio_fingerprints_unique": len(
            {
                row.audio_sha256
                for row in merged
                if row.qc_tag != "duplicate" and row.audio_sha256
            }
        )
        == sum(
            bool(row.audio_sha256)
            for row in merged
            if row.qc_tag != "duplicate"
        ),
        "duplicates": duplicate_rows,
    }
    card = build_dataset_card(
        merged,
        source_files=source_files,
        input_counts=dict(input_counts),
        duplicate_report=duplicate_report,
        split_strategy=split_strategy,
    )
    return merged, duplicate_report, card


def build_dataset_card(
    rows: Sequence[Utterance],
    *,
    source_files: list[dict[str, str]],
    input_counts: dict[str, int],
    duplicate_report: dict[str, Any],
    split_strategy: str,
) -> dict[str, Any]:
    active_rows = [row for row in rows if row.qc_tag != "duplicate"]
    corpora: dict[str, dict[str, Any]] = {}
    for corpus in sorted({_corpus_id(row) for row in active_rows}):
        subset = [row for row in active_rows if _corpus_id(row) == corpus]
        corpora[corpus] = {
            "n": len(subset),
            "hours": sum(row.dur_s for row in subset) / 3600.0,
            "speakers": len({row.effective_speaker_id() for row in subset}),
            "licenses": sorted({row.license or "unknown" for row in subset}),
            "channels": sorted({row.channel or "unknown" for row in subset}),
            "source_urls": sorted({row.source_url or "unknown" for row in subset}),
        }
    return {
        "schema_version": 1,
        "n": len(active_rows),
        "n_manifest": len(rows),
        "hours": sum(row.dur_s for row in active_rows) / 3600.0,
        "speakers": len({row.effective_speaker_id() for row in active_rows}),
        "split_strategy": split_strategy,
        "splits": {
            split: {
                "n": sum(row.split == split for row in active_rows),
                "hours": (
                    sum(row.dur_s for row in active_rows if row.split == split) / 3600.0
                ),
                "speakers": len(
                    {
                        row.effective_speaker_id()
                        for row in active_rows
                        if row.split == split
                    }
                ),
            }
            for split in ("train", "val", "test")
        },
        "input_counts": dict(sorted(input_counts.items())),
        "corpora": corpora,
        "dedup": {
            "n_marked_duplicate": duplicate_report["n_marked_duplicate"],
            "marked_by_reason": duplicate_report["marked_by_reason"],
            "active_audio_fingerprints_unique": duplicate_report[
                "active_audio_fingerprints_unique"
            ],
        },
        "source_manifests": source_files,
    }


def write_merged_outputs(
    output: str | Path,
    rows: list[Utterance],
    duplicate_report: dict[str, Any],
    dataset_card: dict[str, Any],
    *,
    duplicate_report_path: str | Path | None = None,
) -> dict[str, str]:
    path = expand(output)
    write_manifest(path, rows)
    duplicates_path = (
        expand(duplicate_report_path)
        if duplicate_report_path is not None
        else path.with_suffix(".duplicates.json")
    )
    duplicates_path.parent.mkdir(parents=True, exist_ok=True)
    card_path = path.with_suffix(".dataset_card.json")
    duplicates_path.write_text(
        json.dumps(duplicate_report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    card_path.write_text(
        json.dumps(dataset_card, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return {
        "manifest": str(path),
        "duplicate_report": str(duplicates_path),
        "dataset_card": str(card_path),
    }

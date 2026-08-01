#!/usr/bin/env python3
"""Freeze speaker-held-out Track N train/test utterance sets."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Sequence

from minispeech.data.manifest import assert_no_speaker_leak, read_manifest
from minispeech.utils import expand

ROOT = Path(__file__).resolve().parents[1]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _items_sha256(items: list[str]) -> str:
    payload = json.dumps(
        items,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def freeze_trackn_sets(
    manifest_path: str | Path,
    output_dir: str | Path,
    report_path: str | Path,
) -> dict[str, Any]:
    manifest = expand(manifest_path)
    rows = [row for row in read_manifest(manifest) if row.qc_tag != "duplicate"]
    if not rows:
        raise ValueError("merged manifest has no active rows")
    assert_no_speaker_leak(rows)
    output = expand(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    result: dict[str, Any] = {
        "schema_version": 1,
        "manifest": str(manifest),
        "manifest_sha256": _sha256(manifest),
        "sets": {},
    }
    report_lines = [
        "# Track N frozen eval sets",
        f"manifest={manifest}",
        f"manifest_sha256={result['manifest_sha256']}",
    ]
    for split in ("train", "test"):
        selected = sorted(
            (row for row in rows if row.split == split),
            key=lambda row: row.utt_id,
        )
        if not selected:
            raise ValueError(f"merged manifest has no active {split} rows")
        set_id = f"trackn_{split}"
        ids = [row.utt_id for row in selected]
        speakers = {row.effective_speaker_id() for row in selected}
        payload = {
            "id": set_id,
            "version": "1.0.0",
            "description": (
                f"Frozen Track N {split} inventory; speaker-disjoint split from "
                "the merged P1 manifest"
            ),
            "kind": "utt_ids",
            "n": len(ids),
            "source": {
                "manifest": str(manifest),
                "manifest_sha256": result["manifest_sha256"],
                "selection": "active rows (qc_tag != duplicate), split by speaker",
                "speakers": len(speakers),
                "hours": sum(row.dur_s for row in selected) / 3600.0,
            },
            "utt_ids": ids,
        }
        set_path = output / f"{set_id}.json"
        set_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        record = {
            "path": str(set_path),
            "n": len(ids),
            "speakers": len(speakers),
            "hours": payload["source"]["hours"],
            "items_sha256": _items_sha256(ids),
            "file_sha256": _sha256(set_path),
        }
        result["sets"][set_id] = record
        report_lines.extend(
            [
                "",
                f"[{set_id}]",
                f"n={record['n']}",
                f"speakers={record['speakers']}",
                f"hours={record['hours']:.9f}",
                f"items_sha256={record['items_sha256']}",
                f"file_sha256={record['file_sha256']}",
            ]
        )

    train_speakers = {
        row.effective_speaker_id()
        for row in rows
        if row.split == "train"
    }
    test_speakers = {
        row.effective_speaker_id()
        for row in rows
        if row.split == "test"
    }
    overlap = sorted(train_speakers & test_speakers)
    if overlap:
        raise AssertionError(f"speaker overlap between Track N train/test: {overlap}")
    result["speaker_overlap"] = overlap
    report = expand(report_path)
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text("\n".join(report_lines) + "\n", encoding="utf-8", newline="\n")
    result["report"] = str(report)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out-dir", default="configs/eval_sets")
    parser.add_argument("--report", default="out/trackn_eval_hashes.txt")
    args = parser.parse_args(argv)
    try:
        result = freeze_trackn_sets(args.manifest, args.out_dir, args.report)
    except (AssertionError, FileNotFoundError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Freeze speaker-held-out Track N train/val/test/review utterance sets."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

from minispeech.data.manifest import Utterance, assert_no_speaker_leak, read_manifest
from minispeech.utils import expand

ROOT = Path(__file__).resolve().parents[1]

# Train/val/test inventories use only clean active rows.
# review is frozen separately; exclude is documented remainder (not a set).
CLEAN_TRAIN_SELECTION = (
    "qc_tag=clean and split=train; exclude and duplicate excluded from train; "
    "review excluded from train (frozen as trackn_review)"
)
CLEAN_VAL_SELECTION = (
    "qc_tag=clean and split=val; exclude/duplicate/review not included "
    "(review in trackn_review; exclude documented as remainder)"
)
CLEAN_TEST_SELECTION = (
    "qc_tag=clean and split=test; exclude/duplicate/review not included "
    "(review in trackn_review; exclude documented as remainder)"
)
REVIEW_SELECTION = (
    "qc_tag=review (any split); held out of trackn_train/val/test for analysis"
)


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


def _set_stats(selected: list[Utterance]) -> tuple[list[str], set[str], float]:
    ids = [row.utt_id for row in selected]
    speakers = {row.effective_speaker_id() for row in selected}
    hours = sum(row.dur_s for row in selected) / 3600.0
    return ids, speakers, hours


def _write_set(
    *,
    output: Path,
    set_id: str,
    description: str,
    selection: str,
    selected: list[Utterance],
    manifest: Path,
    manifest_sha256: str,
    accounting: dict[str, Any],
) -> dict[str, Any]:
    ids, speakers, hours = _set_stats(selected)
    payload = {
        "id": set_id,
        "version": "2.0.0",
        "description": description,
        "kind": "utt_ids",
        "n": len(ids),
        "source": {
            "manifest": str(manifest),
            "manifest_sha256": manifest_sha256,
            "selection": selection,
            "speakers": len(speakers),
            "hours": hours,
            "accounting": accounting,
        },
        "utt_ids": ids,
    }
    set_path = output / f"{set_id}.json"
    set_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return {
        "path": str(set_path),
        "n": len(ids),
        "speakers": len(speakers),
        "hours": hours,
        "items_sha256": _items_sha256(ids),
        "file_sha256": _sha256(set_path),
    }


def freeze_trackn_sets(
    manifest_path: str | Path,
    output_dir: str | Path,
    report_path: str | Path,
) -> dict[str, Any]:
    manifest = expand(manifest_path)
    all_rows = read_manifest(manifest)
    active = [row for row in all_rows if row.qc_tag != "duplicate"]
    if not active:
        raise ValueError("merged manifest has no active rows")

    none_active = [row.utt_id for row in active if row.qc_tag is None]
    if none_active:
        raise ValueError(
            f"active rows with qc_tag=None forbidden after QC rematerialize; "
            f"n={len(none_active)} example={none_active[:5]}"
        )

    assert_no_speaker_leak(active)
    output = expand(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    clean_train = sorted(
        (row for row in active if row.split == "train" and row.qc_tag == "clean"),
        key=lambda row: row.utt_id,
    )
    clean_val = sorted(
        (row for row in active if row.split == "val" and row.qc_tag == "clean"),
        key=lambda row: row.utt_id,
    )
    clean_test = sorted(
        (row for row in active if row.split == "test" and row.qc_tag == "clean"),
        key=lambda row: row.utt_id,
    )
    review = sorted(
        (row for row in active if row.qc_tag == "review"),
        key=lambda row: row.utt_id,
    )
    exclude = [row for row in active if row.qc_tag == "exclude"]

    if not clean_train:
        raise ValueError("merged manifest has no clean train rows")
    if not clean_test:
        raise ValueError("merged manifest has no clean test rows")
    if not clean_val:
        raise ValueError("merged manifest has no clean val rows")

    bad_train = [
        row.utt_id
        for row in clean_train
        if row.qc_tag in {"exclude", "duplicate"}
    ]
    if bad_train:
        raise AssertionError(
            f"trackn_train contains exclude/duplicate: {bad_train[:5]}"
        )

    n_active = len(active)
    n_sets = len(clean_train) + len(clean_val) + len(clean_test) + len(review)
    n_remainder_exclude = len(exclude)
    if n_sets + n_remainder_exclude != n_active:
        other = Counter(
            str(row.qc_tag)
            for row in active
            if row.qc_tag not in {"clean", "review", "exclude"}
        )
        raise AssertionError(
            f"accounting mismatch: sets={n_sets} + exclude_remainder="
            f"{n_remainder_exclude} != active={n_active}; "
            f"other_tags={dict(other)}"
        )

    accounting = {
        "n_active": n_active,
        "n_trackn_train": len(clean_train),
        "n_trackn_val": len(clean_val),
        "n_trackn_test": len(clean_test),
        "n_trackn_review": len(review),
        "n_remainder_exclude": n_remainder_exclude,
        "hours_remainder_exclude": sum(row.dur_s for row in exclude) / 3600.0,
        "identity": (
            "n_trackn_train + n_trackn_val + n_trackn_test + n_trackn_review "
            "+ n_remainder_exclude == n_active"
        ),
    }

    manifest_sha256 = _sha256(manifest)
    result: dict[str, Any] = {
        "schema_version": 1,
        "manifest": str(manifest),
        "manifest_sha256": manifest_sha256,
        "version": "2.0.0",
        "accounting": accounting,
        "sets": {},
    }
    report_lines = [
        "# Track N frozen eval sets",
        f"manifest={manifest}",
        f"manifest_sha256={manifest_sha256}",
        "version=2.0.0",
        (
            f"accounting: active={n_active} train={len(clean_train)} "
            f"val={len(clean_val)} test={len(clean_test)} "
            f"review={len(review)} remainder_exclude={n_remainder_exclude}"
        ),
        (
            f"hours_remainder_exclude="
            f"{accounting['hours_remainder_exclude']:.9f}"
        ),
    ]

    specs = (
        (
            "trackn_train",
            "Frozen Track N train inventory (clean only); speaker-disjoint",
            CLEAN_TRAIN_SELECTION,
            clean_train,
        ),
        (
            "trackn_val",
            "Frozen Track N val inventory (clean only); speaker-disjoint",
            CLEAN_VAL_SELECTION,
            clean_val,
        ),
        (
            "trackn_test",
            "Frozen Track N test inventory (clean only); speaker-disjoint",
            CLEAN_TEST_SELECTION,
            clean_test,
        ),
        (
            "trackn_review",
            "Frozen Track N review-tag inventory (held out of train/val/test)",
            REVIEW_SELECTION,
            review,
        ),
    )
    for set_id, description, selection, selected in specs:
        record = _write_set(
            output=output,
            set_id=set_id,
            description=description,
            selection=selection,
            selected=selected,
            manifest=manifest,
            manifest_sha256=manifest_sha256,
            accounting=accounting,
        )
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

    train_speakers = {row.effective_speaker_id() for row in clean_train}
    val_speakers = {row.effective_speaker_id() for row in clean_val}
    test_speakers = {row.effective_speaker_id() for row in clean_test}
    overlap = sorted(
        (train_speakers & val_speakers)
        | (train_speakers & test_speakers)
        | (val_speakers & test_speakers)
    )
    if overlap:
        raise AssertionError(
            f"speaker overlap between Track N train/val/test: {overlap[:20]}"
        )
    result["speaker_overlap"] = overlap
    report_lines.extend(["", f"speaker_overlap={overlap}"])

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

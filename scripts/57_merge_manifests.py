#!/usr/bin/env python3
"""Merge manifests, mark cross-corpus duplicates, and hold out speakers."""

from __future__ import annotations

import argparse
import json
import sys
from typing import Sequence

from minispeech.data.merge import merge_manifests, write_merged_outputs


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        action="append",
        required=True,
        help="Input JSONL; repeat for each corpus",
    )
    parser.add_argument(
        "--priority",
        action="append",
        default=None,
        help="Corpus id from highest to lowest priority; repeat in order",
    )
    parser.add_argument("--out", required=True)
    parser.add_argument("--report", default="out/merge_dedup_report.json")
    parser.add_argument("--compute-missing-audio", action="store_true")
    parser.add_argument(
        "--split-strategy",
        choices=("speaker", "preserve"),
        default="speaker",
    )
    parser.add_argument("--train", type=float, default=0.90)
    parser.add_argument("--val", type=float, default=0.05)
    parser.add_argument("--test", type=float, default=0.05)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        rows, duplicates, card = merge_manifests(
            args.manifest,
            compute_missing_audio=bool(args.compute_missing_audio),
            split_strategy=args.split_strategy,
            priority=args.priority,
            train=args.train,
            val=args.val,
            test=args.test,
        )
        paths = write_merged_outputs(
            args.out,
            rows,
            duplicates,
            card,
            duplicate_report_path=args.report,
        )
    except (AssertionError, FileNotFoundError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(
        json.dumps(
            {
                "ok": True,
                **paths,
                "n_active": card["n"],
                "n_manifest": card["n_manifest"],
                "hours": card["hours"],
                "speakers": card["speakers"],
                "n_marked_duplicate": duplicates["n_marked_duplicate"],
                "active_audio_fingerprints_unique": duplicates[
                    "active_audio_fingerprints_unique"
                ],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

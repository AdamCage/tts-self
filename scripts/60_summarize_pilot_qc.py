#!/usr/bin/env python3
"""Summarize the frozen-threshold pilot ASR/DSP QC run."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from minispeech.data.manifest import assert_no_speaker_leak, read_manifest
from minispeech.utils import expand, load_yaml


def summarize(
    manifest_path: str | Path,
    *,
    qc_path: str | Path | None = None,
    clean_max: float = 0.15,
    review_max: float = 0.25,
) -> dict[str, Any]:
    rows = read_manifest(manifest_path)
    assert_no_speaker_leak(rows)
    by_tag: dict[str, dict[str, float | int]] = defaultdict(
        lambda: {"n": 0, "hours": 0.0}
    )
    for row in rows:
        tag = row.qc_tag or "untagged"
        by_tag[tag]["n"] = int(by_tag[tag]["n"]) + 1
        by_tag[tag]["hours"] = float(by_tag[tag]["hours"]) + row.dur_s / 3600.0

    durations = np.asarray([row.dur_s for row in rows], dtype=np.float64)
    duration_percentiles = {
        f"p{p}": float(np.percentile(durations, p)) if durations.size else None
        for p in (0, 25, 50, 75, 90, 95, 99, 100)
    }
    sidecar = {"path": str(qc_path) if qc_path else None, "n": 0, "n_errors": 0}
    if qc_path is not None:
        with expand(qc_path).open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                sidecar["n"] += 1
                if "error" in json.loads(line):
                    sidecar["n_errors"] += 1

    clean_hours = float(by_tag.get("clean", {}).get("hours", 0.0))
    report: dict[str, Any] = {
        "schema_version": 1,
        "manifest": str(expand(manifest_path)),
        "thresholds": {
            "clean_max_cer": clean_max,
            "review_max_cer": review_max,
            "exclude_above_cer": review_max,
        },
        "n": len(rows),
        "hours": sum(row.dur_s for row in rows) / 3600.0,
        "speakers": len({row.effective_speaker_id() for row in rows}),
        "speaker_overlap": [],
        "qc_tags": dict(sorted(by_tag.items())),
        "clean_hours": clean_hours,
        "clean_hours_gate_50": clean_hours >= 50.0,
        "duration_seconds": duration_percentiles,
        "qc_sidecar": sidecar,
    }
    if clean_hours < 50.0:
        report["shortfall_reason"] = (
            "Frozen ASR CER thresholds retained fewer than 50 clean hours; "
            "thresholds were not relaxed."
        )
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--qc", required=True)
    parser.add_argument("--out", default="out/pilot_qc.json")
    parser.add_argument("--clean-max", type=float, default=0.15)
    parser.add_argument("--review-max", type=float, default=0.25)
    args = parser.parse_args(argv)

    data = load_yaml(args.config)["data"]
    report = summarize(
        data["manifest"],
        qc_path=args.qc,
        clean_max=args.clean_max,
        review_max=args.review_max,
    )
    output = expand(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

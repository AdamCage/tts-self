#!/usr/bin/env python3
"""Apply QC tags to manifest from corpus_qc.jsonl (non-destructive; no deletes)."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

from minispeech.data.manifest import read_manifest, write_manifest
from minispeech.utils import expand, load_yaml


def tag_for_cer(cer: float, clean_max: float, review_max: float) -> str:
    if cer <= clean_max:
        return "clean"
    if cer <= review_max:
        return "review"
    return "exclude"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/data_ruslan.yaml")
    ap.add_argument("--qc", default=None, help="corpus_qc.jsonl path")
    ap.add_argument("--clean-max", type=float, default=0.15)
    ap.add_argument("--review-max", type=float, default=0.25)
    ap.add_argument("--review-weight", type=float, default=0.5)
    ap.add_argument("--report", default="docs/corpus_qc.md")
    args = ap.parse_args()
    cfg = load_yaml(args.config)["data"]
    rows = read_manifest(cfg["manifest"])
    qc_path = expand(
        args.qc
        or str(Path(cfg.get("ruslan_dir", "~/datasets/tts/ruslan")).expanduser() / "corpus_qc.jsonl")
    )
    by_id: dict[str, dict] = {}
    with open(qc_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if "utt_id" in obj and "asr_cer" in obj:
                by_id[obj["utt_id"]] = obj

    tags = Counter()
    cers = []
    for r in rows:
        q = by_id.get(r.utt_id)
        if q is None:
            r.qc_tag = r.qc_tag or "review"
            tags[r.qc_tag] += 1
            continue
        cer = float(q["asr_cer"])
        r.asr_cer = cer
        r.asr_wer = float(q.get("asr_wer")) if q.get("asr_wer") is not None else None
        r.qc_tag = tag_for_cer(cer, args.clean_max, args.review_max)
        if r.qc_tag == "review":
            r.sample_weight = args.review_weight
        elif r.qc_tag == "clean":
            r.sample_weight = 1.0
        else:
            r.sample_weight = 0.0
        tags[r.qc_tag] += 1
        cers.append(cer)

    write_manifest(cfg["manifest"], rows)

    report_path = expand(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    pct = {}
    if cers:
        arr = np.asarray(cers)
        for p in (50, 75, 90, 95, 99):
            pct[f"p{p}"] = float(np.percentile(arr, p))
    worst = sorted(by_id.values(), key=lambda x: x.get("asr_cer", 1.0), reverse=True)[:50]
    md = [
        "# Corpus QC",
        "",
        f"Source: `{qc_path}`",
        f"Thresholds: clean≤{args.clean_max}, review≤{args.review_max}, else exclude",
        "",
        "## Tag counts",
        "",
        "```json",
        json.dumps(dict(tags), indent=2),
        "```",
        "",
        "## CER percentiles",
        "",
        "```json",
        json.dumps(pct, indent=2),
        "```",
        "",
        "## Top-50 worst CER",
        "",
    ]
    for w in worst:
        md.append(f"- `{w.get('utt_id')}` CER={w.get('asr_cer'):.3f} `{w.get('wav', '')}`")
    report_path.write_text("\n".join(md), encoding="utf-8")

    # update meta if present
    meta_path = expand(cfg["manifest"]).with_suffix(".meta.json")
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        meta["qc_tags"] = dict(tags)
        meta["qc_thresholds"] = {"clean_max": args.clean_max, "review_max": args.review_max}
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(json.dumps({"tags": dict(tags), "report": str(report_path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Extract a subset of wavs from RUSLAN tar for Phase 0/1."""

from __future__ import annotations

import argparse
import csv
import tarfile
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ruslan-dir", default="~/datasets/tts/ruslan")
    ap.add_argument("--max-wavs", type=int, default=400)
    args = ap.parse_args()
    root = Path(args.ruslan_dir).expanduser().resolve()
    tar_path = root / "wavs_part_001.tar"
    meta_path = root / "metadata.csv"
    wav_dir = root / "wavs"
    wav_dir.mkdir(parents=True, exist_ok=True)

    if not tar_path.exists():
        print(f"Missing {tar_path}")
        return 1

    # metadata: typically id|text or filename|text
    texts: dict[str, str] = {}
    if meta_path.exists():
        with open(meta_path, encoding="utf-8", errors="ignore") as f:
            # try csv then pipe
            sample = f.read(4096)
            f.seek(0)
            delim = "|" if sample.count("|") > sample.count(",") else ","
            reader = csv.reader(f, delimiter=delim)
            for row in reader:
                if not row or row[0].lower() in {"id", "file", "filename", "wav"}:
                    continue
                key = Path(row[0].strip()).stem
                text = row[-1].strip()
                texts[key] = text

    n = 0
    with tarfile.open(tar_path, "r") as tf:
        for m in tf:
            if not m.isfile() or not m.name.lower().endswith(".wav"):
                continue
            name = Path(m.name).name
            target = wav_dir / name
            if not target.exists():
                src = tf.extractfile(m)
                if src is None:
                    continue
                target.write_bytes(src.read())
            stem = target.stem
            txt = texts.get(stem, "")
            if txt:
                target.with_suffix(".txt").write_text(txt, encoding="utf-8")
            n += 1
            if n >= args.max_wavs:
                break
    print(f"Extracted {n} wavs to {wav_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

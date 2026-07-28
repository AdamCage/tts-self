#!/usr/bin/env python3
"""Idempotent extract of RUSLAN wavs from wavs_part_*.tar."""

from __future__ import annotations

import argparse
import csv
import tarfile
from pathlib import Path


def load_texts(meta_path: Path) -> dict[str, str]:
    texts: dict[str, str] = {}
    if not meta_path.exists():
        return texts
    with open(meta_path, encoding="utf-8", errors="ignore") as f:
        sample = f.read(4096)
        f.seek(0)
        delim = "|" if sample.count("|") > sample.count(",") else ","
        reader = csv.reader(f, delimiter=delim)
        for row in reader:
            if not row or row[0].lower() in {"id", "file", "filename", "wav"}:
                continue
            key = Path(row[0].strip()).stem
            texts[key] = row[-1].strip()
    return texts


def extract_tar(
    tar_path: Path,
    wav_dir: Path,
    texts: dict[str, str],
    *,
    max_wavs: int | None,
) -> dict[str, int]:
    stats = {"extracted": 0, "skipped_exists": 0, "no_text": 0, "processed": 0}
    with tarfile.open(tar_path, "r") as tf:
        for m in tf:
            if not m.isfile() or not m.name.lower().endswith(".wav"):
                continue
            name = Path(m.name).name
            target = wav_dir / name
            if target.exists():
                stats["skipped_exists"] += 1
            else:
                src = tf.extractfile(m)
                if src is None:
                    continue
                target.write_bytes(src.read())
                stats["extracted"] += 1
            stem = target.stem
            txt = texts.get(stem, "")
            if txt:
                target.with_suffix(".txt").write_text(txt, encoding="utf-8")
            else:
                stats["no_text"] += 1
            stats["processed"] += 1
            if max_wavs is not None and stats["processed"] >= max_wavs:
                break
    return stats


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ruslan-dir", default="~/datasets/tts/ruslan")
    ap.add_argument("--max-wavs", type=int, default=None, help="Limit per tar (ignored with --all)")
    ap.add_argument("--all", action="store_true", help="Extract all wavs_part_*.tar idempotently")
    args = ap.parse_args()
    root = Path(args.ruslan_dir).expanduser().resolve()
    meta_path = root / "metadata.csv"
    wav_dir = root / "wavs"
    wav_dir.mkdir(parents=True, exist_ok=True)
    texts = load_texts(meta_path)

    if args.all:
        tars = sorted(root.glob("wavs_part_*.tar"))
        max_wavs = None
    else:
        tars = [root / "wavs_part_001.tar"]
        max_wavs = args.max_wavs if args.max_wavs is not None else 400

    if not tars or not any(t.exists() for t in tars):
        print(f"No tar archives under {root}")
        return 1

    total = {"extracted": 0, "skipped_exists": 0, "no_text": 0, "processed": 0}
    for tar_path in tars:
        if not tar_path.exists():
            print(f"Missing {tar_path}")
            continue
        print(f"Extracting {tar_path.name} ...")
        st = extract_tar(tar_path, wav_dir, texts, max_wavs=max_wavs)
        for k, v in st.items():
            total[k] += v
        print(f"  {tar_path.name}: {st}")

    print(f"Total: {total} → {wav_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

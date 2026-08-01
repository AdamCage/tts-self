#!/usr/bin/env python3
"""Download the pinned P1 pilot corpus and record reproducibility metadata."""

from __future__ import annotations

import argparse
import hashlib
import json
import shlex
import shutil
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Sequence

DEFAULT_REPO = "fixie-ai/common_voice_17_0"
DEFAULT_REVISION = "34f78a43893414e7b6e271ba94c1d5e05f18b239"


def _sha256(path: Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _disk(path: Path) -> dict[str, int]:
    usage = shutil.disk_usage(path)
    return {
        "total_bytes": usage.total,
        "used_bytes": usage.used,
        "free_bytes": usage.free,
    }


def build_report(
    destination: Path,
    *,
    repo: str,
    revision: str,
    include: str,
    command: Sequence[str],
    disk_before: dict[str, int],
    disk_after: dict[str, int],
) -> dict[str, object]:
    files = sorted(path for path in destination.rglob("*") if path.is_file())
    artifacts = [
        {
            "path": path.relative_to(destination).as_posix(),
            "size_bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }
        for path in files
        if ".cache" not in path.parts
    ]
    return {
        "schema_version": 1,
        "corpus": "common_voice_ru_17_hf_mirror",
        "url": f"https://huggingface.co/datasets/{repo}",
        "repo": repo,
        "revision": revision,
        "license": "CC0-1.0",
        "gated": False,
        "downloaded_at": datetime.now(UTC).isoformat(),
        "selection_rule": include,
        "destination": str(destination),
        "command": shlex.join(command),
        "n_files": len(artifacts),
        "size_bytes": sum(int(item["size_bytes"]) for item in artifacts),
        "files": artifacts,
        "disk_before": disk_before,
        "disk_after": disk_after,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default=DEFAULT_REPO)
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument("--include", default="ru/*.parquet")
    parser.add_argument(
        "--destination",
        default="~/datasets/tts/common_voice_ru17/source",
    )
    parser.add_argument("--report", default="out/pilot_download.json")
    parser.add_argument(
        "--record-only",
        action="store_true",
        help="Hash an already-downloaded pinned corpus without contacting the Hub",
    )
    args = parser.parse_args(argv)

    destination = Path(args.destination).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    disk_before = _disk(Path.home())
    command = [
        "hf",
        "download",
        args.repo,
        "--type",
        "dataset",
        "--revision",
        args.revision,
        "--include",
        args.include,
        "--local-dir",
        str(destination),
    ]
    if not args.record_only:
        subprocess.run(command, check=True)
    disk_after = _disk(Path.home())

    report = build_report(
        destination,
        repo=args.repo,
        revision=args.revision,
        include=args.include,
        command=command,
        disk_before=disk_before,
        disk_after=disk_after,
    )
    if not report["files"]:
        raise FileNotFoundError(f"no downloaded files under {destination}")
    report_path = Path(args.report).expanduser()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

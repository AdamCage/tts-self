#!/usr/bin/env python3
"""Ingest an audited corpus into the multi-corpus JSONL manifest schema."""

from __future__ import annotations

import argparse
import json
import sys
from typing import Sequence

from minispeech.data.ingest import build_manifest_from_config


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="configs/data_<corpus>.yaml")
    args = parser.parse_args(argv)
    try:
        result = build_manifest_from_config(args.config)
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

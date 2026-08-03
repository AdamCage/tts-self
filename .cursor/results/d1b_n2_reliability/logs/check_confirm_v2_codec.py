"""Fail-closed: every confirm32_v2 utt must exist in full codec token indexes."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path("/mnt/c/projects/tts-self")
confirm = json.loads((ROOT / "configs/train_subsets/d1b_confirm32_v2.json").read_text())
ids = [str(x) for x in confirm["utt_ids"]]
indexes = [
    Path.home() / "datasets/tts/ruslan/codec_tokens/codec_tokens_index.json",
    Path.home() / "datasets/tts/common_voice_ru17/codec_tokens/codec_tokens_index.json",
]
merged: dict = {}
for p in indexes:
    if not p.is_file():
        print(f"MISSING index: {p}", file=sys.stderr)
        sys.exit(2)
    merged.update(json.loads(p.read_text()))

missing = [u for u in ids if u not in merged]
print(
    json.dumps(
        {
            "n": len(ids),
            "found": len(ids) - len(missing),
            "missing_n": len(missing),
            "missing_sample": missing[:5],
            "items_sha256": confirm["items_sha256"],
            "select8_sha": confirm["split_hashes"]["select8"],
        },
        indent=2,
    )
)
sys.exit(1 if missing else 0)

#!/usr/bin/env python3
"""Compute items_sha256 for d1c_smoke_1utt.json (run once before commit)."""
from __future__ import annotations

import json
from pathlib import Path

from minispeech.data.discrete_oracles import items_sha256

path = Path("configs/train_subsets/d1c_smoke_1utt.json")
obj = json.loads(path.read_text(encoding="utf-8"))
digest = items_sha256(obj["utt_ids"])
obj["items_sha256"] = digest
path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
print(digest)

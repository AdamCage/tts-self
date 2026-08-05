#!/usr/bin/env python3
"""Debug D1c eligible pool construction."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path("/mnt/c/projects/tts-self-d1c")
sys.path.insert(0, str(ROOT / "src"))

from minispeech.data.manifest import read_manifest
from minispeech.utils import expand

trackn_train = json.loads((ROOT / "configs/eval_sets/trackn_train.json").read_text())["utt_ids"]
print("n_train", len(trackn_train))
print("sample_train", trackn_train[:3])

ruslan_idx = json.loads(expand("~/datasets/tts/ruslan/codec_tokens/codec_tokens_index.json").read_text())
cv_idx = json.loads(expand("~/datasets/tts/common_voice_ru17/codec_tokens/codec_tokens_index.json").read_text())
print("n_ruslan_codec", len(ruslan_idx), "sample", next(iter(ruslan_idx)))
print("n_cv_codec", len(cv_idx), "sample", next(iter(cv_idx)))

train_set = set(trackn_train)
codec_ids = set(ruslan_idx) | set(cv_idx)
print("intersection", len(train_set & codec_ids))
print("train_not_in_codec", len(train_set - codec_ids))
print("sample_missing", list(train_set - codec_ids)[:5])

# Check ID format mismatch
ruslan_train = [u for u in trackn_train if u.startswith("ruslan_")][:5]
print("ruslan_train_samples", ruslan_train)
print("in_codec", [u in ruslan_idx for u in ruslan_train])
# maybe suffix difference
for u in ruslan_train:
    hits = [k for k in ruslan_idx if k.startswith(u) or u.startswith(k)][:3]
    print("fuzzy", u, hits)

rows = {r.utt_id: r for r in read_manifest(expand("~/datasets/tts/multicorpus/manifest.jsonl"))}
print("manifest_n", len(rows))
u0 = next(iter(train_set & codec_ids), None)
print("example_intersect", u0)
if u0:
    r = rows.get(u0)
    print("row attrs", [a for a in dir(r) if not a.startswith("_")][:40] if r else None)
    if r:
        for k in ("text", "text_norm", "raw_text", "transcript", "qc_tag"):
            print(k, getattr(r, k, "MISSING"))

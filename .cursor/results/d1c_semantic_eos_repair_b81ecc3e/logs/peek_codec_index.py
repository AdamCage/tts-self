#!/usr/bin/env python3
import json
from pathlib import Path

for p in [
    Path("/home/adam/datasets/tts/ruslan/codec_tokens/codec_tokens_index.json"),
    Path("/home/adam/datasets/tts/common_voice_ru17/codec_tokens/codec_tokens_index.json"),
]:
    print(p, "exists", p.exists())
    if not p.exists():
        continue
    data = json.loads(p.read_text(encoding="utf-8"))
    print("type", type(data).__name__, "len", len(data))
    k = next(iter(data))
    v = data[k]
    print("sample_key", k)
    if isinstance(v, dict):
        print("sample_keys", sorted(v.keys()))
        print({kk: v[kk] for kk in list(v)[:15]})
    else:
        print("sample_val", v)

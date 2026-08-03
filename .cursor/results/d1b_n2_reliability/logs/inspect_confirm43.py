#!/usr/bin/env python3
import json
from pathlib import Path

met = Path("/mnt/c/projects/tts-self/out/d1b_n2_reliability/metrics")
for name in (
    "confirm_seed43_selection.json",
    "confirm_seed43_gate24_n2_gate.json",
    "confirm_seed43_full32_n2_gate.json",
):
    p = met / name
    print("===", name, "exists=", p.exists())
    if not p.exists():
        continue
    obj = json.loads(p.read_text(encoding="utf-8"))
    if "selected" in obj:
        print(json.dumps(obj["selected"], indent=2))
    else:
        c = obj.get("criteria") or {}
        print("verdict", obj.get("verdict"))
        print(
            "macro_cer", c.get("macro_cer"),
            "ceiling", c.get("ceiling_copy_plus_margin"),
            "n_cat", c.get("n_catastrophic"),
            "zero_cat", c.get("zero_catastrophic"),
            "cer_pass", c.get("cer_pass"),
            "text_usage", (c.get("text_usage_d1b") or {}).get("pass"),
            "stability", (c.get("stability_d1b") or {}).get("pass"),
        )

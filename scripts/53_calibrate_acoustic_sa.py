#!/usr/bin/env python3
"""Calibrate acoustic stress detector on real RUSLAN (Phase M / M05)."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from minispeech.data.manifest import filter_split, read_manifest
from minispeech.eval.stress_acoustic import calibrate_weights_on_manifest
from minispeech.frontend.stress import accentuate
from minispeech.utils import expand, load_yaml


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default="configs/data_ruslan_local.yaml")
    ap.add_argument("--split", default="val")
    ap.add_argument("--max-items", type=int, default=120)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="out/m05_acoustic_sa_calibration.json")
    args = ap.parse_args()

    dcfg = load_yaml(expand(args.data)).get("data", {})
    rows = filter_split(read_manifest(expand(dcfg["manifest"])), args.split)
    rng = random.Random(args.seed)
    rows = list(rows)
    rng.shuffle(rows)
    items: list[tuple[str, str]] = []
    for u in rows:
        if not u.wav or not Path(u.wav).is_file():
            continue
        text = (u.text_stressed or u.text_norm or u.text_raw or "").strip()
        if not text:
            continue
        stressed = text if "\u0301" in text else accentuate(text)
        if "\u0301" not in stressed:
            continue
        items.append((str(u.wav), stressed))
        if len(items) >= args.max_items:
            break

    weights, report = calibrate_weights_on_manifest(
        items, device=args.device, max_items=args.max_items
    )
    payload = {
        "phase": "M05",
        "n_items_available": len(items),
        "split": args.split,
        "seed": args.seed,
        "weights": {
            "w_dur": weights.w_dur,
            "w_energy": weights.w_energy,
            "w_f0": weights.w_f0,
        },
        **report,
    }
    out = expand(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False))
    return 0 if report.get("pass_ge_0_90") else 2


if __name__ == "__main__":
    raise SystemExit(main())

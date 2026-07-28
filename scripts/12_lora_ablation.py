#!/usr/bin/env python3
"""Phase 5 LoRA ablation scaffold (A frozen / B last-4 / C layers 12-23)."""

from __future__ import annotations

import argparse
import json

from minispeech.utils import expand


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=["A", "B", "C"], required=True)
    ap.add_argument("--out", default="runs/lora_ablation")
    args = ap.parse_args()
    # Full LoRA training requires text hidden cache invalidation and Qwen in the loop.
    # This script records the experiment plan; wire PEFT when Phase 5 starts for real.
    plan = {
        "A": "Qwen frozen (text hidden cache) — baseline",
        "B": "LoRA on blocks around full-attn layers 19,23 + FFN, r=8-16",
        "C": "LoRA on layers 12-23, r=8-16",
        "keep_rule": "Keep LoRA only if WER/stress/prosody improve enough to justify train+TTFB cost",
    }
    out = expand(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"variant_{args.variant}.json").write_text(
        json.dumps({"variant": args.variant, "plan": plan}, indent=2), encoding="utf-8"
    )
    print(json.dumps(plan[args.variant], ensure_ascii=False))
    print("LoRA ablation scaffold written. Run after Phase 4 with live Qwen train loop.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

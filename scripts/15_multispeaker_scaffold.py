#!/usr/bin/env python3
"""Phase 6 multi-speaker scaffold (Dialogs + reference conditioning)."""

from __future__ import annotations

import argparse
import json

from minispeech.utils import expand


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="runs/multispeaker_scaffold")
    args = ap.parse_args()
    plan = {
        "data": "langswap/dialogs-ru-emotional-conversations",
        "speaker": "reference latents as cross-attn memory (VibeVoice-style) + optional ECAPA/WavLM",
        "emotion": "12 Dialogs labels as embeddings",
        "phoneme_channel": "enable frontend/g2p.py as second memory",
        "metrics": ["SIM(WavLM)", "WER", "emotion-cls", "A/B preference"],
        "sim_floor": "0.4-0.5 on unseen is realistic at this data scale",
    }
    out = expand(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "plan.json").write_text(json.dumps(plan, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(plan, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

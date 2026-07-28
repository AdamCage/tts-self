#!/usr/bin/env python3
"""Scaffold / enable non-AR Talker fallback (Phase 2 escape hatch)."""

from __future__ import annotations

import argparse
import json

import torch

from minispeech.infer.generate import load_checkpoint
from minispeech.utils import expand


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, ckpt = load_checkpoint(args.ckpt, device)
    model.enable_nonar_fallback()
    assert model.nonar is not None
    out = expand(args.out or (expand(args.ckpt).parent / "nonar_fallback.pt"))
    torch.save({"model": model.state_dict(), "cfg": ckpt.get("cfg", {}), "mode": "nonar"}, out)
    meta = {
        "switch_condition": "S3 free-running L1 > 3x teacher-forced AND speech collapses after ~3s after input-noise+scheduled-sampling",
        "path": str(out),
    }
    print(json.dumps(meta, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

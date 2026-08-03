#!/usr/bin/env python3
"""Inspect confirm_v2 seed42 checkpoint integrity."""
from __future__ import annotations

from pathlib import Path

import torch

run = Path.home() / "projects/tts-self/runs/discrete_speech_lm_d1b_confirm_v2_seed42"
names = [
    "S1_last.pt",
    "last.pt",
    "candidate_step_0500.pt",
    "candidate_step_0625.pt",
    "candidate_step_0750.pt",
]
for name in names:
    p = run / name
    if not p.exists():
        print(f"{name}: MISSING")
        continue
    sz = p.stat().st_size
    try:
        ck = torch.load(p, map_location="cpu", weights_only=False)
        step = ck.get("optimizer_step") or ck.get("step")
        m = ck.get("metrics") or {}
        print(
            f"{name}: size_gb={sz / 1e9:.3f} step={step} "
            f"loss={m.get('loss')} has_optim={'optimizer' in ck} "
            f"model_keys={len(ck.get('model') or ck.get('state_dict') or {})}"
        )
    except Exception as e:
        print(f"{name}: size_gb={sz / 1e9:.3f} LOAD_FAIL {type(e).__name__}: {e}")

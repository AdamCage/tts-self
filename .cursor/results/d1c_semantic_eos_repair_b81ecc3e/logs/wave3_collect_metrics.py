#!/usr/bin/env python3
"""Collect Wave 3 smoke metrics from Linux clone artifacts."""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import torch

ROOT = Path.home() / "projects" / "tts-self"
PLAN = "d1c_semantic_eos_repair_b81ecc3e"
LISTEN = ROOT / "out" / PLAN / "smoke_1utt" / "listen_demo"
RUN = ROOT / "runs" / "d1c_eos_smoke_1utt"
EXT = Path.home() / "datasets" / "tts" / "d1c_wave3_logs"
RESULTS = ROOT / ".cursor" / "results" / PLAN
WIN_D1C = Path("/mnt/c/projects/tts-self-d1c")
WIN_AUTH = Path("/mnt/c/projects/tts-self")

codes_path = LISTEN / "0000_ruslan_000842_RUSLAN_codes.pt"
payload = torch.load(codes_path, map_location="cpu", weights_only=False)
gen = payload.get("generation") or {}
codes = payload["codes"]
man = json.loads((RUN / "run_manifest.json").read_text(encoding="utf-8"))
infer = json.loads((LISTEN / "infer_index.json").read_text(encoding="utf-8"))
probe = {
    "tip": man.get("git_sha") or man.get("git_commit") or man.get("git"),
    "run_name": "d1c_eos_smoke_1utt",
    "ckpt": str(RUN / "last.pt"),
    "codes_shape": list(codes.shape),
    "codes_max": int(codes.max()),
    "generation": gen,
    "termination_reason": (infer[0].get("termination_reason") if infer else None),
    "n_frames": (infer[0].get("n_frames") if infer else None),
    "wav": (infer[0].get("wav") if infer else None),
    "train_last": None,
}
# last metrics from console if present
console = (EXT / "wave3_gpu_console.log").read_text(encoding="utf-8", errors="replace")
for line in console.splitlines():
    if line.startswith("stage=S1 done last="):
        probe["train_last_line"] = line
        break

out_path = EXT / "wave3_1utt_metrics.json"
out_path.write_text(json.dumps(probe, indent=2, default=str) + "\n", encoding="utf-8")
print(json.dumps(probe, indent=2, default=str))

# mirror logs + wavs
for dest_root in (RESULTS, WIN_D1C / ".cursor" / "results" / PLAN, WIN_AUTH / ".cursor" / "results" / PLAN):
    logdir = dest_root / "logs"
    logdir.mkdir(parents=True, exist_ok=True)
    for p in EXT.glob("wave3_*"):
        shutil.copy2(p, logdir / p.name)

for dest in (
    WIN_AUTH / "out" / PLAN / "smoke_1utt" / "listen_demo",
    WIN_D1C / "out" / PLAN / "smoke_1utt" / "listen_demo",
):
    dest.mkdir(parents=True, exist_ok=True)
    for p in LISTEN.glob("*"):
        if p.is_file():
            shutil.copy2(p, dest / p.name)
    print("mirrored_to", dest)

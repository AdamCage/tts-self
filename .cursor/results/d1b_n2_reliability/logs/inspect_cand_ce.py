import torch
from pathlib import Path

for seed in (42, 43):
    run = Path.home() / f"projects/tts-self/runs/discrete_speech_lm_d1b_confirm_seed{seed}"
    print("===", seed, "===")
    for p in sorted(run.glob("candidate_step_*.pt")):
        ck = torch.load(p, map_location="cpu", weights_only=False)
        m = ck.get("metrics") or {}
        print(
            p.name,
            "opt",
            ck.get("optimizer_step"),
            "loss",
            m.get("loss"),
            "ce_cb0",
            m.get("ce_cb0"),
        )

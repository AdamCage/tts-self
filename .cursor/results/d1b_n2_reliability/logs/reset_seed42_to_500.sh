#!/usr/bin/env bash
set -euo pipefail
tmux kill-session -t d1b_v2 2>/dev/null || true
sleep 1
RUN=$HOME/projects/tts-self/runs/discrete_speech_lm_d1b_confirm_v2_seed42
ls -la "$RUN"/candidate_step_*.pt
rm -f "$RUN/S1_last.pt" "$RUN/last.pt" "$RUN/candidate_step_0625.pt" "$RUN/candidate_step_0750.pt"
cp -f "$RUN/candidate_step_0500.pt" "$RUN/S1_last.pt"
cp -f "$RUN/candidate_step_0500.pt" "$RUN/last.pt"
ls -la "$RUN"/candidate_step_*.pt "$RUN/S1_last.pt"
source "$HOME/projects/tts-self/.venv/bin/activate"
python - <<'PY'
import torch
ck=torch.load("/home/adam/projects/tts-self/runs/discrete_speech_lm_d1b_confirm_v2_seed42/S1_last.pt", map_location="cpu", weights_only=False)
print("reset_step", ck.get("optimizer_step"), "loss", (ck.get("metrics") or {}).get("loss"))
PY

#!/usr/bin/env bash
set -euo pipefail
sed -i 's/\r$//' /mnt/c/projects/tts-self/.cursor/results/d1b_n2_reliability/logs/wave_a_fr_select.sh
cd ~/projects/tts-self
if [ -n "$(git status --porcelain)" ]; then
  echo STILL_DIRTY
  git status --porcelain
  exit 2
fi
tmux kill-session -t d1b_fr_a 2>/dev/null || true
tmux new-session -d -s d1b_fr_a "bash /mnt/c/projects/tts-self/.cursor/results/d1b_n2_reliability/logs/wave_a_fr_select.sh"
sleep 10
tmux capture-pane -t d1b_fr_a -p | tail -50

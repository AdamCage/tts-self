#!/usr/bin/env bash
set -euo pipefail
cd ~/projects/tts-self
if [ -n "$(git status --porcelain)" ]; then
  echo DIRTY
  git status --porcelain
  exit 2
fi
tmux kill-session -t d1b_listen_a 2>/dev/null || true
tmux new-session -d -s d1b_listen_a "bash /mnt/c/projects/tts-self/.cursor/results/d1b_n2_reliability/logs/wave_a_listen.sh; echo LISTEN_EXIT=\$?; bash /mnt/c/projects/tts-self/.cursor/results/d1b_n2_reliability/logs/wave_b_confirm42_train.sh; echo WAVE_B_EXIT=\$?"
sleep 15
tmux capture-pane -t d1b_listen_a -p | tail -40

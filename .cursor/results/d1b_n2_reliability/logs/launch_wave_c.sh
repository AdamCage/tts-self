#!/usr/bin/env bash
set -euo pipefail
sed -i 's/\r$//' /mnt/c/projects/tts-self/.cursor/results/d1b_n2_reliability/logs/wave_c_confirm43_train.sh
sed -i 's/\r$//' /mnt/c/projects/tts-self/.cursor/results/d1b_n2_reliability/logs/wave_c_fr_select.sh
cd ~/projects/tts-self
if [ -n "$(git status --porcelain)" ]; then
  echo DIRTY
  git status --porcelain
  exit 2
fi
tmux kill-session -t d1b_wave_c 2>/dev/null || true
tmux new-session -d -s d1b_wave_c "bash /mnt/c/projects/tts-self/.cursor/results/d1b_n2_reliability/logs/wave_c_confirm43_train.sh; echo TRAIN_C_EXIT=\$?; bash /mnt/c/projects/tts-self/.cursor/results/d1b_n2_reliability/logs/wave_c_fr_select.sh; echo FR_C_EXIT=\$?"
sleep 15
tmux capture-pane -t d1b_wave_c -p | tail -30

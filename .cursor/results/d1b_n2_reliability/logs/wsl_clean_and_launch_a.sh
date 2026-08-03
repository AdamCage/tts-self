#!/usr/bin/env bash
set -euo pipefail
cd ~/projects/tts-self
tmux kill-session -t d1b_dev42 2>/dev/null || true
rm -rf .cursor/results/d1b_n2_reliability
if [ -n "$(git status --porcelain)" ]; then
  echo "STILL_DIRTY:"
  git status --porcelain
  exit 2
fi
echo "CLEAN_OK $(git rev-parse --short HEAD)"
SCRIPT=/mnt/c/projects/tts-self/.cursor/results/d1b_n2_reliability/logs/wave_a_dev42_launch.sh
chmod +x "$SCRIPT"
tmux new-session -d -s d1b_dev42 "bash '$SCRIPT'"
sleep 5
tmux capture-pane -t d1b_dev42 -p | tail -60

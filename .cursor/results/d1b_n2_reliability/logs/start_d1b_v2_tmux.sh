#!/usr/bin/env bash
set -euo pipefail
LOGDIR=/mnt/c/projects/tts-self/.cursor/results/d1b_n2_reliability/logs
sed 's/\r$//' "$LOGDIR/launch_confirm_v2_series.sh" > /tmp/launch_confirm_v2_series.sh
chmod +x /tmp/launch_confirm_v2_series.sh
LOG=$LOGDIR/confirm_v2_series.log
tmux has-session -t d1b_v2 2>/dev/null && tmux kill-session -t d1b_v2 || true
tmux new-session -d -s d1b_v2 "bash /tmp/launch_confirm_v2_series.sh 2>&1 | tee '$LOG'"
sleep 3
tmux ls
echo "--- log ---"
tail -n 50 "$LOG" || true

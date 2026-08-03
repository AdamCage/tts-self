#!/usr/bin/env bash
set -euo pipefail
LOGDIR=/mnt/c/projects/tts-self/.cursor/results/d1b_n2_reliability/logs
sed 's/\r$//' "$LOGDIR/launch_confirm_v2_resume.sh" > /tmp/launch_confirm_v2_resume.sh
chmod +x /tmp/launch_confirm_v2_resume.sh
LOG=$LOGDIR/confirm_v2_series.log
# Append resume marker; keep prior log context.
echo "" >> "$LOG"
echo "=== RESUME after WSL shutdown $(date -Iseconds) ===" >> "$LOG"
tmux has-session -t d1b_v2 2>/dev/null && tmux kill-session -t d1b_v2 || true
tmux new-session -d -s d1b_v2 "bash /tmp/launch_confirm_v2_resume.sh 2>&1 | tee -a '$LOG'"
sleep 3
tmux ls
tail -n 40 "$LOG" || true

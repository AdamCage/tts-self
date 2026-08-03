#!/usr/bin/env bash
set -euo pipefail
LOGDIR=/mnt/c/projects/tts-self/.cursor/results/d1b_n2_reliability/logs
sed 's/\r$//' "$LOGDIR/launch_confirm_v2_linux_out.sh" > /tmp/launch_confirm_v2_linux_out.sh
chmod +x /tmp/launch_confirm_v2_linux_out.sh
# Prefer Linux-side log (C: nearly full); mirror tail to Windows.
LLOG=$HOME/projects/tts-self/out/d1b_n2_reliability/confirm_v2_series_linux.log
mkdir -p "$(dirname "$LLOG")"
WLOG=$LOGDIR/confirm_v2_series.log
echo "" >> "$WLOG"
echo "=== LINUX_OUT relaunch $(date -Iseconds) ===" >> "$WLOG"
df -h /mnt/c | awk 'NR==2{print "free_c="$4}' >> "$WLOG"
tmux has-session -t d1b_v2 2>/dev/null && tmux kill-session -t d1b_v2 || true
tmux new-session -d -s d1b_v2 "bash /tmp/launch_confirm_v2_linux_out.sh 2>&1 | tee '$LLOG' | tee -a '$WLOG'"
sleep 4
tmux ls
tail -n 30 "$LLOG" || true

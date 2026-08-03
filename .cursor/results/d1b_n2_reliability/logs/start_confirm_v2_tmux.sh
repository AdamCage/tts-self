#!/usr/bin/env bash
# Launch confirm_v2 series in tmux; logs on Linux + small mirror on Windows.
set -euo pipefail
SESSION=d1b_confirm_v2
WIN=/mnt/c/projects/tts-self
SCRIPT=$WIN/.cursor/results/d1b_n2_reliability/logs/launch_confirm_v2_linux_out.sh
LINUX_LOG=$HOME/projects/tts-self/out/d1b_n2_reliability/confirm_v2_series_linux.log
mkdir -p "$(dirname "$LINUX_LOG")"
chmod +x "$SCRIPT"
# Kill stale session if any.
tmux kill-session -t "$SESSION" 2>/dev/null || true
tmux new-session -d -s "$SESSION" "bash '$SCRIPT' 2>&1 | tee -a '$LINUX_LOG'; echo EXIT=\$? | tee -a '$LINUX_LOG'; bash"
echo "started tmux session=$SESSION log=$LINUX_LOG"
tmux ls

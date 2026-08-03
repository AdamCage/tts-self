#!/usr/bin/env bash
set -euo pipefail
echo "=== tmux ==="
tmux list-sessions 2>/dev/null || echo none
echo "=== pane ==="
tmux capture-pane -t d1b_fr_b -p 2>/dev/null | tail -25 || true
echo "=== artifacts ==="
ls /mnt/c/projects/tts-self/out/d1b_n2_reliability/confirm_seed42/ 2>/dev/null || true
ls /mnt/c/projects/tts-self/out/d1b_n2_reliability/metrics/confirm_seed42* 2>/dev/null || true
echo "=== log ==="
tail -15 /mnt/c/projects/tts-self/.cursor/results/d1b_n2_reliability/logs/wave_b_fr_select.log 2>/dev/null || true

#!/usr/bin/env bash
set -euo pipefail
echo "=== tmux ==="
tmux list-sessions 2>/dev/null || echo none
echo "=== pane ==="
tmux capture-pane -t d1b_wave_c -p 2>/dev/null | tail -30 || true
echo "=== run dir ==="
ls ~/projects/tts-self/runs/discrete_speech_lm_d1b_confirm_seed43 2>/dev/null | head || echo no_run
echo "=== out ==="
ls /mnt/c/projects/tts-self/out/d1b_n2_reliability/confirm_seed43 2>/dev/null | head || echo no_out
echo "=== metrics ==="
ls /mnt/c/projects/tts-self/out/d1b_n2_reliability/metrics/confirm_seed43* 2>/dev/null || true
tail -8 /mnt/c/projects/tts-self/.cursor/results/d1b_n2_reliability/logs/wave_c_confirm43_train.log 2>/dev/null | tr '\r' '\n' | tail -5
tail -8 /mnt/c/projects/tts-self/.cursor/results/d1b_n2_reliability/logs/wave_c_fr_select.log 2>/dev/null | tr '\r' '\n' | tail -5

#!/usr/bin/env bash
set -euo pipefail
echo "=== tmux ==="
tmux list-sessions 2>/dev/null || echo none
echo "=== pane ==="
tmux capture-pane -t d1b_listen_a -p 2>/dev/null | tail -35 || true
echo "=== listen wavs ==="
ls /mnt/c/projects/tts-self/out/d1b_n2_reliability/dev_seed42/listen_demo 2>/dev/null | head || true
echo "=== confirm run ==="
ls ~/projects/tts-self/runs/discrete_speech_lm_d1b_confirm_seed42 2>/dev/null | head || echo "no confirm dir yet"
echo "=== GPU ==="
nvidia-smi --query-compute-apps=pid,used_memory,process_name --format=csv 2>/dev/null || true
echo "=== log tails ==="
tail -5 /mnt/c/projects/tts-self/.cursor/results/d1b_n2_reliability/logs/wave_a_listen.log 2>/dev/null || true
tail -8 /mnt/c/projects/tts-self/.cursor/results/d1b_n2_reliability/logs/wave_b_confirm42_train.log 2>/dev/null || true

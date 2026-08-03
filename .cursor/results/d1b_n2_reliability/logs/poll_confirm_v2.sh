#!/usr/bin/env bash
set -euo pipefail
OUT=$HOME/projects/tts-self/out/d1b_n2_reliability
LOG=$OUT/confirm_v2_series_linux.log
echo "=== processes ==="
ps -ef | grep -E '74_eval|72_train|76_select|73_infer|launch_confirm' | grep -v grep || true
echo "=== seed42 outs ==="
ls -1 "$OUT/confirm_v2_seed42" 2>/dev/null || true
echo "=== seed42 gates ==="
find "$OUT/confirm_v2_seed42" -name n2_gate.json 2>/dev/null | sort || true
echo "=== seed43 ==="
ls -1 "$OUT/confirm_v2_seed43" 2>/dev/null || echo '(none yet)'
ls -lah ~/projects/tts-self/runs/discrete_speech_lm_d1b_confirm_v2_seed43 2>/dev/null | head -20 || true
echo "=== log markers ==="
tr '\r' '\n' < "$LOG" | grep -E '===|macro_cer|verdict|ERROR|Bus error|MISSING|TRUNCATED|finished|CLEAN_SHA|wrote candidate|tf_ce' | tail -50
echo "=== tmux ==="
tmux ls 2>/dev/null || true

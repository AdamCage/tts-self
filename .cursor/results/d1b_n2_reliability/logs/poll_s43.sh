#!/usr/bin/env bash
set -eu
LOG=$HOME/projects/tts-self/out/d1b_n2_reliability/confirm_v2_seed43_series.log
OUT=$HOME/projects/tts-self/out/d1b_n2_reliability/confirm_v2_seed43
echo "=== procs ==="
ps -ef | grep -E '72_train|74_eval|76_select|73_infer|run_seed43' | grep -v grep || true
echo "=== run dir ==="
ls -lah ~/projects/tts-self/runs/discrete_speech_lm_d1b_confirm_v2_seed43/ 2>/dev/null | head -25 || true
echo "=== outs ==="
ls -1 "$OUT" 2>/dev/null || echo none
echo "=== gates ==="
find "$OUT" -name n2_gate.json 2>/dev/null | sort || true
echo "=== log markers ==="
grep -E '===|CLEAN_SHA|ERROR|Bus error|MISSING|TRUNCATED|finished|EXIT=|macro_cer|selected|verdict' "$LOG" 2>/dev/null | tail -40 || true
echo "=== gpu ==="
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader || true
tmux ls 2>/dev/null || true

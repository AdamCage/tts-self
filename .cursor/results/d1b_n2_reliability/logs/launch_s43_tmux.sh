#!/usr/bin/env bash
set -eu
tmux kill-session -t d1b_confirm_v2_s43 2>/dev/null || true
LOG=$HOME/projects/tts-self/out/d1b_n2_reliability/confirm_v2_seed43_series.log
rm -f "$LOG"
mkdir -p "$HOME/projects/tts-self/out/d1b_n2_reliability"
python3 - <<'PY'
from pathlib import Path
s = Path('/mnt/c/projects/tts-self/.cursor/results/d1b_n2_reliability/logs/run_seed43_only.sh')
t = s.read_bytes().replace(b'\r\n', b'\n').replace(b'\r', b'\n')
if b'maste\n' in t or b'origin maste' in t:
    raise SystemExit('corrupt maste still present')
d = Path('/tmp/run_seed43_only.sh')
d.write_bytes(t)
d.chmod(0o755)
print('script ok')
PY
tmux new-session -d -s d1b_confirm_v2_s43 "bash /tmp/run_seed43_only.sh 2>&1 | tee -a $LOG; echo EXIT=\$? | tee -a $LOG; bash"
sleep 30
echo "=== log ==="
tail -n 40 "$LOG"
echo "=== procs ==="
ps -ef | grep -E '72_train|run_seed43' | grep -v grep || true
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader || true

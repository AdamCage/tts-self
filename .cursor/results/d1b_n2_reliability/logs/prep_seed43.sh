#!/usr/bin/env bash
set -eu
cd ~/projects/tts-self
export HF_HOME=$HOME/hf_cache
export PATH="$HOME/.local/bin:$PATH"
unset MINISPEECH_ALLOW_DIRTY

rm -rf .cursor/results/d1b_n2_reliability
source .venv/bin/activate
uv pip install -e . --no-deps
python -c "import minispeech.train.discrete_lm_trainer as m; p=m.__file__; assert p.startswith('/home/'), p; print('import', p)"

git fetch origin
git merge --ff-only origin/master
if [ -n "$(git status --porcelain)" ]; then
  echo DIRTY
  git status --porcelain
  exit 2
fi
echo "CLEAN_SHA=$(git rev-parse HEAD)"

# Ensure run_seed43 is LF on Linux tmp
python3 - <<'PY'
from pathlib import Path
src = Path('/mnt/c/projects/tts-self/.cursor/results/d1b_n2_reliability/logs/run_seed43_only.sh')
dst = Path('/tmp/run_seed43_only.sh')
text = src.read_bytes().replace(b'\r\n', b'\n').replace(b'\r', b'\n')
# harden git sync lines
text = text.replace(b'git pull --ff-only origin master', b'git fetch origin && git merge --ff-only origin/master')
dst.write_bytes(text)
dst.chmod(0o755)
print('wrote', dst, 'bytes', len(text))
PY

tmux kill-session -t d1b_confirm_v2 2>/dev/null || true
SESSION=d1b_confirm_v2_s43
LINUX_LOG=$HOME/projects/tts-self/out/d1b_n2_reliability/confirm_v2_seed43_series.log
mkdir -p "$(dirname "$LINUX_LOG")"
tmux kill-session -t "$SESSION" 2>/dev/null || true
tmux new-session -d -s "$SESSION" "bash /tmp/run_seed43_only.sh 2>&1 | tee -a '$LINUX_LOG'; echo EXIT=\$? | tee -a '$LINUX_LOG'; bash"
echo "started $SESSION"
tmux ls

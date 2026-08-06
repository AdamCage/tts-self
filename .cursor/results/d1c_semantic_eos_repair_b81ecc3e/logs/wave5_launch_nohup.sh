#!/usr/bin/env bash
set -euo pipefail
cd ~/projects/tts-self
unset GIT_DIR GIT_WORK_TREE
EXT="$HOME/datasets/tts/d1c_wave5_logs"
mkdir -p "$EXT"

git fetch origin
git checkout d1c/semantic-eos
git reset --hard origin/d1c/semantic-eos
if [ -n "$(git status --porcelain)" ]; then
  while IFS= read -r line; do
    path="${line:3}"
    case "$path" in
      .cursor/results/*)
        mkdir -p "$EXT/pre_bak/$(dirname "$path")"
        mv -f "$path" "$EXT/pre_bak/$path" 2>/dev/null || true
        ;;
    esac
  done < <(git status --porcelain)
fi
test -z "$(git status --porcelain)" || { git status -sb; exit 2; }
TIP=$(git rev-parse HEAD)
echo CLEAN_TIP=$TIP

cp -f .cursor/results/d1c_semantic_eos_repair_b81ecc3e/logs/wave5_seed102_run.sh "$EXT/wave5_seed102_run.sh"
python3 - <<'PY'
from pathlib import Path
p=Path.home()/"datasets/tts/d1c_wave5_logs/wave5_seed102_run.sh"
p.write_bytes(p.read_bytes().replace(b"\r\n", b"\n").replace(b"\r", b"\n"))
PY
chmod +x "$EXT/wave5_seed102_run.sh"
test -z "$(git status --porcelain)"

nohup bash "$EXT/wave5_seed102_run.sh" > "$EXT/wave5_seed102_nohup.outer.log" 2>&1 &
echo $! > "$EXT/wave5_seed102.pid"
echo STARTED pid=$(cat "$EXT/wave5_seed102.pid") tip=$TIP
sleep 4
head -n 20 "$EXT/wave5_seed102_nohup.outer.log" || true

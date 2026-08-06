#!/usr/bin/env bash
# Full Wave 4 relaunch on current clean tip (after contract reseal).
set -euo pipefail
cd ~/projects/tts-self
unset GIT_DIR GIT_WORK_TREE
EXT="$HOME/datasets/tts/d1c_wave4_logs"
mkdir -p "$EXT"

git fetch origin
git checkout d1c/semantic-eos
git reset --hard origin/d1c/semantic-eos
if [ -n "$(git status --porcelain)" ]; then
  while IFS= read -r line; do
    path="${line:3}"
    case "$path" in
      .cursor/results/*)
        mkdir -p "$EXT/pre_full_bak/$(dirname "$path")"
        mv -f "$path" "$EXT/pre_full_bak/$path" 2>/dev/null || rm -rf "$path"
        ;;
    esac
  done < <(git status --porcelain)
fi
test -z "$(git status --porcelain)" || { git status -sb; exit 2; }
TIP=$(git rev-parse HEAD)
echo "CLEAN_TIP=$TIP"

# Ensure no stale run dir with matching name
if [ -d runs/d1c_eos_dev_w1_seed101 ]; then
  echo "REFUSING: run dir exists; quarantine first"
  exit 3
fi

cp -f .cursor/results/d1c_semantic_eos_repair_b81ecc3e/logs/wave4_w1_seed101_run.sh \
  "$EXT/wave4_w1_seed101_run.sh"
sed -i 's/\r$//' "$EXT/wave4_w1_seed101_run.sh"
chmod +x "$EXT/wave4_w1_seed101_run.sh"
test -z "$(git status --porcelain)"

nohup bash "$EXT/wave4_w1_seed101_run.sh" > "$EXT/wave4_w1_full_nohup.outer.log" 2>&1 &
echo $! > "$EXT/wave4_w1_full.pid"
echo "FULL_WAVE4_STARTED pid=$(cat "$EXT/wave4_w1_full.pid") tip=$TIP"
sleep 3
head -n 25 "$EXT/wave4_w1_full_nohup.outer.log" || true

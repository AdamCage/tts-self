#!/usr/bin/env bash
# Launcher: pull clean tip, run Wave 3 GPU via /tmp copy (no dirty tree).
set -euo pipefail
unset GIT_DIR GIT_WORK_TREE MINISPEECH_ALLOW_DIRTY
cd ~/projects/tts-self
git fetch origin
git checkout d1c/semantic-eos
git pull --ff-only origin d1c/semantic-eos
if [ -n "$(git status --porcelain)" ]; then
  echo DIRTY_WORKTREE
  git status -sb
  exit 2
fi
echo "LAUNCH_TIP=$(git rev-parse --short HEAD)"
SCRIPT_REPO=.cursor/results/d1c_semantic_eos_repair_b81ecc3e/logs/wave3_gpu_run.sh
SCRIPT_TMP=/tmp/wave3_gpu_run_$$.sh
tr -d '\r' < "$SCRIPT_REPO" > "$SCRIPT_TMP"
chmod +x "$SCRIPT_TMP"
export SMOKE_STEPS="${SMOKE_STEPS:-80}"
bash "$SCRIPT_TMP"
rc=$?
rm -f "$SCRIPT_TMP"
exit $rc

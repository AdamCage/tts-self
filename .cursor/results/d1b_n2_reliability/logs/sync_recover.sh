#!/usr/bin/env bash
set -euo pipefail
sed -i 's/\r$//' /mnt/c/projects/tts-self/.cursor/results/d1b_n2_reliability/logs/recover_best.sh
cd ~/projects/tts-self
git fetch /mnt/c/projects/tts-self master
git merge --ff-only FETCH_HEAD
echo "HEAD=$(git rev-parse HEAD)"
if [ -n "$(git status --porcelain)" ]; then
  echo DIRTY
  git status --porcelain
  exit 2
fi
bash /mnt/c/projects/tts-self/.cursor/results/d1b_n2_reliability/logs/recover_best.sh

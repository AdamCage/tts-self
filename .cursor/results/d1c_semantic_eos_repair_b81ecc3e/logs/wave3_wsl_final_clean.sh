#!/usr/bin/env bash
set -euo pipefail
cd ~/projects/tts-self
git fetch origin
git reset --hard origin/d1c/semantic-eos
# drop leftover untracked wave3 logs if present
rm -f .cursor/results/d1c_semantic_eos_repair_b81ecc3e/logs/wave3_focused_pytest.log \
      .cursor/results/d1c_semantic_eos_repair_b81ecc3e/logs/wave3_full_pytest.log
# re-checkout tracked logs
git checkout HEAD -- .cursor/results/d1c_semantic_eos_repair_b81ecc3e/logs/wave3_focused_pytest.log \
  .cursor/results/d1c_semantic_eos_repair_b81ecc3e/logs/wave3_full_pytest.log || true
if [ -n "$(git status --porcelain)" ]; then
  echo DIRTY
  git status -sb
  exit 2
fi
echo "CLEAN tip=$(git rev-parse --short HEAD)"

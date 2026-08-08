#!/usr/bin/env bash
set -euo pipefail
cd ~/projects/tts-self
git fetch origin
git checkout d1c/semantic-eos
git pull --ff-only origin d1c/semantic-eos
if [ -n "$(git status --porcelain)" ]; then
  echo DIRTY_WORKTREE
  git status -sb
  exit 2
fi
echo "SHA=$(git rev-parse HEAD)"
echo "SHORT=$(git rev-parse --short HEAD)"
source .venv/bin/activate
export HF_HOME="$HOME/hf_cache"
unset MINISPEECH_ALLOW_DIRTY || true
uv pip install -e . --no-deps
python scripts/00_env_check.py

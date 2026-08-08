#!/usr/bin/env bash
set -euo pipefail
cd ~/projects/tts-self
mkdir -p ~/tmp
if [ -d listen_service ]; then
  dest=~/tmp/listen_service_aside_wave3
  rm -rf "$dest"
  mv listen_service "$dest"
  echo "MOVED_LISTEN_SERVICE_TO=$dest"
fi
if [ -n "$(git status --porcelain)" ]; then
  echo DIRTY_WORKTREE
  git status -sb
  exit 2
fi
echo CLEAN
echo "SHA=$(git rev-parse HEAD)"
echo "SHORT=$(git rev-parse --short HEAD)"
source .venv/bin/activate
export HF_HOME="$HOME/hf_cache"
unset MINISPEECH_ALLOW_DIRTY || true
uv pip install -e . --no-deps
python scripts/00_env_check.py

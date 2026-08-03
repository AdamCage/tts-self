#!/usr/bin/env bash
set -euo pipefail
cd ~/projects/tts-self
git remote -v
git branch -av | head -30
git status -sb
git rev-parse HEAD
git status --porcelain | wc -l
git ls-remote --heads origin | head

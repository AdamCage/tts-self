#!/usr/bin/env bash
set -euo pipefail
cd /mnt/c/projects/tts-self-d1c
export GIT_DIR=/mnt/c/projects/tts-self/.git/worktrees/tts-self-d1c
export GIT_WORK_TREE=/mnt/c/projects/tts-self-d1c
export PYTHONPATH=/mnt/c/projects/tts-self-d1c/src
export HF_HOME="${HF_HOME:-$HOME/hf_cache}"
# shellcheck disable=SC1091
source /home/adam/projects/tts-self/.venv/bin/activate
LOG=.cursor/results/d1c_semantic_eos_repair_b81ecc3e/logs/repair_round2_full_pytest.log
{
  echo "=== D1c INFRA repair round2 full pytest ==="
  date -u
  echo "rootdir=$(pwd)"
  echo "GIT_DIR=$GIT_DIR"
  echo "GIT_WORK_TREE=$GIT_WORK_TREE"
  git rev-parse HEAD
  git rev-parse --short HEAD
  echo "---"
  python -m pytest -v --tb=line
} 2>&1 | tee "$LOG"

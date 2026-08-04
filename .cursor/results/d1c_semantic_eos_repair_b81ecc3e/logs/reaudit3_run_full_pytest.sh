#!/usr/bin/env bash
set -euo pipefail
cd /mnt/c/projects/tts-self-d1c
export GIT_DIR=/mnt/c/projects/tts-self/.git/worktrees/tts-self-d1c
export GIT_WORK_TREE=/mnt/c/projects/tts-self-d1c
export PYTHONPATH=/mnt/c/projects/tts-self-d1c/src
export HF_HOME="${HF_HOME:-$HOME/hf_cache}"
# Honour no-GPU constraint for this repair round (masks bf16 CUDA unit test → skip).
export CUDA_VISIBLE_DEVICES=
# shellcheck disable=SC1091
source /home/adam/projects/tts-self/.venv/bin/activate
LOG=.cursor/results/d1c_semantic_eos_repair_b81ecc3e/logs/reaudit3_full_pytest.log
{
  echo "=== D1c INFRA REAUDIT round3 independent full pytest ==="
  date -u
  echo "rootdir=$(pwd)"
  echo "GIT_DIR=$GIT_DIR"
  echo "GIT_WORK_TREE=$GIT_WORK_TREE"
  echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES-<unset>}"
  git rev-parse HEAD
  git rev-parse --short HEAD
  echo "note: Windows worktree via /mnt/c requires GIT_DIR/GIT_WORK_TREE override"
  echo "---"
  python -m pytest -v --tb=line
} 2>&1 | tee "$LOG"

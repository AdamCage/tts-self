#!/usr/bin/env bash
# REAUDIT ROUND2 — independent reproduction of the round-2 pytest claim (397 passed, 2 skipped).
# Uses the same GIT_DIR override the repair agent documented, so the /mnt/c worktree
# git_sha() lookup resolves. Run from the D1c worktree.
set -uo pipefail

cd /mnt/c/projects/tts-self-d1c
source /home/adam/projects/tts-self/.venv/bin/activate
export HF_HOME="$HOME/hf_cache"
export CUDA_VISIBLE_DEVICES=
export GIT_DIR=/mnt/c/projects/tts-self/.git/worktrees/tts-self-d1c
export GIT_WORK_TREE=/mnt/c/projects/tts-self-d1c

LOG=.cursor/results/d1c_semantic_eos_repair_b81ecc3e/logs/reaudit2_full_pytest_gitenv.log

{
  echo "=== REAUDIT ROUND2 independent full-suite reproduction (GIT_DIR override) ==="
  date -Is
  echo "rootdir=$(pwd)"
  echo "GIT_DIR=${GIT_DIR}"
  echo "GIT_WORK_TREE=${GIT_WORK_TREE}"
  echo "git_sha=$(git rev-parse HEAD)"
  echo "git_sha_short=$(git rev-parse --short HEAD)"
  echo "dirty_attested_paths=$(git status --porcelain -- src scripts configs tests | wc -l)"
  python -c 'import sys; print("python", sys.version.split()[0])'
  echo
  python -m pytest -q 2>&1 | tail -n 30
} 2>&1 | tee "${LOG}"

echo "--- saved ${LOG} ($(stat -c%s "${LOG}") bytes) ---"

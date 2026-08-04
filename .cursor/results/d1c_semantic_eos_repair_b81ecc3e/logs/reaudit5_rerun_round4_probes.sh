#!/usr/bin/env bash
# REAUDIT_ROUND5 — re-run the ROUND-4 probes verbatim as a regression harness.
# Expected intentional delta: reaudit4 E5 (`--checkpoint-step` invents a step for
# best.pt) must now FAIL, because round 5 deliberately closed that contract.
# Everything else must still pass.
set -u
cd /mnt/c/projects/tts-self-d1c || exit 1
export CUDA_VISIBLE_DEVICES=
export HF_HOME=/home/adam/hf_cache
export PYTHONPATH=/mnt/c/projects/tts-self-d1c/src
export GIT_DIR=/mnt/c/projects/tts-self/.git/worktrees/tts-self-d1c
export GIT_WORK_TREE=/mnt/c/projects/tts-self-d1c
PY=/home/adam/projects/tts-self/.venv/bin/python
L=.cursor/results/d1c_semantic_eos_repair_b81ecc3e/logs

echo "===== RERUN reaudit4_probe_dupckpt.py (the round-4 attack, verbatim) ====="
"$PY" "$L/reaudit4_probe_dupckpt.py" 2>&1 | tee "$L/reaudit5_rerun_r4_dupckpt.log"

echo
echo "===== RERUN reaudit4_probe_prod.py (round-4 production regression, verbatim) ====="
"$PY" "$L/reaudit4_probe_prod.py" 2>&1 | tee "$L/reaudit5_rerun_r4_prod.log" | grep -E '^[A-Z][0-9a-z]* (PASS|FAIL|INFO)|^SUMMARY|^  FAILED'

echo
echo "===== RERUN reaudit4_probe_mutation.py (verbatim) ====="
"$PY" "$L/reaudit4_probe_mutation.py" 2>&1 | tee "$L/reaudit5_rerun_r4_mutation.log" | grep -E '^X[0-9] |^SUMMARY|^  FAILED'

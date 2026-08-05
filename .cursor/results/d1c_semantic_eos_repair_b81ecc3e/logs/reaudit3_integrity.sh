#!/usr/bin/env bash
# REAUDIT round3 post-probe integrity: no attested source was mutated.
cd /mnt/c/projects/tts-self-d1c || exit 1
export GIT_DIR=/mnt/c/projects/tts-self/.git/worktrees/tts-self-d1c
export GIT_WORK_TREE=/mnt/c/projects/tts-self-d1c
echo "=== git status (src scripts configs tests) ==="
git status --porcelain -- src scripts configs tests
echo "=== HEAD ==="
git rev-parse HEAD
echo "=== recipe_source_manifest verify AFTER all probes ==="
CUDA_VISIBLE_DEVICES= PYTHONPATH=/mnt/c/projects/tts-self-d1c/src \
  /home/adam/projects/tts-self/.venv/bin/python - <<'PY'
import json
from pathlib import Path
from minispeech.data.d1b_oracles import load_eval_contract, verify_recipe_source_manifest
root = Path("/mnt/c/projects/tts-self-d1c")
c = load_eval_contract(root / "configs/eval_contracts/d1c_semantic_eos_v2_0_0.json")
m = c["recipe_source_manifest"]
print("n_locked =", len(m))
print("mismatches =", verify_recipe_source_manifest(m, root=root))
print("contract sha256 =", c["sha256"])
PY
echo "=== checkpoint payload carries the step (fix is one line) ==="
grep -n 'candidate_step_' src/minispeech/train/discrete_lm_trainer.py | head -3
echo "=== evaluator discards the loaded payload step ==="
sed -n '156,165p' scripts/74_eval_discrete_lm.py

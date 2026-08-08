#!/usr/bin/env bash
set -euo pipefail
cd ~/projects/tts-self
echo "SHA=$(git rev-parse HEAD)"
echo "SHORT=$(git rev-parse --short HEAD)"
if [ -n "$(git status --porcelain)" ]; then
  echo DIRTY_AFTER_TESTS
  git status --porcelain
else
  echo CLEAN_AFTER_TESTS
fi
python3 - <<'PY'
import json
from pathlib import Path
p = Path("configs/train_subsets/d1c_confirm_seed_commitment.json")
d = json.loads(p.read_text(encoding="utf-8"))
print("commitment=", d.get("commitment"))
print("plaintext_seed_present=", d.get("plaintext_seed_present"))
PY
ls -la .cursor/results/d1c_semantic_eos_repair_b81ecc3e/logs/wave3_focused_pytest.log \
      .cursor/results/d1c_semantic_eos_repair_b81ecc3e/logs/wave3_full_pytest.log
# mirror logs to Windows worktree
WT=/mnt/c/projects/tts-self-d1c/.cursor/results/d1c_semantic_eos_repair_b81ecc3e/logs
cp -a .cursor/results/d1c_semantic_eos_repair_b81ecc3e/logs/wave3_focused_pytest.log "$WT/"
cp -a .cursor/results/d1c_semantic_eos_repair_b81ecc3e/logs/wave3_full_pytest.log "$WT/"
echo LOGS_MIRRORED

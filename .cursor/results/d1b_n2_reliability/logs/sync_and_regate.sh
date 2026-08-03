#!/usr/bin/env bash
set -euo pipefail
cd ~/projects/tts-self
# Sync from Windows authoring clone (local path, no remote required)
git fetch /mnt/c/projects/tts-self master
git merge --ff-only FETCH_HEAD
echo "HEAD=$(git rev-parse HEAD)"
test -z "$(git status --porcelain)" || { echo DIRTY; git status --porcelain; exit 2; }
source .venv/bin/activate
export HF_HOME=$HOME/hf_cache
pytest -q tests/test_discrete_n2_protocol.py
python /mnt/c/projects/tts-self/.cursor/results/d1b_n2_reliability/logs/patch_regate.py
python scripts/76_select_discrete_checkpoint.py \
  --candidates-json /mnt/c/projects/tts-self/out/d1b_n2_reliability/metrics/dev_seed42_candidate_rows.json \
  --selection-subset dev32 \
  --require-subset dev32 \
  --verify-frozen-subset dev32 \
  --out runs/discrete_speech_lm_d1b_dev_seed42/checkpoint_selection.json \
  --promote-best \
  --best-path runs/discrete_speech_lm_d1b_dev_seed42/best.pt
cp -f runs/discrete_speech_lm_d1b_dev_seed42/checkpoint_selection.json \
  /mnt/c/projects/tts-self/out/d1b_n2_reliability/metrics/dev_seed42_selection.json
# Promote best candidate gate as official DEV N2 gate
python - <<'PY'
import json, shutil
from pathlib import Path
sel = json.loads(Path("runs/discrete_speech_lm_d1b_dev_seed42/checkpoint_selection.json").read_text())
step = int(sel["selected"]["step"])
src = Path(f"/mnt/c/projects/tts-self/out/d1b_n2_reliability/dev_seed42/cand_{step:04d}_n2/n2_gate.json")
dst = Path("/mnt/c/projects/tts-self/out/d1b_n2_reliability/metrics/dev_seed42_n2_gate.json")
shutil.copy2(src, dst)
g = json.loads(dst.read_text())
print("DEV gate verdict", g["verdict"], "n_cat", g["criteria"].get("n_catastrophic"), "step", step)
PY

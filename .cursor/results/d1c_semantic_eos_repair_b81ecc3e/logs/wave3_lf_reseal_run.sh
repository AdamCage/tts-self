#!/usr/bin/env bash
set -euo pipefail
cd ~/projects/tts-self
source .venv/bin/activate
export HF_HOME="$HOME/hf_cache"
export PYTHONPATH="$PWD/src"
export CUDA_VISIBLE_DEVICES=
# Ensure clean LF sources (no prior dirty seal edits)
git checkout -- configs/eval_contracts/d1c_semantic_eos_v2_0_0.json \
  configs/eval_contracts/d1c_semantic_eos_v2_0_1.json \
  configs/discrete_speech_lm_d1c_semantic_eos_w1.yaml \
  configs/discrete_speech_lm_d1c_semantic_eos_w2.yaml \
  .cursor/results/d1c_semantic_eos_repair_b81ecc3e/artifacts/d1c_semantic_eos_v2_0_0.json \
  .cursor/results/d1c_semantic_eos_repair_b81ecc3e/artifacts/repair_round7_hashes.json || true

python .cursor/results/d1c_semantic_eos_repair_b81ecc3e/logs/repair_round7_rebind_hashes.py \
  | tee .cursor/results/d1c_semantic_eos_repair_b81ecc3e/logs/wave3_lf_reseal.log

# Mark artifact as wave3 LF reseal
python - <<'PY'
import json
from pathlib import Path
p = Path(".cursor/results/d1c_semantic_eos_repair_b81ecc3e/artifacts/repair_round7_hashes.json")
d = json.loads(p.read_text(encoding="utf-8"))
d["round"] = "7_lf_reseal_wave3"
d["note"] = "Resealed on WSL LF after round-7 Windows CRLF seal poison"
p.write_text(json.dumps(d, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
Path(".cursor/results/d1c_semantic_eos_repair_b81ecc3e/artifacts/repair_wave3_lf_reseal_hashes.json").write_text(
    json.dumps(d, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
)
print("wrote wave3 lf reseal artifact")
PY

# Mirror sealed files to Windows d1c worktree for commit (LF bytes)
WT=/mnt/c/projects/tts-self-d1c
for f in \
  configs/eval_contracts/d1c_semantic_eos_v2_0_0.json \
  configs/eval_contracts/d1c_semantic_eos_v2_0_1.json \
  configs/discrete_speech_lm_d1c_semantic_eos_w1.yaml \
  configs/discrete_speech_lm_d1c_semantic_eos_w2.yaml \
  .cursor/results/d1c_semantic_eos_repair_b81ecc3e/artifacts/d1c_semantic_eos_v2_0_0.json \
  .cursor/results/d1c_semantic_eos_repair_b81ecc3e/artifacts/repair_round7_hashes.json \
  .cursor/results/d1c_semantic_eos_repair_b81ecc3e/artifacts/repair_wave3_lf_reseal_hashes.json \
  .cursor/results/d1c_semantic_eos_repair_b81ecc3e/logs/wave3_lf_reseal.log
do
  mkdir -p "$(dirname "$WT/$f")"
  cp -a "$f" "$WT/$f"
done
echo MIRRORED_TO_WORKTREE

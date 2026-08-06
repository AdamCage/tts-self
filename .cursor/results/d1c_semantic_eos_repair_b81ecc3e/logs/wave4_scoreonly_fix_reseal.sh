#!/usr/bin/env bash
set -euo pipefail
cd ~/projects/tts-self
unset GIT_DIR GIT_WORK_TREE
EXT="$HOME/datasets/tts/d1c_wave4_logs"
WIN=/mnt/c/projects/tts-self-d1c
mkdir -p "$EXT"

cp -f "$WIN/scripts/74_eval_discrete_lm.py" scripts/74_eval_discrete_lm.py
sed -i 's/\r$//' scripts/74_eval_discrete_lm.py

source .venv/bin/activate
export HF_HOME="$HOME/hf_cache"
export PYTHONPATH="$PWD/src"
export CUDA_VISIBLE_DEVICES=

# Quick unit: score-only key selection for semantic vs legacy
python - <<'PY' | tee "$EXT/wave4_score_only_infer_keys_probe.log"
from pathlib import Path
import importlib.util
from minispeech.data.d1b_oracles import load_eval_contract
from minispeech.infer.discrete_config import GENERATE_INFER_KEYS, SEMANTIC_EOS_INFER_DEFAULTS

spec = importlib.util.spec_from_file_location("e74", "scripts/74_eval_discrete_lm.py")
# Don't exec full script (argparse side effects); just validate key sets.
peek = load_eval_contract("configs/eval_contracts/d1c_semantic_eos_v2_0_0.json")
assert (peek.get("termination") or {}).get("mode") == "semantic_eos_v1"
semantic_keys = tuple(SEMANTIC_EOS_INFER_DEFAULTS)
legacy = set(GENERATE_INFER_KEYS) - set(semantic_keys)
assert "stop_threshold" in legacy
assert "stop_threshold" not in semantic_keys
print("PROBE_OK", {"semantic_n": len(semantic_keys), "legacy_extra": sorted(legacy)})
PY

cp -f "$WIN/.cursor/results/d1c_semantic_eos_repair_b81ecc3e/logs/wave4_clean_sha_reseal.py" \
  .cursor/results/d1c_semantic_eos_repair_b81ecc3e/logs/wave4_clean_sha_reseal.py
sed -i 's/\r$//' .cursor/results/d1c_semantic_eos_repair_b81ecc3e/logs/wave4_clean_sha_reseal.py
python .cursor/results/d1c_semantic_eos_repair_b81ecc3e/logs/wave4_clean_sha_reseal.py \
  | tee "$EXT/wave4_post_scoreonly_fix_reseal.log"

# Focused regression on prior subset alias tests + a couple 74-related if cheap
python -m pytest -q tests/test_d1c_oracles.py \
  | tee "$EXT/wave4_scoreonly_fix_pytest.log"

for f in \
  scripts/74_eval_discrete_lm.py \
  configs/eval_contracts/d1c_semantic_eos_v2_0_0.json \
  configs/eval_contracts/d1c_semantic_eos_v2_0_1.json \
  configs/discrete_speech_lm_d1c_semantic_eos_w1.yaml \
  configs/discrete_speech_lm_d1c_semantic_eos_w2.yaml \
  .cursor/results/d1c_semantic_eos_repair_b81ecc3e/artifacts/wave4_clean_sha_reseal_hashes.json \
  .cursor/results/d1c_semantic_eos_repair_b81ecc3e/artifacts/d1c_semantic_eos_v2_0_0.json
 do
  mkdir -p "$WIN/$(dirname "$f")"
  cp -f "$f" "$WIN/$f"
done
cp -f "$EXT/wave4_score_only_infer_keys_probe.log" "$WIN/.cursor/results/d1c_semantic_eos_repair_b81ecc3e/logs/"
cp -f "$EXT/wave4_post_scoreonly_fix_reseal.log" "$WIN/.cursor/results/d1c_semantic_eos_repair_b81ecc3e/logs/"
cp -f "$EXT/wave4_scoreonly_fix_pytest.log" "$WIN/.cursor/results/d1c_semantic_eos_repair_b81ecc3e/logs/"
echo SCOREONLY_FIX_READY

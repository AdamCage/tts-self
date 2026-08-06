#!/usr/bin/env bash
set -euo pipefail
cd ~/projects/tts-self
unset GIT_DIR GIT_WORK_TREE
EXT="$HOME/datasets/tts/d1c_wave4_logs"
WIN=/mnt/c/projects/tts-self-d1c
mkdir -p "$EXT"

cp -f "$WIN/src/minispeech/data/d1b_oracles.py" src/minispeech/data/d1b_oracles.py
cp -f "$WIN/tests/test_d1c_oracles.py" tests/test_d1c_oracles.py
sed -i 's/\r$//' src/minispeech/data/d1b_oracles.py tests/test_d1c_oracles.py

source .venv/bin/activate
export HF_HOME="$HOME/hf_cache"
export PYTHONPATH="$PWD/src"
export CUDA_VISIBLE_DEVICES=

cp -f "$WIN/.cursor/results/d1c_semantic_eos_repair_b81ecc3e/logs/wave4_clean_sha_reseal.py" \
  .cursor/results/d1c_semantic_eos_repair_b81ecc3e/logs/wave4_clean_sha_reseal.py
sed -i 's/\r$//' .cursor/results/d1c_semantic_eos_repair_b81ecc3e/logs/wave4_clean_sha_reseal.py
python .cursor/results/d1c_semantic_eos_repair_b81ecc3e/logs/wave4_clean_sha_reseal.py \
  | tee "$EXT/wave4_post_split_rebind_reseal.log"

python -m pytest -q tests/test_d1c_oracles.py \
  tests/test_d1b_oracles.py::test_reject_contradictory_subset_label_vs_hash \
  | tee "$EXT/wave4_split_rebind_pytest.log"

for f in \
  src/minispeech/data/d1b_oracles.py \
  tests/test_d1c_oracles.py \
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
cp -f "$EXT/wave4_split_rebind_pytest.log" "$WIN/.cursor/results/d1c_semantic_eos_repair_b81ecc3e/logs/"
cp -f "$EXT/wave4_post_split_rebind_reseal.log" "$WIN/.cursor/results/d1c_semantic_eos_repair_b81ecc3e/logs/"
echo SPLIT_REBIND_READY

#!/usr/bin/env bash
set -euo pipefail
cd ~/projects/tts-self
unset GIT_DIR GIT_WORK_TREE
FILES=(
  configs/eval_contracts/d1c_semantic_eos_v2_0_0.json
  configs/eval_contracts/d1c_semantic_eos_v2_0_1.json
  configs/discrete_speech_lm_d1c_semantic_eos_w1.yaml
  configs/discrete_speech_lm_d1c_semantic_eos_w2.yaml
  .cursor/results/d1c_semantic_eos_repair_b81ecc3e/artifacts/wave4_clean_sha_reseal_hashes.json
  .cursor/results/d1c_semantic_eos_repair_b81ecc3e/artifacts/d1c_semantic_eos_v2_0_0.json
  .cursor/results/d1c_semantic_eos_repair_b81ecc3e/logs/wave4_clean_sha_reseal.py
  .cursor/results/d1c_semantic_eos_repair_b81ecc3e/logs/wave4_clean_sha_reseal.log
  .cursor/results/d1c_semantic_eos_repair_b81ecc3e/logs/wave4_clean_sha_reseal_run.sh
  .cursor/results/d1c_semantic_eos_repair_b81ecc3e/logs/wave4_w1_seed101_run.sh
)
for f in "${FILES[@]}"; do
  if [[ -f "$f" ]]; then
    mkdir -p "/mnt/c/projects/tts-self-d1c/$(dirname "$f")"
    cp -f "$f" "/mnt/c/projects/tts-self-d1c/$f"
  fi
done
echo MIRROR_OK

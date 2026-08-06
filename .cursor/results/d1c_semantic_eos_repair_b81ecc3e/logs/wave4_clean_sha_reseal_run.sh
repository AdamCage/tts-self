#!/usr/bin/env bash
set -euo pipefail
cd ~/projects/tts-self
unset GIT_DIR GIT_WORK_TREE

# Restore tip configs (discard poisoned reseal), then re-apply clean_sha reseal.
git checkout -- \
  configs/eval_contracts/d1c_semantic_eos_v2_0_0.json \
  configs/eval_contracts/d1c_semantic_eos_v2_0_1.json \
  configs/discrete_speech_lm_d1c_semantic_eos_w1.yaml \
  configs/discrete_speech_lm_d1c_semantic_eos_w2.yaml

cp /mnt/c/projects/tts-self-d1c/.cursor/results/d1c_semantic_eos_repair_b81ecc3e/logs/wave4_clean_sha_reseal.py \
  .cursor/results/d1c_semantic_eos_repair_b81ecc3e/logs/

source .venv/bin/activate
export HF_HOME="$HOME/hf_cache"
export PYTHONPATH="$PWD/src"
export CUDA_VISIBLE_DEVICES=

python .cursor/results/d1c_semantic_eos_repair_b81ecc3e/logs/wave4_clean_sha_reseal.py \
  | tee .cursor/results/d1c_semantic_eos_repair_b81ecc3e/logs/wave4_clean_sha_reseal.log

FILES=(
  configs/eval_contracts/d1c_semantic_eos_v2_0_0.json
  configs/eval_contracts/d1c_semantic_eos_v2_0_1.json
  configs/discrete_speech_lm_d1c_semantic_eos_w1.yaml
  configs/discrete_speech_lm_d1c_semantic_eos_w2.yaml
  .cursor/results/d1c_semantic_eos_repair_b81ecc3e/artifacts/wave4_clean_sha_reseal_hashes.json
  .cursor/results/d1c_semantic_eos_repair_b81ecc3e/artifacts/d1c_semantic_eos_v2_0_0.json
  .cursor/results/d1c_semantic_eos_repair_b81ecc3e/logs/wave4_clean_sha_reseal.py
  .cursor/results/d1c_semantic_eos_repair_b81ecc3e/logs/wave4_clean_sha_reseal.log
)
for f in "${FILES[@]}"; do
  if [[ -f "$f" ]]; then
    mkdir -p "/mnt/c/projects/tts-self-d1c/$(dirname "$f")"
    cp -f "$f" "/mnt/c/projects/tts-self-d1c/$f"
  fi
done

python - <<'PY'
import json
c = json.load(open("configs/eval_contracts/d1c_semantic_eos_v2_0_0.json", encoding="utf-8"))
print("repro", c["repro"])
print("sha", c["sha256"])
print("recipe", c["recipe_sha256"])
print("manifest", c["recipe_source_manifest_sha256"])
print("w1", c["recipe_source_manifest"]["configs/discrete_speech_lm_d1c_semantic_eos_w1.yaml"])
print("w2", c["recipe_source_manifest"]["configs/discrete_speech_lm_d1c_semantic_eos_w2.yaml"])
# Expect yaml digests unchanged from Wave3 LF reseal tip.
assert c["recipe_source_manifest"]["configs/discrete_speech_lm_d1c_semantic_eos_w1.yaml"] == \
    "91125b5abfed71c8160e374b8e21a593cd4495d15c888ccd4e7f7b82dec4d768"
assert c["recipe_source_manifest"]["configs/discrete_speech_lm_d1c_semantic_eos_w2.yaml"] == \
    "4890e51af27150996a1a80a3cdba1af92d0e6de27230b536fb55cde65c79660e"
print("YAML_DIGESTS_PRESERVED")
PY

git status -sb | head -20

#!/usr/bin/env bash
# REAUDIT ROUND2 probe C — live scripts/72_train_discrete_lm.py D1c fail-closed behaviour.
# No GPU, no training: every case must abort in the provenance/guard section.
set -uo pipefail
cd /mnt/c/projects/tts-self-d1c
export PYTHONPATH=/mnt/c/projects/tts-self-d1c/src
export HF_HOME="${HF_HOME:-$HOME/hf_cache}"
export CUDA_VISIBLE_DEVICES=""
# shellcheck disable=SC1091
source /home/adam/projects/tts-self/.venv/bin/activate

TMP=/tmp/reaudit2/cfg
mkdir -p "$TMP"
W1=configs/discrete_speech_lm_d1c_semantic_eos_w1.yaml

# stale-hash injection (temp copy, worktree untouched); filename keeps the d1c marker
STALE=$TMP/discrete_speech_lm_d1c_semantic_eos_stale.yaml
python - <<'PY'
import yaml, pathlib
src = pathlib.Path("configs/discrete_speech_lm_d1c_semantic_eos_w1.yaml")
dst = pathlib.Path("/tmp/reaudit2/cfg/discrete_speech_lm_d1c_semantic_eos_stale.yaml")
cfg = yaml.safe_load(src.read_text(encoding="utf-8"))
cfg["train"]["eval_contract_hash"] = "a59ada9c1e0a64f067b27dd2b4d93ca270702bd41881746d1bb9a437d2f33772"
cfg["train"]["run_name"] = "reaudit2_stale_probe"
dst.write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), encoding="utf-8")
print("wrote", dst)
PY

# missing-declared-hash injection
NOHASH=$TMP/discrete_speech_lm_d1c_semantic_eos_nohash.yaml
python - <<'PY'
import yaml, pathlib
src = pathlib.Path("configs/discrete_speech_lm_d1c_semantic_eos_w1.yaml")
dst = pathlib.Path("/tmp/reaudit2/cfg/discrete_speech_lm_d1c_semantic_eos_nohash.yaml")
cfg = yaml.safe_load(src.read_text(encoding="utf-8"))
cfg["train"].pop("eval_contract_hash", None)
cfg["train"]["run_name"] = "reaudit2_nohash_probe"
dst.write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), encoding="utf-8")
print("wrote", dst)
PY

run_case() {
  local label="$1"; shift
  echo "======== CASE $label"
  echo "CMD: python scripts/72_train_discrete_lm.py $*"
  timeout 300 python scripts/72_train_discrete_lm.py "$@" 2>&1 | tail -25
  echo "EXIT=${PIPESTATUS[0]}"
  echo
}

run_case "C1 canonical w1, no --allow-dirty (expect require_clean_sha refusal)" \
  --config "$W1" --subset dev32 --subset-split full32 --codec-tokens-dir /tmp/reaudit2/nonexistent

run_case "C2 canonical w1 + --skip-eval-contract (expect D1c refusal)" \
  --config "$W1" --subset dev32 --subset-split full32 --skip-eval-contract --codec-tokens-dir /tmp/reaudit2/nonexistent

run_case "C3 canonical w1, NO --subset (expect silent-first-N refusal or cfg subset)" \
  --config "$W1" --codec-tokens-dir /tmp/reaudit2/nonexistent

run_case "C4 stale eval_contract_hash + --allow-dirty (expect fail-closed mismatch)" \
  --config "$STALE" --subset dev32 --subset-split full32 --allow-dirty --codec-tokens-dir /tmp/reaudit2/nonexistent

run_case "C5 missing eval_contract_hash + --allow-dirty (expect refuse silent overwrite)" \
  --config "$NOHASH" --subset dev32 --subset-split full32 --allow-dirty --codec-tokens-dir /tmp/reaudit2/nonexistent

run_case "C6 canonical w1 + --allow-dirty (expect WARNING + proceed past contract)" \
  --config "$W1" --subset dev32 --subset-split full32 --allow-dirty --codec-tokens-dir /tmp/reaudit2/nonexistent

echo "======== static: is_d1c_cfg detection"
grep -n "is_d1c_cfg\|is_d1b_cfg\|is_formal_cfg" scripts/72_train_discrete_lm.py

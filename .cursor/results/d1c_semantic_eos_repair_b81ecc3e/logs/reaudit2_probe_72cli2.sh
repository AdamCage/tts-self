#!/usr/bin/env bash
# REAUDIT ROUND2 probe C2 — reach the deeper scripts/72 D1c guards with a correct subset
# and a synthetic contract whose repro.clean_sha_available=true (temp copies only).
set -uo pipefail
cd /mnt/c/projects/tts-self-d1c
export PYTHONPATH=/mnt/c/projects/tts-self-d1c/src
export HF_HOME="${HF_HOME:-$HOME/hf_cache}"
export CUDA_VISIBLE_DEVICES=""
# shellcheck disable=SC1091
source /home/adam/projects/tts-self/.venv/bin/activate

python - <<'PY'
import hashlib, json, pathlib, sys
sys.path.insert(0, "src")
import yaml
from minispeech.data.d1b_oracles import eval_contract_content_sha256

tmp = pathlib.Path("/tmp/reaudit2/cfg"); tmp.mkdir(parents=True, exist_ok=True)
c = json.loads(pathlib.Path("configs/eval_contracts/d1c_semantic_eos_v2_0_0.json").read_text(encoding="utf-8"))
c["repro"] = dict(c.get("repro") or {}); c["repro"]["clean_sha_available"] = True
c.pop("sha256", None)
sha = eval_contract_content_sha256(c)
c["sha256"] = sha
cp = tmp / "d1c_semantic_eos_v2_0_0_cleansha.json"
cp.write_text(json.dumps(c, ensure_ascii=False, indent=2), encoding="utf-8")
print("synthetic contract sha:", sha)

base = yaml.safe_load(pathlib.Path("configs/discrete_speech_lm_d1c_semantic_eos_w1.yaml").read_text(encoding="utf-8"))

def emit(name, mutate):
    cfg = json.loads(json.dumps(base))
    cfg["train"]["eval_contract_path"] = str(cp)
    cfg["train"]["eval_contract_hash"] = sha
    cfg["train"]["run_name"] = f"reaudit2_{name}"
    mutate(cfg)
    p = tmp / f"discrete_speech_lm_d1c_semantic_eos_{name}.yaml"
    p.write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), encoding="utf-8")
    print("wrote", p)

emit("ok", lambda c: None)
emit("nohash2", lambda c: c["train"].pop("eval_contract_hash", None))
emit("stale2", lambda c: c["train"].__setitem__("eval_contract_hash", "a"*64))
PY

SUB=configs/train_subsets/d1c_dev32.json
TMP=/tmp/reaudit2/cfg

run_case() {
  local label="$1"; shift
  echo "======== CASE $label"
  echo "CMD: python scripts/72_train_discrete_lm.py $*"
  timeout 300 python scripts/72_train_discrete_lm.py "$@" 2>&1 | tail -22
  echo "EXIT=${PIPESTATUS[0]}"
  echo
}

run_case "D1 clean-sha contract, declared hash correct, NO --allow-dirty (must pass contract, stop later)" \
  --config "$TMP/discrete_speech_lm_d1c_semantic_eos_ok.yaml" --subset "$SUB" --subset-split full32 --codec-tokens-dir /tmp/reaudit2/nonexistent

run_case "D2 clean-sha contract, MISSING declared eval_contract_hash (must refuse silent overwrite)" \
  --config "$TMP/discrete_speech_lm_d1c_semantic_eos_nohash2.yaml" --subset "$SUB" --subset-split full32 --codec-tokens-dir /tmp/reaudit2/nonexistent

run_case "D3 clean-sha contract, STALE declared hash (must fail closed)" \
  --config "$TMP/discrete_speech_lm_d1c_semantic_eos_stale2.yaml" --subset "$SUB" --subset-split full32 --codec-tokens-dir /tmp/reaudit2/nonexistent

run_case "D4 clean-sha contract + --allow-dirty (must print WARNING about MINISPEECH_ALLOW_DIRTY)" \
  --config "$TMP/discrete_speech_lm_d1c_semantic_eos_ok.yaml" --subset "$SUB" --subset-split full32 --allow-dirty --codec-tokens-dir /tmp/reaudit2/nonexistent

echo "======== CLI alias check: does --subset dev32 point at D1b or D1c?"
python - <<'PY'
import sys
sys.path.insert(0, "src")
from minispeech.data.d1b_oracles import SUBSET_ALIASES, resolve_subset_spec, load_frozen_utt_ids
for a in ("dev32", "select8", "gate24", "full32"):
    p, s = resolve_subset_spec(a)
    try:
        ids, _, d = load_frozen_utt_ids(a)
        print(f"alias {a!r} -> {p} split={s} items_sha256={d} n={len(ids)}")
    except Exception as e:
        print(f"alias {a!r} -> {p} split={s} ERROR {e}")
print("d1c aliases present:", [k for k in SUBSET_ALIASES if "d1c" in k] or "NONE")
PY

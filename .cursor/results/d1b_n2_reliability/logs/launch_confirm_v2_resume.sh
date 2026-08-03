#!/usr/bin/env bash
# Resume interrupted confirm_v2_seed42 train, then complete FR+seed43.
set -euo pipefail
cd ~/projects/tts-self
export HF_HOME=$HOME/hf_cache
unset MINISPEECH_ALLOW_DIRTY

WIN=/mnt/c/projects/tts-self
WIN_LOG=$WIN/.cursor/results/d1b_n2_reliability/logs
WIN_OUT=$WIN/out/d1b_n2_reliability
WIN_MET=$WIN_OUT/metrics
CODEC="$HOME/datasets/tts/ruslan/codec_tokens,$HOME/datasets/tts/common_voice_ru17/codec_tokens"
SELECT8_SHA=6667da2f8fc2c3706deff48a79489103f739419041f3fb492f5822647075a0ed
CFG=configs/discrete_speech_lm_d1b_443m.yaml

mkdir -p "$WIN_LOG" "$WIN_MET"
git fetch origin
git pull --ff-only origin master
test -z "$(git status --porcelain)" || { echo DIRTY; git status --porcelain; exit 2; }
SHA=$(git rev-parse HEAD)
echo "CLEAN_SHA=$SHA resume=$(date -Iseconds)" | tee -a "$WIN_LOG/confirm_v2_sha.txt"

finish_seed() {
  local SEED=$1
  local SKIP_TRAIN=${2:-0}
  local RUN=runs/discrete_speech_lm_d1b_confirm_v2_seed${SEED}
  local TAG=confirm_v2_seed${SEED}
  local LOG=$WIN_LOG/${TAG}
  mkdir -p "$WIN_OUT/$TAG" "$LOG"

  if [ "$SKIP_TRAIN" -eq 0 ]; then
    echo "=== TRAIN $TAG sha=$SHA ===" | tee -a "$LOG/train.log"
    source .venv/bin/activate
    python scripts/72_train_discrete_lm.py \
      --config "$CFG" \
      --codec-tokens-dir "$CODEC" \
      --subset confirm32 \
      --subset-split full32 \
      --run-name "discrete_speech_lm_d1b_confirm_v2_seed${SEED}" \
      --seed "$SEED" \
      2>&1 | tee -a "$LOG/train.log"
  elif [ ! -f "$RUN/candidate_step_0750.pt" ]; then
    echo "=== RESUME TRAIN $TAG sha=$SHA ===" | tee -a "$LOG/train.log"
    source .venv/bin/activate
    python scripts/72_train_discrete_lm.py \
      --config "$CFG" \
      --codec-tokens-dir "$CODEC" \
      --subset confirm32 \
      --subset-split full32 \
      --run-name "discrete_speech_lm_d1b_confirm_v2_seed${SEED}" \
      --seed "$SEED" \
      --resume true \
      2>&1 | tee -a "$LOG/train.log"
  else
    echo "=== TRAIN already complete $TAG ===" | tee -a "$LOG/train.log"
  fi

  STEPS=(250 375 500 625 750)
  CAND_JSON=$WIN_MET/${TAG}_select8_candidate_rows.json
  echo '[' > "$CAND_JSON.tmp"
  local first=1
  for step in "${STEPS[@]}"; do
    local ckpt
    ckpt=$(printf "%s/candidate_step_%04d.pt" "$RUN" "$step")
    test -f "$ckpt" || { echo "MISSING $ckpt"; exit 3; }
    local tag
    tag=$(printf "cand_%04d" "$step")
    local out_dir=$WIN_OUT/$TAG/${tag}_select8_n2
    local wav_dir=$WIN_OUT/$TAG/${tag}_select8_wavs
    if [ -f "$out_dir/candidate_row.json" ]; then
      echo "=== skip select8 step=$step (exists) ===" | tee -a "$LOG/fr_select.log"
    else
      mkdir -p "$out_dir" "$wav_dir"
      echo "=== decode select8 step=$step ===" | tee -a "$LOG/fr_select.log"
      source .venv_qwen_tts/bin/activate
      python scripts/74_eval_discrete_lm.py \
        --ckpt "$ckpt" \
        --config "$CFG" \
        --codec-tokens-dir "$CODEC" \
        --subset select8 \
        --decode-only \
        --out "$out_dir" \
        --wav-dir "$wav_dir" \
        2>&1 | tee -a "$LOG/fr_select.log"
      deactivate || true
      source .venv/bin/activate
      python scripts/74_eval_discrete_lm.py \
        --score-only "$out_dir/n2_decode_bundle.json" \
        --out "$out_dir" \
        --asr-config configs/asr.yaml \
        --n2-gate \
        2>&1 | tee -a "$LOG/fr_select.log"
      python - <<PY
import json
from pathlib import Path
from minispeech.eval.discrete_n2 import candidate_from_eval_report, teacher_forced_ce_from_ckpt_metrics
from minispeech.data.d1b_oracles import load_frozen_utt_ids
import torch
gate = json.loads(Path("$out_dir/n2_gate.json").read_text())
model = (gate.get("reports") or {}).get("model") or {}
_,_,digest = load_frozen_utt_ids("select8")
tf = gate.get("teacher_forced_ce")
if tf is None:
    ck = torch.load("$ckpt", map_location="cpu", weights_only=False)
    tf = teacher_forced_ce_from_ckpt_metrics(ck.get("metrics") or {})
row = candidate_from_eval_report(model, ckpt="$ckpt", step=$step, selection_subset="select8", teacher_forced_ce=tf)
row["items_sha256"] = digest
row["gate_verdict"] = gate.get("verdict")
row["teacher_forced_ce"] = tf
Path("$out_dir/candidate_row.json").write_text(json.dumps(row, indent=2)+"\n")
print(json.dumps({"step": $step, "tf_ce": tf, "macro_cer": row.get("macro_cer")}, indent=2))
PY
    fi
    if [ "$first" -eq 1 ]; then first=0; else echo ',' >> "$CAND_JSON.tmp"; fi
    cat "$out_dir/candidate_row.json" >> "$CAND_JSON.tmp"
  done
  echo ']' >> "$CAND_JSON.tmp"
  mv "$CAND_JSON.tmp" "$CAND_JSON"

  python scripts/76_select_discrete_checkpoint.py \
    --candidates-json "$CAND_JSON" \
    --selection-subset select8 \
    --require-subset select8 \
    --verify-frozen-subset select8 \
    --require-items-sha256 "$SELECT8_SHA" \
    --out "$RUN/checkpoint_selection.json" \
    --promote-best \
    --best-path "$RUN/best.pt" \
    2>&1 | tee -a "$LOG/fr_select.log"
  cp -f "$RUN/checkpoint_selection.json" "$WIN_MET/${TAG}_selection.json"

  for SPLIT in gate24 full32; do
    local out_dir=$WIN_OUT/$TAG/best_${SPLIT}_n2
    local wav_dir=$WIN_OUT/$TAG/best_${SPLIT}_wavs
    if [ -f "$out_dir/n2_gate.json" ]; then
      echo "=== skip $SPLIT (exists) ===" | tee -a "$LOG/fr_select.log"
    else
      mkdir -p "$out_dir" "$wav_dir"
      echo "=== decode $SPLIT best ===" | tee -a "$LOG/fr_select.log"
      source .venv_qwen_tts/bin/activate
      python scripts/74_eval_discrete_lm.py \
        --ckpt "$RUN/best.pt" \
        --config "$CFG" \
        --codec-tokens-dir "$CODEC" \
        --subset "$SPLIT" \
        --decode-only \
        --out "$out_dir" \
        --wav-dir "$wav_dir" \
        2>&1 | tee -a "$LOG/fr_select.log"
      deactivate || true
      source .venv/bin/activate
      python scripts/74_eval_discrete_lm.py \
        --score-only "$out_dir/n2_decode_bundle.json" \
        --out "$out_dir" \
        --asr-config configs/asr.yaml \
        --n2-gate \
        2>&1 | tee -a "$LOG/fr_select.log"
    fi
    cp -f "$out_dir/n2_gate.json" "$WIN_MET/${TAG}_${SPLIT}_n2_gate.json"
  done

  if [ ! -d "$WIN_OUT/$TAG/listen_demo" ] || [ -z "$(ls -A "$WIN_OUT/$TAG/listen_demo" 2>/dev/null || true)" ]; then
    source .venv_qwen_tts/bin/activate
    python scripts/73_infer_discrete_lm.py \
      --ckpt "$RUN/best.pt" \
      --config "$CFG" \
      --codec-tokens-dir "$CODEC" \
      --subset select8 \
      --max-utts 8 \
      --decode \
      --out "$WIN_OUT/$TAG/listen_demo" \
      2>&1 | tee -a "$LOG/listen.log"
  fi
}

# Resume seed42 train if incomplete, then full finish; seed43 from scratch.
finish_seed 42 1
finish_seed 43 0
echo "=== CONFIRM v2 series finished ===" | tee -a "$WIN_LOG/confirm_v2_sha.txt"

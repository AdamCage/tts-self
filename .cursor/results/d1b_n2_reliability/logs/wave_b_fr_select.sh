#!/usr/bin/env bash
# Wave B — select8 FR candidate selection, then one-shot gate24 + full32 on best.
set -euo pipefail
cd ~/projects/tts-self
export HF_HOME=$HOME/hf_cache
unset MINISPEECH_ALLOW_DIRTY
WIN_LOG=/mnt/c/projects/tts-self/.cursor/results/d1b_n2_reliability/logs
WIN_OUT=/mnt/c/projects/tts-self/out/d1b_n2_reliability
WIN_MET=$WIN_OUT/metrics
RUN=runs/discrete_speech_lm_d1b_confirm_seed42
CODEC="$HOME/datasets/tts/ruslan/codec_tokens,$HOME/datasets/tts/common_voice_ru17/codec_tokens"
mkdir -p "$WIN_LOG" "$WIN_MET" "$WIN_OUT/confirm_seed42"
test -z "$(git status --porcelain)" || { echo DIRTY; git status --porcelain; exit 2; }

STEPS=(250 375 500 625 750)
CAND_JSON="$WIN_MET/confirm_seed42_select8_candidate_rows.json"
echo '[' > "$CAND_JSON.tmp"
first=1
SELECT8_SHA=fa083ed564809c01037292b1324bced72d219610cef42757b9c5155489d62651

for step in "${STEPS[@]}"; do
  ckpt=$(printf "%s/candidate_step_%04d.pt" "$RUN" "$step")
  tag=$(printf "cand_%04d" "$step")
  out_dir="$WIN_OUT/confirm_seed42/${tag}_select8_n2"
  wav_dir="$WIN_OUT/confirm_seed42/${tag}_select8_wavs"
  mkdir -p "$out_dir" "$wav_dir"
  echo "=== decode select8 $ckpt ===" | tee -a "$WIN_LOG/wave_b_fr_select.log"
  source .venv_qwen_tts/bin/activate
  python scripts/74_eval_discrete_lm.py \
    --ckpt "$ckpt" \
    --config configs/discrete_speech_lm_d1b_443m.yaml \
    --codec-tokens-dir "$CODEC" \
    --subset select8 \
    --decode-only \
    --out "$out_dir" \
    --wav-dir "$wav_dir" \
    2>&1 | tee -a "$WIN_LOG/wave_b_fr_select.log"
  deactivate || true
  source .venv/bin/activate
  python scripts/74_eval_discrete_lm.py \
    --score-only "$out_dir/n2_decode_bundle.json" \
    --out "$out_dir" \
    --asr-config configs/asr.yaml \
    --n2-gate \
    2>&1 | tee -a "$WIN_LOG/wave_b_fr_select.log"
  python - <<PY
import json
from pathlib import Path
from minispeech.eval.discrete_n2 import candidate_from_eval_report
from minispeech.data.d1b_oracles import load_frozen_utt_ids
gate = json.loads(Path("$out_dir/n2_gate.json").read_text())
model = (gate.get("reports") or {}).get("model") or {}
_,_,digest = load_frozen_utt_ids("select8")
row = candidate_from_eval_report(model, ckpt="$ckpt", step=$step, selection_subset="select8")
row["items_sha256"] = digest
row["gate_verdict"] = gate.get("verdict")
Path("$out_dir/candidate_row.json").write_text(json.dumps(row, indent=2)+"\n")
print(json.dumps(row, indent=2))
PY
  if [ "$first" -eq 1 ]; then first=0; else echo ',' >> "$CAND_JSON.tmp"; fi
  cat "$out_dir/candidate_row.json" >> "$CAND_JSON.tmp"
done
echo ']' >> "$CAND_JSON.tmp"
mv "$CAND_JSON.tmp" "$CAND_JSON"

source .venv/bin/activate
python scripts/76_select_discrete_checkpoint.py \
  --candidates-json "$CAND_JSON" \
  --selection-subset select8 \
  --require-subset select8 \
  --verify-frozen-subset select8 \
  --require-items-sha256 "$SELECT8_SHA" \
  --out "$RUN/checkpoint_selection.json" \
  --promote-best \
  --best-path "$RUN/best.pt" \
  2>&1 | tee -a "$WIN_LOG/wave_b_fr_select.log"
cp -f "$RUN/checkpoint_selection.json" "$WIN_MET/confirm_seed42_selection.json"

# One-shot gate24 then full32 on promoted best — no recipe changes after.
for SPLIT in gate24 full32; do
  out_dir="$WIN_OUT/confirm_seed42/best_${SPLIT}_n2"
  wav_dir="$WIN_OUT/confirm_seed42/best_${SPLIT}_wavs"
  mkdir -p "$out_dir" "$wav_dir"
  echo "=== decode $SPLIT best ===" | tee -a "$WIN_LOG/wave_b_fr_select.log"
  source .venv_qwen_tts/bin/activate
  python scripts/74_eval_discrete_lm.py \
    --ckpt "$RUN/best.pt" \
    --config configs/discrete_speech_lm_d1b_443m.yaml \
    --codec-tokens-dir "$CODEC" \
    --subset "$SPLIT" \
    --decode-only \
    --out "$out_dir" \
    --wav-dir "$wav_dir" \
    2>&1 | tee -a "$WIN_LOG/wave_b_fr_select.log"
  deactivate || true
  source .venv/bin/activate
  python scripts/74_eval_discrete_lm.py \
    --score-only "$out_dir/n2_decode_bundle.json" \
    --out "$out_dir" \
    --asr-config configs/asr.yaml \
    --n2-gate \
    2>&1 | tee -a "$WIN_LOG/wave_b_fr_select.log"
  cp -f "$out_dir/n2_gate.json" "$WIN_MET/confirm_seed42_${SPLIT}_n2_gate.json"
done

# Listen demo
source .venv_qwen_tts/bin/activate
python scripts/73_infer_discrete_lm.py \
  --ckpt "$RUN/best.pt" \
  --config configs/discrete_speech_lm_d1b_443m.yaml \
  --codec-tokens-dir "$CODEC" \
  --subset select8 \
  --max-utts 8 \
  --decode \
  --out "$WIN_OUT/confirm_seed42/listen_demo" \
  2>&1 | tee -a "$WIN_LOG/wave_b_listen.log"

echo "=== Wave B FR+gates done ===" | tee -a "$WIN_LOG/wave_b_fr_select.log"

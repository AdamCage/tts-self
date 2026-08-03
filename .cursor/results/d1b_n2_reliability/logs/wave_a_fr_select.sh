#!/usr/bin/env bash
# Wave A — offline FR candidate decode/score/select on DEV32.
set -euo pipefail
cd ~/projects/tts-self
export HF_HOME=$HOME/hf_cache
unset MINISPEECH_ALLOW_DIRTY

WIN_LOG=/mnt/c/projects/tts-self/.cursor/results/d1b_n2_reliability/logs
WIN_OUT=/mnt/c/projects/tts-self/out/d1b_n2_reliability
WIN_MET=$WIN_OUT/metrics
RUN=runs/discrete_speech_lm_d1b_dev_seed42
CODEC="$HOME/datasets/tts/ruslan/codec_tokens,$HOME/datasets/tts/common_voice_ru17/codec_tokens"
mkdir -p "$WIN_LOG" "$WIN_MET" "$WIN_OUT/dev_seed42"

test -z "$(git status --porcelain)" || { echo DIRTY; git status --porcelain; exit 2; }

STEPS=(250 375 500 625 750)
CAND_JSON="$WIN_MET/dev_seed42_candidate_rows.json"
echo '[' > "$CAND_JSON.tmp"
first=1

for step in "${STEPS[@]}"; do
  ckpt=$(printf "%s/candidate_step_%04d.pt" "$RUN" "$step")
  tag=$(printf "cand_%04d" "$step")
  out_dir="$WIN_OUT/dev_seed42/${tag}_n2"
  wav_dir="$WIN_OUT/dev_seed42/${tag}_wavs"
  mkdir -p "$out_dir" "$wav_dir"
  echo "=== decode $ckpt ===" | tee -a "$WIN_LOG/wave_a_fr_select.log"
  source .venv_qwen_tts/bin/activate
  python scripts/74_eval_discrete_lm.py \
    --ckpt "$ckpt" \
    --config configs/discrete_speech_lm_d1b_443m.yaml \
    --codec-tokens-dir "$CODEC" \
    --subset dev32 \
    --decode-only \
    --out "$out_dir" \
    --wav-dir "$wav_dir" \
    2>&1 | tee -a "$WIN_LOG/wave_a_fr_select.log"
  deactivate || true
  source .venv/bin/activate
  echo "=== score $tag ===" | tee -a "$WIN_LOG/wave_a_fr_select.log"
  python scripts/74_eval_discrete_lm.py \
    --score-only "$out_dir/n2_decode_bundle.json" \
    --out "$out_dir" \
    --asr-config configs/asr.yaml \
    --n2-gate \
    2>&1 | tee -a "$WIN_LOG/wave_a_fr_select.log"
  # Extract candidate row from n2_gate / model report
  python - <<PY
import json
from pathlib import Path
from minispeech.eval.discrete_n2 import candidate_from_eval_report, items_sha256
from minispeech.data.d1b_oracles import load_frozen_utt_ids
gate = json.loads(Path("$out_dir/n2_gate.json").read_text())
model = (gate.get("reports") or {}).get("model") or {}
ids,_,digest = load_frozen_utt_ids("dev32")
row = candidate_from_eval_report(
    model,
    ckpt="$ckpt",
    step=$step,
    selection_subset="dev32",
)
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
  --selection-subset dev32 \
  --require-subset dev32 \
  --verify-frozen-subset dev32 \
  --out "$RUN/checkpoint_selection.json" \
  --promote-best \
  --best-path "$RUN/best.pt" \
  2>&1 | tee -a "$WIN_LOG/wave_a_fr_select.log"
cp -f "$RUN/checkpoint_selection.json" "$WIN_MET/dev_seed42_selection.json"
echo "=== Wave A FR select done ===" | tee -a "$WIN_LOG/wave_a_fr_select.log"

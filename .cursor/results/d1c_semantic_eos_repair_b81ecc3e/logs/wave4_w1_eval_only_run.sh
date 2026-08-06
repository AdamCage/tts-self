#!/usr/bin/env bash
# Wave 4 eval-only resume: select8 candidates → select → listen → gate24 → full32
# Assumes runs/d1c_eos_dev_w1_seed101/candidate_step_*.pt already exist.
set -euo pipefail

PLAN_SLUG=d1c_semantic_eos_repair_b81ecc3e
RUN_NAME=d1c_eos_dev_w1_seed101
SEED=101
CONFIG=configs/discrete_speech_lm_d1c_semantic_eos_w1.yaml
CONTRACT=configs/eval_contracts/d1c_semantic_eos_v2_0_0.json
SUBSET=configs/train_subsets/d1c_dev32.json
CODEC_DIRS="$HOME/datasets/tts/ruslan/codec_tokens,$HOME/datasets/tts/common_voice_ru17/codec_tokens"
EXT_LOGDIR="$HOME/datasets/tts/d1c_wave4_logs"
OUT_ROOT="out/${PLAN_SLUG}/dev/w1_seed101"
WIN_OUT="/mnt/c/projects/tts-self/out/${PLAN_SLUG}/dev/w1_seed101"
WIN_D1C="/mnt/c/projects/tts-self-d1c/out/${PLAN_SLUG}/dev/w1_seed101"
WIN_RES="/mnt/c/projects/tts-self-d1c/.cursor/results/${PLAN_SLUG}"
RESULTS="$HOME/projects/tts-self/.cursor/results/${PLAN_SLUG}"

unset GIT_DIR GIT_WORK_TREE MINISPEECH_ALLOW_DIRTY
cd ~/projects/tts-self

if [ -n "$(git status --porcelain)" ]; then
  echo DIRTY_WORKTREE
  git status -sb
  exit 2
fi
TIP=$(git rev-parse HEAD)
SHORT=$(git rev-parse --short HEAD)

mkdir -p "$EXT_LOGDIR" "$OUT_ROOT" "$RESULTS/logs" "$RESULTS/artifacts" \
  "$WIN_OUT" "$WIN_D1C" "$WIN_RES/logs" "$WIN_RES/artifacts"
CONSOLE_LOG="$EXT_LOGDIR/wave4_w1_eval_console.log"
exec > >(tee -a "$CONSOLE_LOG") 2>&1

echo "=== Wave 4 DEV W1 eval resume $(date -Is) tip=$TIP ==="

source .venv/bin/activate
export HF_HOME="$HOME/hf_cache"
export PYTHONPATH="$PWD/src"
uv pip install -e . --no-deps

RUN=runs/${RUN_NAME}
test -f "$RUN/candidate_step_0750.pt"
ls -la "$RUN"/candidate_step_*.pt | tee "$EXT_LOGDIR/wave4_w1_candidates.txt"

STEPS=(250 375 500 625 750)
REPORTS=()
for step in "${STEPS[@]}"; do
  ckpt=$(printf "%s/candidate_step_%04d.pt" "$RUN" "$step")
  tag=$(printf "cand_%04d" "$step")
  out_dir="$OUT_ROOT/select8/${tag}"
  wav_dir="$out_dir/wavs"
  mkdir -p "$out_dir" "$wav_dir"
  echo "=== DECODE select8 $ckpt ==="
  # shellcheck disable=SC1091
  source .venv_qwen_tts/bin/activate
  export HF_HOME="$HOME/hf_cache"
  export PYTHONPATH="$PWD/src"
  python scripts/74_eval_discrete_lm.py \
    --ckpt "$ckpt" \
    --config "$CONFIG" \
    --eval-contract "$CONTRACT" \
    --codec-tokens-dir "$CODEC_DIRS" \
    --subset "$SUBSET" \
    --subset-split select8 \
    --speaker-map configs/speaker_maps/trackn_v2.json \
    --decode-only \
    --n2-gate \
    --out "$out_dir" \
    --wav-dir "$wav_dir" \
    --checkpoint-step "$step" \
    2>&1 | tee -a "$EXT_LOGDIR/wave4_w1_select8_eval.log"
  # shellcheck disable=SC1091
  source .venv/bin/activate
  export PYTHONPATH="$PWD/src"
  echo "=== SCORE select8 $tag ==="
  python scripts/74_eval_discrete_lm.py \
    --score-only "$out_dir/n2_decode_bundle.json" \
    --out "$out_dir" \
    --asr-config configs/asr.yaml \
    --eval-contract "$CONTRACT" \
    --n2-gate \
    2>&1 | tee -a "$EXT_LOGDIR/wave4_w1_select8_eval.log"
  report="$out_dir/discrete_eval_report_v2.json"
  test -f "$report" || { echo "MISSING_FORMAL_REPORT $out_dir"; ls -la "$out_dir"; exit 3; }
  REPORTS+=("$report")
  echo "FORMAL_REPORT $report"
done

echo "=== SELECT d1c formal ==="
python scripts/76_select_discrete_checkpoint.py \
  --d1c-formal \
  --formal-reports "${REPORTS[@]}" \
  --out "$RUN/checkpoint_selection.json" \
  --promote-best \
  --best-path "$RUN/best.pt" \
  --eval-contract "$CONTRACT" \
  2>&1 | tee "$EXT_LOGDIR/wave4_w1_selection.log"

test -f "$RUN/best.pt"
cp -f "$RUN/checkpoint_selection.json" "$RESULTS/artifacts/selection_w1_seed101.json"
cp -f "$RUN/checkpoint_selection.json" "$WIN_RES/artifacts/selection_w1_seed101.json"

for which in best last; do
  ckpt="$RUN/${which}.pt"
  listen="$OUT_ROOT/${which}_listen_demo"
  mkdir -p "$listen"
  echo "=== LISTEN $which ==="
  source .venv/bin/activate
  export PYTHONPATH="$PWD/src"
  python scripts/73_infer_discrete_lm.py \
    --ckpt "$ckpt" \
    --config "$CONFIG" \
    --eval-contract "$CONTRACT" \
    --codec-tokens-dir "$CODEC_DIRS" \
    --subset "$SUBSET" \
    --subset-split select8 \
    --speaker-map configs/speaker_maps/trackn_v2.json \
    --out "$listen" \
    --seed "$SEED" \
    --codes-only \
    2>&1 | tee -a "$EXT_LOGDIR/wave4_w1_listen.log"
  source .venv_qwen_tts/bin/activate
  export PYTHONPATH="$PWD/src"
  python scripts/73_infer_discrete_lm.py \
    --ckpt "$ckpt" \
    --config "$CONFIG" \
    --eval-contract "$CONTRACT" \
    --codec-tokens-dir "$CODEC_DIRS" \
    --subset "$SUBSET" \
    --subset-split select8 \
    --speaker-map configs/speaker_maps/trackn_v2.json \
    --out "$listen" \
    --seed "$SEED" \
    --decode \
    2>&1 | tee -a "$EXT_LOGDIR/wave4_w1_listen.log"
  source .venv/bin/activate
  export PYTHONPATH="$PWD/src"
  mkdir -p "$WIN_OUT/${which}_listen_demo" "$WIN_D1C/${which}_listen_demo"
  cp -f "$listen"/*.wav "$WIN_OUT/${which}_listen_demo/" 2>/dev/null || true
  cp -f "$listen"/*.wav "$WIN_D1C/${which}_listen_demo/" 2>/dev/null || true
done

for split in gate24 full32; do
  out_dir="$OUT_ROOT/${split}"
  wav_dir="$out_dir/wavs"
  mkdir -p "$out_dir" "$wav_dir"
  echo "=== DECODE $split best.pt ==="
  source .venv_qwen_tts/bin/activate
  export PYTHONPATH="$PWD/src"
  python scripts/74_eval_discrete_lm.py \
    --ckpt "$RUN/best.pt" \
    --config "$CONFIG" \
    --eval-contract "$CONTRACT" \
    --codec-tokens-dir "$CODEC_DIRS" \
    --subset "$SUBSET" \
    --subset-split "$split" \
    --speaker-map configs/speaker_maps/trackn_v2.json \
    --decode-only \
    --n2-gate \
    --out "$out_dir" \
    --wav-dir "$wav_dir" \
    2>&1 | tee -a "$EXT_LOGDIR/wave4_w1_${split}_eval.log"
  source .venv/bin/activate
  export PYTHONPATH="$PWD/src"
  echo "=== SCORE $split ==="
  python scripts/74_eval_discrete_lm.py \
    --score-only "$out_dir/n2_decode_bundle.json" \
    --out "$out_dir" \
    --asr-config configs/asr.yaml \
    --eval-contract "$CONTRACT" \
    --n2-gate \
    2>&1 | tee -a "$EXT_LOGDIR/wave4_w1_${split}_eval.log"
  cp -f "$out_dir/n2_gate.json" "$RESULTS/artifacts/gate_w1_seed101_${split}.json" 2>/dev/null || true
  cp -f "$out_dir/n2_gate.json" "$WIN_RES/artifacts/gate_w1_seed101_${split}.json" 2>/dev/null || true
  if [[ -f "$out_dir/discrete_eval_report_v2.json" ]]; then
    cp -f "$out_dir/discrete_eval_report_v2.json" \
      "$RESULTS/artifacts/eval_report_w1_seed101_${split}.json"
    cp -f "$out_dir/discrete_eval_report_v2.json" \
      "$WIN_RES/artifacts/eval_report_w1_seed101_${split}.json"
  fi
done

cp -f "$EXT_LOGDIR"/wave4_w1_*.log "$RESULTS/logs/" 2>/dev/null || true
cp -f "$EXT_LOGDIR"/wave4_w1_*.txt "$RESULTS/logs/" 2>/dev/null || true
cp -f "$EXT_LOGDIR"/wave4_w1_*.log "$WIN_RES/logs/" 2>/dev/null || true
cp -f "$EXT_LOGDIR"/wave4_w1_*.txt "$WIN_RES/logs/" 2>/dev/null || true

echo "=== Wave 4 DEV W1 seed101 DONE $(date -Is) tip=$SHORT ==="

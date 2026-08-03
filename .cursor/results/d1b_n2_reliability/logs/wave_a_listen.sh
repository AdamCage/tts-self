#!/usr/bin/env bash
# Mandatory checkpoint-listen after best.pt (DEV seed42).
set -euo pipefail
cd ~/projects/tts-self
export HF_HOME=$HOME/hf_cache
unset MINISPEECH_ALLOW_DIRTY
test -z "$(git status --porcelain)" || { echo DIRTY; git status --porcelain; exit 2; }
WIN_OUT=/mnt/c/projects/tts-self/out/d1b_n2_reliability/dev_seed42/listen_demo
WIN_LOG=/mnt/c/projects/tts-self/.cursor/results/d1b_n2_reliability/logs
mkdir -p "$WIN_OUT" "$WIN_LOG"
CODEC="$HOME/datasets/tts/ruslan/codec_tokens,$HOME/datasets/tts/common_voice_ru17/codec_tokens"
source .venv_qwen_tts/bin/activate
python scripts/73_infer_discrete_lm.py \
  --ckpt runs/discrete_speech_lm_d1b_dev_seed42/best.pt \
  --config configs/discrete_speech_lm_d1b_443m.yaml \
  --codec-tokens-dir "$CODEC" \
  --subset dev32 \
  --max-utts 8 \
  --decode \
  --out "$WIN_OUT" \
  2>&1 | tee "$WIN_LOG/wave_a_listen.log"
echo "listen wavs in $WIN_OUT"

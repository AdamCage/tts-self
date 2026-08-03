#!/usr/bin/env bash
# Wave C — CONFIRM32 seed43 train (frozen recipe). Only after Wave B train done.
set -euo pipefail
cd ~/projects/tts-self
export HF_HOME=$HOME/hf_cache
unset MINISPEECH_ALLOW_DIRTY
WIN_LOG=/mnt/c/projects/tts-self/.cursor/results/d1b_n2_reliability/logs
mkdir -p "$WIN_LOG"
test -z "$(git status --porcelain)" || { echo DIRTY; git status --porcelain; exit 2; }
source .venv/bin/activate
python scripts/72_train_discrete_lm.py \
  --config configs/discrete_speech_lm_d1b_443m.yaml \
  --codec-tokens-dir "$HOME/datasets/tts/ruslan/codec_tokens,$HOME/datasets/tts/common_voice_ru17/codec_tokens" \
  --subset confirm32 \
  --subset-split full32 \
  --run-name discrete_speech_lm_d1b_confirm_seed43 \
  --seed 43 \
  2>&1 | tee "$WIN_LOG/wave_c_confirm43_train.log"

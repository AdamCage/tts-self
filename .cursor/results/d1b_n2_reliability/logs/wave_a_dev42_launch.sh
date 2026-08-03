#!/usr/bin/env bash
# Wave A — DEV32 seed42 train (D1b). Logs live on Windows mount (not WSL git tree).
set -euo pipefail
cd ~/projects/tts-self
source .venv/bin/activate
export HF_HOME=$HOME/hf_cache
unset MINISPEECH_ALLOW_DIRTY

WIN_LOG="/mnt/c/projects/tts-self/.cursor/results/d1b_n2_reliability/logs"
WIN_OUT="/mnt/c/projects/tts-self/out/d1b_n2_reliability"
mkdir -p "$WIN_LOG" "$WIN_OUT/frozen" runs
cp -f configs/eval_contracts/d1b_n2_reliability.json "$WIN_OUT/frozen/eval_contract.json"
cp -f configs/speaker_maps/trackn_v2.json "$WIN_OUT/frozen/speaker_map.json"

{
  echo "=== Wave A preflight $(date -Is) ==="
  git rev-parse HEAD
  echo "porcelain_bytes=$(git status --porcelain | wc -c)"
  git status --porcelain
  python -V
  python - <<'PY'
import torch
print("torch", torch.__version__, "cuda", torch.cuda.is_available(), torch.cuda.get_device_name(0))
PY
  nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv
} | tee "$WIN_LOG/wave_a_preflight.log"

test -z "$(git status --porcelain)" || { echo "DIRTY TREE — abort"; exit 2; }

CODEC_DIRS="$HOME/datasets/tts/ruslan/codec_tokens,$HOME/datasets/tts/common_voice_ru17/codec_tokens"

python scripts/72_train_discrete_lm.py \
  --config configs/discrete_speech_lm_d1b_443m.yaml \
  --subset dev32 \
  --run-name discrete_speech_lm_d1b_dev_seed42 \
  --seed 42 \
  --stage S1 \
  --codec-tokens-dir "$CODEC_DIRS" \
  2>&1 | tee "$WIN_LOG/wave_a_dev42_train.log"

echo "=== Wave A train done $(date -Is) ===" | tee -a "$WIN_LOG/wave_a_dev42_train.log"

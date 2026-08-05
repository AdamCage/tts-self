#!/usr/bin/env bash
# Internal launcher for WSL (uses uv + repo .venv).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

export HF_HOME="${HF_HOME:-$HOME/hf_cache}"
export PATH="$HOME/.local/bin:$PATH"

if [[ ! -x "${ROOT}/.venv/bin/python" ]]; then
  echo "missing ${ROOT}/.venv — create with: uv venv .venv && uv pip install -e '.[qwen_tts,stress]'" >&2
  exit 1
fi

# shellcheck disable=SC1091
source "${ROOT}/.venv/bin/activate"

uv pip install -r listen_service/requirements.txt

if [[ ! -f listen_service/.env ]]; then
  cp listen_service/.env.example listen_service/.env
fi

# Force known-good local paths if still placeholders / missing
if [[ ! -f "${CKPT_PATH:-}" ]]; then
  export CKPT_PATH="${ROOT}/runs/discrete_speech_lm_d1b_confirm_v2_seed43/best.pt"
fi
if [[ ! -d "${TOKENIZER_PATH:-}" ]]; then
  export TOKENIZER_PATH="${HOME}/datasets/tts/models/Qwen3.5-0.8B-Base"
fi
export DEVICE="${DEVICE:-cuda}"
export HOST="${HOST:-127.0.0.1}"
export PORT="${PORT:-7860}"
export SPEAKER_ID="${SPEAKER_ID:-1088}"

# Refresh .env for the app's dotenv loader
cat > listen_service/.env <<EOF
CKPT_PATH=${CKPT_PATH}
TOKENIZER_PATH=${TOKENIZER_PATH}
DEVICE=${DEVICE}
HOST=${HOST}
PORT=${PORT}
SPEAKER_ID=${SPEAKER_ID}
HF_HOME=${HF_HOME}
EOF

.venv/bin/python - <<'PY'
import gradio
import torch

print("gradio", gradio.__version__)
print("cuda", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu", torch.cuda.get_device_name(0))
PY

exec .venv/bin/python listen_service/app.py --host "${HOST}" --port "${PORT}" --preload

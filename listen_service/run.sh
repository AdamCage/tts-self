#!/usr/bin/env bash
# Launch FastAPI listen UI (uses uv + repo/service .venv).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "${ROOT}/.." && pwd)"
cd "${ROOT}"

if [[ -f "${ROOT}/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "${ROOT}/.env"
  set +a
fi

export HF_HOME="${HF_HOME:-$HOME/hf_cache}"
export PATH="$HOME/.local/bin:$PATH"

PYTHON="${ROOT}/.venv/bin/python"
if [[ ! -x "${PYTHON}" ]]; then
  PYTHON="${REPO}/.venv/bin/python"
fi
if [[ ! -x "${PYTHON}" ]]; then
  PYTHON="$(command -v python3 || command -v python)"
fi

if command -v uv >/dev/null 2>&1; then
  uv pip install -r "${ROOT}/requirements.txt" --python "${PYTHON}"
fi

exec "${PYTHON}" "${ROOT}/app.py" --host "${HOST:-127.0.0.1}" --port "${PORT:-7860}"

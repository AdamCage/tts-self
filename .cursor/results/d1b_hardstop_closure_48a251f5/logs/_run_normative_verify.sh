#!/usr/bin/env bash
set -euo pipefail
cd "${HOME}/projects/tts-self"
git fetch origin
git reset --hard origin/master
git clean -fd
SHA=$(git rev-parse HEAD)
PORC=$(git status --porcelain | wc -l | tr -d ' ')
echo "CLEAN_SHA=${SHA}"
echo "PORCELAIN_COUNT=${PORC}"
test "${PORC}" = "0"

export PATH="${HOME}/.local/bin:${PATH}"
# shellcheck disable=SC1091
source .venv/bin/activate
export HF_HOME="${HOME}/hf_cache"
export PYTHONPATH=src
unset MINISPEECH_ALLOW_DIRTY || true
uv pip install -e . --no-deps -q

# Strip CRLF into a temp script so we do not dirty the tracked tree before the check.
TMP_VERIFY=$(mktemp)
tr -d '\r' < .cursor/results/d1b_hardstop_closure_48a251f5/logs/hardstop_closure_verify.sh > "${TMP_VERIFY}"
chmod +x "${TMP_VERIFY}"
bash "${TMP_VERIFY}"
RC=$?
rm -f "${TMP_VERIFY}"
echo "VERIFY_DONE rc=${RC} CLEAN_SHA=${SHA}"
exit "${RC}"

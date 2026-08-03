#!/usr/bin/env bash
# Clean WSL worktree, reinstall editable from Linux FS, run seed43 only.
set -euo pipefail
cd ~/projects/tts-self
export HF_HOME=$HOME/hf_cache
unset MINISPEECH_ALLOW_DIRTY

# Remove accidental untracked results that dirtied the tree.
rm -rf .cursor/results/d1b_n2_reliability

source .venv/bin/activate
echo "pth before:"; cat .venv/lib/python3.11/site-packages/_editable_impl_minispeech.pth || true
uv pip install -e . --no-deps
echo "pth after:"; cat .venv/lib/python3.11/site-packages/_editable_impl_minispeech.pth
python -c "import minispeech.train.discrete_lm_trainer as m; print('import', m.__file__)"

git fetch origin
git pull --ff-only origin master
test -z "$(git status --porcelain)" || { echo DIRTY; git status --porcelain; exit 2; }
SHA=$(git rev-parse HEAD)
echo "CLEAN_SHA=$SHA"

# Patch launcher pipefail for train: re-run seed43 via finish_seed logic inline.
WIN=/mnt/c/projects/tts-self
WIN_LOG=$WIN/.cursor/results/d1b_n2_reliability/logs
WIN_MET=$WIN/out/d1b_n2_reliability/metrics
OUT=$HOME/projects/tts-self/out/d1b_n2_reliability
CODEC="$HOME/datasets/tts/ruslan/codec_tokens,$HOME/datasets/tts/common_voice_ru17/codec_tokens"
SELECT8_SHA=6667da2f8fc2c3706deff48a79489103f739419041f3fb492f5822647075a0ed
CFG=configs/discrete_speech_lm_d1b_443m.yaml
mkdir -p "$WIN_LOG" "$WIN_MET" "$OUT"

# Source finish_seed by running a one-seed wrapper.
# shellcheck disable=SC1091
export CLEAN_SHA_OVERRIDE=$SHA
bash "$WIN/.cursor/results/d1b_n2_reliability/logs/run_seed43_only.sh"

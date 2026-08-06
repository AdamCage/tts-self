#!/usr/bin/env bash
set -euo pipefail
cd ~/projects/tts-self
git fetch origin
git checkout d1c/semantic-eos
git reset --hard origin/d1c/semantic-eos
# drop untracked wave3 logs from prior attempt (keep aside)
mkdir -p ~/tmp/wave3_wsl_scratch
if [ -n "$(git status --porcelain)" ]; then
  # move untracked results logs aside if any
  git status --porcelain | awk '{print $2}' | while read -r f; do
    if [ -e "$f" ]; then
      mkdir -p ~/tmp/wave3_wsl_scratch/"$(dirname "$f")"
      mv "$f" ~/tmp/wave3_wsl_scratch/"$f" 2>/dev/null || rm -rf "$f"
    fi
  done
fi
if [ -n "$(git status --porcelain)" ]; then
  echo DIRTY_WORKTREE
  git status -sb
  exit 2
fi
echo CLEAN
echo "SHA=$(git rev-parse HEAD)"
echo "SHORT=$(git rev-parse --short HEAD)"
source .venv/bin/activate
export HF_HOME="$HOME/hf_cache"
export PYTHONPATH="$PWD/src"
export CUDA_VISIBLE_DEVICES=
unset MINISPEECH_ALLOW_DIRTY || true
uv pip install -e . --no-deps
python - <<'PY'
from pathlib import Path
from minispeech.data.d1b_oracles import load_eval_contract, verify_recipe_source_manifest
c=load_eval_contract('configs/eval_contracts/d1c_semantic_eos_v2_0_0.json')
mm=verify_recipe_source_manifest(c['recipe_source_manifest'], root=Path('.'))
assert mm==[], mm
print('SEAL_OK', c['sha256'][:16], c['recipe_source_manifest_sha256'][:16])
PY

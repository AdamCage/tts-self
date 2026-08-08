#!/usr/bin/env bash
set -euo pipefail
cd ~/projects/tts-self
if [ -n "$(git status --porcelain)" ]; then
  echo DIRTY_WORKTREE
  git status -sb
  exit 2
fi
source .venv/bin/activate
export HF_HOME="$HOME/hf_cache"
export PYTHONPATH="$PWD/src"
unset MINISPEECH_ALLOW_DIRTY || true
RESULTS="$PWD/.cursor/results/d1c_semantic_eos_repair_b81ecc3e/logs"
mkdir -p "$RESULTS"

echo "=== focused Wave3 unit gates ==="
{
  python -m pytest -q tests/test_d1c_semantic_eos.py
  python -m pytest -q tests/test_discrete_eval_report_schema.py
  python -m pytest -q tests/test_d1c_oracles.py
  python -m pytest -q tests/test_d1c_selector_integration.py
  python -m pytest -q tests/test_discrete_stop_config.py
  python -m pytest -q tests/test_discrete_speech_lm.py
  python -m pytest -q tests/test_discrete_lm_trainer_steps.py tests/test_discrete_lm_resume.py
  python -m pytest -q tests/test_d1c_infra_repair.py
} 2>&1 | tee "$RESULTS/wave3_focused_pytest.log"

echo "=== full pytest ==="
python -m pytest -q 2>&1 | tee "$RESULTS/wave3_full_pytest.log"

echo "WAVE3_CPU_GATES_DONE"

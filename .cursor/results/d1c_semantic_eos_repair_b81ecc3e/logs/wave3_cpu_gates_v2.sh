#!/usr/bin/env bash
set -euo pipefail
cd ~/projects/tts-self
source .venv/bin/activate
export HF_HOME="$HOME/hf_cache"
export PYTHONPATH="$PWD/src"
export CUDA_VISIBLE_DEVICES=
RESULTS="$PWD/.cursor/results/d1c_semantic_eos_repair_b81ecc3e/logs"
mkdir -p "$RESULTS"
WT=/mnt/c/projects/tts-self-d1c/.cursor/results/d1c_semantic_eos_repair_b81ecc3e/logs

echo "=== focused Wave3 unit gates (CUDA masked) ==="
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
cp -a "$RESULTS/wave3_focused_pytest.log" "$WT/wave3_focused_pytest.log"

echo "=== full pytest (CUDA masked) ==="
python -m pytest -q 2>&1 | tee "$RESULTS/wave3_full_pytest.log"
cp -a "$RESULTS/wave3_full_pytest.log" "$WT/wave3_full_pytest.log"
echo WAVE3_CPU_GATES_DONE

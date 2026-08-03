#!/usr/bin/env bash
set -euo pipefail
ROOT=/mnt/c/projects/tts-self/out/d1b_n2_reliability/dev_seed42
echo "=== wav counts cand_0250 ==="
for d in model copy_synth wrong_utt random_codes shuffle_text mask_text; do
  n=$(ls "$ROOT/cand_0250_wavs/$d" 2>/dev/null | wc -l)
  echo "$d: $n"
done
echo "=== bundle/gate files ==="
ls -la "$ROOT"/cand_*_n2/n2_*.json 2>/dev/null || true
ls -la "$ROOT"/cand_*_n2/candidate_row.json 2>/dev/null || true
echo "=== GPU ==="
nvidia-smi --query-compute-apps=pid,used_memory,process_name --format=csv 2>/dev/null || true
echo "=== tmux ==="
tmux list-sessions 2>/dev/null || true
echo "=== pane tail ==="
tmux capture-pane -t d1b_fr_a -p 2>/dev/null | tail -30 || true
echo "=== log tail ==="
tail -20 /mnt/c/projects/tts-self/.cursor/results/d1b_n2_reliability/logs/wave_a_fr_select.log 2>/dev/null || true

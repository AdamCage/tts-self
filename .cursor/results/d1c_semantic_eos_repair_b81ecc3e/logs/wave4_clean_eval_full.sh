#!/usr/bin/env bash
# Full clean-tree Wave4 eval (decode+score) using existing candidates on tip c7e3ee0.
set -euo pipefail
cd ~/projects/tts-self
unset GIT_DIR GIT_WORK_TREE MINISPEECH_ALLOW_DIRTY
EXT="$HOME/datasets/tts/d1c_wave4_logs"
PLAN=d1c_semantic_eos_repair_b81ecc3e
OUT_ROOT="out/${PLAN}/dev/w1_seed101"
RUN=runs/d1c_eos_dev_w1_seed101
CONFIG=configs/discrete_speech_lm_d1c_semantic_eos_w1.yaml
CONTRACT=configs/eval_contracts/d1c_semantic_eos_v2_0_0.json
SUBSET=configs/train_subsets/d1c_dev32.json
CODEC_DIRS="$HOME/datasets/tts/ruslan/codec_tokens,$HOME/datasets/tts/common_voice_ru17/codec_tokens"
SEED=101
WIN_OUT="/mnt/c/projects/tts-self/out/${PLAN}/dev/w1_seed101"
WIN_D1C="/mnt/c/projects/tts-self-d1c/out/${PLAN}/dev/w1_seed101"
WIN_RES="/mnt/c/projects/tts-self-d1c/.cursor/results/${PLAN}"

mkdir -p "$EXT/results_staging/artifacts" "$EXT/results_staging/logs" "$OUT_ROOT"

# Move any untracked results noise out of the worktree
if [ -n "$(git status --porcelain)" ]; then
  while IFS= read -r line; do
    path="${line:3}"
    case "$path" in
      .cursor/results/*)
        mkdir -p "$EXT/dirty_bak/$(dirname "$path")"
        mv -f "$path" "$EXT/dirty_bak/$path" 2>/dev/null || true
        ;;
    esac
  done < <(git status --porcelain)
fi
test -z "$(git status --porcelain)" || { echo STILL_DIRTY; git status -sb; exit 2; }
TIP=$(git rev-parse HEAD)
echo "CLEAN_TIP=$TIP"

source .venv/bin/activate
export HF_HOME="$HOME/hf_cache"
export PYTHONPATH="$PWD/src"
python - <<'PY'
from minispeech.utils import git_is_dirty, git_sha
assert not git_is_dirty()
print("LIVE_CLEAN", git_sha())
PY

LOG="$EXT/wave4_w1_clean_eval.console.log"
exec > >(tee -a "$LOG") 2>&1

test -f "$RUN/candidate_step_0750.pt"
REPORTS=()
STEPS=(250 375 500 625 750)
for step in "${STEPS[@]}"; do
  ckpt=$(printf "%s/candidate_step_%04d.pt" "$RUN" "$step")
  tag=$(printf "cand_%04d" "$step")
  out_dir="$OUT_ROOT/select8/${tag}"
  wav_dir="$out_dir/wavs"
  mkdir -p "$out_dir" "$wav_dir"
  echo "=== DECODE select8 $ckpt ==="
  source .venv_qwen_tts/bin/activate
  export HF_HOME="$HOME/hf_cache"
  export PYTHONPATH="$PWD/src"
  python scripts/74_eval_discrete_lm.py \
    --ckpt "$ckpt" \
    --config "$CONFIG" \
    --eval-contract "$CONTRACT" \
    --codec-tokens-dir "$CODEC_DIRS" \
    --subset "$SUBSET" \
    --subset-split select8 \
    --speaker-map configs/speaker_maps/trackn_v2.json \
    --decode-only \
    --n2-gate \
    --out "$out_dir" \
    --wav-dir "$wav_dir" \
    --checkpoint-step "$step"
  source .venv/bin/activate
  export PYTHONPATH="$PWD/src"
  echo "=== SCORE select8 $tag ==="
  python scripts/74_eval_discrete_lm.py \
    --score-only "$out_dir/n2_decode_bundle.json" \
    --out "$out_dir" \
    --asr-config configs/asr.yaml \
    --eval-contract "$CONTRACT" \
    --n2-gate
  report="$out_dir/discrete_eval_report_v2.json"
  python - <<PY
import json
from pathlib import Path
r=json.loads(Path("$report").read_text())
p=r.get("provenance") or {}
assert p.get("git_dirty") is False, p
print("REPORT_CLEAN", "$tag", p.get("git_sha"))
PY
  REPORTS+=("$report")
done

# Ensure still clean before selection (no results written into git tree)
test -z "$(git status --porcelain)" || { echo DIRTY_MID; git status -sb; exit 3; }

python scripts/76_select_discrete_checkpoint.py \
  --d1c-formal \
  --formal-reports "${REPORTS[@]}" \
  --out "$RUN/checkpoint_selection.json" \
  --promote-best \
  --best-path "$RUN/best.pt" \
  --eval-contract "$CONTRACT"
cp -f "$RUN/checkpoint_selection.json" "$EXT/results_staging/artifacts/selection_w1_seed101.json"

for which in best last; do
  ckpt="$RUN/${which}.pt"
  listen="$OUT_ROOT/${which}_listen_demo"
  mkdir -p "$listen"
  source .venv/bin/activate
  export PYTHONPATH="$PWD/src"
  python scripts/73_infer_discrete_lm.py \
    --ckpt "$ckpt" --config "$CONFIG" --eval-contract "$CONTRACT" \
    --codec-tokens-dir "$CODEC_DIRS" --subset "$SUBSET" --subset-split select8 \
    --speaker-map configs/speaker_maps/trackn_v2.json --out "$listen" --seed "$SEED" --codes-only
  source .venv_qwen_tts/bin/activate
  export PYTHONPATH="$PWD/src"
  python scripts/73_infer_discrete_lm.py \
    --ckpt "$ckpt" --config "$CONFIG" --eval-contract "$CONTRACT" \
    --codec-tokens-dir "$CODEC_DIRS" --subset "$SUBSET" --subset-split select8 \
    --speaker-map configs/speaker_maps/trackn_v2.json --out "$listen" --seed "$SEED" --decode
  source .venv/bin/activate
  export PYTHONPATH="$PWD/src"
  mkdir -p "$WIN_OUT/${which}_listen_demo" "$WIN_D1C/${which}_listen_demo"
  cp -f "$listen"/*.wav "$WIN_OUT/${which}_listen_demo/" 2>/dev/null || true
  cp -f "$listen"/*.wav "$WIN_D1C/${which}_listen_demo/" 2>/dev/null || true
done

for split in gate24 full32; do
  out_dir="$OUT_ROOT/${split}"
  wav_dir="$out_dir/wavs"
  mkdir -p "$out_dir" "$wav_dir"
  source .venv_qwen_tts/bin/activate
  export PYTHONPATH="$PWD/src"
  python scripts/74_eval_discrete_lm.py \
    --ckpt "$RUN/best.pt" --config "$CONFIG" --eval-contract "$CONTRACT" \
    --codec-tokens-dir "$CODEC_DIRS" --subset "$SUBSET" --subset-split "$split" \
    --speaker-map configs/speaker_maps/trackn_v2.json --decode-only --n2-gate \
    --out "$out_dir" --wav-dir "$wav_dir"
  source .venv/bin/activate
  export PYTHONPATH="$PWD/src"
  python scripts/74_eval_discrete_lm.py \
    --score-only "$out_dir/n2_decode_bundle.json" --out "$out_dir" \
    --asr-config configs/asr.yaml --eval-contract "$CONTRACT" --n2-gate
  python - <<PY
import json
from pathlib import Path
g=json.loads(Path("$out_dir/n2_gate.json").read_text())
p=g.get("provenance") or {}
assert p.get("git_dirty") is False and p.get("git_dirty_score") is False, p
assert g.get("verdict")=="PASS", g.get("verdict")
print("GATE_CLEAN_PASS", "$split")
PY
  cp -f "$out_dir/n2_gate.json" "$EXT/results_staging/artifacts/gate_w1_seed101_${split}.json"
  cp -f "$out_dir/discrete_eval_report_v2.json" "$EXT/results_staging/artifacts/eval_report_w1_seed101_${split}.json"
done

test -z "$(git status --porcelain)" || { echo DIRTY_END; git status -sb; exit 4; }
mkdir -p "$WIN_RES/artifacts" "$WIN_RES/logs"
cp -f "$EXT/results_staging/artifacts/"* "$WIN_RES/artifacts/"
cp -f "$LOG" "$WIN_RES/logs/wave4_w1_clean_eval.console.log"
echo "WAVE4_CLEAN_EVAL_DONE tip=$TIP"

#!/usr/bin/env bash
# Wave 5 — DEV seed102 replication of locked W1 recipe (clean-tree formal).
set -euo pipefail
PLAN_SLUG=d1c_semantic_eos_repair_b81ecc3e
RUN_NAME=d1c_eos_dev_locked_seed102
SEED=102
CONFIG=configs/discrete_speech_lm_d1c_semantic_eos_w1.yaml
CONTRACT=configs/eval_contracts/d1c_semantic_eos_v2_0_0.json
SUBSET=configs/train_subsets/d1c_dev32.json
CODEC_DIRS="$HOME/datasets/tts/ruslan/codec_tokens,$HOME/datasets/tts/common_voice_ru17/codec_tokens"
EXT_LOGDIR="$HOME/datasets/tts/d1c_wave5_logs"
OUT_ROOT="out/${PLAN_SLUG}/dev/seed102"
WIN_OUT="/mnt/c/projects/tts-self/out/${PLAN_SLUG}/dev/seed102"
WIN_D1C="/mnt/c/projects/tts-self-d1c/out/${PLAN_SLUG}/dev/seed102"
WIN_RES="/mnt/c/projects/tts-self-d1c/.cursor/results/${PLAN_SLUG}"
STAGE="$EXT_LOGDIR/results_staging"

unset GIT_DIR GIT_WORK_TREE MINISPEECH_ALLOW_DIRTY
cd ~/projects/tts-self
mkdir -p "$EXT_LOGDIR" "$STAGE/artifacts" "$STAGE/logs" "$OUT_ROOT" "$WIN_OUT" "$WIN_D1C"

git fetch origin
git checkout d1c/semantic-eos
git reset --hard origin/d1c/semantic-eos
# quarantine untracked results noise
if [ -n "$(git status --porcelain)" ]; then
  while IFS= read -r line; do
    path="${line:3}"
    case "$path" in
      .cursor/results/*)
        mkdir -p "$EXT_LOGDIR/dirty_bak/$(dirname "$path")"
        mv -f "$path" "$EXT_LOGDIR/dirty_bak/$path" 2>/dev/null || true
        ;;
    esac
  done < <(git status --porcelain)
fi
test -z "$(git status --porcelain)" || { echo DIRTY; git status -sb; exit 2; }
TIP=$(git rev-parse HEAD)
SHORT=$(git rev-parse --short HEAD)

CONSOLE="$EXT_LOGDIR/wave5_seed102_console.log"
exec > >(tee -a "$CONSOLE") 2>&1
echo "=== Wave 5 seed102 start $(date -Is) tip=$TIP ==="

source .venv/bin/activate
export HF_HOME="$HOME/hf_cache"
export PYTHONPATH="$PWD/src"
uv pip install -e . --no-deps
python scripts/00_env_check.py
python - <<'PY'
from minispeech.utils import git_is_dirty, git_sha
assert not git_is_dirty()
print("LIVE_CLEAN", git_sha())
PY

if [ -d "runs/${RUN_NAME}" ]; then
  mv "runs/${RUN_NAME}" "runs/${RUN_NAME}_quarantine_$(date +%s)"
fi

python scripts/72_train_discrete_lm.py \
  --config "$CONFIG" \
  --eval-contract "$CONTRACT" \
  --subset "$SUBSET" \
  --subset-split full32 \
  --run-name "$RUN_NAME" \
  --seed "$SEED" \
  --stage S1 \
  --codec-tokens-dir "$CODEC_DIRS" \
  2>&1 | tee "$EXT_LOGDIR/wave5_seed102_train.log"

RUN=runs/${RUN_NAME}
test -f "$RUN/candidate_step_0750.pt"

REPORTS=()
STEPS=(250 375 500 625 750)
for step in "${STEPS[@]}"; do
  ckpt=$(printf "%s/candidate_step_%04d.pt" "$RUN" "$step")
  tag=$(printf "cand_%04d" "$step")
  out_dir="$OUT_ROOT/select8/${tag}"
  wav_dir="$out_dir/wavs"
  mkdir -p "$out_dir" "$wav_dir"
  source .venv_qwen_tts/bin/activate
  export PYTHONPATH="$PWD/src"
  python scripts/74_eval_discrete_lm.py \
    --ckpt "$ckpt" --config "$CONFIG" --eval-contract "$CONTRACT" \
    --codec-tokens-dir "$CODEC_DIRS" --subset "$SUBSET" --subset-split select8 \
    --speaker-map configs/speaker_maps/trackn_v2.json --decode-only --n2-gate \
    --out "$out_dir" --wav-dir "$wav_dir" --checkpoint-step "$step" \
    2>&1 | tee -a "$EXT_LOGDIR/wave5_seed102_select8.log"
  source .venv/bin/activate
  export PYTHONPATH="$PWD/src"
  python scripts/74_eval_discrete_lm.py \
    --score-only "$out_dir/n2_decode_bundle.json" --out "$out_dir" \
    --asr-config configs/asr.yaml --eval-contract "$CONTRACT" --n2-gate \
    2>&1 | tee -a "$EXT_LOGDIR/wave5_seed102_select8.log"
  report="$out_dir/discrete_eval_report_v2.json"
  python - <<PY
import json
from pathlib import Path
r=json.loads(Path("$report").read_text())
assert (r.get("provenance") or {}).get("git_dirty") is False
print("REPORT_CLEAN", "$tag")
PY
  REPORTS+=("$report")
done

test -z "$(git status --porcelain)" || { echo DIRTY_MID; git status -sb; exit 3; }

python scripts/76_select_discrete_checkpoint.py \
  --d1c-formal --formal-reports "${REPORTS[@]}" \
  --out "$RUN/checkpoint_selection.json" --promote-best --best-path "$RUN/best.pt" \
  --eval-contract "$CONTRACT" \
  2>&1 | tee "$EXT_LOGDIR/wave5_seed102_selection.log"
cp -f "$RUN/checkpoint_selection.json" "$STAGE/artifacts/selection_seed102.json"

for which in best last; do
  listen="$OUT_ROOT/${which}_listen_demo"
  mkdir -p "$listen"
  source .venv/bin/activate; export PYTHONPATH="$PWD/src"
  python scripts/73_infer_discrete_lm.py \
    --ckpt "$RUN/${which}.pt" --config "$CONFIG" --eval-contract "$CONTRACT" \
    --codec-tokens-dir "$CODEC_DIRS" --subset "$SUBSET" --subset-split select8 \
    --speaker-map configs/speaker_maps/trackn_v2.json --out "$listen" --seed "$SEED" --codes-only
  source .venv_qwen_tts/bin/activate; export PYTHONPATH="$PWD/src"
  python scripts/73_infer_discrete_lm.py \
    --ckpt "$RUN/${which}.pt" --config "$CONFIG" --eval-contract "$CONTRACT" \
    --codec-tokens-dir "$CODEC_DIRS" --subset "$SUBSET" --subset-split select8 \
    --speaker-map configs/speaker_maps/trackn_v2.json --out "$listen" --seed "$SEED" --decode
  source .venv/bin/activate; export PYTHONPATH="$PWD/src"
  mkdir -p "$WIN_OUT/${which}_listen_demo" "$WIN_D1C/${which}_listen_demo"
  cp -f "$listen"/*.wav "$WIN_OUT/${which}_listen_demo/" 2>/dev/null || true
  cp -f "$listen"/*.wav "$WIN_D1C/${which}_listen_demo/" 2>/dev/null || true
done

for split in gate24 full32; do
  out_dir="$OUT_ROOT/${split}"; wav_dir="$out_dir/wavs"; mkdir -p "$out_dir" "$wav_dir"
  source .venv_qwen_tts/bin/activate; export PYTHONPATH="$PWD/src"
  python scripts/74_eval_discrete_lm.py \
    --ckpt "$RUN/best.pt" --config "$CONFIG" --eval-contract "$CONTRACT" \
    --codec-tokens-dir "$CODEC_DIRS" --subset "$SUBSET" --subset-split "$split" \
    --speaker-map configs/speaker_maps/trackn_v2.json --decode-only --n2-gate \
    --out "$out_dir" --wav-dir "$wav_dir" 2>&1 | tee -a "$EXT_LOGDIR/wave5_seed102_${split}.log"
  source .venv/bin/activate; export PYTHONPATH="$PWD/src"
  python scripts/74_eval_discrete_lm.py \
    --score-only "$out_dir/n2_decode_bundle.json" --out "$out_dir" \
    --asr-config configs/asr.yaml --eval-contract "$CONTRACT" --n2-gate \
    2>&1 | tee -a "$EXT_LOGDIR/wave5_seed102_${split}.log"
  python - <<PY
import json
from pathlib import Path
g=json.loads(Path("$out_dir/n2_gate.json").read_text())
p=g.get("provenance") or {}
assert p.get("git_dirty") is False and p.get("git_dirty_score") is False, p
print("GATE", "$split", g.get("verdict"), "dirty", p.get("git_dirty"))
PY
  cp -f "$out_dir/n2_gate.json" "$STAGE/artifacts/gate_seed102_${split}.json"
  cp -f "$out_dir/discrete_eval_report_v2.json" "$STAGE/artifacts/eval_report_seed102_${split}.json" 2>/dev/null || true
done

test -z "$(git status --porcelain)" || { echo DIRTY_END; git status -sb; exit 4; }
mkdir -p "$WIN_RES/artifacts" "$WIN_RES/logs"
cp -f "$STAGE/artifacts/"* "$WIN_RES/artifacts/"
cp -f "$EXT_LOGDIR"/wave5_seed102_*.log "$WIN_RES/logs/" 2>/dev/null || true
cp -f "$CONSOLE" "$WIN_RES/logs/"
echo "=== Wave 5 seed102 DONE $(date -Is) tip=$SHORT ==="

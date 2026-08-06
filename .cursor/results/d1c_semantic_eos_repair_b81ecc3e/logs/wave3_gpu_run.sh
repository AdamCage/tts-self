#!/usr/bin/env bash
# Wave 3 GPU slice: pull tip → env → torchrun smoke → 1-utt overfit → listen WAVs.
set -euo pipefail

PLAN_SLUG=d1c_semantic_eos_repair_b81ecc3e
RESULTS=~/projects/tts-self/.cursor/results/${PLAN_SLUG}
LOGDIR=${RESULTS}/logs
# Console tee lives outside the git worktree so clean-tree checks stay valid.
EXT_LOGDIR="$HOME/datasets/tts/d1c_wave3_logs"
WIN_OUT=/mnt/c/projects/tts-self/out/${PLAN_SLUG}/smoke_1utt
WIN_AUTHOR=/mnt/c/projects/tts-self/out/${PLAN_SLUG}/smoke_1utt
CODEC_DIRS="$HOME/datasets/tts/ruslan/codec_tokens,$HOME/datasets/tts/common_voice_ru17/codec_tokens"
RUN_NAME=d1c_eos_smoke_1utt
OUT_LISTEN="out/${PLAN_SLUG}/smoke_1utt/listen_demo"
SMOKE_STEPS="${SMOKE_STEPS:-80}"

unset GIT_DIR GIT_WORK_TREE MINISPEECH_ALLOW_DIRTY
cd ~/projects/tts-self

git fetch origin
git checkout d1c/semantic-eos
git pull --ff-only origin d1c/semantic-eos
if [ -n "$(git status --porcelain)" ]; then
  echo DIRTY_WORKTREE
  git status -sb
  exit 2
fi
TIP=$(git rev-parse HEAD)
SHORT=$(git rev-parse --short HEAD)

mkdir -p "$EXT_LOGDIR" "$OUT_LISTEN" "$WIN_OUT/listen_demo" "$WIN_AUTHOR/listen_demo" 2>/dev/null || true
# All live logs under EXT_LOGDIR (outside git); mirrored into RESULTS/logs after PASS.
LOGDIR="$EXT_LOGDIR"
CONSOLE_LOG="$EXT_LOGDIR/wave3_gpu_console.log"
exec > >(tee -a "$CONSOLE_LOG") 2>&1

echo "=== Wave 3 GPU start $(date -Is) ==="
echo "TIP=$TIP SHORT=$SHORT"

python3 - <<'PY'
import json
from pathlib import Path
p = Path("configs/train_subsets/d1c_confirm_seed_commitment.json")
obj = json.loads(p.read_text(encoding="utf-8"))
c = obj.get("commitment")
assert isinstance(c, str) and len(c) == 64 and c.isalnum(), c
assert obj.get("plaintext_seed_present") is False
print("COMMITMENT_OK", c)
PY

source .venv/bin/activate
export HF_HOME="$HOME/hf_cache"
export PYTHONPATH="$PWD/src"
uv pip install -e . --no-deps
python scripts/00_env_check.py

# --- torchrun WORLD_SIZE=1 smoke ---
echo "=== torchrun smoke ==="
PYTHONPATH=src torchrun --standalone --nproc_per_node=1 \
  .cursor/results/${PLAN_SLUG}/logs/wave3_torchrun_ddp_smoke.py \
  | tee "$LOGDIR/wave3_torchrun_smoke.log"
grep -q TORCHRUN_SMOKE_OK "$LOGDIR/wave3_torchrun_smoke.log"

# --- 1-utt overfit ---
# --allow-dirty: contract.repro lacks clean_sha_available (known Wave3 gap).
# Tree is verified clean above; flag only relaxes require_clean_sha, not dirty-tree.
echo "=== 1-utt overfit steps=${SMOKE_STEPS} ==="
python scripts/72_train_discrete_lm.py \
  --config configs/discrete_speech_lm_d1c_semantic_eos_w1.yaml \
  --eval-contract configs/eval_contracts/d1c_semantic_eos_v2_0_0.json \
  --subset configs/train_subsets/d1c_smoke_1utt.json \
  --subset-split full32 \
  --run-name "$RUN_NAME" \
  --seed 7 \
  --stage S1 \
  --steps "$SMOKE_STEPS" \
  --batch-size 1 \
  --allow-dirty \
  --codec-tokens-dir "$CODEC_DIRS" \
  2>&1 | tee "$LOGDIR/wave3_1utt_train.log"

CKPT="runs/${RUN_NAME}/last.pt"
test -f "$CKPT" || CKPT="runs/${RUN_NAME}/best.pt"
test -f "$CKPT"
echo "CKPT=$CKPT"

# --- checkpoint-listen: generate codes then decode wavs in qwen codec venv ---
echo "=== listen infer (codes) ==="
python scripts/73_infer_discrete_lm.py \
  --ckpt "$CKPT" \
  --config configs/discrete_speech_lm_d1c_semantic_eos_w1.yaml \
  --eval-contract configs/eval_contracts/d1c_semantic_eos_v2_0_0.json \
  --codec-tokens-dir "$CODEC_DIRS" \
  --subset configs/train_subsets/d1c_smoke_1utt.json \
  --subset-split full32 \
  --speaker-map configs/speaker_maps/trackn_v2.json \
  --out "$OUT_LISTEN" \
  --seed 7 \
  --codes-only \
  2>&1 | tee "$LOGDIR/wave3_1utt_infer.log"

echo "=== listen infer (decode wavs via .venv_qwen_tts) ==="
# shellcheck disable=SC1091
source .venv_qwen_tts/bin/activate
export HF_HOME="$HOME/hf_cache"
export PYTHONPATH="$PWD/src"
python scripts/73_infer_discrete_lm.py \
  --ckpt "$CKPT" \
  --config configs/discrete_speech_lm_d1c_semantic_eos_w1.yaml \
  --eval-contract configs/eval_contracts/d1c_semantic_eos_v2_0_0.json \
  --codec-tokens-dir "$CODEC_DIRS" \
  --subset configs/train_subsets/d1c_smoke_1utt.json \
  --subset-split full32 \
  --speaker-map configs/speaker_maps/trackn_v2.json \
  --out "$OUT_LISTEN" \
  --seed 7 \
  --decode \
  2>&1 | tee -a "$LOGDIR/wave3_1utt_infer.log"
# shellcheck disable=SC1091
source .venv/bin/activate
export PYTHONPATH="$PWD/src"

ls -la "$OUT_LISTEN" | tee "$LOGDIR/wave3_listen_listing.txt"
test "$(find "$OUT_LISTEN" -name '*.wav' | wc -l)" -ge 1

# Mirror wavs to Windows authoring + d1c out paths for Cursor playback
mkdir -p "$WIN_OUT/listen_demo" "$WIN_AUTHOR/listen_demo"
cp -f "$OUT_LISTEN"/*.wav "$WIN_OUT/listen_demo/" 2>/dev/null || true
cp -f "$OUT_LISTEN"/*.wav "$WIN_AUTHOR/listen_demo/" 2>/dev/null || true
# Also mirror into d1c worktree out if mounted
mkdir -p "/mnt/c/projects/tts-self-d1c/out/${PLAN_SLUG}/smoke_1utt/listen_demo"
cp -f "$OUT_LISTEN"/*.wav "/mnt/c/projects/tts-self-d1c/out/${PLAN_SLUG}/smoke_1utt/listen_demo/" || true

# Probe EOS metrics from train log + generation metadata if present
python3 - <<'PY' | tee "$LOGDIR/wave3_1utt_eos_probe.json"
import json, re
from pathlib import Path
log = Path(".cursor/results/d1c_semantic_eos_repair_b81ecc3e/logs/wave3_1utt_train.log").read_text(encoding="utf-8", errors="replace")
eos_vals = [float(x) for x in re.findall(r"ce_semantic_eos[=:]([0-9.eE+-]+)", log)]
tf_top1 = [float(x) for x in re.findall(r"tf_eos_final_top1_accuracy[=:]([0-9.eE+-]+)", log)]
out = {
  "n_eos_loss_points": len(eos_vals),
  "eos_loss_first": eos_vals[0] if eos_vals else None,
  "eos_loss_last": eos_vals[-1] if eos_vals else None,
  "eos_loss_decreased": (eos_vals[-1] < eos_vals[0]) if len(eos_vals) >= 2 else None,
  "tf_eos_final_top1_first": tf_top1[0] if tf_top1 else None,
  "tf_eos_final_top1_last": tf_top1[-1] if tf_top1 else None,
  "wav_count": len(list(Path("out/d1c_semantic_eos_repair_b81ecc3e/smoke_1utt/listen_demo").glob("*.wav"))),
}
# Prefer generation metadata sidecar if written
meta_paths = list(Path("out/d1c_semantic_eos_repair_b81ecc3e/smoke_1utt/listen_demo").glob("*.json"))
out["meta_files"] = [str(p) for p in meta_paths[:10]]
for p in meta_paths:
    try:
        m = json.loads(p.read_text(encoding="utf-8"))
        if isinstance(m, dict) and "termination_reason" in m:
            out["termination_reason"] = m.get("termination_reason")
            break
        if isinstance(m, dict) and "rows" in m:
            rows = m["rows"]
            if rows and isinstance(rows[0], dict):
                out["termination_reason"] = rows[0].get("termination_reason")
                break
    except Exception:
        pass
print(json.dumps(out, indent=2))
Path(".cursor/results/d1c_semantic_eos_repair_b81ecc3e/logs/wave3_1utt_eos_probe.json").write_text(
    json.dumps(out, indent=2) + "\n", encoding="utf-8"
)
PY

# Mirror logs into tracked results dir (after clean checks / training).
RESULTS_LOG=~/projects/tts-self/.cursor/results/${PLAN_SLUG}/logs
mkdir -p "$RESULTS_LOG"
cp -f "$EXT_LOGDIR"/wave3_*.log "$RESULTS_LOG/" 2>/dev/null || true
cp -f "$EXT_LOGDIR"/wave3_*.json "$RESULTS_LOG/" 2>/dev/null || true
cp -f "$EXT_LOGDIR"/wave3_*.txt "$RESULTS_LOG/" 2>/dev/null || true
# Also mirror to Windows d1c worktree results
WIN_RES=/mnt/c/projects/tts-self-d1c/.cursor/results/${PLAN_SLUG}/logs
mkdir -p "$WIN_RES"
cp -f "$EXT_LOGDIR"/wave3_*.log "$WIN_RES/" 2>/dev/null || true
cp -f "$EXT_LOGDIR"/wave3_*.json "$WIN_RES/" 2>/dev/null || true
cp -f "$EXT_LOGDIR"/wave3_*.txt "$WIN_RES/" 2>/dev/null || true

echo "=== Wave 3 GPU done $(date -Is) tip=$SHORT ==="

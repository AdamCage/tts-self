#!/usr/bin/env bash
set -euo pipefail
cd ~/projects/tts-self
source .venv/bin/activate
RUN=runs/discrete_speech_lm_d1b_dev_seed42
# Remove destroyed self-symlink; step 250 weights are gone.
rm -f "$RUN/candidate_step_0250.pt" "$RUN/best.pt"
# Filter candidate rows to ckpts that still exist on disk.
python - <<'PY'
import json
from pathlib import Path
met = Path("/mnt/c/projects/tts-self/out/d1b_n2_reliability/metrics/dev_seed42_candidate_rows.json")
rows = json.loads(met.read_text())
kept = []
for r in rows:
    p = Path(r["ckpt"])
    if not p.is_absolute():
        p = Path.cwd() / p
    ok = p.is_file() and not p.is_symlink()
    r["ckpt_on_disk"] = bool(ok)
    if ok:
        kept.append(r)
    print(f"step={r['step']} on_disk={ok} verdict={r.get('gate_verdict')}")
if not kept:
    raise SystemExit("no candidates left on disk")
met.write_text(json.dumps(kept, indent=2) + "\n")
# Keep full history note
note = Path("/mnt/c/projects/tts-self/out/d1b_n2_reliability/metrics/dev_seed42_step250_loss.md")
note.write_text(
    "# DEV step 250 checkpoint lost\n\n"
    "Promote used Path.resolve() on best.pt symlink and unlinked the real "
    "candidate_step_0250.pt (self-loop). Fixed in scripts/76_select_discrete_checkpoint.py. "
    "FR scores for step 250 remain under cand_0250_n2/; selection proceeds among "
    "remaining on-disk candidates (625/750 PASS).\n"
)
print("kept", len(kept), "candidates")
PY
source .venv/bin/activate
python scripts/76_select_discrete_checkpoint.py \
  --candidates-json /mnt/c/projects/tts-self/out/d1b_n2_reliability/metrics/dev_seed42_candidate_rows.json \
  --selection-subset dev32 \
  --require-subset dev32 \
  --verify-frozen-subset dev32 \
  --out "$RUN/checkpoint_selection.json" \
  --promote-best \
  --best-path "$RUN/best.pt"
cp -f "$RUN/checkpoint_selection.json" \
  /mnt/c/projects/tts-self/out/d1b_n2_reliability/metrics/dev_seed42_selection.json
python - <<'PY'
import json, shutil
from pathlib import Path
sel = json.loads(Path("runs/discrete_speech_lm_d1b_dev_seed42/checkpoint_selection.json").read_text())
step = int(sel["selected"]["step"])
src = Path(f"/mnt/c/projects/tts-self/out/d1b_n2_reliability/dev_seed42/cand_{step:04d}_n2/n2_gate.json")
dst = Path("/mnt/c/projects/tts-self/out/d1b_n2_reliability/metrics/dev_seed42_n2_gate.json")
shutil.copy2(src, dst)
g = json.loads(dst.read_text())
print("selected", step, "verdict", g["verdict"], "n_cat", g["criteria"]["n_catastrophic"])
print("best.pt", Path("runs/discrete_speech_lm_d1b_dev_seed42/best.pt").stat().st_size)
PY

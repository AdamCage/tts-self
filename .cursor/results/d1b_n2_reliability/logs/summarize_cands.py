#!/usr/bin/env python3
import json
from pathlib import Path

root = Path("/mnt/c/projects/tts-self/out/d1b_n2_reliability/dev_seed42")
for p in sorted(root.glob("cand_*_n2/candidate_row.json")):
    row = json.loads(p.read_text(encoding="utf-8"))
    cat = Path(p.parent / "n2_catastrophic_rows.json")
    cats = json.loads(cat.read_text(encoding="utf-8")) if cat.exists() else []
    print(
        f"step={row.get('step')} macro_cer={row.get('macro_cer'):.4f} "
        f"n_cat={row.get('n_catastrophic')} runaway={row.get('runaway_or_repetition')} "
        f"verdict={row.get('gate_verdict')} dur={row.get('mean_dur_ratio'):.3f}"
    )
    for c in cats:
        print(f"  CAT {c.get('utt_id')} cer={c.get('cer'):.3f} frames={c.get('n_frames')}/{c.get('gt_frames')} stop={c.get('stop_reason')}")
        print(f"    hyp={c.get('hyp')[:120]!r}")

sel = Path("/mnt/c/projects/tts-self/out/d1b_n2_reliability/metrics/dev_seed42_selection.json")
if sel.exists():
    print("SELECTION:", json.dumps(json.loads(sel.read_text())["selected"], indent=2, ensure_ascii=False))
else:
    print("selection not ready")

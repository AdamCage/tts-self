#!/usr/bin/env python3
"""Summarize confirm_v2 seed42/43 gates and write RESULTS artifacts."""
from __future__ import annotations

import json
from pathlib import Path

base = Path.home() / "projects/tts-self"
win = Path("/mnt/c/projects/tts-self")
out = base / "out/d1b_n2_reliability"
res = win / ".cursor/results/d1b_n2_reliability"
met = win / "out/d1b_n2_reliability/metrics"


def load(p: Path) -> dict:
    return json.loads(p.read_text(encoding="utf-8"))


summary: dict = {
    "sha": "0db69d608eb3d3d2f8fd9ba535fb92eb457fc06b",
    "bee3f00_parent": "bee3f009661bb30472fc02682f429787c38b46cb",
    "confirm_oracle_items_sha256": "63d7e351602e62d26196d944cd1f7a9e928a241e70d968f6eb58c0d1b7a746b9",
    "select8_sha256": "6667da2f8fc2c3706deff48a79489103f739419041f3fb492f5822647075a0ed",
    "eval_contract_sha256": "d04295d3ac9864afe5002a8470baead8f5f57a3c8f8d38831fe134aa7feada92",
    "seeds": {},
}

for seed in (42, 43):
    tag = f"confirm_v2_seed{seed}"
    run = base / f"runs/discrete_speech_lm_d1b_confirm_v2_seed{seed}"
    sel = load(run / "checkpoint_selection.json")
    g24 = load(out / tag / "best_gate24_n2/n2_gate.json")
    g32 = load(out / tag / "best_full32_n2/n2_gate.json")
    c24 = g24.get("criteria") or {}
    c32 = g32.get("criteria") or {}
    stab24 = c24.get("stability_d1b") or {}
    stab32 = c32.get("stability_d1b") or {}
    selected = sel.get("selected") or {}
    summary["seeds"][str(seed)] = {
        "run_dir": str(run),
        "selected_step": selected.get("step"),
        "selected_ckpt": selected.get("ckpt"),
        "selected_tf_ce": selected.get("teacher_forced_ce"),
        "select8_macro_cer": selected.get("macro_cer"),
        "select8_gate_eligible": selected.get("gate_eligible"),
        "gate24_verdict": g24.get("verdict"),
        "gate24_macro_cer": c24.get("macro_cer"),
        "gate24_cer_pass": c24.get("cer_pass"),
        "gate24_text_usage_pass": (c24.get("text_usage_d1b") or {}).get("pass"),
        "gate24_stability": {
            "runaway_or_repetition": stab24.get("runaway_or_repetition"),
            "phrase_loop_repetition": stab24.get("phrase_loop_repetition"),
            "hard_cap_count": stab24.get("hard_cap_count"),
            "hard_cap_ok": stab24.get("hard_cap_ok"),
            "mean_dur_ratio": stab24.get("mean_dur_ratio"),
        },
        "full32_verdict": g32.get("verdict"),
        "full32_macro_cer": c32.get("macro_cer"),
        "full32_cer_pass": c32.get("cer_pass"),
        "full32_stability": {
            "runaway_or_repetition": stab32.get("runaway_or_repetition"),
            "phrase_loop_repetition": stab32.get("phrase_loop_repetition"),
            "hard_cap_count": stab32.get("hard_cap_count"),
            "hard_cap_ok": stab32.get("hard_cap_ok"),
            "mean_dur_ratio": stab32.get("mean_dur_ratio"),
        },
    }
    print(f"seed{seed}", json.dumps(summary["seeds"][str(seed)], indent=2))

pass_both = all(
    summary["seeds"][s]["gate24_verdict"] == "PASS"
    and summary["seeds"][s]["full32_verdict"] == "PASS"
    for s in ("42", "43")
)
summary["formal_verdict"] = "PASS" if pass_both else "HARD-STOP"
print("FORMAL", summary["formal_verdict"])

met.mkdir(parents=True, exist_ok=True)
(res / "metrics").mkdir(parents=True, exist_ok=True)
(res / "metrics/confirm_v2_summary.json").write_text(
    json.dumps(summary, indent=2) + "\n", encoding="utf-8"
)
(met / "confirm_v2_summary.json").write_text(
    json.dumps(summary, indent=2) + "\n", encoding="utf-8"
)

# listen index (Linux primary + Windows mirrors if present)
listen = {"plan_slug": "d1b_n2_reliability", "entries": []}
for seed in (42, 43):
    tag = f"confirm_v2_seed{seed}"
    linux_demo = out / tag / "listen_demo"
    win_demo = win / "out/d1b_n2_reliability" / tag / "listen_demo"
    wavs = sorted(linux_demo.glob("*.wav")) if linux_demo.is_dir() else []
    for i, w in enumerate(wavs, 1):
        entry = {
            "seed": seed,
            "utt": w.name,
            "linux": str(w),
        }
        mirror = win_demo / f"utt_{i}.wav"
        if mirror.is_file():
            entry["windows"] = str(mirror).replace("/mnt/c/", "C:/")
        listen["entries"].append(entry)

(res / "listen_index.json").write_text(json.dumps(listen, indent=2) + "\n", encoding="utf-8")

gate = {
    "plan_slug": "d1b_n2_reliability",
    "formal_verdict": summary["formal_verdict"],
    "prior_pass_status": "INVALID_WITHDRAWN",
    "gpu_code_sha": summary["sha"],
    "selection_fix_sha": summary["bee3f00_parent"],
    "atomic_save_sha": summary["sha"],
    "eval_contract_sha256": summary["eval_contract_sha256"],
    "confirm_oracle_items_sha256": summary["confirm_oracle_items_sha256"],
    "confirm_oracle_path": "configs/train_subsets/d1b_confirm32_v2.json",
    "confirm_oracle_seed": 20260804,
    "do_not_proceed_to": ["1h", "5h", "full"],
    "confirm_v2": summary["seeds"],
    "reasons": [
        "Prior PASS withdrawn (null TF-CE selection, score-only contract bypass, GT max_frames, hard_cap/phrase-loop, burned CONFIRM v1).",
        "confirm_v2 series completed on clean SHA 0db69d6 (atomic save on bee3f00 fixes).",
        "Both confirm seeds FAIL N2_D1b on stability (phrase_loop_repetition / hard_cap) despite CER below ceiling; formal PASS not restored.",
    ],
}
(res / "d1b_reliability_gate.json").write_text(
    json.dumps(gate, indent=2) + "\n", encoding="utf-8"
)
print("wrote gate + listen_index")

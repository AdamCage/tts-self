#!/usr/bin/env python3
"""Patch DEV candidate gates after Jaccard stress-normalization fix (no re-ASR)."""
from __future__ import annotations

import json
from pathlib import Path

from minispeech.eval.discrete_n2 import (
    annotate_catastrophic_rows,
    candidate_from_eval_report,
    enrich_model_report,
)

ROOT = Path("/mnt/c/projects/tts-self/out/d1b_n2_reliability/dev_seed42")
MET = Path("/mnt/c/projects/tts-self/out/d1b_n2_reliability/metrics")
STEPS = (250, 375, 500, 625, 750)
DIGEST = "c886ba31f3305546d9b4938e1cd0b7a2fa4fce738d7da342e9da2a30a28b172c"


def main() -> None:
    rows_out = []
    for step in STEPS:
        out_dir = ROOT / f"cand_{step:04d}_n2"
        gate = json.loads((out_dir / "n2_gate.json").read_text(encoding="utf-8"))
        model_rows = json.loads((out_dir / "n2_model_rows.json").read_text(encoding="utf-8"))
        annotated = annotate_catastrophic_rows(model_rows)
        cats = [r for r in annotated if r.get("catastrophic")]
        n_cat = len(cats)
        zero_cat = n_cat == 0

        # Preserve prior hard criteria; only refresh catastrophic fields + verdict.
        crit = dict(gate.get("criteria") or {})
        crit["n_catastrophic"] = n_cat
        crit["zero_catastrophic"] = zero_cat

        # Recompute D1b hard verdict from stored flags + new zero_cat.
        text_d1b = crit.get("text_usage_d1b") or {}
        stab = crit.get("stability_d1b") or {}
        hard = [
            bool(crit.get("cer_pass")),
            bool(crit.get("bootstrap_sep_wrong_utt")),
            bool(crit.get("bootstrap_sep_random")),
            bool(crit.get("no_runaway_repetition")),
            bool(crit.get("cb0_healthy")),
            bool(text_d1b.get("pass", False)),
            bool(stab.get("pass", False)),
            zero_cat,
        ]
        # If text_usage_d1b missing (shouldn't), fall back to legacy soft flags
        if "pass" not in text_d1b:
            hard[5] = (
                bool(crit.get("text_usage_shuffle_hurts"))
                and bool(crit.get("text_usage_mask_hurts"))
                and bool(crit.get("text_usage_prefix_cf_changes"))
            )
        if "pass" not in stab:
            hard[6] = bool(crit.get("no_runaway_repetition")) and bool(crit.get("cb0_healthy"))

        verdict = "PASS" if all(hard) else "FAIL"
        gate["criteria"] = crit
        gate["verdict"] = verdict
        gate["regate_note"] = (
            "Patched n_catastrophic/verdict after word_token_jaccard "
            "normalize_for_asr fix; ASR hyps/CER unchanged."
        )
        # Update model report catastrophic fields if present
        if "reports" in gate and "model" in gate["reports"]:
            gate["reports"]["model"]["n_catastrophic"] = n_cat
            gate["reports"]["model"]["catastrophic_rows"] = [
                {k: v for k, v in r.items() if k != "edit_counts"} for r in cats
            ]

        (out_dir / "n2_gate.json").write_text(json.dumps(gate, indent=2) + "\n")
        (out_dir / "n2_model_rows.json").write_text(
            json.dumps(annotated, indent=2, ensure_ascii=False) + "\n"
        )
        (out_dir / "n2_catastrophic_rows.json").write_text(
            json.dumps(cats, indent=2, ensure_ascii=False) + "\n"
        )

        # Rebuild candidate row from enriched model report stub
        model_rep = enrich_model_report({"rows": annotated})
        ckpt = f"runs/discrete_speech_lm_d1b_dev_seed42/candidate_step_{step:04d}.pt"
        row = candidate_from_eval_report(
            model_rep, ckpt=ckpt, step=step, selection_subset="dev32"
        )
        row["items_sha256"] = DIGEST
        row["gate_verdict"] = verdict
        (out_dir / "candidate_row.json").write_text(json.dumps(row, indent=2) + "\n")
        rows_out.append(row)
        print(
            f"step={step} verdict={verdict} n_cat={n_cat} "
            f"hard={hard} macro={row.get('macro_cer')}"
        )

    MET.mkdir(parents=True, exist_ok=True)
    (MET / "dev_seed42_candidate_rows.json").write_text(
        json.dumps(rows_out, indent=2) + "\n"
    )
    print("done")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Re-annotate catastrophic + re-run D1b N2 gate from scored model rows (no ASR)."""
from __future__ import annotations

import json
from pathlib import Path

from minispeech.eval.discrete_n2 import (
    annotate_catastrophic_rows,
    candidate_from_eval_report,
    enrich_model_report,
    evaluate_n2_gate_d1b,
    load_frozen_utt_ids,
)

ROOT = Path("/mnt/c/projects/tts-self/out/d1b_n2_reliability/dev_seed42")
MET = Path("/mnt/c/projects/tts-self/out/d1b_n2_reliability/metrics")
STEPS = (250, 375, 500, 625, 750)


def _slim_report(rep: dict) -> dict:
    """Rebuild a minimal report for non-model conditions from prior gate criteria."""
    return rep


def main() -> None:
    ids, _, digest = __import__("minispeech.data.d1b_oracles", fromlist=["load_frozen_utt_ids"]).load_frozen_utt_ids(
        "dev32"
    )
    rows_out = []
    for step in STEPS:
        tag = f"cand_{step:04d}"
        out_dir = ROOT / f"{tag}_n2"
        gate_path = out_dir / "n2_gate.json"
        model_rows_path = out_dir / "n2_model_rows.json"
        old_gate = json.loads(gate_path.read_text(encoding="utf-8"))
        model_rows = json.loads(model_rows_path.read_text(encoding="utf-8"))
        # Re-annotate model
        model_rep = enrich_model_report({"rows": model_rows, **{
            k: v for k, v in (old_gate.get("reports") or {}).get("model", {}).items()
            if k not in ("rows", "catastrophic_rows", "n_catastrophic", "macro_cer", "mean_cer", "micro_cer")
        }})
        # Other conditions: keep prior enriched reports (aggregates unchanged by jaccard fix)
        reports = dict(old_gate.get("reports") or {})
        # Build stub reports with mean_cer etc. from old reports; inject dummy rows if needed
        def stub(name: str) -> dict:
            r = dict(reports.get(name) or {})
            # ensure required fields for evaluate_n2_gate_d1b
            if "rows" not in r:
                r["rows"] = []
            if "mean_cer" not in r and "macro_cer" in r:
                r["mean_cer"] = r["macro_cer"]
            if "cer_ci95_low" not in r:
                r["cer_ci95_low"] = float(r.get("macro_cer", r.get("mean_cer", 1.0)))
            if "cer_ci95_high" not in r:
                r["cer_ci95_high"] = float(r.get("macro_cer", r.get("mean_cer", 1.0)))
            return r

        copy_rep = stub("copy_synth")
        wrong_rep = stub("wrong_utt")
        rand_rep = stub("random_codes")
        shuf_rep = stub("shuffle_text")
        mask_rep = stub("mask_text")

        # Prefer recompute from decode bundle hyps if present after prior score
        # Fall back: use old criteria values by packing into enrich-friendly reports
        crit = old_gate.get("criteria") or {}
        for rep, key in (
            (copy_rep, "copy_synth_cer_subset"),
            (wrong_rep, "wrong_utt_cer"),
            (rand_rep, "random_codes_cer"),
            (shuf_rep, "shuffle_cer"),
            (mask_rep, "mask_cer"),
        ):
            if rep.get("mean_cer") is None and key in crit:
                rep["mean_cer"] = crit[key]
                rep["macro_cer"] = crit[key]
                rep.setdefault("cer_ci95_low", crit[key])
                rep.setdefault("cer_ci95_high", crit[key])

        gate = evaluate_n2_gate_d1b(
            model_rep,
            copy_rep,
            wrong_rep,
            rand_rep,
            shuf_rep,
            mask_rep,
            copy_synth_ceiling=float(crit.get("copy_synth_cer_subset", 0.0228)),
        )
        model_rows_new = gate.pop("model_rows")
        cat_rows = gate.pop("catastrophic_rows", [])
        gate["reports"] = gate.pop("enriched_reports")
        gate["artifacts"] = old_gate.get("artifacts") or {}
        gate["regate_note"] = "Re-gated after word_token_jaccard normalize_for_asr fix; no re-ASR."
        gate_path.write_text(json.dumps(gate, indent=2), encoding="utf-8")
        model_rows_path.write_text(
            json.dumps(model_rows_new, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        (out_dir / "n2_catastrophic_rows.json").write_text(
            json.dumps(cat_rows, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        ckpt = f"runs/discrete_speech_lm_d1b_dev_seed42/candidate_step_{step:04d}.pt"
        row = candidate_from_eval_report(
            gate["reports"]["model"],
            ckpt=ckpt,
            step=step,
            selection_subset="dev32",
        )
        row["items_sha256"] = digest
        row["gate_verdict"] = gate.get("verdict")
        (out_dir / "candidate_row.json").write_text(json.dumps(row, indent=2) + "\n")
        rows_out.append(row)
        print(
            f"step={step} verdict={gate['verdict']} n_cat={gate['criteria'].get('n_catastrophic')} "
            f"macro={gate['criteria'].get('macro_cer')}"
        )

    MET.mkdir(parents=True, exist_ok=True)
    (MET / "dev_seed42_candidate_rows.json").write_text(
        json.dumps(rows_out, indent=2) + "\n", encoding="utf-8"
    )
    print("wrote", MET / "dev_seed42_candidate_rows.json")


if __name__ == "__main__":
    main()

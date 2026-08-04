"""Probe 6: selector eligibility criteria with *consistently re-hashed* reports,
so refusals cannot be attributed to tamper detection instead of the gate itself."""

from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "src"))

RESULTS: list[tuple[str, str, str]] = []


def rec(b: str, s: str, d: str) -> None:
    RESULTS.append((b, s, d))
    print(f"REAUDIT {b} {s} {d}", flush=True)


def load_script(name: str):
    spec = importlib.util.spec_from_file_location(name.replace(".py", ""), ROOT / "scripts" / name)
    mod = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


CONDITIONS = ("model", "copy_synth", "wrong_utt", "random_codes", "shuffle_text", "mask_text")


def main() -> int:
    m74 = load_script("74_eval_discrete_lm.py")
    m76 = load_script("76_select_discrete_checkpoint.py")
    from minispeech.data.d1b_oracles import load_eval_contract
    from minispeech.data.discrete_oracles import load_formal_subset
    from minispeech.eval.discrete_report_schema import rows_sha256, sha256_json

    contract = load_eval_contract(ROOT / "configs/eval_contracts/d1c_semantic_eos_v2_0_0.json")
    sel = load_formal_subset("select8", contract=contract)
    uids = list(sel["utt_ids"])

    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        wav_dir = tmp / "wavs"
        wav_dir.mkdir(parents=True)

        def base_rows(**gen_over) -> list[dict]:
            out = []
            for i, uid in enumerate(uids):
                wav = wav_dir / f"{i:04d}.wav"
                wav.write_bytes(b"RIFFfakeWAVE")
                gen = {
                    "termination_mode": "semantic_eos_v1",
                    "termination_reason": "learned_eos",
                    "n_decoded_frames": 40,
                    "eos_selected": True,
                    "early_eos_suppressed_count": 0,
                    "max_cb0_run": 3,
                    "generated_codes_sha256": "a" * 64,
                }
                gen.update(gen_over)
                out.append(
                    {
                        "ordinal": i, "utt_id": uid, "condition": "model",
                        "n_frames": 40, "gt_frames": 42, "cer": 0.08,
                        "ref": "привет мир", "hyp": "привет мир",
                        "wav": str(wav), "stop_reason": gen["termination_reason"],
                        "generation": gen,
                    }
                )
            return out

        def emit(**gen_over) -> dict:
            reports = {c: {"condition": c, "n": len(uids), "rows": base_rows(**gen_over)} for c in CONDITIONS}
            out = m74._emit_d1c_v2_from_gate(
                out_dir=tmp / f"eval_{abs(hash(json.dumps(gen_over, sort_keys=True)))}",
                contract=contract,
                ckpt=str(tmp / "c.pt"),
                provenance={
                    "subset_label": "select8",
                    "subset_path": str(ROOT / "configs/train_subsets/d1c_dev32.json"),
                    "subset_file_sha256": sel["file_sha256"],
                    "checkpoint_step": 750,
                },
                infer=dict(contract.get("infer") or {}),
                teacher_forced_ce={"loss": 3.21, "ce_cb0": 2.0, "ce_residual": 1.2},
                ordered_utt_ids=uids,
                condition_reports=reports,
                gate={"verdict": "PASS", "criteria": {"macro_cer": 0.08, "n_catastrophic": 0}},
            )
            return json.loads(Path(out).read_text(encoding="utf-8"))

        (tmp / "c.pt").write_bytes(b"fake")

        def five(report: dict, tag: str) -> list[Path]:
            paths = []
            for step in (250, 375, 500, 625, 750):
                r = json.loads(json.dumps(report))
                r["checkpoint"]["step"] = step
                r["checkpoint"]["sha256"] = f"{step:064d}"
                for c in CONDITIONS:
                    r["conditions"][c]["rows_sha256"] = rows_sha256(r["conditions"][c]["rows"])
                r.pop("report_sha256", None)
                r["report_sha256"] = sha256_json(r)
                p = tmp / f"{tag}_{step}.json"
                p.write_text(json.dumps(r, ensure_ascii=False), encoding="utf-8")
                paths.append(p)
            return paths

        # (1) all rows terminate by safety_max_frames -> must be refused
        rep_safety = emit(termination_reason="safety_max_frames", eos_selected=False)
        try:
            res = m76.select_d1c_formal_reports(five(rep_safety, "safety"))
            rec("B6", "FAIL",
                f"selector SELECTED an all-safety_max_frames candidate: step={res['selected']['step']}")
        except ValueError as exc:
            rec("B6", "PASS", f"all-safety candidates refused: {exc!s:.90}")

        # (2) early_eos_suppressed_count=7 and max_cb0_run=99 -> plan 11 forbids both
        rep_leak = emit(early_eos_suppressed_count=7, max_cb0_run=99)
        try:
            res = m76.select_d1c_formal_reports(five(rep_leak, "leak"))
            rec("B6", "FAIL",
                "selector SELECTED a candidate with early_eos_suppressed_count=7 and "
                f"max_cb0_run=99: step={res['selected']['step']} "
                "(plan 11 requires early_eos_suppressed_count==0 and CB0 run<=12)")
        except ValueError as exc:
            rec("B6", "PASS", f"suppression/repeat violations refused: {exc!s:.90}")

        # (3) empty generation (0 frames) -> plan 11 forbids
        rep_empty = emit(n_decoded_frames=0)
        rep_empty2 = json.loads(json.dumps(rep_empty))
        for c in CONDITIONS:
            for row in rep_empty2["conditions"][c]["rows"]:
                row["n_frames"] = 0
        try:
            res = m76.select_d1c_formal_reports(five(rep_empty2, "empty"))
            rec("B6", "FAIL",
                f"selector SELECTED a candidate with zero-frame generations: step={res['selected']['step']}")
        except ValueError as exc:
            rec("B6", "PASS", f"empty generations refused: {exc!s:.90}")

        # (4) duration ratio far outside [0.5, 2.0] -> plan 11 forbids
        rep_dur = json.loads(json.dumps(emit()))
        for c in CONDITIONS:
            for row in rep_dur["conditions"][c]["rows"]:
                row["n_frames"] = 400  # gt_frames=42 -> ratio ~9.5
        try:
            res = m76.select_d1c_formal_reports(five(rep_dur, "dur"))
            rec("B6", "FAIL",
                "selector SELECTED a candidate with per-row duration ratio ~9.5x "
                f"(plan 11 requires [0.5,2.0]): step={res['selected']['step']}")
        except ValueError as exc:
            rec("B6", "PASS", f"duration-ratio violations refused: {exc!s:.90}")

        # (5) catastrophic rows explicitly flagged -> must be refused
        rep_cat = json.loads(json.dumps(emit()))
        for c in CONDITIONS:
            for row in rep_cat["conditions"][c]["rows"]:
                row["catastrophic"] = True
        try:
            res = m76.select_d1c_formal_reports(five(rep_cat, "cat"))
            rec("B6", "PASS" if not res["selected"]["gate_eligible"] else "FAIL",
                f"explicit catastrophic=True: gate_eligible={res['selected']['gate_eligible']}")
        except ValueError as exc:
            rec("B6", "PASS", f"catastrophic rows refused: {exc!s:.90}")

    print("\n=== REAUDIT PROBE6 SUMMARY ===")
    for b, s, d in RESULTS:
        print(f"{s:5s} {b:4s} {d}")
    print(f"\nTOTAL checks={len(RESULTS)} fail_or_error={sum(1 for r in RESULTS if r[1] in ('FAIL','ERROR'))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

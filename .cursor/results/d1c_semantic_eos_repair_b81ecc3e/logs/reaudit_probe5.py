"""Probe 5: cascade the v2 emitter past the missing-ordinal defect to enumerate
every remaining fail-open in the evaluator -> selector chain."""

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
    from minispeech.eval.discrete_report_schema import sha256_json

    contract = load_eval_contract(ROOT / "configs/eval_contracts/d1c_semantic_eos_v2_0_0.json")
    sel = load_formal_subset("select8", contract=contract)
    uids = list(sel["utt_ids"])

    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        wav_dir = tmp / "wavs"
        wav_dir.mkdir(parents=True)

        def rows(condition: str) -> list[dict]:
            out = []
            for i, uid in enumerate(uids):
                wav = wav_dir / f"{i:04d}.wav"
                wav.write_bytes(b"RIFFfakeWAVE")
                out.append(
                    {
                        "ordinal": i,  # <- the field scripts/74 forgets to stamp
                        "utt_id": uid,
                        "condition": condition,
                        "n_frames": 40,
                        "gt_frames": 42,
                        "cer": 0.08,
                        "ref": "привет мир",
                        "hyp": "привет мир",
                        "wav": str(wav),
                        "stop_reason": "learned_eos",
                        "generation": {
                            "termination_mode": "semantic_eos_v1",
                            "termination_reason": "learned_eos",
                            "n_decoded_frames": 40,
                            "eos_selected": True,
                            "early_eos_suppressed_count": 0,
                            "max_cb0_run": 3,
                            "generated_codes_sha256": "a" * 64,
                        },
                    }
                )
            return out

        condition_reports = {c: {"condition": c, "n": len(uids), "rows": rows(c)} for c in CONDITIONS}
        gate = {"verdict": "PASS", "criteria": {"macro_cer": 0.08, "n_catastrophic": 0}}
        ckpt = tmp / "candidate_step_0750.pt"
        ckpt.write_bytes(b"fake")
        out = m74._emit_d1c_v2_from_gate(
            out_dir=tmp / "eval_out",
            contract=contract,
            ckpt=str(ckpt),
            provenance={
                "subset_label": "select8",
                "subset_path": str(ROOT / "configs/train_subsets/d1c_dev32.json"),
                "subset_file_sha256": sel["file_sha256"],
                "checkpoint_step": 750,
            },
            infer=dict(contract.get("infer") or {}),
            # ADVERSARIAL: no EOS TF metrics supplied at all, matching the real
            # evaluator's teacher_forced_ce() which returns only loss/ce_cb0/ce_residual
            teacher_forced_ce={"loss": 3.21, "ce_cb0": 2.0, "ce_residual": 1.2},
            ordered_utt_ids=uids,
            condition_reports=condition_reports,
            gate=gate,
        )
        rec("B4", "INFO", f"with ordinal stamped, emitter succeeded -> {Path(out).name}")
        report = json.loads(Path(out).read_text(encoding="utf-8"))
        tf = report["teacher_forced_metrics"]
        rec("B4", "FAIL",
            f"fabricated TF provenance: teacher_forced_ce_source={tf.get('teacher_forced_ce_source')!r} "
            f"tf_eos_final_top1_accuracy={tf.get('tf_eos_final_top1_accuracy')} "
            f"tf_eos_early_top1_false_positive_count={tf.get('tf_eos_early_top1_false_positive_count')} "
            "(never measured by the evaluator)")
        rec("B4", "FAIL" if report.get("formal_eligible") is True else "PASS",
            f"formal_eligible hardcoded True by emitter: {report.get('formal_eligible')}")
        wav0 = report["conditions"]["model"]["rows"][0]["wav"]
        rec("B4", "FAIL" if wav0["file_sha256"] == wav0["pcm_sha256"] else "PASS",
            f"pcm_sha256 is a copy of file_sha256 (no real PCM hash): "
            f"{wav0['file_sha256'][:12]}… == {wav0['pcm_sha256'][:12]}…")
        cat = report["conditions"]["model"]["rows"][0].get("catastrophic")
        rec("B4", "FAIL" if cat is None else "PASS",
            f"row.catastrophic never stamped by evaluator -> selector n_catastrophic always 0 (got {cat!r})")

        # Selector: five identical reports differing only in step
        paths = []
        for step in (250, 375, 500, 625, 750):
            r = json.loads(json.dumps(report))
            r["checkpoint"]["step"] = step
            r["checkpoint"]["sha256"] = f"{step:064d}"
            r.pop("report_sha256", None)
            r["report_sha256"] = sha256_json(r)
            p = tmp / f"rep_{step}.json"
            p.write_text(json.dumps(r, ensure_ascii=False), encoding="utf-8")
            paths.append(p)
        result = m76.select_d1c_formal_reports(paths)
        s = result["selected"]
        rec("B6", "FAIL",
            f"selector declared candidate ELIGIBLE from a report with zero measured EOS-TF "
            f"metrics: step={s['step']} gate_eligible={s['gate_eligible']} macro_cer={s['macro_cer']}")

        # Does the selector notice a report whose rows say safety termination?
        bad = json.loads(json.dumps(report))
        for row in bad["conditions"]["model"]["rows"]:
            row["generation"]["termination_reason"] = "safety_max_frames"
        bpaths = []
        for step in (250, 375, 500, 625, 750):
            r = json.loads(json.dumps(bad))
            r["checkpoint"]["step"] = step
            r["checkpoint"]["sha256"] = f"{step:064d}"
            r.pop("report_sha256", None)
            r["report_sha256"] = sha256_json(r)
            p = tmp / f"bad_{step}.json"
            p.write_text(json.dumps(r, ensure_ascii=False), encoding="utf-8")
            bpaths.append(p)
        try:
            m76.select_d1c_formal_reports(bpaths)
            rec("B6", "FAIL", "selector selected a candidate whose every row hit safety_max_frames")
        except ValueError as exc:
            rec("B6", "PASS", f"selector refuses all-safety candidates: {exc!s:.80}")

        # Does the selector notice early_eos_suppressed_count > 0 / cb0 run > 12?
        leak = json.loads(json.dumps(report))
        for row in leak["conditions"]["model"]["rows"]:
            row["generation"]["early_eos_suppressed_count"] = 7
            row["generation"]["max_cb0_run"] = 99
        lpaths = []
        for step in (250, 375, 500, 625, 750):
            r = json.loads(json.dumps(leak))
            r["checkpoint"]["step"] = step
            r["checkpoint"]["sha256"] = f"{step:064d}"
            r.pop("report_sha256", None)
            r["report_sha256"] = sha256_json(r)
            p = tmp / f"leak_{step}.json"
            p.write_text(json.dumps(r, ensure_ascii=False), encoding="utf-8")
            lpaths.append(p)
        try:
            res = m76.select_d1c_formal_reports(lpaths)
            rec("B6", "FAIL",
                "selector selected a candidate with early_eos_suppressed_count=7 and "
                f"max_cb0_run=99 (plan 11 forbids both): step={res['selected']['step']}")
        except ValueError as exc:
            rec("B6", "PASS", f"selector refuses suppression/repeat violations: {exc!s:.80}")

    print("\n=== REAUDIT PROBE5 SUMMARY ===")
    for b, s, d in RESULTS:
        print(f"{s:5s} {b:4s} {d}")
    print(f"\nTOTAL checks={len(RESULTS)} fail_or_error={sum(1 for r in RESULTS if r[1] in ('FAIL','ERROR'))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

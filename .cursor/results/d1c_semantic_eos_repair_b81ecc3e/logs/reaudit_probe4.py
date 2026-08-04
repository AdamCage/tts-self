"""Independent adversarial probe, part 4: end-to-end reachability of the D1c v2 report path.

Feeds realistic evaluator-shaped rows (exactly as ``free_running_asr_eval`` builds
them in scripts/74) through ``_emit_d1c_v2_from_gate`` to test whether the D1c
formal report emitter actually works, or only appears wired.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "src"))

RESULTS: list[tuple[str, str, str]] = []


def rec(blocker: str, status: str, detail: str) -> None:
    RESULTS.append((blocker, status, detail))
    print(f"REAUDIT {blocker} {status} {detail}", flush=True)


def load_script(name: str):
    path = ROOT / "scripts" / name
    spec = importlib.util.spec_from_file_location(name.replace(".py", ""), path)
    mod = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


CONDITIONS = ("model", "copy_synth", "wrong_utt", "random_codes", "shuffle_text", "mask_text")


def evaluator_shaped_rows(uids: list[str], condition: str, wav_dir: Path) -> list[dict]:
    """Replicate scripts/74 free_running_asr_eval row_out exactly (lines ~421-434)."""
    rows = []
    for i, uid in enumerate(uids):
        wav = wav_dir / f"{i:04d}_{uid}.wav"
        wav.write_bytes(b"RIFF0000WAVEfake")
        row = {
            "utt_id": uid,
            "condition": condition,
            "n_frames": 40,
            "gt_frames": 42,
            "cer": 0.08,
            "ref": "привет мир",
            "hyp": "привет мир",
            "wav": str(wav),
            "stop_reason": "learned_eos" if condition == "model" else condition,
        }
        if condition in ("model", "wrong_utt", "shuffle_text", "mask_text"):
            row["generation"] = {
                "termination_mode": "semantic_eos_v1",
                "termination_reason": "learned_eos",
                "n_decoded_frames": 40,
                "eos_selected": True,
                "early_eos_suppressed_count": 0,
                "max_cb0_run": 3,
                "generated_codes_sha256": "a" * 64,
            }
        rows.append(row)
    return rows


def main() -> int:
    m74 = load_script("74_eval_discrete_lm.py")
    from minispeech.data.d1b_oracles import load_eval_contract
    from minispeech.data.discrete_oracles import load_formal_subset

    contract = load_eval_contract(ROOT / "configs/eval_contracts/d1c_semantic_eos_v2_0_0.json")
    rec("B4", "PASS" if m74._is_d1c_semantic_contract(contract) else "FAIL",
        "contract 2.0.0 recognised as D1c semantic contract")
    sel = load_formal_subset("select8", contract=contract)
    uids = [u.replace("/", "_") for u in sel["utt_ids"]]

    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        wav_dir = tmp / "wavs"
        wav_dir.mkdir(parents=True, exist_ok=True)
        condition_reports = {
            c: {"condition": c, "n": len(uids), "rows": evaluator_shaped_rows(uids, c, wav_dir),
                "infer": dict(contract.get("infer") or {})}
            for c in CONDITIONS
        }
        gate = {
            "verdict": "PASS",
            "criteria": {
                "macro_cer": 0.08, "fr_cer": 0.08, "n_catastrophic": 0,
                "max_abs_log_duration_error": 0.05, "mean_abs_log_duration_error": 0.02,
                "eligible": True,
            },
        }
        provenance = {
            "git_sha": "deadbeef", "git_dirty": False,
            "subset_label": "select8",
            "subset_path": str(ROOT / "configs/train_subsets/d1c_dev32.json"),
            "subset_file_sha256": sel["file_sha256"],
            "checkpoint_step": 750,
        }
        ckpt = tmp / "candidate_step_0750.pt"
        ckpt.write_bytes(b"fake-checkpoint")
        try:
            out = m74._emit_d1c_v2_from_gate(
                out_dir=tmp / "eval_out",
                contract=contract,
                ckpt=str(ckpt),
                provenance=provenance,
                infer=dict(contract.get("infer") or {}),
                teacher_forced_ce={"loss": 3.21, "ce_cb0": 2.0, "ce_residual": 1.2},
                ordered_utt_ids=uids,
                condition_reports=condition_reports,
                gate=gate,
            )
            rec("B4", "PASS", f"_emit_d1c_v2_from_gate produced {out}")
            report = json.loads(Path(out).read_text(encoding="utf-8"))
            tf = report["teacher_forced_metrics"]
            rec("B4", "INFO",
                f"emitted teacher_forced_metrics={json.dumps(tf, ensure_ascii=False)}")
            rec(
                "B4",
                "FAIL" if tf.get("teacher_forced_ce_source") == "eval_computed"
                and "tf_eos_final_top1_accuracy" in tf and tf["tf_eos_final_top1_accuracy"] == 1.0
                else "PASS",
                "emitted report claims eval_computed + tf_eos_final_top1_accuracy=1.0 "
                "even though the evaluator never measured EOS TF metrics",
            )
            # Now feed it to the selector to see whether eligibility is real
            m76 = load_script("76_select_discrete_checkpoint.py")
            paths = []
            for step in (250, 375, 500, 625, 750):
                r = json.loads(json.dumps(report))
                r["checkpoint"]["step"] = step
                r["checkpoint"]["sha256"] = f"{step:064d}"
                r.pop("report_sha256", None)
                from minispeech.eval.discrete_report_schema import sha256_json

                r["report_sha256"] = sha256_json(r)
                p = tmp / f"rep_{step}.json"
                p.write_text(json.dumps(r, ensure_ascii=False), encoding="utf-8")
                paths.append(p)
            result = m76.select_d1c_formal_reports(paths)
            rec(
                "B4",
                "INFO",
                f"selector accepted fabricated-TF reports; selected step="
                f"{result['selected']['step']} gate_eligible={result['selected']['gate_eligible']}",
            )
        except Exception as exc:  # noqa: BLE001
            import traceback

            traceback.print_exc()
            rec(
                "B4",
                "FAIL",
                f"_emit_d1c_v2_from_gate RAISED on evaluator-shaped rows: "
                f"{type(exc).__name__}: {exc}",
            )

    print("\n=== REAUDIT PROBE4 SUMMARY ===")
    for b, s, d in RESULTS:
        print(f"{s:5s} {b:4s} {d}")
    fails = [r for r in RESULTS if r[1] in ("FAIL", "ERROR")]
    print(f"\nTOTAL checks={len(RESULTS)} fail_or_error={len(fails)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

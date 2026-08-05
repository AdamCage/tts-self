#!/usr/bin/env python3
"""REAUDIT_ROUND5 — the guard must hold at the PRODUCTION CLI, not just in-process.

Every prior probe (mine and the repair agent's) calls
``select_d1c_formal_reports`` in-process. This probe shells out to
``python scripts/76_select_discrete_checkpoint.py --d1c-formal ...`` and checks
the process exit code, the message, and whether ``checkpoint_selection.json``
was written. A guard that raises but still writes the selection file, or that is
bypassed by the CLI branch, would be invisible to an in-process probe.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "src"))

from minispeech.eval.discrete_report_schema import (  # noqa: E402
    build_formal_report,
    ordered_utt_ids_sha256,
    sha256_json,
)

PY = "/home/adam/projects/tts-self/.venv/bin/python"
SEL = ROOT / "scripts/76_select_discrete_checkpoint.py"
STEPS = [250, 375, 500, 625, 750]
IDS = ["u0", "u1"]
CONDS = ("model", "copy_synth", "wrong_utt", "random_codes", "shuffle_text", "mask_text")

RESULTS: list[tuple[str, str, str]] = []


def rec(tag: str, ok, msg: str) -> None:
    status = "INFO" if ok is None else ("PASS" if ok else "FAIL")
    RESULTS.append((tag, status, msg))
    print(f"{tag} {status} {msg}", flush=True)


def conditions_block() -> dict:
    out = {}
    for name in CONDS:
        rows = []
        for i, uid in enumerate(IDS):
            rows.append(
                {
                    "utt_id": uid,
                    "ordinal": i,
                    "condition": name,
                    "cer": 0.0 if name in ("model", "copy_synth") else 0.9,
                    "catastrophic": False,
                    "n_frames": 10,
                    "gt_frames": 10,
                    "abs_log_duration_error": 0.0,
                    "normalized_reference_sha256": "a" * 64,
                    "generated_codes_sha256": hashlib.sha256(
                        f"{name}{uid}".encode()
                    ).hexdigest(),
                    "generation": {
                        "termination_reason": "learned_eos",
                        "n_decoded_frames": 10,
                        "early_eos_suppressed_count": 0,
                        "max_cb0_run": 2,
                    },
                    "wav": {
                        "path": f"{name}_{uid}.wav",
                        "file_sha256": "c" * 64,
                        "pcm_sha256": "d" * 64,
                    },
                }
            )
        out[name] = {
            "subset_key": "select8",
            "subset_path": "configs/frozen/d1c_dev32.json",
            "subset_file_sha256": "e" * 64,
            "expected_count": len(IDS),
            "evaluated_count": len(IDS),
            "ordered_utt_ids_sha256": ordered_utt_ids_sha256(IDS),
            "rows": rows,
        }
    return out


def make_report(step: int, ckpt: Path, sha: str) -> dict:
    return build_formal_report(
        formal_eligible=True,
        checkpoint={
            "path": str(ckpt),
            "sha256": sha,
            "step": step,
            "schema": "minispeech.discrete_lm_checkpoint",
            "schema_version": "2.0.0",
        },
        provenance={
            "eval_contract_sha256": "1" * 64,
            "recipe_sha256": "2" * 64,
            "asr_sha256": "3" * 64,
            "checkpoint_step": step,
        },
        infer={"max_frames": 750},
        teacher_forced_metrics={
            "teacher_forced_ce_source": "eval_computed",
            "total_ce": 1.0 + step / 10000.0,
            "tf_eos_final_top1_accuracy": 1.0,
            "tf_eos_early_top1_false_positive_count": 0,
        },
        row_binding={
            "ordered_utt_ids": IDS,
            "ordered_utt_ids_sha256": ordered_utt_ids_sha256(IDS),
        },
        conditions=conditions_block(),
        summaries={
            "model_macro_cer": 0.0,
            "n_catastrophic": 0,
            "max_abs_log_duration_error": 0.0,
            "mean_abs_log_duration_error": 0.0,
            "runaway_or_repetition": False,
            "phrase_loop_repetition": False,
        },
        gates={"eligible": True, "text_usage": {"pass": True}, "runaway_or_repetition": False},
    )


def run_cli(paths: list[Path], out: Path) -> tuple[int, str]:
    proc = subprocess.run(  # noqa: S603
        [PY, str(SEL), "--d1c-formal", "--formal-reports", *[str(p) for p in paths],
         "--out", str(out)],
        cwd=str(ROOT), capture_output=True, text=True, check=False,
        env={
            "PATH": "/usr/bin:/bin", "CUDA_VISIBLE_DEVICES": "", "HOME": "/home/adam",
            "HF_HOME": "/home/adam/hf_cache", "PYTHONPATH": str(ROOT / "src"),
            "GIT_DIR": "/mnt/c/projects/tts-self/.git/worktrees/tts-self-d1c",
            "GIT_WORK_TREE": str(ROOT),
        },
    )
    return proc.returncode, (proc.stderr or proc.stdout).strip()


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="reaudit5_cli_") as tmp:
        td = Path(tmp)

        # ---- legit set: five distinct candidate files
        legit_dir = td / "legit"
        legit_dir.mkdir(parents=True)
        legit = []
        for s in STEPS:
            ck = legit_dir / f"candidate_step_{s:04d}.pt"
            ck.write_bytes(f"distinct-checkpoint-bytes-{s}".encode())
            rep = make_report(s, ck, hashlib.sha256(ck.read_bytes()).hexdigest())
            p = legit_dir / f"report_{s}.json"
            p.write_text(json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8")
            legit.append(p)
        out_ok = td / "sel_ok.json"
        rc, msg = run_cli(legit, out_ok)
        sel = json.loads(out_ok.read_text("utf-8")) if out_ok.is_file() else {}
        rec("K1", rc == 0 and out_ok.is_file(),
            f"CLI accepts the legit set: rc={rc} wrote={out_ok.is_file()} "
            f"selected_step={(sel.get('selected') or {}).get('step')} msg={msg[:80]!r}")

        # ---- attack set: one physical checkpoint published five times
        atk_dir = td / "attack"
        atk_dir.mkdir(parents=True)
        blob = b"one-physical-checkpoint-bytes"
        shared = hashlib.sha256(blob).hexdigest()
        atk = []
        for s in STEPS:
            ck = atk_dir / f"candidate_step_{s:04d}.pt"
            ck.write_bytes(blob)
            rep = make_report(s, ck, shared)
            p = atk_dir / f"report_{s}.json"
            p.write_text(json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8")
            atk.append(p)
        out_bad = td / "sel_attack.json"
        rc, msg = run_cli(atk, out_bad)
        rec("K2", rc != 0 and "duplicate checkpoint.sha256" in msg,
            f"CLI refuses the duplicate-checkpoint set: rc={rc} msg={msg[:130]!r}")
        rec("K3", not out_bad.is_file(),
            f"no checkpoint_selection.json written on refusal (exists={out_bad.is_file()})")

        # ---- forged sha set
        f_dir = td / "forged"
        f_dir.mkdir(parents=True)
        forged = []
        for s in STEPS:
            ck = f_dir / f"candidate_step_{s:04d}.pt"
            ck.write_bytes(f"real-bytes-{s}".encode())
            live = hashlib.sha256(ck.read_bytes()).hexdigest()
            rep = make_report(s, ck, live)
            if s == 500:
                rep["checkpoint"]["sha256"] = "f" * 64
                rep.pop("report_sha256", None)
                rep["report_sha256"] = sha256_json(rep)
            p = f_dir / f"report_{s}.json"
            p.write_text(json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8")
            forged.append(p)
        out_f = td / "sel_forged.json"
        rc, msg = run_cli(forged, out_f)
        rec("K4", rc != 0 and "does not match file" in msg,
            f"CLI refuses a forged checkpoint.sha256: rc={rc} msg={msg[:130]!r}")
        rec("K5", not out_f.is_file(), f"no selection written (exists={out_f.is_file()})")

        # ---- missing checkpoint file
        g_dir = td / "gone"
        g_dir.mkdir(parents=True)
        gone = []
        for s in STEPS:
            ck = g_dir / f"candidate_step_{s:04d}.pt"
            ck.write_bytes(f"gone-bytes-{s}".encode())
            rep = make_report(s, ck, hashlib.sha256(ck.read_bytes()).hexdigest())
            p = g_dir / f"report_{s}.json"
            p.write_text(json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8")
            gone.append(p)
            if s == 625:
                ck.unlink()
        out_g = td / "sel_gone.json"
        rc, msg = run_cli(gone, out_g)
        rec("K6", rc != 0 and "not readable" in msg,
            f"CLI refuses an unreachable checkpoint.path: rc={rc} msg={msg[:120]!r}")

        # ---- historical/manual inputs still refused in formal mode
        proc = subprocess.run(  # noqa: S603
            [PY, str(SEL), "--d1c-formal", "--candidates-json", str(legit[0]),
             "--out", str(td / "x.json")],
            cwd=str(ROOT), capture_output=True, text=True, check=False,
            env={"PATH": "/usr/bin:/bin", "CUDA_VISIBLE_DEVICES": "", "HOME": "/home/adam",
                 "HF_HOME": "/home/adam/hf_cache", "PYTHONPATH": str(ROOT / "src")},
        )
        rec("K7", proc.returncode != 0,
            f"formal mode still refuses flat/manual candidate inputs: rc={proc.returncode} "
            f"msg={(proc.stderr or '').strip()[:100]!r}")

    fails = sum(1 for _, st, _ in RESULTS if st == "FAIL")
    print(f"\nSUMMARY fails={fails} total={len(RESULTS)}")
    for tag, st, msg in RESULTS:
        if st == "FAIL":
            print(f"  FAILED {tag}: {msg}")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())

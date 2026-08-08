#!/usr/bin/env python3
"""ROUND7 live probe: identical-weights / forged-stamp attack (I1/K7) + legit accept.

CPU only. Builds evaluator-shaped formal reports over real torch checkpoints and
drives production ``select_d1c_formal_reports`` + ``scripts/76 --d1c-formal``.
No GPU, no seed invent, no commit.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "src"))
RESULTS = ROOT / ".cursor" / "results" / "d1c_semantic_eos_repair_b81ecc3e"
LOG = RESULTS / "logs" / "repair_round7_probe_identical_weights.log"

STEPS = (250, 375, 500, 625, 750)
LINES: list[str] = []


def _load_script(name: str):
    path = ROOT / "scripts" / name
    spec = importlib.util.spec_from_file_location(path.stem, path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def rec(fid: str, ok: bool | None, msg: str) -> None:
    tag = "PASS" if ok is True else ("FAIL" if ok is False else "INFO")
    LINES.append(f"{fid} {tag} {msg}")
    print(LINES[-1], flush=True)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _eligible_report(
    *,
    ckpt: Path,
    step: int,
    hashes: dict[str, str],
    cer: float = 0.05,
) -> dict:
    from minispeech.eval.discrete_report_schema import (
        ordered_utt_ids_sha256,
        rows_sha256,
        sha256_json,
    )

    live = _sha(ckpt)
    ids = ["u0"]
    ids_sha = ordered_utt_ids_sha256(ids)
    row = {
        "utt_id": "u0",
        "ordinal": 0,
        "cer": cer,
        "catastrophic": False,
        "n_frames": 10,
        "gt_frames": 10,
        "generation": {
            "termination_reason": "learned_eos",
            "n_decoded_frames": 10,
            "early_eos_suppressed_count": 0,
            "max_cb0_run": 2,
        },
        "wav": {"path": "x.wav", "file_sha256": "a" * 64, "pcm_sha256": "b" * 64},
        "normalized_reference_sha256": "c" * 64,
        "generated_codes_sha256": "d" * 64,
    }
    cond = {
        "subset_key": "select8",
        "subset_path": "x",
        "subset_file_sha256": "3" * 64,
        "expected_count": 1,
        "evaluated_count": 1,
        "ordered_utt_ids_sha256": ids_sha,
        "rows": [row],
        "rows_sha256": rows_sha256([row]),
    }
    report = {
        "schema": "minispeech.discrete_eval_report",
        "schema_version": "2.0.0",
        "formal_eligible": True,
        "checkpoint": {
            "path": str(ckpt),
            "sha256": live,
            "step": int(step),
            "schema": "minispeech.discrete_lm_checkpoint",
            "schema_version": "2.0.0",
            "hashes": dict(hashes),
        },
        "provenance": {
            "eval_contract_sha256": hashes["eval_contract"],
            "recipe_sha256": hashes["recipe"],
            "asr_sha256": "1" * 64,
        },
        "infer": {"max_frames": 750},
        "teacher_forced_metrics": {
            "teacher_forced_ce_source": "eval_computed",
            "total_ce": 1.0,
            "tf_eos_final_top1_accuracy": 1.0,
            "tf_eos_early_top1_false_positive_count": 0,
        },
        "row_binding": {
            "ordered_utt_ids": ids,
            "ordered_utt_ids_sha256": ids_sha,
        },
        "conditions": {
            name: dict(cond)
            for name in (
                "model",
                "copy_synth",
                "wrong_utt",
                "random_codes",
                "shuffle_text",
                "mask_text",
            )
        },
        "summaries": {
            "model_macro_cer": cer,
            "n_catastrophic": 0,
            "max_abs_log_duration_error": 0.0,
            "mean_abs_log_duration_error": 0.0,
            "phrase_loop_repetition": False,
            "runaway_or_repetition": False,
        },
        "gates": {
            "eligible": True,
            "text_usage": {"pass": True},
            "runaway_or_repetition": False,
        },
    }
    report["report_sha256"] = sha256_json(report)
    return report


def main() -> int:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    from minispeech.data.d1b_oracles import load_eval_contract
    from minispeech.eval.discrete_report_schema import sha256_json, validate_formal_report
    from minispeech.models.discrete_speech_lm import DiscreteSpeechLM, DiscreteSpeechLMConfig

    mod76 = _load_script("76_select_discrete_checkpoint.py")
    contract = load_eval_contract(
        ROOT / "configs" / "eval_contracts" / "d1c_semantic_eos_v2_0_0.json"
    )
    live_hashes = {
        "eval_contract": str(contract["sha256"]),
        "recipe": str(contract["recipe_sha256"]),
        "recipe_source_manifest": str(contract["recipe_source_manifest_sha256"]),
        "run_manifest": "r" * 64,
    }

    cfg = DiscreteSpeechLMConfig.tiny()
    cfg.termination_mode = "semantic_eos_v1"
    cfg.eos_id = cfg.codebook_size
    cfg.codec_vocab_size = cfg.codebook_size
    cfg.semantic_vocab_size = cfg.codebook_size + 1
    base = DiscreteSpeechLM(cfg)
    shared = {k: v.detach().clone() for k, v in base.state_dict().items()}

    def save(path: Path, *, step: int, state: dict, junk: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model": {k: v.detach().clone() for k, v in state.items()},
                "step": int(step),
                "optimizer_step": int(step),
                "hashes": dict(live_hashes),
                "junk": junk,
            },
            path,
        )

    def write_reports(td: Path, ckpts: list[Path]) -> list[Path]:
        td.mkdir(parents=True, exist_ok=True)
        paths = []
        for ck, step in zip(ckpts, STEPS):
            report = _eligible_report(ckpt=ck, step=step, hashes=live_hashes)
            validate_formal_report(report)  # real schema
            p = td / f"report_{step}.json"
            p.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            paths.append(p)
        return paths

    def run_cli(reports: list[Path], out: Path) -> subprocess.CompletedProcess[str]:
        env = dict(os.environ)
        env["CUDA_VISIBLE_DEVICES"] = ""
        env["PYTHONPATH"] = str(ROOT / "src") + os.pathsep + env.get("PYTHONPATH", "")
        return subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts" / "76_select_discrete_checkpoint.py"),
                "--d1c-formal",
                "--formal-reports",
                *[str(p) for p in reports],
                "--out",
                str(out),
            ],
            capture_output=True,
            text=True,
            cwd=str(ROOT),
            env=env,
        )

    with tempfile.TemporaryDirectory(prefix="d1c_r7_") as td_s:
        td = Path(td_s)

        # ---- I1 / K7 attack ----
        attack_ck = []
        for s in STEPS:
            ck = td / "attack" / f"candidate_step_{s:04d}.pt"
            save(ck, step=s, state=shared, junk="x" * (s % 7 + 1))
            attack_ck.append(ck)
        wdig = {mod76._model_state_sha256(c) for c in attack_ck}
        fdig = {_sha(c) for c in attack_ck}
        rec(
            "I0",
            len(wdig) == 1 and len(fdig) == 5,
            f"attack setup weight_digests={len(wdig)}/5 file_sha={len(fdig)}/5",
        )
        attack_reports = write_reports(td / "attack_reports", attack_ck)
        try:
            mod76.select_d1c_formal_reports(attack_reports)
            rec("I1", False, "selector ACCEPTED identical-weight forged-stamp set")
        except ValueError as exc:
            rec(
                "I1",
                "identical model weights" in str(exc),
                f"selector refused: {str(exc)[:160]}",
            )
        out_attack = td / "attack_selection.json"
        proc = run_cli(attack_reports, out_attack)
        rec(
            "K7",
            proc.returncode != 0 and not out_attack.is_file(),
            f"CLI rc={proc.returncode} wrote={out_attack.is_file()} "
            f"err={(proc.stderr or proc.stdout)[:160]}",
        )

        # ---- Legitimate distinct weights ----
        legit_ck = []
        for i, s in enumerate(STEPS):
            state = {k: v.detach().clone() for k, v in shared.items()}
            first = next(iter(state))
            state[first] = state[first] + (0.01 * (i + 1))
            ck = td / "legit" / f"candidate_step_{s:04d}.pt"
            save(ck, step=s, state=state, junk=f"legit-{s}")
            legit_ck.append(ck)
        wdig = {mod76._model_state_sha256(c) for c in legit_ck}
        rec("L0", len(wdig) == 5, f"legit distinct_weight_digests={len(wdig)}/5")
        legit_reports = write_reports(td / "legit_reports", legit_ck)
        try:
            sel = mod76.select_d1c_formal_reports(legit_reports)
            rec(
                "L1",
                int(sel["selected"]["step"]) in STEPS,
                f"selector ACCEPTS legit set selected_step={sel['selected']['step']}",
            )
        except ValueError as exc:
            rec("L1", False, f"selector refused legit set: {exc}")
        out_legit = td / "legit_selection.json"
        proc = run_cli(legit_reports, out_legit)
        rec(
            "L2",
            proc.returncode == 0 and out_legit.is_file(),
            f"CLI rc={proc.returncode} wrote={out_legit.is_file()} "
            f"out={(proc.stdout or '')[:120]}",
        )

        # ---- J1 hashes vs provenance ----
        skewed = []
        for i, p in enumerate(legit_reports):
            d = json.loads(p.read_text(encoding="utf-8"))
            d["checkpoint"]["hashes"]["eval_contract"] = "8" * 64
            d.pop("report_sha256", None)
            d["report_sha256"] = sha256_json(d)
            sp = td / f"j1_{i}.json"
            sp.write_text(json.dumps(d), encoding="utf-8")
            skewed.append(sp)
        try:
            mod76.select_d1c_formal_reports(skewed)
            rec("J1", False, "selector ACCEPTED hashes/provenance skew")
        except ValueError as exc:
            rec(
                "J1",
                "checkpoint.hashes.eval_contract" in str(exc),
                f"selector refused skew: {str(exc)[:160]}",
            )

    n_pass = sum(1 for line in LINES if " PASS " in line)
    n_fail = sum(1 for line in LINES if " FAIL " in line)
    summary = f"SUMMARY pass={n_pass} fail={n_fail} total={len(LINES)}"
    LINES.append(summary)
    print(summary, flush=True)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    LOG.write_text("\n".join(LINES) + "\n", encoding="utf-8")
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())

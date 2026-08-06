#!/usr/bin/env python3
"""REAUDIT_ROUND7 — independent attack of identical-weights / J1 / bypasses.

Drives production ``select_d1c_formal_reports`` and ``scripts/76 --d1c-formal``.
Does not trust repair-agent logs. CPU only; no seed invent; no commit.
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
RES = ROOT / ".cursor" / "results" / "d1c_semantic_eos_repair_b81ecc3e"
LOG = RES / "logs" / "reaudit7_probe_attack.log"

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
    provenance_overrides: dict[str, str] | None = None,
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
    prov = {
        "eval_contract_sha256": hashes["eval_contract"],
        "recipe_sha256": hashes["recipe"],
        "asr_sha256": "1" * 64,
    }
    if provenance_overrides:
        prov.update(provenance_overrides)
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
        "provenance": prov,
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


def main() -> int:  # noqa: C901, PLR0912, PLR0915
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    from minispeech.data.d1b_oracles import load_eval_contract
    from minispeech.eval.discrete_report_schema import sha256_json, validate_formal_report
    from minispeech.models.discrete_speech_lm import DiscreteSpeechLM, DiscreteSpeechLMConfig

    mod76 = _load_script("76_select_discrete_checkpoint.py")
    # Also import 74 helpers for rename / invent spot checks.
    mod74 = _load_script("74_eval_discrete_lm.py")

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

    def save(path: Path, *, step: int, state: dict | None, junk: str, **extra) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        blob: dict = {
            "step": int(step),
            "optimizer_step": int(step),
            "hashes": dict(live_hashes),
            "junk": junk,
        }
        if state is not None:
            blob["model"] = {k: v.detach().clone() for k, v in state.items()}
        blob.update(extra)
        torch.save(blob, path)

    def write_reports(
        td: Path,
        ckpts: list[Path],
        *,
        hash_overrides: dict[str, str] | None = None,
        provenance_overrides: dict[str, str] | None = None,
    ) -> list[Path]:
        td.mkdir(parents=True, exist_ok=True)
        hashes = dict(live_hashes)
        if hash_overrides:
            hashes.update(hash_overrides)
        paths = []
        for ck, step in zip(ckpts, STEPS):
            report = _eligible_report(
                ckpt=ck,
                step=step,
                hashes=hashes,
                provenance_overrides=provenance_overrides,
            )
            validate_formal_report(report)
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

    with tempfile.TemporaryDirectory(prefix="d1c_reaudit7_") as td_s:
        td = Path(td_s)

        # ---- I1 / K7 identical-weights forged stamps ----
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
                f"selector refused: {str(exc)[:180]}",
            )
        out_attack = td / "attack_selection.json"
        proc = run_cli(attack_reports, out_attack)
        rec(
            "K7",
            proc.returncode != 0 and not out_attack.is_file(),
            f"CLI rc={proc.returncode} wrote={out_attack.is_file()} "
            f"err={(proc.stderr or proc.stdout)[:180]}",
        )

        # ---- Legit five distinct weight digests ----
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
            f"out={(proc.stdout or '')[:140]}",
        )

        # ---- J1 hashes vs provenance skew ----
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
                "checkpoint.hashes.eval_contract" in str(exc)
                or "provenance.eval_contract_sha256" in str(exc),
                f"selector refused skew: {str(exc)[:180]}",
            )
        out_j1 = td / "j1_selection.json"
        proc = run_cli(skewed, out_j1)
        rec(
            "J1c",
            proc.returncode != 0 and not out_j1.is_file(),
            f"CLI J1 rc={proc.returncode} wrote={out_j1.is_file()} "
            f"err={(proc.stderr or proc.stdout)[:160]}",
        )

        # ---- Bypass surfaces ----
        # B1: empty model state_dict
        empty_ck = []
        for s in STEPS:
            ck = td / "empty" / f"candidate_step_{s:04d}.pt"
            save(ck, step=s, state={}, junk=f"empty-{s}")
            empty_ck.append(ck)
        empty_reports = write_reports(td / "empty_reports", empty_ck)
        try:
            mod76.select_d1c_formal_reports(empty_reports)
            rec("B1", False, "selector ACCEPTED empty model state_dict set")
        except ValueError as exc:
            rec(
                "B1",
                "non-empty model" in str(exc) or "state_dict" in str(exc),
                f"empty state_dict refused: {str(exc)[:160]}",
            )

        # B2: missing model key entirely
        miss_ck = []
        for s in STEPS:
            ck = td / "nomodel" / f"candidate_step_{s:04d}.pt"
            save(ck, step=s, state=None, junk=f"nomodel-{s}")
            miss_ck.append(ck)
        miss_reports = write_reports(td / "nomodel_reports", miss_ck)
        try:
            mod76.select_d1c_formal_reports(miss_reports)
            rec("B2", False, "selector ACCEPTED checkpoints with no model key")
        except ValueError as exc:
            rec(
                "B2",
                "non-empty model" in str(exc) or "state_dict" in str(exc),
                f"missing model key refused: {str(exc)[:160]}",
            )

        # B3: forged model_state_sha256 field in report (should be ignored; live digest wins)
        forged_reports = []
        for i, (p, ck) in enumerate(zip(legit_reports, legit_ck)):
            d = json.loads(p.read_text(encoding="utf-8"))
            # Plant a fake digest field that would collide if trusted.
            d["checkpoint"]["model_state_sha256"] = "f" * 64
            d["checkpoint"]["weight_digest"] = "f" * 64
            d.pop("report_sha256", None)
            d["report_sha256"] = sha256_json(d)
            sp = td / f"forge_digest_{i}.json"
            sp.write_text(json.dumps(d), encoding="utf-8")
            forged_reports.append(sp)
        # Attack variant: identical weights BUT forged distinct digests in report JSON.
        attack_forged = []
        for i, (p, ck) in enumerate(zip(attack_reports, attack_ck)):
            d = json.loads(p.read_text(encoding="utf-8"))
            d["checkpoint"]["model_state_sha256"] = f"{i:064x}"
            d["checkpoint"]["weight_digest"] = f"{i:064x}"
            d.pop("report_sha256", None)
            d["report_sha256"] = sha256_json(d)
            sp = td / f"attack_forge_{i}.json"
            sp.write_text(json.dumps(d), encoding="utf-8")
            attack_forged.append(sp)
        try:
            mod76.select_d1c_formal_reports(attack_forged)
            rec(
                "B3",
                False,
                "BYPASS: forged report weight_digest fields made identical-weight set ACCEPT",
            )
        except ValueError as exc:
            rec(
                "B3",
                "identical model weights" in str(exc),
                f"forged report digests ignored; live refuse: {str(exc)[:160]}",
            )
        # Legit set with forged fields should still accept (fields ignored).
        try:
            sel = mod76.select_d1c_formal_reports(forged_reports)
            rec(
                "B3b",
                int(sel["selected"]["step"]) in STEPS,
                f"extraneous report digest fields ignored on legit set step={sel['selected']['step']}",
            )
        except ValueError as exc:
            rec("B3b", False, f"extraneous fields broke legit accept: {exc}")

        # B4: partial collision (4 distinct + 1 clone of first) must refuse
        partial_ck = []
        for i, s in enumerate(STEPS):
            state = {k: v.detach().clone() for k, v in shared.items()}
            first = next(iter(state))
            if i == 4:
                # Clone step-250 weights exactly.
                state = {k: v.detach().clone() for k, v in shared.items()}
                first = next(iter(state))
                state[first] = state[first] + 0.01  # same delta as i=0
            else:
                state[first] = state[first] + (0.01 * (i + 1))
            ck = td / "partial" / f"candidate_step_{s:04d}.pt"
            save(ck, step=s, state=state, junk=f"partial-{s}")
            partial_ck.append(ck)
        partial_w = {mod76._model_state_sha256(c) for c in partial_ck}
        partial_reports = write_reports(td / "partial_reports", partial_ck)
        try:
            mod76.select_d1c_formal_reports(partial_reports)
            rec(
                "B4",
                False,
                f"partial collision ACCEPTED (distinct_weight_digests={len(partial_w)}/5)",
            )
        except ValueError as exc:
            rec(
                "B4",
                "identical model weights" in str(exc) and len(partial_w) < 5,
                f"partial collision refused digests={len(partial_w)}/5: {str(exc)[:140]}",
            )

        # ---- Prior regressions (quick) ----
        # R1: rename — no payload step → resolve_checkpoint_step refuses
        rename_ck = td / "rename" / "candidate_step_0250.pt"
        rename_ck.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model": shared, "hashes": dict(live_hashes), "junk": "rename"}, rename_ck)
        try:
            mod74.resolve_checkpoint_step(
                ckpt_path=rename_ck,
                payload={"model": shared, "hashes": dict(live_hashes)},
                require=True,
            )
            rec("R1", False, "rename (no payload step) was ACCEPTED by resolve_checkpoint_step")
        except (SystemExit, ValueError, RuntimeError) as exc:
            rec(
                "R1",
                "payload['step']" in str(exc) or "step" in str(exc).lower(),
                f"rename/no-step refused: {str(exc)[:140]}",
            )

        # R2: CLI invent — no payload step + --checkpoint-step confirms invent
        invent_ck = td / "invent" / "best.pt"
        invent_ck.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model": shared, "junk": "invent-only"}, invent_ck)
        try:
            mod74.resolve_checkpoint_step(
                ckpt_path=invent_ck,
                payload={"model": shared, "junk": "invent-only"},
                cli_step=375,
                require=True,
            )
            rec("R2", False, "CLI invent (best.pt + --checkpoint-step) ACCEPTED")
        except (SystemExit, ValueError, RuntimeError) as exc:
            rec(
                "R2",
                "cannot invent" in str(exc) or "payload['step']" in str(exc),
                f"CLI invent refused: {str(exc)[:160]}",
            )

        # R3: distinct file sha still enforced (duplicate sha refuses)
        dup_ck = []
        # Two steps pointing at the SAME file path by rewriting reports.
        same = td / "dup" / "candidate_step_0250.pt"
        save(same, step=250, state=shared, junk="dup")
        # Build five distinct files then rewrite report[1] to claim same sha+path as report[0]
        # Use two reports with identical sha by hardlinking content... simpler: two ckpts with same bytes
        bytes250 = same.read_bytes()
        for s in STEPS:
            ck = td / "dupset" / f"candidate_step_{s:04d}.pt"
            if s == 250:
                ck.parent.mkdir(parents=True, exist_ok=True)
                ck.write_bytes(bytes250)
            elif s == 375:
                ck.parent.mkdir(parents=True, exist_ok=True)
                ck.write_bytes(bytes250)  # identical file bytes → same sha
            else:
                state = {k: v.detach().clone() for k, v in shared.items()}
                first = next(iter(state))
                state[first] = state[first] + float(s) * 0.001
                save(ck, step=s, state=state, junk=f"dup-{s}")
            dup_ck.append(ck)
        # Fix basenames: 375 has identical bytes to 250 but wrong step stamp inside — still same sha
        # Need proper step stamps with identical bytes impossible unless we only collide sha via copy.
        # Re-do: write five with distinct weights, then force report sha collision by pointing two
        # reports at the same path.
        five = []
        for i, s in enumerate(STEPS):
            state = {k: v.detach().clone() for k, v in shared.items()}
            first = next(iter(state))
            state[first] = state[first] + (0.02 * (i + 1))
            ck = td / "sha" / f"candidate_step_{s:04d}.pt"
            save(ck, step=s, state=state, junk=f"sha-{s}")
            five.append(ck)
        sha_reports = write_reports(td / "sha_reports", five)
        # Mutate step=375 report to reuse step=250 path+sha (duplicate sha across candidates).
        r0 = json.loads(sha_reports[0].read_text(encoding="utf-8"))
        r1 = json.loads(sha_reports[1].read_text(encoding="utf-8"))
        r1["checkpoint"]["path"] = r0["checkpoint"]["path"]
        r1["checkpoint"]["sha256"] = r0["checkpoint"]["sha256"]
        # Keep step=375 and basename mismatch will fire first — instead keep path as
        # candidate_step_0375.pt but copy bytes so sha matches 250.
        five[1].write_bytes(five[0].read_bytes())
        r1 = json.loads(sha_reports[1].read_text(encoding="utf-8"))
        r1["checkpoint"]["sha256"] = _sha(five[1])
        r1["checkpoint"]["path"] = str(five[1])
        r1.pop("report_sha256", None)
        r1["report_sha256"] = sha256_json(r1)
        sha_reports[1].write_text(json.dumps(r1), encoding="utf-8")
        # Also need shared run hashes + distinct basenames — file sha now collides.
        try:
            mod76.select_d1c_formal_reports(sha_reports)
            rec("R3", False, "duplicate checkpoint.sha256 ACCEPTED")
        except ValueError as exc:
            rec(
                "R3",
                "duplicate checkpoint.sha256" in str(exc) or "identical model weights" in str(exc),
                f"distinct-sha / identity refuse: {str(exc)[:160]}",
            )

    n_pass = sum(1 for line in LINES if " PASS " in line)
    n_fail = sum(1 for line in LINES if " FAIL " in line)
    n_info = sum(1 for line in LINES if " INFO " in line)
    summary = f"SUMMARY pass={n_pass} fail={n_fail} info={n_info} total={len(LINES)}"
    LINES.append(summary)
    print(summary, flush=True)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    LOG.write_text("\n".join(LINES) + "\n", encoding="utf-8")
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())

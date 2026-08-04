#!/usr/bin/env python3
"""REAUDIT_ROUND5 — independent attack probe on the round-5 §8.4 closure.

Round 5 claims the round-4 MEDIUM (one physical checkpoint published as all five
candidates) is closed fail-closed. This probe does NOT trust that claim.

Differences from `logs/repair_round5_probe_dupckpt.py` (which I do not trust):

* Its N2 replaces ``validate_formal_report`` with ``lambda data, **kw: data``,
  so the selector refusal was only ever demonstrated against schema-bypassed
  forgeries. Here the duplicate-set reports pass the **real, unpatched** schema
  (identical files ⇒ claimed sha == live sha) before the selector sees them.
* Its N1 stubs ``_load_model`` with a lambda that returns a payload without a
  step. Here ``_load_model`` is the **real** function on real ``torch.save``d
  checkpoints for every run.
* It never checks that the *legitimate* five-candidate set is still accepted
  (a fail-closed rule that refuses everything is not a fix). Block L does.
* It never attacks the residual surface: a step can still be asserted by
  **renaming** a checkpoint, because ``resolve_checkpoint_step`` accepts a
  filename-only derivation with no payload step. Block M attacks that.

Disclosed stubs (same seam as reaudit3/reaudit4): ``load_codec_items``,
``enforce_eval_contract``, ``_resolve_speaker_registry``,
``_ref_texts_from_manifest``, ``free_running_asr_eval``, ``load_yaml``.
For eligibility runs only, the two EOS scalars from the real
``teacher_forced_ce`` are overridden to 1.0 / 0 (an untrained tiny model cannot
reach the strict ``== 1.0`` the selector requires); block A keeps
``teacher_forced_ce`` fully real.
"""
from __future__ import annotations

import contextlib
import copy
import hashlib
import importlib.util
import io
import json
import os
import sys
import tempfile
import traceback
from dataclasses import asdict
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from minispeech.data.d1b_oracles import load_eval_contract  # noqa: E402
from minispeech.eval.discrete_report_schema import (  # noqa: E402
    sha256_json,
    validate_formal_report,
)
from minispeech.frontend.normalize import normalize_for_asr  # noqa: E402
from minispeech.models.discrete_speech_lm import (  # noqa: E402
    DiscreteSpeechLM,
    DiscreteSpeechLMConfig,
)

REAL_CONTRACT = ROOT / "configs/eval_contracts/d1c_semantic_eos_v2_0_0.json"
D1B_CONTRACT = ROOT / "configs/eval_contracts/d1b_n2_reliability.json"
STEPS = [250, 375, 500, 625, 750]
RESULTS: list[tuple[str, str, str]] = []


def rec(tag: str, ok, msg: str) -> None:
    status = "INFO" if ok is None else ("PASS" if ok else "FAIL")
    RESULTS.append((tag, status, msg))
    print(f"{tag} {status} {msg}", flush=True)


def load_script(name: str):
    path = ROOT / "scripts" / name
    spec = importlib.util.spec_from_file_location(name.replace(".", "_"), path)
    mod = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(mod)
    return mod


def sha_file(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main() -> int:  # noqa: C901, PLR0912, PLR0915
    mod = load_script("74_eval_discrete_lm.py")
    mod76 = load_script("76_select_discrete_checkpoint.py")
    contract = load_eval_contract(REAL_CONTRACT)

    base = DiscreteSpeechLMConfig.tiny()
    base.termination_mode = "semantic_eos_v1"
    base.eos_id = base.codebook_size
    base.codec_vocab_size = base.codebook_size
    base.semantic_vocab_size = base.codebook_size + 1
    model_ref = DiscreteSpeechLM(base)
    mdict = asdict(base)
    mdict["termination"] = {
        "mode": "semantic_eos_v1",
        "eos_id": int(base.eos_id),
        "semantic_vocab_size": int(base.semantic_vocab_size),
        "codec_vocab_size": int(base.codec_vocab_size),
    }
    cfg = {"model": mdict, "data": {"pad_token_id": 0}, "infer": dict(contract["infer"])}

    utts = ["u0", "u1", "u2"]
    refs = {u: "привет мир" for u in utts}
    rows = [
        {
            "utt_id": u,
            "input_ids": torch.tensor([3 + i, 4 + i, 5 + i]),
            "codes": torch.randint(0, base.codebook_size, (4 + i, base.n_codebooks)),
            "speaker_id": i % 2,
        }
        for i, u in enumerate(utts)
    ]
    hyp_for = {
        "model": "привет мир",
        "copy_synth": "привет мир",
        "wrong_utt": "щщщщщщ щщщ",
        "random_codes": "щщщщщщщщщщ",
        "shuffle_text": "щщщщщщ щщщ",
        "mask_text": "щщщщщщщщщщ",
    }
    seed_box = {"n": 0}

    def fake_fr(model_, rows_, **kw):
        condition = kw["condition"]
        out_wav_dir = Path(kw["out_wav_dir"])
        c = kw["cfg"]
        icfg = dict(
            mod.resolve_infer_settings(
                c,
                infer_overrides=kw.get("infer_overrides"),
                max_frames_override=kw["max_frames"],
                codebook_size=int((c.get("model") or {}).get("codebook_size", 2048)),
            )
        )
        icfg["termination_mode"] = "semantic_eos_v1"
        rows_out, gen_codes, cers, edits = [], [], [], []
        for i, r in enumerate(rows_):
            uid = str(r["utt_id"])
            seed_box["n"] += 1
            pcm = np.linspace(-0.3, 0.3, 2400 + (seed_box["n"] % 7), dtype=np.float32)
            wav = out_wav_dir / f"{i:04d}_{uid}.wav"
            wav.parent.mkdir(parents=True, exist_ok=True)
            sf.write(str(wav), pcm, 24000)
            hyp = hyp_for[condition]
            ref_n, hyp_n = normalize_for_asr(refs[uid]), normalize_for_asr(hyp)
            cer = float(mod.compute_cer([ref_n], [hyp_n]))
            edits.append(mod.edits_from_jiwer_chars(ref_n, hyp_n))
            n_frames = int(r["codes"].shape[0])
            codes_t = torch.randint(0, 2048, (n_frames, 16))
            gen_codes.append(codes_t)
            cers.append(cer)
            rows_out.append(
                {
                    "utt_id": uid, "ordinal": i, "condition": condition,
                    "n_frames": n_frames, "gt_frames": n_frames, "cer": cer,
                    "ref": refs[uid], "hyp": hyp, "wav": str(wav),
                    "stop_reason": "learned_eos",
                    "generation": {
                        "termination_reason": "learned_eos",
                        "n_decoded_frames": n_frames,
                        "early_eos_suppressed_count": 0,
                        "max_cb0_run": 2,
                    },
                    "generated_codes_sha256": hashlib.sha256(
                        codes_t.numpy().astype(np.int32).tobytes()
                    ).hexdigest(),
                }
            )
        boot = mod.bootstrap_mean_cer(cers)
        return mod.enrich_model_report(
            {
                "condition": condition, "n": len(rows_out), "mean_cer": boot["mean"],
                "cer_ci95_low": boot["ci95_low"], "cer_ci95_high": boot["ci95_high"],
                "cb0": mod.cb0_stats(gen_codes), "rows": rows_out, "edit_counts": edits,
                "infer": {k: v for k, v in icfg.items() if not str(k).startswith("_")},
            }
        )

    real_tf_ce = mod.teacher_forced_ce

    def perfect_tf_ce(m_, batch, device):
        out = dict(real_tf_ce(m_, batch, device))
        out["tf_eos_final_top1_accuracy"] = 1.0
        out["tf_eos_early_top1_false_positive_count"] = 0
        return out

    def install_stubs(*, contract_obj=None, perfect_eos=True):
        mod.load_codec_items = lambda *a, **k: rows
        mod.enforce_eval_contract = lambda *a, **k: (
            contract if contract_obj is None else contract_obj
        )
        mod._resolve_speaker_registry = lambda **k: (
            object(), "configs/speaker_maps/trackn_v2.json",
        )
        mod._ref_texts_from_manifest = lambda ids, m: dict(refs)
        mod.free_running_asr_eval = fake_fr
        mod.load_yaml = lambda p: {"infer": dict(contract["infer"])}
        mod.teacher_forced_ce = perfect_tf_ce if perfect_eos else real_tf_ce

    def save_ckpt(path: Path, step, *, junk=None):
        payload = {
            "model": model_ref.state_dict(),
            "cfg": cfg,
            "schema": "minispeech.discrete_lm_checkpoint",
            "schema_version": "2.0.0",
        }
        if step is not None:
            payload["step"] = int(step)
            payload["optimizer_step"] = int(step)
            payload["global_step"] = int(step)
        if junk is not None:
            payload["_probe_junk"] = junk
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(payload, path)
        return path

    def run_main(argv: list[str]):
        buf = io.StringIO()
        old = sys.argv
        sys.argv = ["74_eval_discrete_lm.py", *argv]
        tb = se = ""
        rc = None
        try:
            with contextlib.redirect_stdout(buf):
                rc = mod.main()
        except SystemExit as exc:
            se = str(exc)
            rc = exc.code if isinstance(exc.code, int) else None
        except BaseException:  # noqa: BLE001
            tb = traceback.format_exc()
        finally:
            sys.argv = old
        return rc, buf.getvalue(), tb, se

    def base_argv(ckpt: Path, out: Path, wav: Path, td: Path, extra=None):
        return [
            "--ckpt", str(ckpt),
            "--codec-tokens-dir", str(td / "tokens"),
            "--max-utts", "3",
            "--n2-gate",
            "--out", str(out),
            "--wav-dir", str(wav),
            "--eval-contract", str(REAL_CONTRACT),
            "--config", "configs/discrete_speech_lm_d1b_443m.yaml",
        ] + list(extra or [])

    with tempfile.TemporaryDirectory(prefix="reaudit5_") as tmp:
        td = Path(tmp)

        # =============================================================== A
        # A. Round-4 attack, reproduced verbatim on the production path:
        #    ONE best.pt, five --checkpoint-step values.
        install_stubs(perfect_eos=True)
        one = save_ckpt(td / "attack" / "best.pt", None)
        emitted, exits = [], []
        for s in STEPS:
            outd = td / f"a_{s}"
            rc, so, tb, se = run_main(base_argv(one, outd, td / f"wav_a_{s}", td,
                                                ["--checkpoint-step", str(s)]))
            exits.append(se or (tb.strip().splitlines()[-1] if tb else f"rc={rc}"))
            if (outd / "discrete_eval_report_v2.json").is_file():
                emitted.append(s)
        rec(
            "A1",
            not emitted,
            f"round-4 attack: five main() runs on ONE best.pt emitted "
            f"{len(emitted)} formal reports (expect 0); steps_emitted={emitted}",
        )
        rec(
            "A2",
            all("cannot invent" in e for e in exits),
            f"all five refusals are 'cannot invent': {exits[0][:130]}",
        )
        # A3: refusal happens BEFORE any eval work (no gate json either)
        leaked = [s for s in STEPS if (td / f"a_{s}" / "n2_gate.json").is_file()]
        rec("A3", not leaked, f"no n2_gate.json leaked from refused runs: {leaked}")

        # A4: real teacher-forced path unpatched, legitimate ckpt still works
        install_stubs(perfect_eos=False)
        ck750 = save_ckpt(td / "legit" / "candidate_step_0750.pt", 750)
        rc, so, tb, se = run_main(base_argv(ck750, td / "a4", td / "wav_a4", td))
        rep_a4 = {}
        if (td / "a4" / "discrete_eval_report_v2.json").is_file():
            rep_a4 = json.loads((td / "a4" / "discrete_eval_report_v2.json").read_text("utf-8"))
        gate_a4 = {}
        if (td / "a4" / "n2_gate.json").is_file():
            gate_a4 = json.loads((td / "a4" / "n2_gate.json").read_text("utf-8"))
        tfm = rep_a4.get("teacher_forced_metrics") or {}
        rec(
            "A4",
            (rep_a4.get("checkpoint") or {}).get("step") == 750 and rc == 0,
            f"REAL _load_model + REAL tf metrics -> step="
            f"{(rep_a4.get('checkpoint') or {}).get('step')} rc={rc} exit={se!r}",
        )
        rec(
            "A5",
            tfm.get("teacher_forced_ce_source") == "eval_computed"
            and tfm.get("tf_eos_final_top1_accuracy") is not None,
            f"FAIL#4 EOS plumbing: source={tfm.get('teacher_forced_ce_source')} "
            f"acc={tfm.get('tf_eos_final_top1_accuracy')} "
            f"fp={tfm.get('tf_eos_early_top1_false_positive_count')} (untrained -> not 1.0)",
        )
        rec(
            "A6",
            gate_a4.get("gate") == "N2_D1c_semantic_eos_v1"
            and "termination_d1c" in (gate_a4.get("criteria") or {}),
            f"evaluate_n2_gate_d1c wired: gate={gate_a4.get('gate')} "
            f"step={gate_a4.get('step')}",
        )

        # =============================================================== L
        # L. Legitimate five-candidate set must STILL be accepted end to end.
        install_stubs(perfect_eos=True)
        legit_paths, legit_steps, legit_shas = [], [], []
        for s in STEPS:
            ck = save_ckpt(td / "legit" / f"candidate_step_{s:04d}.pt", s)
            outd = td / f"l_{s}"
            rc, so, tb, se = run_main(base_argv(ck, outd, td / f"wav_l_{s}", td))
            if tb:
                print(f"[L {s}] {tb}")
            rp = outd / "discrete_eval_report_v2.json"
            if rp.is_file():
                d = json.loads(rp.read_text("utf-8"))
                legit_paths.append(rp)
                legit_steps.append((d.get("checkpoint") or {}).get("step"))
                legit_shas.append((d.get("checkpoint") or {}).get("sha256"))
            else:
                legit_steps.append(None)
        rec("L1", legit_steps == STEPS, f"five real main() runs stamped steps={legit_steps}")
        rec(
            "L2",
            len(set(legit_shas)) == 5,
            f"five real candidate files have {len(set(legit_shas))} distinct sha256",
        )
        selection = None
        if len(legit_paths) == 5:
            try:
                selection = mod76.select_d1c_formal_reports(list(legit_paths))
                got = sorted(int(c["step"]) for c in selection["candidates"])
                rec("L3", got == STEPS,
                    f"selector ACCEPTS untouched legitimate production reports steps={got} "
                    "(round-5 rules are not a blanket refusal)")
                rec("L4", selection.get("selected") is not None,
                    f"selector selected step={(selection.get('selected') or {}).get('step')}")
            except Exception as exc:  # noqa: BLE001
                rec("L3", False, f"REGRESSION: selector refused the legit set: {type(exc).__name__}: {exc}")
                rec("L4", False, "no selection")
        else:
            rec("L3", False, f"only {len(legit_paths)} reports emitted")
            rec("L4", False, "no selection")

        # =============================================================== D
        # D. Duplicate identity, through the REAL unpatched schema.
        if len(legit_paths) == 5:
            dup_dir = td / "dup"
            dup_dir.mkdir(parents=True, exist_ok=True)
            # ONE physical payload, copied byte-identically to five candidate names
            src_blob = (td / "legit" / "candidate_step_0250.pt").read_bytes()
            dup_sha = hashlib.sha256(src_blob).hexdigest()
            dup_reports, schema_ok = [], 0
            for i, s in enumerate(STEPS):
                alias = dup_dir / f"candidate_step_{s:04d}.pt"
                alias.write_bytes(src_blob)
                d = copy.deepcopy(json.loads(legit_paths[i].read_text("utf-8")))
                d["checkpoint"]["step"] = s
                d["checkpoint"]["path"] = str(alias)
                d["checkpoint"]["sha256"] = dup_sha
                d.pop("report_sha256", None)
                d["report_sha256"] = sha256_json(d)
                try:
                    validate_formal_report(d)  # REAL schema, not stubbed
                    schema_ok += 1
                except Exception as exc:  # noqa: BLE001
                    print(f"[D schema {s}] {exc}")
                p = dup_dir / f"report_{s}.json"
                p.write_text(json.dumps(d, ensure_ascii=False), encoding="utf-8")
                dup_reports.append(p)
            rec(
                "D1",
                schema_ok == 5,
                f"{schema_ok}/5 duplicate-identity reports pass the REAL schema "
                "(claimed sha == live sha of identical files) — so the selector is "
                "the only layer that can refuse this shape",
            )
            try:
                mod76.select_d1c_formal_reports(dup_reports)
                rec("D2", False, "SELECTOR ACCEPTED five reports with identical checkpoint.sha256")
            except ValueError as exc:
                rec("D2", "duplicate checkpoint.sha256" in str(exc),
                    f"selector hard-refuses duplicate sha256: {exc}")
            except Exception as exc:  # noqa: BLE001
                rec("D2", False, f"unexpected error type {type(exc).__name__}: {exc}")

            # D3: partial duplicate (only two of five collide) must also refuse
            part = []
            for i, s in enumerate(STEPS):
                d = copy.deepcopy(json.loads(legit_paths[i].read_text("utf-8")))
                if s == 375:
                    alias = dup_dir / "p_candidate_step_0375.pt"
                    alias.write_bytes((td / "legit" / "candidate_step_0250.pt").read_bytes())
                    tgt = dup_dir / "candidate_step_0375.pt"
                    tgt.write_bytes(alias.read_bytes())
                    d["checkpoint"]["path"] = str(tgt)
                    d["checkpoint"]["sha256"] = sha_file(tgt)
                    d.pop("report_sha256", None)
                    d["report_sha256"] = sha256_json(d)
                p = dup_dir / f"part_{s}.json"
                p.write_text(json.dumps(d, ensure_ascii=False), encoding="utf-8")
                part.append(p)
            try:
                mod76.select_d1c_formal_reports(part)
                rec("D3", False, "selector accepted a set where 2 of 5 share one checkpoint")
            except ValueError as exc:
                rec("D3", "duplicate checkpoint.sha256" in str(exc),
                    f"partial duplicate (2 of 5) refused: {str(exc)[:120]}")

        # =============================================================== H
        # H. Claimed sha vs on-disk bytes, at both layers.
        if len(legit_paths) == 5:
            d = copy.deepcopy(json.loads(legit_paths[0].read_text("utf-8")))
            d["checkpoint"]["sha256"] = "0" * 64
            d.pop("report_sha256", None)
            d["report_sha256"] = sha256_json(d)
            try:
                validate_formal_report(d)
                rec("H1", False, "schema accepted checkpoint.sha256 != file bytes")
            except ValueError as exc:
                rec("H1", "does not match file" in str(exc), f"schema refuses forged sha: {exc}")
            bad = td / "h" / "bad.json"
            bad.parent.mkdir(parents=True, exist_ok=True)
            bad.write_text(json.dumps(d, ensure_ascii=False), encoding="utf-8")
            try:
                mod76.select_d1c_formal_reports([bad] + list(legit_paths[1:]))
                rec("H2", False, "selector accepted a forged-sha report")
            except ValueError as exc:
                rec("H2", True, f"selector refuses forged sha end to end: {str(exc)[:120]}")

            # H3: selector's OWN hash verification, isolated from the schema.
            #     (Schema skips the check when the path is unreachable, so this
            #      proves the selector adds an independent layer.)
            reps = [json.loads(p.read_text("utf-8")) for p in legit_paths]
            r_bad = copy.deepcopy(reps)
            r_bad[2]["checkpoint"]["sha256"] = "b" * 64
            try:
                mod76._require_distinct_checkpoint_identities(r_bad)
                rec("H3", False, "selector helper accepted claimed sha != live sha")
            except ValueError as exc:
                rec("H3", "sha256 mismatch" in str(exc),
                    f"selector helper verifies live bytes: {str(exc)[:130]}")

            # H4: unreadable path -> schema silently skips, selector must refuse
            r_gone = copy.deepcopy(reps)
            r_gone[1]["checkpoint"]["path"] = str(td / "does_not_exist" / "candidate_step_0375.pt")
            r_gone[1].pop("report_sha256", None)
            r_gone[1]["report_sha256"] = sha256_json(r_gone[1])
            schema_gone_ok = True
            try:
                validate_formal_report(r_gone[1])
            except Exception:  # noqa: BLE001
                schema_gone_ok = False
            try:
                mod76._require_distinct_checkpoint_identities(r_gone)
                rec("H4", False, "selector accepted an unreadable checkpoint.path")
            except ValueError as exc:
                rec("H4", "not readable" in str(exc),
                    f"unreadable path: schema_accepts={schema_gone_ok} selector_refuses "
                    f"({str(exc)[:90]})")

            # H5: basename must agree with the report's own step
            r_name = copy.deepcopy(reps)
            renamed = td / "h" / "best.pt"
            renamed.write_bytes((td / "legit" / "candidate_step_0250.pt").read_bytes())
            r_name[0]["checkpoint"]["path"] = str(renamed)
            r_name[0]["checkpoint"]["sha256"] = sha_file(renamed)
            try:
                mod76._require_distinct_checkpoint_identities(r_name)
                rec("H5", False, "selector accepted basename 'best.pt' for step 250")
            except ValueError as exc:
                rec("H5", "does not match expected" in str(exc),
                    f"basename rule enforced: {str(exc)[:120]}")

            # H6: cross-wired basename (step 250 report pointing at step 0750 file)
            r_cross = copy.deepcopy(reps)
            r_cross[0]["checkpoint"]["path"] = str(td / "legit" / "candidate_step_0750.pt")
            r_cross[0]["checkpoint"]["sha256"] = sha_file(td / "legit" / "candidate_step_0750.pt")
            try:
                mod76._require_distinct_checkpoint_identities(r_cross)
                rec("H6", False, "selector accepted step=250 report bound to candidate_step_0750.pt")
            except ValueError as exc:
                rec("H6", True, f"cross-wired basename refused: {str(exc)[:120]}")

            # H7: symlink aliasing (same inode, five candidate names)
            if os.name != "nt":
                sl_dir = td / "sym"
                sl_dir.mkdir(parents=True, exist_ok=True)
                target = td / "legit" / "candidate_step_0250.pt"
                r_sym = copy.deepcopy(reps)
                try:
                    for i, s in enumerate(STEPS):
                        link = sl_dir / f"candidate_step_{s:04d}.pt"
                        if link.exists() or link.is_symlink():
                            link.unlink()
                        os.symlink(target, link)
                        r_sym[i]["checkpoint"]["path"] = str(link)
                        r_sym[i]["checkpoint"]["sha256"] = sha_file(link)
                    mod76._require_distinct_checkpoint_identities(r_sym)
                    rec("H7", False, "selector accepted five symlinks to ONE checkpoint")
                except ValueError as exc:
                    rec("H7", True, f"symlink aliasing refused: {str(exc)[:110]}")
                except OSError as exc:
                    rec("H7", None, f"symlink unavailable: {exc}")
            else:
                rec("H7", None, "symlink test skipped on this platform")

        # =============================================================== C
        # C. --checkpoint-step is confirm-only on the formal path.
        install_stubs(perfect_eos=True)
        ck500 = save_ckpt(td / "legit" / "candidate_step_0500.pt", 500)
        rc, so, tb, se = run_main(base_argv(ck500, td / "c1", td / "wav_c1", td,
                                            ["--checkpoint-step", "500"]))
        r = td / "c1" / "discrete_eval_report_v2.json"
        got = json.loads(r.read_text("utf-8"))["checkpoint"]["step"] if r.is_file() else None
        rec("C1", got == 500, f"CLI confirm of derived step accepted -> step={got} exit={se!r}")

        rc, so, tb, se = run_main(base_argv(ck500, td / "c2", td / "wav_c2", td,
                                            ["--checkpoint-step", "625"]))
        rec(
            "C2",
            "conflicts" in se and not (td / "c2" / "discrete_eval_report_v2.json").is_file(),
            f"foreign CLI step on a derived checkpoint refused: {se[:110]!r}",
        )
        rc, so, tb, se = run_main(base_argv(one, td / "c3", td / "wav_c3", td,
                                            ["--checkpoint-step", "625"]))
        rec(
            "C3",
            "cannot invent" in se and not (td / "c3" / "discrete_eval_report_v2.json").is_file(),
            f"CLI invent on best.pt refused: {se[:110]!r}",
        )
        # C4: payload-derived step on a non-candidate filename may be confirmed
        ck_pay = save_ckpt(td / "pay" / "best.pt", 625)
        rc, so, tb, se = run_main(base_argv(ck_pay, td / "c4", td / "wav_c4", td,
                                            ["--checkpoint-step", "625"]))
        r = td / "c4" / "discrete_eval_report_v2.json"
        got = json.loads(r.read_text("utf-8"))["checkpoint"]["step"] if r.is_file() else None
        rec("C4", got == 625, f"payload-derived step confirmed via CLI -> step={got}")
        # C5: non-formal (require=False) diagnostic invent still allowed by contract
        got5 = mod.resolve_checkpoint_step(
            ckpt_path=one, payload={}, cli_step=750, allowed_steps=STEPS, require=False
        )
        rec("C5", got5 == 750, f"documented: require=False still allows invent -> {got5}")
        # C6: out-of-set step still refused
        ck999 = save_ckpt(td / "legit" / "candidate_step_0999.pt", 999)
        rc, so, tb, se = run_main(base_argv(ck999, td / "c6", td / "wav_c6", td))
        rec("C6", "not in contract" in se, f"out-of-set step refused: {se[:100]!r}")
        # C7: payload/filename disagreement still refused
        ck_mm = save_ckpt(td / "mm" / "candidate_step_0750.pt", 500)
        rc, so, tb, se = run_main(base_argv(ck_mm, td / "c7", td / "wav_c7", td))
        rec(
            "C7",
            "mismatch" in se and not (td / "c7" / "discrete_eval_report_v2.json").is_file(),
            f"payload/filename mismatch refused: {se[:110]!r}",
        )

        # =============================================================== M
        # M. NEW SURFACE: the CLI can no longer invent a step — but `mv` can.
        #    One trained state, saved five times under five candidate names with
        #    NO payload step. Bytes differ (junk key) so the distinct-sha rule
        #    is satisfied; the step comes from the filename alone.
        install_stubs(perfect_eos=True)
        m_paths, m_steps, m_shas = [], [], []
        for s in STEPS:
            ck = save_ckpt(td / "rename" / f"candidate_step_{s:04d}.pt", None, junk="x" * (s % 17 + 1))
            outd = td / f"m_{s}"
            rc, so, tb, se = run_main(base_argv(ck, outd, td / f"wav_m_{s}", td))
            rp = outd / "discrete_eval_report_v2.json"
            if rp.is_file():
                d = json.loads(rp.read_text("utf-8"))
                m_paths.append(rp)
                m_steps.append((d.get("checkpoint") or {}).get("step"))
                m_shas.append((d.get("checkpoint") or {}).get("sha256"))
            else:
                m_steps.append(f"refused:{se[:40]}")
        rec(
            "M1",
            None,
            f"rename-only attack: five reports from ONE model state with NO payload "
            f"step -> steps={m_steps} distinct_sha={len(set(m_shas))}",
        )
        if len(m_paths) == 5:
            try:
                sel_m = mod76.select_d1c_formal_reports(list(m_paths))
                rec(
                    "M2",
                    False,
                    "SELECTOR ACCEPTED five renamed copies of ONE model state "
                    f"(selected step={sel_m['selected']['step']}): the round-5 rules "
                    "prove file distinctness, not training-step provenance; "
                    "`--checkpoint-step` invent is closed but `mv`/re-save is not",
                )
            except Exception as exc:  # noqa: BLE001
                rec("M2", True, f"selector refused the rename attack: {type(exc).__name__}: {exc}")
        else:
            rec("M2", True, f"evaluator refused rename-only checkpoints ({len(m_paths)}/5 emitted)")

        # M3: is a payload step REQUIRED on the formal path?
        got_m3 = None
        err_m3 = ""
        try:
            got_m3 = mod.resolve_checkpoint_step(
                ckpt_path="runs/x/candidate_step_0625.pt",
                payload={"model": {}},  # no step of any kind
                allowed_steps=STEPS,
                require=True,
            )
        except SystemExit as exc:
            err_m3 = str(exc)
        rec(
            "M3",
            None,
            f"formal path with NO payload step: resolve -> {got_m3} err={err_m3!r} "
            "(filename alone is authoritative)",
        )

        # =============================================================== G
        # G. D1b contract must not receive the D1c gate / formal report.
        if D1B_CONTRACT.is_file():
            install_stubs(contract_obj=load_eval_contract(D1B_CONTRACT), perfect_eos=True)
            rc, so, tb, se = run_main(base_argv(ck750, td / "g3", td / "wav_g3", td))
            gp3 = td / "g3" / "n2_gate.json"
            g3 = json.loads(gp3.read_text("utf-8")) if gp3.is_file() else {}
            rec(
                "G1",
                g3.get("gate") != "N2_D1c_semantic_eos_v1"
                and not (td / "g3" / "discrete_eval_report_v2.json").is_file(),
                f"D1b contract keeps the D1b gate: gate={g3.get('gate')!r} "
                f"formal_v2_emitted={(td / 'g3' / 'discrete_eval_report_v2.json').is_file()}",
            )
            # G2: non-formal path (D1b) still tolerates --checkpoint-step invent
            rc, so, tb, se = run_main(base_argv(one, td / "g2", td / "wav_g2", td,
                                                ["--checkpoint-step", "625"]))
            gp2 = td / "g2" / "n2_gate.json"
            g2 = json.loads(gp2.read_text("utf-8")) if gp2.is_file() else {}
            rec(
                "G2",
                None,
                f"non-formal D1b + --checkpoint-step 625 -> gate.step={g2.get('step')} "
                f"exit={se[:60]!r} (documented require=False behaviour)",
            )
            install_stubs(perfect_eos=True)
        else:
            rec("G1", None, "D1b contract not found")

    fails = sum(1 for _, st, _ in RESULTS if st == "FAIL")
    print(f"\nSUMMARY fails={fails} total={len(RESULTS)}")
    for tag, st, msg in RESULTS:
        if st == "FAIL":
            print(f"  FAILED {tag}: {msg}")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())

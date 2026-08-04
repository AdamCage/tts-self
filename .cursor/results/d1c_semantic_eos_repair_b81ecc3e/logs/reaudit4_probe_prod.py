#!/usr/bin/env python3
"""REAUDIT_ROUND4 decisive production-path probe (independent of the repair agent).

Differences from `logs/repair_round4_probe_step.py` (which I do not trust):

* `_load_model` is **NOT stubbed**. A real `candidate_step_XXXX.pt` is written with
  `torch.save` and production `_load_model` does the real `torch.load` + rebuild.
  The repair probe replaced `_load_model` with a lambda that *hands back* the
  payload, so it could not detect a `_load_model` that discards the payload.
* The five-report selector §8.4 set is produced by **five real `main()` runs** with
  five real checkpoint files, and the emitted report JSONs are fed to
  `select_d1c_formal_reports` **untouched**. The repair probe ran `main()` once and
  hand-patched `checkpoint.step` into four deep-copies.
* Adds main()-level negatives (mismatch / missing / CLI conflict / out-of-set),
  schema negatives, selector controls, and D1c-gate load-bearing checks.

Stubs (disclosed): `load_codec_items`, `enforce_eval_contract`,
`_resolve_speaker_registry`, `_ref_texts_from_manifest`, `free_running_asr_eval`,
`load_yaml` — all GPU/codec/ASR/dataset-bound outer deps, same seam as reaudit3.
For the *eligibility* runs only, the two EOS scalars returned by the real
`teacher_forced_ce` are overridden to the perfect values (1.0 / 0); everything else
(model forward, CE, aggregation, projection, gate, emitter, schema, selector) is real.
Run C1 keeps `teacher_forced_ce` fully unpatched.
"""
from __future__ import annotations

import ast
import contextlib
import copy
import hashlib
import importlib.util
import io
import json
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


def semantic_cfg_and_model():
    base = DiscreteSpeechLMConfig.tiny()
    base.termination_mode = "semantic_eos_v1"
    base.eos_id = base.codebook_size
    base.codec_vocab_size = base.codebook_size
    base.semantic_vocab_size = base.codebook_size + 1
    model = DiscreteSpeechLM(base)
    mdict = asdict(base)
    mdict["termination"] = {
        "mode": "semantic_eos_v1",
        "eos_id": int(base.eos_id),
        "semantic_vocab_size": int(base.semantic_vocab_size),
        "codec_vocab_size": int(base.codec_vocab_size),
    }
    cfg = {"model": mdict, "data": {"pad_token_id": 0}}
    return cfg, model


def run_main(mod, argv: list[str]):
    """Return (rc, stdout, traceback_text, systemexit_text)."""
    buf = io.StringIO()
    old = sys.argv
    sys.argv = ["74_eval_discrete_lm.py", *argv]
    tb = ""
    se = ""
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


def main() -> int:  # noqa: C901, PLR0912, PLR0915
    mod = load_script("74_eval_discrete_lm.py")
    mod76 = load_script("76_select_discrete_checkpoint.py")
    src = (ROOT / "scripts/74_eval_discrete_lm.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    fn_main = next(
        n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main"
    )
    call_names = {
        n.func.id
        for n in ast.walk(fn_main)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
    }

    # ---------------------------------------------------------------- A: AST
    rec("A1", "resolve_checkpoint_step" in call_names, "main() calls resolve_checkpoint_step")
    rec("A2", "evaluate_n2_gate_d1c" in call_names, "main() calls evaluate_n2_gate_d1c")
    rec("A3", "--checkpoint-step" in src, "CLI exposes --checkpoint-step")
    load_fn = next(
        n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_load_model"
    )
    ret3 = [
        n
        for n in ast.walk(load_fn)
        if isinstance(n, ast.Return) and isinstance(n.value, ast.Tuple)
    ]
    rec(
        "A4",
        bool(ret3) and all(len(r.value.elts) == 3 for r in ret3),
        f"_load_model returns 3-tuple on every return path (n_returns={len(ret3)})",
    )

    contract = load_eval_contract(REAL_CONTRACT)
    cfg_model, model_ref = semantic_cfg_and_model()
    codebook = model_ref.codebook_size
    ncb = model_ref.n_codebooks
    utts = ["u0", "u1", "u2"]
    refs = {u: "привет мир" for u in utts}
    rows = [
        {
            "utt_id": u,
            "input_ids": torch.tensor([3 + i, 4 + i, 5 + i]),
            "codes": torch.randint(0, codebook, (4 + i, ncb)),
            "speaker_id": i % 2,
        }
        for i, u in enumerate(utts)
    ]

    cfg = dict(cfg_model)
    cfg["infer"] = dict(contract["infer"])

    hyp_for = {
        "model": "привет мир",
        "copy_synth": "привет мир",
        "wrong_utt": "щщщщщщ щщщ",
        "random_codes": "щщщщщщщщщщ",
        "shuffle_text": "щщщщщщ щщщ",
        "mask_text": "щщщщщщщщщщ",
    }

    def make_fake_fr(seed_box):
        def fake_free_running(model_, rows_, **kw):
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
            rows_out, gen_codes, cers, edit_counts = [], [], [], []
            for i, r in enumerate(rows_):
                uid = str(r["utt_id"])
                seed_box["n"] += 1
                pcm = np.linspace(-0.3, 0.3, 2400 + (seed_box["n"] % 7), dtype=np.float32)
                wav = out_wav_dir / f"{i:04d}_{uid}.wav"
                wav.parent.mkdir(parents=True, exist_ok=True)
                sf.write(str(wav), pcm, 24000)
                hyp = hyp_for[condition]
                ref_n = normalize_for_asr(refs[uid])
                hyp_n = normalize_for_asr(hyp)
                cer = float(mod.compute_cer([ref_n], [hyp_n]))
                edit_counts.append(mod.edits_from_jiwer_chars(ref_n, hyp_n))
                n_frames = int(r["codes"].shape[0])
                codes_t = torch.randint(0, 2048, (n_frames, 16))
                gen_codes.append(codes_t)
                cers.append(cer)
                rows_out.append(
                    {
                        "utt_id": uid,
                        "ordinal": i,
                        "condition": condition,
                        "n_frames": n_frames,
                        "gt_frames": int(r["codes"].shape[0]),
                        "cer": cer,
                        "ref": refs[uid],
                        "hyp": hyp,
                        "wav": str(wav),
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
            report = {
                "condition": condition,
                "n": len(rows_out),
                "mean_cer": boot["mean"],
                "cer_ci95_low": boot["ci95_low"],
                "cer_ci95_high": boot["ci95_high"],
                "cb0": mod.cb0_stats(gen_codes),
                "rows": rows_out,
                "edit_counts": edit_counts,
                "infer": {k: v for k, v in icfg.items() if not str(k).startswith("_")},
            }
            return mod.enrich_model_report(report)

        return fake_free_running

    real_tf_ce = mod.teacher_forced_ce

    def perfect_tf_ce(m_, batch, device):
        out = dict(real_tf_ce(m_, batch, device))
        out["tf_eos_final_top1_accuracy"] = 1.0
        out["tf_eos_early_top1_false_positive_count"] = 0
        return out

    def install_stubs(seed_box, *, contract_obj=None, perfect_eos=False):
        mod.load_codec_items = lambda *a, **k: rows
        mod.enforce_eval_contract = lambda *a, **k: (
            contract if contract_obj is None else contract_obj
        )
        mod._resolve_speaker_registry = lambda **k: (
            object(),
            "configs/speaker_maps/trackn_v2.json",
        )
        mod._ref_texts_from_manifest = lambda ids, m: dict(refs)
        mod.free_running_asr_eval = make_fake_fr(seed_box)
        mod.load_yaml = lambda p: {"infer": dict(contract["infer"])}
        mod.teacher_forced_ce = perfect_tf_ce if perfect_eos else real_tf_ce

    def save_ckpt(path: Path, step, *, with_cfg=True, payload_step_key="step"):
        payload = {
            "model": model_ref.state_dict(),
            "cfg": cfg if with_cfg else None,
            "schema": "minispeech.discrete_lm_checkpoint",
            "schema_version": "2.0.0",
        }
        if step is not None:
            payload[payload_step_key] = int(step)
            if payload_step_key == "step":
                payload["global_step"] = int(step)
                payload["optimizer_step"] = int(step)
        if not with_cfg:
            payload.pop("cfg")
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(payload, path)
        return payload

    def base_argv(ckpt: Path, out: Path, wav: Path, td: Path, extra=None):
        argv = [
            "--ckpt", str(ckpt),
            "--codec-tokens-dir", str(td / "tokens"),
            "--max-utts", "3",
            "--n2-gate",
            "--out", str(out),
            "--wav-dir", str(wav),
            "--eval-contract", str(REAL_CONTRACT),
            "--config", "configs/discrete_speech_lm_d1b_443m.yaml",
        ]
        return argv + list(extra or [])

    with tempfile.TemporaryDirectory() as tmp:
        td = Path(tmp)
        seed_box = {"n": 0}

        # ------------------------------------------------ B: REAL _load_model
        ck750 = td / "ckpts" / "candidate_step_0750.pt"
        payload750 = save_ckpt(ck750, 750)
        try:
            loaded = mod._load_model(ck750, torch.device("cpu"))
            ok_len = isinstance(loaded, tuple) and len(loaded) == 3
            got_step = mod.checkpoint_step_from_payload(loaded[2]) if ok_len else None
            rec("B1", ok_len, f"REAL _load_model returned tuple len={len(loaded)}")
            rec(
                "B2",
                got_step == 750,
                f"REAL _load_model 3rd element carries payload step={got_step} "
                "(NOT stubbed - this is the check the repair probe could not make)",
            )
            rec(
                "B3",
                isinstance(loaded[0], torch.nn.Module),
                f"REAL _load_model rebuilt a live model ({type(loaded[0]).__name__})",
            )
        except Exception as exc:  # noqa: BLE001
            rec("B1", False, f"REAL _load_model raised: {exc}")
            rec("B2", False, "no payload")
            rec("B3", False, "no model")

        rec("B4", mod.parse_candidate_step_from_path(ck750) == 750, "filename parse -> 750")
        rec("B5", mod.parse_candidate_step_from_path(td / "best.pt") is None, "best.pt -> None")

        # ------------------------------- C1: real main(), REAL teacher_forced_ce
        install_stubs(seed_box, perfect_eos=False)
        out_c1 = td / "c1"
        rc, so, tb, se = run_main(
            mod, base_argv(ck750, out_c1, td / "wav_c1", td)
        )
        rec("C1a", rc == 0 and not tb and not se, f"main() rc={rc} tb={bool(tb)} exit={se!r}")
        if tb:
            print(tb)
        v2 = out_c1 / "discrete_eval_report_v2.json"
        rec("C1b", v2.is_file(), "production main() emitted formal v2 report")
        rep_c1 = json.loads(v2.read_text(encoding="utf-8")) if v2.is_file() else {}
        gate_c1 = {}
        gp = out_c1 / "n2_gate.json"
        if gp.is_file():
            gate_c1 = json.loads(gp.read_text(encoding="utf-8"))
        rec(
            "C1c",
            (rep_c1.get("checkpoint") or {}).get("step") == 750,
            f"REAL _load_model + REAL tf metrics -> report checkpoint.step="
            f"{(rep_c1.get('checkpoint') or {}).get('step')}",
        )
        rec("C1d", gate_c1.get("step") == 750, f"gate.step={gate_c1.get('step')}")
        rec(
            "C1e",
            (gate_c1.get("provenance") or {}).get("checkpoint_step") == 750,
            f"provenance.checkpoint_step={(gate_c1.get('provenance') or {}).get('checkpoint_step')}",
        )
        rec(
            "C1f",
            gate_c1.get("gate") == "N2_D1c_semantic_eos_v1",
            f"gate id={gate_c1.get('gate')} (evaluate_n2_gate_d1c, not D1b)",
        )
        rec(
            "C1g",
            "termination_d1c" in (gate_c1.get("criteria") or {}),
            f"criteria.termination_d1c present={list((gate_c1.get('criteria') or {}).get('termination_d1c') or {})[:3]}",
        )
        tfm = rep_c1.get("teacher_forced_metrics") or {}
        rec(
            "C1h",
            tfm.get("teacher_forced_ce_source") == "eval_computed",
            f"tf source={tfm.get('teacher_forced_ce_source')} "
            f"acc={tfm.get('tf_eos_final_top1_accuracy')} fp={tfm.get('tf_eos_early_top1_false_positive_count')}",
        )
        try:
            validate_formal_report(rep_c1)
            rec("C1i", True, "unpatched validate_formal_report accepts the report")
        except Exception as exc:  # noqa: BLE001
            rec("C1i", False, f"validation: {exc}")

        # C2: step is not a constant — same code, different real checkpoint
        ck250 = td / "ckpts" / "candidate_step_0250.pt"
        save_ckpt(ck250, 250)
        rc, so, tb, se = run_main(mod, base_argv(ck250, td / "c2", td / "wav_c2", td))
        rep_c2 = {}
        p2 = td / "c2" / "discrete_eval_report_v2.json"
        if p2.is_file():
            rep_c2 = json.loads(p2.read_text(encoding="utf-8"))
        rec(
            "C2a",
            (rep_c2.get("checkpoint") or {}).get("step") == 250,
            f"CONTROL different real ckpt -> step={(rep_c2.get('checkpoint') or {}).get('step')} "
            "(not a hardcoded 750)",
        )

        # ---------------------------------- D: five REAL main() runs -> selector
        install_stubs(seed_box, perfect_eos=True)
        report_paths = []
        emitted_steps = []
        for s in STEPS:
            ck = td / "ckpts" / f"candidate_step_{s:04d}.pt"
            save_ckpt(ck, s)
            outd = td / f"d_{s}"
            rc, so, tb, se = run_main(mod, base_argv(ck, outd, td / f"wav_d_{s}", td))
            if tb:
                print(f"[D {s}] {tb}")
            rp = outd / "discrete_eval_report_v2.json"
            if not rp.is_file():
                rec("D0", False, f"step {s}: main() produced no formal report; rc={rc} exit={se!r}")
                emitted_steps.append(None)
                continue
            data = json.loads(rp.read_text(encoding="utf-8"))
            emitted_steps.append((data.get("checkpoint") or {}).get("step"))
            report_paths.append(rp)
        rec(
            "D1",
            emitted_steps == STEPS,
            f"five REAL main() runs stamped steps={emitted_steps} (expected {STEPS})",
        )
        rec("D2", len(report_paths) == 5, f"five untouched production reports on disk: {len(report_paths)}")

        selection = None
        if len(report_paths) == 5:
            try:
                selection = mod76.select_d1c_formal_reports(list(report_paths))
                got = sorted(int(c["step"]) for c in selection["candidates"])
                rec("D3", got == STEPS, f"selector accepted UNTOUCHED production reports steps={got}")
                rec(
                    "D4",
                    selection.get("selected") is not None
                    and int(selection["selected"]["step"]) in STEPS,
                    f"selector selected step={(selection.get('selected') or {}).get('step')}",
                )
            except Exception as exc:  # noqa: BLE001
                is_step_err = "candidate steps must be exactly" in str(exc)
                rec("D3", not is_step_err, f"selector raised: {type(exc).__name__}: {exc}")
                rec("D4", False, "no selection")
        else:
            rec("D3", False, "cannot run selector: fewer than five production reports")
            rec("D4", False, "no selection")

        # D5 control: hand-zero one production report's step -> selector must refuse
        if len(report_paths) == 5:
            zdir = td / "zero"
            zdir.mkdir(parents=True, exist_ok=True)
            zpaths = []
            for i, rp in enumerate(report_paths):
                d = json.loads(rp.read_text(encoding="utf-8"))
                if i == 0:
                    d["checkpoint"]["step"] = 0
                    d.pop("report_sha256", None)
                    d["report_sha256"] = sha256_json(d)
                zp = zdir / f"r{i}.json"
                zp.write_text(json.dumps(d, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
                zpaths.append(zp)
            try:
                mod76.select_d1c_formal_reports(zpaths)
                rec("D5", False, "CONTROL step=0 report was accepted by the selector")
            except Exception as exc:  # noqa: BLE001
                rec("D5", True, f"CONTROL step=0 refused: {type(exc).__name__}: {exc}")

        # ---------------------------------------- E: main()-level step negatives
        install_stubs(seed_box, perfect_eos=True)

        ck_mm = td / "ckpts_mm" / "candidate_step_0750.pt"
        save_ckpt(ck_mm, 500)  # payload says 500, filename says 750
        rc, so, tb, se = run_main(mod, base_argv(ck_mm, td / "e1", td / "wav_e1", td))
        rec(
            "E1",
            bool(se) and "mismatch" in se and not (td / "e1" / "discrete_eval_report_v2.json").is_file(),
            f"payload/filename mismatch fails closed: exit={se!r}",
        )

        ck_ns = td / "ckpts_ns" / "best.pt"
        save_ckpt(ck_ns, None)
        rc, so, tb, se = run_main(mod, base_argv(ck_ns, td / "e2", td / "wav_e2", td))
        rec(
            "E2",
            bool(se) and "unresolved" in se and not (td / "e2" / "discrete_eval_report_v2.json").is_file(),
            f"missing payload step + non-candidate filename fails closed: exit={se!r}",
        )

        rc, so, tb, se = run_main(
            mod,
            base_argv(ck750, td / "e3", td / "wav_e3", td, ["--checkpoint-step", "375"]),
        )
        rec("E3", bool(se) and "conflicts" in se, f"--checkpoint-step conflict fails closed: exit={se!r}")

        rc, so, tb, se = run_main(
            mod,
            base_argv(ck_ns, td / "e4", td / "wav_e4", td, ["--checkpoint-step", "999"]),
        )
        rec(
            "E4",
            bool(se) and "not in contract" in se,
            f"--checkpoint-step outside candidate_steps fails closed: exit={se!r}",
        )

        rc, so, tb, se = run_main(
            mod,
            base_argv(ck_ns, td / "e5", td / "wav_e5", td, ["--checkpoint-step", "625"]),
        )
        r5 = td / "e5" / "discrete_eval_report_v2.json"
        d5 = json.loads(r5.read_text(encoding="utf-8")) if r5.is_file() else {}
        rec(
            "E5",
            (d5.get("checkpoint") or {}).get("step") == 625,
            f"--checkpoint-step override on best.pt -> step={(d5.get('checkpoint') or {}).get('step')}",
        )

        # E6: legacy fallback key only (optimizer_step, no 'step')
        ck_opt = td / "ckpts_opt" / "best.pt"
        save_ckpt(ck_opt, 500, payload_step_key="optimizer_step")
        rc, so, tb, se = run_main(mod, base_argv(ck_opt, td / "e6", td / "wav_e6", td))
        r6 = td / "e6" / "discrete_eval_report_v2.json"
        d6 = json.loads(r6.read_text(encoding="utf-8")) if r6.is_file() else {}
        rec(
            "E6",
            (d6.get("checkpoint") or {}).get("step") == 500,
            f"payload optimizer_step fallback -> step={(d6.get('checkpoint') or {}).get('step')} exit={se!r}",
        )

        # ------------------------------------------------ F: schema negatives
        from minispeech.eval.discrete_report_schema import validate_formal_report as vfr

        base_rep = copy.deepcopy(rep_c1)
        for tag, bad_step, expect in (
            ("F1", 0, "positive integer"),
            ("F2", -5, "positive integer"),
            ("F3", True, "positive integer"),
            ("F4", "750", "positive integer"),
            ("F5", 750.0, "positive integer"),
        ):
            d = copy.deepcopy(base_rep)
            d["checkpoint"]["step"] = bad_step
            d.pop("report_sha256", None)
            d["report_sha256"] = sha256_json(d)
            try:
                vfr(d)
                rec(tag, False, f"schema accepted checkpoint.step={bad_step!r}")
            except ValueError as exc:
                rec(tag, expect in str(exc), f"step={bad_step!r} refused: {exc}")
        d = copy.deepcopy(base_rep)
        d["checkpoint"].pop("step", None)
        d.pop("report_sha256", None)
        d["report_sha256"] = sha256_json(d)
        try:
            vfr(d)
            rec("F6", False, "schema accepted missing checkpoint.step")
        except ValueError as exc:
            rec("F6", True, f"missing step refused: {exc}")
        d = copy.deepcopy(base_rep)
        try:
            vfr(d, allowed_candidate_steps=[250, 375])
            rec("F7", False, "allowed_candidate_steps not enforced")
        except ValueError as exc:
            rec("F7", "not in allowed candidate_steps" in str(exc), f"membership enforced: {exc}")

        # ------------------------------- G: D1c gate load-bearing (not a label)
        from minispeech.eval.discrete_n2 import evaluate_n2_gate_d1b, evaluate_n2_gate_d1c

        def cond_report(name, *, reason="learned_eos", cer=0.0):
            rws = []
            edits = []
            for i, u in enumerate(utts):
                hyp = hyp_for[name]
                ref_n = normalize_for_asr(refs[u])
                hyp_n = normalize_for_asr(hyp)
                edits.append(mod.edits_from_jiwer_chars(ref_n, hyp_n))
                rws.append(
                    {
                        "utt_id": u,
                        "ordinal": i,
                        "condition": name,
                        "n_frames": 4 + i,
                        "gt_frames": 4 + i,
                        "cer": cer,
                        "ref": refs[u],
                        "hyp": hyp,
                        "stop_reason": reason,
                        "generation": {
                            "termination_reason": reason,
                            "n_decoded_frames": 4 + i,
                            "early_eos_suppressed_count": 0,
                            "max_cb0_run": 2,
                        },
                    }
                )
            return mod.enrich_model_report(
                {
                    "condition": name,
                    "n": len(rws),
                    "mean_cer": cer,
                    "rows": rws,
                    "edit_counts": edits,
                    "cb0": {"max_run": 2},
                }
            )

        def gate_args(reason):
            return (
                cond_report("model", reason=reason, cer=0.0),
                cond_report("copy_synth", cer=0.0),
                cond_report("wrong_utt", cer=0.9),
                cond_report("random_codes", cer=0.9),
                cond_report("shuffle_text", cer=0.9),
                cond_report("mask_text", cer=0.9),
            )

        g_d1b = evaluate_n2_gate_d1b(*gate_args("safety_max_frames"), copy_synth_ceiling=0.5)
        g_d1c = evaluate_n2_gate_d1c(
            *gate_args("safety_max_frames"),
            teacher_forced_metrics={
                "tf_eos_final_top1_accuracy": 1.0,
                "tf_eos_early_top1_false_positive_count": 0,
            },
            copy_synth_ceiling=0.5,
        )
        rec(
            "G1",
            g_d1c.get("gate") == "N2_D1c_semantic_eos_v1"
            and g_d1c["criteria"]["termination_d1c"]["pass"] is False,
            f"D1c gate flags all-safety_max_frames run: d1b_verdict={g_d1b.get('verdict')} "
            f"d1c_verdict={g_d1c.get('verdict')} term_pass={g_d1c['criteria']['termination_d1c']['pass']}",
        )
        g_d1c_ok = evaluate_n2_gate_d1c(
            *gate_args("learned_eos"),
            teacher_forced_metrics={
                "tf_eos_final_top1_accuracy": 1.0,
                "tf_eos_early_top1_false_positive_count": 0,
            },
            copy_synth_ceiling=0.5,
        )
        rec(
            "G2",
            g_d1c_ok["criteria"]["termination_d1c"]["pass"] is True,
            f"D1c termination passes on a clean learned_eos run (verdict={g_d1c_ok.get('verdict')})",
        )

        # G3: a D1b contract must NOT get the D1c gate / formal D1c report
        if D1B_CONTRACT.is_file():
            d1b_contract = load_eval_contract(D1B_CONTRACT)
            install_stubs(seed_box, contract_obj=d1b_contract, perfect_eos=True)
            rc, so, tb, se = run_main(mod, base_argv(ck750, td / "g3", td / "wav_g3", td))
            gp3 = td / "g3" / "n2_gate.json"
            g3 = json.loads(gp3.read_text(encoding="utf-8")) if gp3.is_file() else {}
            rec(
                "G3",
                g3.get("gate") != "N2_D1c_semantic_eos_v1",
                f"D1b contract keeps the D1b gate: gate={g3.get('gate')!r} "
                f"formal_v2_emitted={(td / 'g3' / 'discrete_eval_report_v2.json').is_file()}",
            )
            install_stubs(seed_box, perfect_eos=True)
        else:
            rec("G3", None, f"D1b contract not found at {D1B_CONTRACT}")

        # --------------------------------- H: score-only step recovery
        outd = td / "h_decode"
        ckh = td / "ckpts" / "candidate_step_0625.pt"
        save_ckpt(ckh, 625)
        rc, so, tb, se = run_main(
            mod,
            [
                "--ckpt", str(ckh),
                "--codec-tokens-dir", str(td / "tokens"),
                "--max-utts", "3",
                "--decode-only",
                "--out", str(outd),
                "--wav-dir", str(td / "wav_h"),
                "--eval-contract", str(REAL_CONTRACT),
                "--config", "configs/discrete_speech_lm_d1b_443m.yaml",
            ],
        )
        bundle_p = outd / "n2_decode_bundle.json"
        bundle_ok = bundle_p.is_file()
        bstep = None
        if bundle_ok:
            b = json.loads(bundle_p.read_text(encoding="utf-8"))
            bstep = (b.get("provenance") or {}).get("checkpoint_step")
        rec("H1", bundle_ok and bstep == 625, f"--decode-only bundle carries checkpoint_step={bstep}")

    fails = sum(1 for _, st, _ in RESULTS if st == "FAIL")
    print(f"\nSUMMARY fails={fails} total={len(RESULTS)}")
    for tag, st, msg in RESULTS:
        if st == "FAIL":
            print(f"  FAILED {tag}: {msg}")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""REAUDIT ROUND 3 — decisive production-path probe for FAIL #4.

Unlike the repair agent's probe (which calls the helpers directly), this drives
``scripts/74_eval_discrete_lm.py::main()`` itself through argparse, stubbing ONLY
the GPU / codec / ASR-bound outer dependencies.

Real code exercised end to end:
    teacher_forced_ce -> DataLoader loop -> aggregate_teacher_forced_batches
    -> teacher_forced_metrics_for_emitter -> evaluate_n2_gate_d1b
    -> text_usage_acceptance / stability_acceptance
    -> _emit_d1c_v2_from_gate -> emit_d1c_v2_report
    -> build_formal_report -> validate_formal_report
and, for --score-only, ``_score_n2_bundle`` (the function main() delegates to).
"""
from __future__ import annotations

import ast
import contextlib
import hashlib
import importlib.util
import io
import json
import sys
import tempfile
import traceback
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from minispeech.data.d1b_oracles import load_eval_contract  # noqa: E402
from minispeech.frontend.normalize import normalize_for_asr  # noqa: E402
from minispeech.eval.discrete_report_schema import (  # noqa: E402
    validate_formal_report,
)
from minispeech.models.discrete_speech_lm import (  # noqa: E402
    DiscreteSpeechLM,
    DiscreteSpeechLMConfig,
)

REAL_CONTRACT = ROOT / "configs/eval_contracts/d1c_semantic_eos_v2_0_0.json"
RESULTS: list[tuple[str, str, str]] = []


def rec(tag: str, ok: bool | None, msg: str) -> None:
    status = "INFO" if ok is None else ("PASS" if ok else "FAIL")
    RESULTS.append((tag, status, msg))
    print(f"{tag} {status} {msg}", flush=True)


def load_script(name: str):
    path = ROOT / "scripts" / name
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(mod)
    return mod


def semantic_model() -> DiscreteSpeechLM:
    cfg = DiscreteSpeechLMConfig.tiny()
    cfg.termination_mode = "semantic_eos_v1"
    cfg.eos_id = cfg.codebook_size
    cfg.codec_vocab_size = cfg.codebook_size
    cfg.semantic_vocab_size = cfg.codebook_size + 1
    return DiscreteSpeechLM(cfg)


# --------------------------------------------------------------------------
# G0 — AST proof that main() wires aggregation -> projection -> emitter
# --------------------------------------------------------------------------
def probe_ast(mod) -> None:
    src = (ROOT / "scripts/74_eval_discrete_lm.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    fn = next(
        n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main"
    )
    calls = [n for n in ast.walk(fn) if isinstance(n, ast.Call)]
    names = {n.func.id for n in calls if isinstance(n.func, ast.Name)}
    rec(
        "G0a",
        "aggregate_teacher_forced_batches" in names,
        "main() calls aggregate_teacher_forced_batches",
    )
    rec(
        "G0b",
        "teacher_forced_metrics_for_emitter" in names,
        "main() calls teacher_forced_metrics_for_emitter",
    )
    # tf_ce must be assigned from teacher_forced_metrics_for_emitter(...)
    tf_src = None
    for node in ast.walk(fn):
        if (
            isinstance(node, ast.Assign)
            and any(
                isinstance(t, ast.Name) and t.id == "tf_ce" for t in node.targets
            )
            and isinstance(node.value, ast.Call)
            and isinstance(node.value.func, ast.Name)
        ):
            tf_src = node.value.func.id
    rec("G0c", tf_src == "teacher_forced_metrics_for_emitter", f"tf_ce := {tf_src}()")
    # emitter call must receive teacher_forced_ce=tf_ce (a Name, not a subscript)
    emit_calls = [
        c
        for c in calls
        if isinstance(c.func, ast.Name) and c.func.id == "_emit_d1c_v2_from_gate"
    ]
    kw_ok = []
    for c in emit_calls:
        for kw in c.keywords:
            if kw.arg == "teacher_forced_ce":
                kw_ok.append(ast.dump(kw.value)[:40])
    rec(
        "G0d",
        len(emit_calls) == 1 and kw_ok == ["Name(id='tf_ce', ctx=Load())"],
        f"main() emit call kwarg teacher_forced_ce={kw_ok}",
    )
    # decode bundle must carry the same dict
    rec(
        "G0e",
        '"teacher_forced_ce": tf_ce' in src,
        "decode bundle carries tf_ce dict",
    )
    # the round-2 bug signature must be gone
    rec(
        "G0f",
        'tf_ce = summary.get("loss")' not in src,
        "old bare-float reduction 'tf_ce = summary.get(\"loss\")' absent",
    )


# --------------------------------------------------------------------------
# shared harness: run main() with only outer deps stubbed
# --------------------------------------------------------------------------
def build_env(mod, td: Path, model, n_utts: int = 3):
    codebook = model.codebook_size
    ncb = model.n_codebooks
    contract = load_eval_contract(REAL_CONTRACT)

    utts = [f"u{i}" for i in range(n_utts)]
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

    wav_seed = {"n": 0}

    def fake_free_running(model_, rows_, **kw):
        condition = kw["condition"]
        out_wav_dir = Path(kw["out_wav_dir"])
        cfg = kw["cfg"]
        icfg = mod.resolve_infer_settings(
            cfg,
            infer_overrides=kw.get("infer_overrides"),
            max_frames_override=kw["max_frames"],
            codebook_size=int((cfg.get("model") or {}).get("codebook_size", 2048)),
        )
        icfg = dict(icfg)
        icfg["termination_mode"] = "semantic_eos_v1"
        rows_out = []
        gen_codes = []
        cers = []
        edit_counts = []
        hyp_for = {
            "model": "привет мир",
            "copy_synth": "привет мир",
            "wrong_utt": "щщщщщщ щщщ",
            "random_codes": "щщщщщщщщщщ",
            "shuffle_text": "щщщщщщ щщщ",
            "mask_text": "щщщщщщщщщщ",
        }
        for i, r in enumerate(rows_):
            uid = str(r["utt_id"])
            wav_seed["n"] += 1
            pcm = np.linspace(-0.3, 0.3, 2400 + wav_seed["n"], dtype=np.float32)
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

    cfg = {
        "model": {
            "codebook_size": codebook,
            "n_codebooks": ncb,
            "termination": {"mode": "semantic_eos_v1", "eos_id": model.eos_id},
        },
        "data": {"pad_token_id": 0},
        "infer": dict(contract["infer"]),
    }

    ckpt = td / "fake_best.pt"
    torch.save({"model": {}}, ckpt)

    mod._load_model = lambda p, d: (model, json.loads(json.dumps(cfg)))
    mod.load_codec_items = lambda *a, **k: rows
    mod.enforce_eval_contract = lambda *a, **k: contract
    mod._resolve_speaker_registry = lambda **k: (
        object(),
        "configs/speaker_maps/trackn_v2.json",
    )
    mod._ref_texts_from_manifest = lambda ids, m: dict(refs)
    mod.free_running_asr_eval = fake_free_running
    mod.load_yaml = lambda p: {"infer": dict(contract["infer"])}
    return contract, ckpt, utts


def run_main(mod, argv: list[str]) -> tuple[int | None, str, str]:
    """Return (rc, stdout, traceback-or-empty)."""
    buf = io.StringIO()
    old = sys.argv
    sys.argv = ["74_eval_discrete_lm.py", *argv]
    tb = ""
    rc: int | None = None
    try:
        with contextlib.redirect_stdout(buf):
            rc = mod.main()
    except BaseException:  # noqa: BLE001 - probing for unhandled tracebacks
        tb = traceback.format_exc()
    finally:
        sys.argv = old
    return rc, buf.getvalue(), tb


def main() -> int:  # noqa: C901
    mod = load_script("74_eval_discrete_lm.py")
    probe_ast(mod)

    torch.manual_seed(0)
    model = semantic_model()

    with tempfile.TemporaryDirectory() as tmp:
        td = Path(tmp)
        contract, ckpt, utts = build_env(mod, td, model)

        base = [
            "--ckpt",
            str(ckpt),
            "--codec-tokens-dir",
            str(td / "tokens"),
            "--max-utts",
            "3",
            "--n2-gate",
            "--config",
            "configs/discrete_speech_lm_d1b_443m.yaml",
        ]

        # ---------------- G1: real main(), measured path ----------------
        out1 = td / "run_measured"
        rc, so, tb = run_main(
            mod, [*base, "--out", str(out1), "--wav-dir", str(td / "wav1")]
        )
        rec("G1a", rc == 0 and not tb, f"main(--n2-gate) rc={rc} traceback={bool(tb)}")
        if tb:
            print(tb)
        v2 = out1 / "discrete_eval_report_v2.json"
        rec("G1b", v2.is_file(), f"production main() wrote {v2.name}")
        gate1 = json.loads((out1 / "n2_gate.json").read_text(encoding="utf-8"))
        if v2.is_file():
            report = json.loads(v2.read_text(encoding="utf-8"))
            tf = report["teacher_forced_metrics"]
            rec("G1c", report.get("formal_eligible") is True, "formal_eligible=True")
            rec(
                "G1d",
                tf.get("teacher_forced_ce_source") == "eval_computed",
                f"teacher_forced_ce_source={tf.get('teacher_forced_ce_source')!r}",
            )
            rec(
                "G1e",
                isinstance(tf.get("tf_eos_final_top1_accuracy"), (int, float)),
                f"tf_eos_final_top1_accuracy={tf.get('tf_eos_final_top1_accuracy')!r}",
            )
            rec(
                "G1f",
                isinstance(tf.get("tf_eos_early_top1_false_positive_count"), int),
                f"tf_eos_early_fp={tf.get('tf_eos_early_top1_false_positive_count')!r}",
            )
            rec("G1g", "ce_semantic_eos" in tf, f"ce_semantic_eos={tf.get('ce_semantic_eos')}")
            rec(
                "G1h",
                isinstance(tf.get("total_ce"), (int, float)),
                f"total_ce={tf.get('total_ce')}",
            )
            # untrained tiny model must NOT report a perfect EOS result
            rec(
                "G1i",
                float(tf.get("tf_eos_final_top1_accuracy", 1.0)) != 1.0
                or int(tf.get("tf_eos_early_top1_false_positive_count", 0)) != 0,
                "untrained model does not report the 1.0/0 default"
                f" (acc={tf.get('tf_eos_final_top1_accuracy')},"
                f" fp={tf.get('tf_eos_early_top1_false_positive_count')})",
            )
            # unpatched schema validation
            try:
                validate_formal_report(report)
                rec("G1j", True, "unpatched validate_formal_report accepts the report")
            except Exception as exc:  # noqa: BLE001
                rec("G1j", False, f"validate_formal_report rejected: {exc}")
            rec(
                "G1k",
                "text_usage" in report["gates"]
                and "phrase_loop_repetition" in report["summaries"]
                and "runaway_or_repetition" in report["summaries"]
                and "runaway_or_repetition" in report["gates"],
                "§11 fields present: gates.text_usage / summaries.phrase_loop_repetition"
                " / runaway_or_repetition",
            )
            rec(
                "G1l",
                report["gates"]["text_usage"].get("pass")
                == bool((gate1.get("text_usage_acceptance") or {}).get("pass")),
                "gates.text_usage.pass mirrors the live gate acceptance block"
                f" (report={report['gates']['text_usage'].get('pass')},"
                f" gate={(gate1.get('text_usage_acceptance') or {}).get('pass')})",
            )
            # eval_summary.json keeps the aggregate too
            summ = json.loads((out1 / "eval_summary.json").read_text(encoding="utf-8"))
            rec(
                "G1m",
                summ.get("teacher_forced_ce_source") == "eval_computed",
                f"eval_summary teacher_forced_ce_source={summ.get('teacher_forced_ce_source')!r}"
                f" n_batches={summ.get('n_batches')}",
            )
            # independent recompute of the aggregation from raw per-batch metrics
            from minispeech.data.collate_codec_seq import collate_codec_seq

            batches = []
            rows_probe = mod.load_codec_items(None, None)
            for i in range(0, len(rows_probe), 4):
                chunk = rows_probe[i : i + 4]
                b = collate_codec_seq(
                    [dict(r) for r in chunk],
                    pad_token_id=0,
                    speech_eos_id=model.eos_id,
                    termination_mode="semantic_eos_v1",
                )
                batches.append(mod.teacher_forced_ce(model, b, torch.device("cpu")))
            exp_acc = sum(
                float(b["tf_eos_final_top1_accuracy"]) for b in batches
            ) / len(batches)
            exp_fp = sum(
                int(b["tf_eos_early_top1_false_positive_count"]) for b in batches
            )
            rec(
                "G1o",
                report["checkpoint"]["step"] != 0,
                "checkpoint.step stamped by the REAL main() run: "
                f"{report['checkpoint']['step']} "
                "(0 => the selector's five-step requirement can never be met)",
            )
            rec(
                "G1n",
                abs(float(tf["tf_eos_final_top1_accuracy"]) - exp_acc) < 1e-9
                and int(tf["tf_eos_early_top1_false_positive_count"]) == exp_fp,
                "emitted EOS metrics reproduce an independent recompute"
                f" (acc {tf['tf_eos_final_top1_accuracy']} vs {exp_acc};"
                f" fp {tf['tf_eos_early_top1_false_positive_count']} vs {exp_fp})",
            )

        # ---------- G2: reinstate the round-2 bare-float bug ----------
        real_proj = mod.teacher_forced_metrics_for_emitter
        mod.teacher_forced_metrics_for_emitter = lambda s: s.get("loss")
        out2 = td / "run_barefloat"
        rc2, so2, tb2 = run_main(
            mod, [*base, "--out", str(out2), "--wav-dir", str(td / "wav2")]
        )
        mod.teacher_forced_metrics_for_emitter = real_proj
        rec(
            "G2a",
            rc2 == 0 and not tb2,
            f"bare-float injection: rc={rc2} unhandled_traceback={bool(tb2)}",
        )
        if tb2:
            print(tb2)
        rec(
            "G2b",
            (out2 / "n2_gate.json").is_file(),
            "gate JSON still written before the refusal",
        )
        rec(
            "G2c",
            not (out2 / "discrete_eval_report_v2.json").is_file(),
            "no formal v2 report emitted from a bare float (fail-closed)",
        )
        rec(
            "G2d",
            "refused" in so2 or "skipped" in so2,
            "refusal reason printed: "
            + next(
                (
                    ln
                    for ln in so2.splitlines()
                    if "refused" in ln or "skipped" in ln
                ),
                "<none>",
            ),
        )

        # ---------- G3: reinstate the round-2 discarding aggregation ----------
        real_agg = mod.aggregate_teacher_forced_batches

        def legacy_agg(batch_metrics):
            totals = {"loss": 0.0, "ce_cb0": 0.0, "ce_residual": 0.0}
            for m in batch_metrics:
                for k in totals:
                    totals[k] += float(m[k])
            n = max(len(batch_metrics), 1)
            return {k: v / n for k, v in totals.items()}

        mod.aggregate_teacher_forced_batches = legacy_agg
        out3 = td / "run_legacyagg"
        rc3, so3, tb3 = run_main(
            mod, [*base, "--out", str(out3), "--wav-dir", str(td / "wav3")]
        )
        mod.aggregate_teacher_forced_batches = real_agg
        rec(
            "G3a",
            not (out3 / "discrete_eval_report_v2.json").is_file(),
            "legacy discarding aggregation => NO formal report "
            "(the new aggregation is load-bearing, not decorative)",
        )
        rec(
            "G3b",
            rc3 == 0 and not tb3,
            f"legacy aggregation still fails soft: rc={rc3} traceback={bool(tb3)}",
        )

        # ---------- G4: decode-only bundle carries the measured dict ----------
        out4 = td / "run_decode"
        rc4, so4, tb4 = run_main(
            mod,
            [
                "--ckpt",
                str(ckpt),
                "--codec-tokens-dir",
                str(td / "tokens"),
                "--max-utts",
                "3",
                "--decode-only",
                "--config",
                "configs/discrete_speech_lm_d1b_443m.yaml",
                "--out",
                str(out4),
                "--wav-dir",
                str(td / "wav4"),
            ],
        )
        bundle_path = out4 / "n2_decode_bundle.json"
        rec("G4a", rc4 == 0 and not tb4 and bundle_path.is_file(), "decode-only bundle written")
        if tb4:
            print(tb4)
        bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
        btf = bundle.get("teacher_forced_ce")
        rec(
            "G4b",
            isinstance(btf, dict)
            and btf.get("teacher_forced_ce_source") == "eval_computed",
            f"bundle.teacher_forced_ce is a measured dict: {type(btf).__name__}"
            f" source={btf.get('teacher_forced_ce_source') if isinstance(btf, dict) else btf!r}",
        )

        # ---------- G5..G7: --score-only production path ----------
        class _StubASR:
            def transcribe(self, audio, sr, use_cache=False):  # noqa: ARG002
                return "привет мир"

        mod.load_asr_engine = lambda cfgp: _StubASR()
        real_assert_bundle = mod.assert_score_only_d1b_bundle
        real_assert_wavs = mod.assert_score_only_wavs_present
        mod.assert_score_only_d1b_bundle = lambda b, **k: contract
        mod.assert_score_only_wavs_present = lambda b, **k: None

        def score_run(tag: str, bundle_obj, out_name: str):
            p = td / f"bundle_{out_name}.json"
            p.write_text(json.dumps(bundle_obj, ensure_ascii=False), encoding="utf-8")
            o = td / f"score_{out_name}"
            return p, o, run_main(
                mod,
                [
                    "--score-only",
                    str(p),
                    "--out",
                    str(o),
                    "--eval-contract",
                    str(REAL_CONTRACT),
                ],
            )

        _, o5, (rc5, so5, tb5) = score_run("G5", bundle, "measured")
        rec("G5a", rc5 == 0 and not tb5, f"--score-only measured rc={rc5} traceback={bool(tb5)}")
        if tb5:
            print(tb5)
        v2s = o5 / "discrete_eval_report_v2.json"
        rec("G5b", v2s.is_file(), "--score-only + measured TF => formal v2 report")
        if v2s.is_file():
            rs = json.loads(v2s.read_text(encoding="utf-8"))
            rec(
                "G5c",
                rs["teacher_forced_metrics"]["teacher_forced_ce_source"]
                == "eval_computed"
                and rs["provenance"].get("score_only") is True,
                "score-only report is stamped eval_computed with score_only provenance",
            )
            try:
                validate_formal_report(rs)
                rec("G5d", True, "score-only report passes unpatched validation")
            except Exception as exc:  # noqa: BLE001
                rec("G5d", False, f"score-only report rejected: {exc}")

        bare_bundle = json.loads(json.dumps(bundle))
        bare_bundle["teacher_forced_ce"] = float(btf["loss"])
        _, o6, (rc6, so6, tb6) = score_run("G6", bare_bundle, "bare")
        rec("G6a", rc6 == 0 and not tb6, f"--score-only bare float rc={rc6} traceback={bool(tb6)}")
        if tb6:
            print(tb6)
        rec(
            "G6b",
            not (o6 / "discrete_eval_report_v2.json").is_file()
            and (o6 / "n2_gate.json").is_file(),
            "--score-only bare float: no formal report, gate JSON intact",
        )
        rec(
            "G6c",
            "skipped" in so6 or "refused" in so6,
            "skip message: "
            + next(
                (ln for ln in so6.splitlines() if "skipped" in ln or "refused" in ln),
                "<none>",
            ),
        )

        db_bundle = json.loads(json.dumps(bundle))
        db_bundle["teacher_forced_ce"] = {
            **btf,
            "teacher_forced_ce_source": "decode_bundle",
        }
        _, o7, (rc7, so7, tb7) = score_run("G7", db_bundle, "decodebundle")
        rec("G7a", rc7 == 0 and not tb7, f"--score-only decode_bundle rc={rc7} traceback={bool(tb7)}")
        rec(
            "G7b",
            not (o7 / "discrete_eval_report_v2.json").is_file(),
            "decode_bundle provenance never yields a formal report",
        )

        # score-only where EOS metrics were silently dropped but source kept
        drop_bundle = json.loads(json.dumps(bundle))
        drop_bundle["teacher_forced_ce"] = {
            "loss": btf["loss"],
            "total_ce": btf["loss"],
            "teacher_forced_ce_source": "eval_computed",
        }
        _, o8, (rc8, so8, tb8) = score_run("G8", drop_bundle, "noeos")
        rec("G8a", rc8 == 0 and not tb8, f"--score-only eval_computed w/o EOS rc={rc8} traceback={bool(tb8)}")
        rec(
            "G8b",
            not (o8 / "discrete_eval_report_v2.json").is_file(),
            "eval_computed source without measured EOS is still refused "
            "(no fabricated 1.0/0)",
        )

        mod.assert_score_only_d1b_bundle = real_assert_bundle
        mod.assert_score_only_wavs_present = real_assert_wavs

    n_fail = sum(1 for _, s, _ in RESULTS if s == "FAIL")
    print(f"\nSUMMARY pass={sum(1 for _, s, _ in RESULTS if s == 'PASS')} fail={n_fail}")
    print("ROUND3_REAUDIT_MAINPATH_" + ("PASS" if n_fail == 0 else "FAIL"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

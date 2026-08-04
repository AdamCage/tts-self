#!/usr/bin/env python3
"""ROUND 5 live probe: round-4 duplicate-checkpoint attack must now FAIL closed.

Reproduces REAUDIT_ROUND4 N1–N3 against production entry points:
  * five main() runs against ONE best.pt with different --checkpoint-step
    → formal path refuses inventing steps (CLI confirm-only)
  * five reports with identical checkpoint.sha256
    → select_d1c_formal_reports refuses
  * claimed checkpoint.sha256 ≠ file bytes
    → schema / selector refuse
"""
from __future__ import annotations

import contextlib
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


def main() -> int:
    mod = load_script("74_eval_discrete_lm.py")
    mod76 = load_script("76_select_discrete_checkpoint.py")
    contract = load_eval_contract(REAL_CONTRACT)

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

    def fake_fr(model_, rows_, **kw):
        condition = kw["condition"]
        out_wav_dir = Path(kw["out_wav_dir"])
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
        for i, r in enumerate(rows_):
            uid = str(r["utt_id"])
            pcm = np.linspace(-0.3, 0.3, 2400 + i, dtype=np.float32)
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
                    "gt_frames": n_frames,
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

    with tempfile.TemporaryDirectory(prefix="r5_dupckpt_") as td_s:
        td = Path(td_s)
        best = td / "best.pt"
        torch.save({"model": {}, "cfg": cfg}, best)  # no payload step

        mod._load_model = lambda p, d: (model, dict(cfg), {"cfg": cfg})
        mod.load_codec_items = lambda *a, **k: rows
        mod.enforce_eval_contract = lambda *a, **k: contract
        mod._resolve_speaker_registry = lambda **k: (
            object(),
            "configs/speaker_maps/trackn_v2.json",
        )
        mod._ref_texts_from_manifest = lambda ids, man: dict(refs)
        mod.free_running_asr_eval = fake_fr
        mod.load_yaml = lambda p: {"infer": dict(contract["infer"])}

        # --- N1/N2 attack surface: CLI invent on single best.pt ---
        invent_exits: list[str] = []
        for s in STEPS:
            argv = [
                "74_eval_discrete_lm.py",
                "--ckpt",
                str(best),
                "--codec-tokens-dir",
                str(td / "tokens"),
                "--max-utts",
                "3",
                "--n2-gate",
                "--out",
                str(td / f"out_{s}"),
                "--wav-dir",
                str(td / f"wav_{s}"),
                "--eval-contract",
                str(REAL_CONTRACT),
                "--config",
                "configs/discrete_speech_lm_d1b_443m.yaml",
                "--checkpoint-step",
                str(s),
            ]
            old = sys.argv
            buf = io.StringIO()
            exit_msg = ""
            rc = None
            try:
                sys.argv = argv
                with contextlib.redirect_stdout(buf):
                    try:
                        rc = mod.main()
                    except SystemExit as e:
                        exit_msg = str(e)
                        rc = int(e.code) if isinstance(e.code, int) else 1
            except Exception:
                exit_msg = traceback.format_exc()
                rc = 1
            finally:
                sys.argv = old
            invent_exits.append(exit_msg or f"rc={rc}")
            v2 = td / f"out_{s}" / "discrete_eval_report_v2.json"
            if v2.is_file():
                rec("N1", False, f"step={s} produced formal report (attack not closed)")
                break
        else:
            invent_ok = all("cannot invent" in m or "unresolved" in m for m in invent_exits)
            rec(
                "N1",
                invent_ok,
                f"five best.pt + --checkpoint-step invents refused: {invent_exits[0][:120]}",
            )

        # Direct resolve contract
        try:
            mod.resolve_checkpoint_step(
                ckpt_path=best,
                payload={},
                cli_step=625,
                allowed_steps=STEPS,
                require=True,
            )
            rec("N1b", False, "resolve allowed invent on best.pt")
        except SystemExit as e:
            rec("N1b", "cannot invent" in str(e), f"resolve invent refuse: {e}")

        # --- Selector duplicate-hash refuse (forged reports, schema bypassed) ---
        shared = b"one-physical-ckpt"
        shared_sha = hashlib.sha256(shared).hexdigest()
        paths = []
        for s in STEPS:
            alias = td / "dup" / f"candidate_step_{s:04d}.pt"
            alias.parent.mkdir(parents=True, exist_ok=True)
            alias.write_bytes(shared)
            rep = {
                "schema": "minispeech.discrete_eval_report",
                "schema_version": "2.0.0",
                "formal_eligible": True,
                "checkpoint": {
                    "path": str(alias),
                    "sha256": shared_sha,
                    "step": s,
                    "schema": "minispeech.discrete_lm_checkpoint",
                    "schema_version": "2.0.0",
                },
                "provenance": {
                    "eval_contract_sha256": "e" * 64,
                    "recipe_sha256": "f" * 64,
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
                    "ordered_utt_ids": ["u0"],
                    "ordered_utt_ids_sha256": "2" * 64,
                },
                "conditions": {
                    name: {
                        "subset_key": "select8",
                        "subset_path": "x",
                        "subset_file_sha256": "3" * 64,
                        "expected_count": 1,
                        "evaluated_count": 1,
                        "ordered_utt_ids_sha256": "2" * 64,
                        "rows": [
                            {
                                "utt_id": "u0",
                                "cer": 0.0,
                                "catastrophic": False,
                                "n_frames": 10,
                                "gt_frames": 10,
                                "generation": {
                                    "termination_reason": "learned_eos",
                                    "n_decoded_frames": 10,
                                    "early_eos_suppressed_count": 0,
                                    "max_cb0_run": 2,
                                },
                                "wav": {
                                    "path": "x.wav",
                                    "file_sha256": "a" * 64,
                                    "pcm_sha256": "b" * 64,
                                },
                                "normalized_reference_sha256": "c" * 64,
                                "generated_codes_sha256": "d" * 64,
                            }
                        ],
                        "rows_sha256": "4" * 64,
                    }
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
                    "model_macro_cer": 0.0,
                    "n_catastrophic": 0,
                    "max_abs_log_duration_error": 0.0,
                    "mean_abs_log_duration_error": 0.0,
                },
                "gates": {"eligible": True, "text_usage": {"pass": True}},
                "report_sha256": "5" * 64,
            }
            p = td / f"dup_report_{s}.json"
            p.write_text(json.dumps(rep), encoding="utf-8")
            paths.append(p)

        import minispeech.eval.discrete_report_schema as schema

        old_v = schema.validate_formal_report
        schema.validate_formal_report = lambda data, **kwargs: data  # type: ignore[assignment]
        mod76.validate_formal_report = schema.validate_formal_report
        try:
            try:
                mod76.select_d1c_formal_reports(paths)
                rec("N2", False, "selector ACCEPTED duplicate checkpoint.sha256 set")
            except ValueError as e:
                rec(
                    "N2",
                    "duplicate checkpoint.sha256" in str(e),
                    f"selector refused duplicate hashes: {e}",
                )
        finally:
            schema.validate_formal_report = old_v
            mod76.validate_formal_report = old_v

        # --- N3: hash mismatch vs on-disk file ---
        ck = td / "candidate_step_0250.pt"
        ck.write_bytes(b"real-bytes")
        live = hashlib.sha256(b"real-bytes").hexdigest()
        forged = {
            "schema": "minispeech.discrete_eval_report",
            "schema_version": "2.0.0",
            "formal_eligible": True,
            "checkpoint": {
                "path": str(ck),
                "sha256": "0" * 64,
                "step": 250,
                "schema": "x",
                "schema_version": "2",
            },
            "provenance": {"eval_contract_sha256": "1" * 64},
            "infer": {"max_frames": 750},
            "teacher_forced_metrics": {
                "teacher_forced_ce_source": "eval_computed",
                "total_ce": 1.0,
            },
            "row_binding": {
                "ordered_utt_ids": ["u0"],
                "ordered_utt_ids_sha256": hashlib.sha256(
                    json.dumps(["u0"], separators=(",", ":")).encode()
                ).hexdigest(),
            },
            "conditions": {},
            "report_sha256": "x",
        }
        # Minimal: call schema check on checkpoint fields via validate when
        # conditions incomplete — exercise the hash check path directly.
        try:
            # Build a complete-enough report for validate_formal_report hash check.
            from minispeech.eval.discrete_report_schema import (
                build_formal_report,
                ordered_utt_ids_sha256,
            )

            ids = ["u0"]
            conditions = {
                name: {
                    "subset_key": "select8",
                    "subset_path": "s.json",
                    "subset_file_sha256": "e" * 64,
                    "expected_count": 1,
                    "evaluated_count": 1,
                    "ordered_utt_ids_sha256": ordered_utt_ids_sha256(ids),
                    "rows": [
                        {
                            "utt_id": "u0",
                            "ordinal": 0,
                            "normalized_reference_sha256": "a" * 64,
                            "generated_codes_sha256": "b" * 64,
                            "generation": {"termination_reason": "learned_eos"},
                            "wav": {
                                "path": "u.wav",
                                "file_sha256": "c" * 64,
                                "pcm_sha256": "d" * 64,
                            },
                            "cer": 0.0,
                        }
                    ],
                }
                for name in (
                    "model",
                    "copy_synth",
                    "wrong_utt",
                    "random_codes",
                    "shuffle_text",
                    "mask_text",
                )
            }
            good = build_formal_report(
                formal_eligible=True,
                checkpoint={
                    "path": str(ck),
                    "sha256": live,
                    "step": 250,
                    "schema": "x",
                    "schema_version": "2",
                },
                provenance={"eval_contract_sha256": "1" * 64},
                infer={"max_frames": 750},
                teacher_forced_metrics={
                    "teacher_forced_ce_source": "eval_computed",
                    "total_ce": 1.0,
                },
                row_binding={
                    "ordered_utt_ids": ids,
                    "ordered_utt_ids_sha256": ordered_utt_ids_sha256(ids),
                },
                conditions=conditions,
            )
            good["checkpoint"]["sha256"] = "0" * 64
            good.pop("report_sha256", None)
            good["report_sha256"] = sha256_json(good)
            try:
                validate_formal_report(good)
                rec("N3", False, "schema accepted mismatched checkpoint.sha256")
            except ValueError as e:
                rec(
                    "N3",
                    "checkpoint.sha256 does not match" in str(e),
                    f"schema refused hash mismatch: {e}",
                )
        except Exception as e:
            rec("N3", False, f"N3 setup error: {e}")

        # Confirm CLI can still confirm a derived step (non-attack).
        cand = td / "candidate_step_0625.pt"
        torch.save({"model": {}, "step": 625, "cfg": cfg}, cand)
        got = mod.resolve_checkpoint_step(
            ckpt_path=cand,
            payload={"step": 625},
            cli_step=625,
            allowed_steps=STEPS,
            require=True,
        )
        rec("C1", got == 625, f"CLI confirm of derived step still works: {got}")

    n_fail = sum(1 for _, s, _ in RESULTS if s == "FAIL")
    n_pass = sum(1 for _, s, _ in RESULTS if s == "PASS")
    print(f"\nSUMMARY pass={n_pass} fail={n_fail}", flush=True)
    return 1 if n_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())

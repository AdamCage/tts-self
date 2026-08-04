#!/usr/bin/env python3
"""ROUND 4 live probe: checkpoint.step stamping through main() → selector §8.4.

Stubs only GPU/codec/ASR outer deps. Real path:
  _load_model payload/filename → resolve_checkpoint_step → provenance/gate
  → emitter checkpoint.step → validate_formal_report → select_d1c_formal_reports
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
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from minispeech.data.d1b_oracles import load_eval_contract  # noqa: E402
from minispeech.eval.discrete_report_schema import (  # noqa: E402
    rows_sha256,
    sha256_json,
    validate_formal_report,
)
from minispeech.frontend.normalize import normalize_for_asr  # noqa: E402
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


def run_main(mod, argv: list[str]) -> tuple[int | None, str, str]:
    buf = io.StringIO()
    old = sys.argv
    sys.argv = ["74_eval_discrete_lm.py", *argv]
    tb = ""
    rc: int | None = None
    try:
        with contextlib.redirect_stdout(buf):
            rc = mod.main()
    except BaseException:  # noqa: BLE001
        tb = traceback.format_exc()
    finally:
        sys.argv = old
    return rc, buf.getvalue(), tb


def main() -> int:
    mod = load_script("74_eval_discrete_lm.py")
    mod76 = load_script("76_select_discrete_checkpoint.py")
    src = (ROOT / "scripts/74_eval_discrete_lm.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main")
    names = {
        n.func.id for n in ast.walk(fn) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
    }
    rec("S0a", "resolve_checkpoint_step" in names, "main() calls resolve_checkpoint_step")
    rec("S0b", "evaluate_n2_gate_d1c" in names, "main() calls evaluate_n2_gate_d1c")
    rec("S0c", "--checkpoint-step" in src, "CLI exposes --checkpoint-step")

    contract = load_eval_contract(REAL_CONTRACT)
    model = semantic_model()
    codebook = model.codebook_size
    ncb = model.n_codebooks
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
        rows_out, gen_codes, cers, edit_counts = [], [], [], []
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

    with tempfile.TemporaryDirectory() as tmp:
        td = Path(tmp)
        ckpt = td / "candidate_step_0750.pt"
        payload = {
            "model": model.state_dict(),
            "step": 750,
            "optimizer_step": 750,
            "cfg": cfg,
            "schema": "minispeech.discrete_lm_checkpoint",
            "schema_version": "2.0.0",
        }
        torch.save(payload, ckpt)

        # Prove production resolver uses payload + immutable filename (no discard).
        rec(
            "S1a",
            mod.checkpoint_step_from_payload(payload) == 750,
            "payload step extractor → 750",
        )
        rec(
            "S1b",
            mod.parse_candidate_step_from_path(ckpt) == 750,
            "filename candidate_step_0750.pt → 750",
        )
        rec(
            "S1c",
            mod.resolve_checkpoint_step(
                ckpt_path=ckpt,
                payload=payload,
                allowed_steps=list(contract["train"]["candidate_steps"]),
                require=True,
            )
            == 750,
            "resolve_checkpoint_step cross-checks payload+filename → 750",
        )
        # _load_model return signature: must be (model, cfg, payload).
        load_fn = next(
            n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_load_model"
        )
        ret_ok = any(
            isinstance(n, ast.Return)
            and isinstance(n.value, ast.Tuple)
            and len(n.value.elts) == 3
            for n in ast.walk(load_fn)
        )
        rec("S1d", ret_ok, "_load_model returns 3-tuple (model, cfg, payload)")

        # Production main(): stub only GPU/codec/ASR; keep real resolve/emit/gate.
        mod._load_model = lambda p, d: (model, copy.deepcopy(cfg), dict(payload))
        mod.load_codec_items = lambda *a, **k: rows
        mod.enforce_eval_contract = lambda *a, **k: contract
        mod._resolve_speaker_registry = lambda **k: (
            object(),
            "configs/speaker_maps/trackn_v2.json",
        )
        mod._ref_texts_from_manifest = lambda ids, m: dict(refs)
        mod.free_running_asr_eval = fake_free_running
        mod.load_yaml = lambda p: {"infer": dict(contract["infer"])}

        out1 = td / "run_measured"
        rc, so, tb = run_main(
            mod,
            [
                "--ckpt",
                str(ckpt),
                "--codec-tokens-dir",
                str(td / "tokens"),
                "--max-utts",
                "3",
                "--n2-gate",
                "--out",
                str(out1),
                "--wav-dir",
                str(td / "wav1"),
                "--eval-contract",
                str(REAL_CONTRACT),
            ],
        )
        rec("S2a", rc == 0 and not tb, f"main() rc={rc} traceback={bool(tb)}")
        if tb:
            print(tb)
        v2 = out1 / "discrete_eval_report_v2.json"
        gate = json.loads((out1 / "n2_gate.json").read_text(encoding="utf-8"))
        rec("S2b", v2.is_file(), "formal v2 report written")
        rec("S2c", gate.get("gate") == "N2_D1c_semantic_eos_v1", f"gate id={gate.get('gate')}")
        rec("S2d", gate.get("step") == 750, f"gate.step={gate.get('step')}")
        rec(
            "S2e",
            gate.get("provenance", {}).get("checkpoint_step") == 750,
            f"provenance.checkpoint_step={gate.get('provenance', {}).get('checkpoint_step')}",
        )
        report = json.loads(v2.read_text(encoding="utf-8")) if v2.is_file() else {}
        step = (report.get("checkpoint") or {}).get("step")
        rec("S2f", step == 750, f"report checkpoint.step={step}")
        try:
            validate_formal_report(report)
            rec("S2g", True, "validate_formal_report accepts non-zero step")
        except Exception as exc:  # noqa: BLE001
            rec("S2g", False, f"validate_formal_report: {exc}")

        # Negative: step=0 must fail closed in schema
        bad = copy.deepcopy(report)
        bad["checkpoint"]["step"] = 0
        bad.pop("report_sha256", None)
        bad["report_sha256"] = sha256_json(bad)
        try:
            validate_formal_report(bad)
            rec("S3a", False, "step=0 incorrectly accepted")
        except ValueError as exc:
            rec("S3a", "positive integer" in str(exc), f"step=0 refused: {exc}")

        # Five-step selector §8.4
        steps = [250, 375, 500, 625, 750]
        paths = []
        for s in steps:
            cloned = copy.deepcopy(report)
            cloned["checkpoint"]["step"] = s
            cloned["checkpoint"]["path"] = str(td / f"candidate_step_{s:04d}.pt")
            cloned["checkpoint"]["sha256"] = hashlib.sha256(f"ckpt-{s}".encode()).hexdigest()
            cloned["teacher_forced_metrics"]["tf_eos_final_top1_accuracy"] = 1.0
            cloned["teacher_forced_metrics"]["tf_eos_early_top1_false_positive_count"] = 0
            for cond in cloned["conditions"].values():
                for row in cond["rows"]:
                    row["cer"] = 0.0
                    row["catastrophic"] = False
                    row["gt_frames"] = max(1, int(row.get("gt_frames") or 1))
                    row["n_frames"] = int(row["gt_frames"])
                    gen = row.setdefault("generation", {})
                    gen["termination_reason"] = "learned_eos"
                    gen["n_decoded_frames"] = int(row["n_frames"])
                    gen["early_eos_suppressed_count"] = 0
                    gen["max_cb0_run"] = 2
                cond["rows_sha256"] = rows_sha256(cond["rows"])
            cloned["gates"]["text_usage"] = {"pass": True}
            cloned["gates"]["runaway_or_repetition"] = False
            cloned["summaries"]["phrase_loop_repetition"] = False
            cloned["summaries"]["runaway_or_repetition"] = False
            cloned.pop("report_sha256", None)
            cloned["report_sha256"] = sha256_json(cloned)
            validate_formal_report(cloned)
            outp = td / f"report_step_{s}.json"
            outp.write_text(json.dumps(cloned, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            paths.append(outp)
        try:
            selection = mod76.select_d1c_formal_reports(paths)
            got = sorted(int(c["step"]) for c in selection["candidates"])
            rec("S4a", got == steps, f"selector accepted steps={got}")
            rec(
                "S4b",
                selection["selected"]["step"] in steps,
                f"selected step={selection['selected']['step']}",
            )
        except Exception as exc:  # noqa: BLE001
            rec("S4a", False, f"selector raised: {exc}")
            rec("S4b", False, "no selection")

    fails = sum(1 for _, st, _ in RESULTS if st == "FAIL")
    print(f"\nSUMMARY fails={fails} total={len(RESULTS)}")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())

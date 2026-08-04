#!/usr/bin/env python3
"""Round-6 live probe: rename/re-save attack refuse + legitimate set accept.

CPU only. Drives production resolve_checkpoint_step / require_checkpoint_payload_hashes
/ main() / select_d1c_formal_reports. No GPU, no seed invent, no commit.
"""

from __future__ import annotations

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
RESULTS = ROOT / ".cursor" / "results" / "d1c_semantic_eos_repair_b81ecc3e"
LOG = RESULTS / "logs" / "repair_round6_probe_rename.log"

STEPS = (250, 375, 500, 625, 750)
CONTRACT_PATH = ROOT / "configs" / "eval_contracts" / "d1c_semantic_eos_v2_0_0.json"


def _load_script(name: str):
    path = ROOT / "scripts" / name
    spec = importlib.util.spec_from_file_location(path.stem, path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def rec(fid: str, ok: bool | None, msg: str, lines: list[str]) -> None:
    tag = "PASS" if ok is True else ("FAIL" if ok is False else "INFO")
    lines.append(f"{fid} {tag} {msg}")
    print(lines[-1])


def main() -> int:
    from minispeech.data.d1b_oracles import load_eval_contract
    from minispeech.eval.discrete_report_schema import (
        rows_sha256,
        sha256_json,
        validate_formal_report,
    )
    from minispeech.frontend.normalize import normalize_for_asr
    from minispeech.models.discrete_speech_lm import (
        DiscreteSpeechLM,
        DiscreteSpeechLMConfig,
    )

    lines: list[str] = []
    mod = _load_script("74_eval_discrete_lm.py")
    mod76 = _load_script("76_select_discrete_checkpoint.py")
    contract = load_eval_contract(CONTRACT_PATH)
    live_hashes = {
        "eval_contract": str(contract["sha256"]),
        "recipe": str(contract["recipe_sha256"]),
        "recipe_source_manifest": str(contract["recipe_source_manifest_sha256"]),
        "run_manifest": "r" * 64,
    }

    # --- A: resolve refuses filename-only on formal path ---
    got = None
    err = ""
    try:
        got = mod.resolve_checkpoint_step(
            ckpt_path="runs/x/candidate_step_0625.pt",
            payload={"model": {}},
            allowed_steps=STEPS,
            require=True,
        )
    except SystemExit as exc:
        err = str(exc)
    rec(
        "A1",
        got is None and "payload['step']" in err,
        f"filename-only require=True refused: got={got} err={err[:120]!r}",
        lines,
    )

    # --- B: hashes required ---
    try:
        mod.require_checkpoint_payload_hashes({"step": 500})
        rec("B1", False, "missing hashes did not raise", lines)
    except SystemExit as exc:
        rec("B1", "hashes" in str(exc), f"missing hashes refused: {exc}", lines)
    try:
        mod.require_checkpoint_payload_hashes(
            {"step": 500, "hashes": dict(live_hashes)},
            expected_eval_contract="0" * 64,
        )
        rec("B2", False, "eval_contract mismatch did not raise", lines)
    except SystemExit as exc:
        rec("B2", "mismatch" in str(exc), f"eval_contract mismatch refused: {exc}", lines)
    got_h = mod.require_checkpoint_payload_hashes(
        {"step": 500, "hashes": dict(live_hashes)},
        expected_eval_contract=live_hashes["eval_contract"],
    )
    rec("B3", got_h["run_manifest"] == live_hashes["run_manifest"], "legit hashes accepted", lines)

    cfg_tiny = DiscreteSpeechLMConfig.tiny()
    cfg_tiny.termination_mode = "semantic_eos_v1"
    cfg_tiny.eos_id = cfg_tiny.codebook_size
    cfg_tiny.codec_vocab_size = cfg_tiny.codebook_size
    cfg_tiny.semantic_vocab_size = cfg_tiny.codebook_size + 1
    model = DiscreteSpeechLM(cfg_tiny)
    cfg = {
        "model": {
            "codebook_size": model.codebook_size,
            "n_codebooks": model.n_codebooks,
            "termination": {"mode": "semantic_eos_v1", "eos_id": model.eos_id},
        },
        "data": {"pad_token_id": 0},
        "infer": dict(contract["infer"]),
    }
    utts = ["u0", "u1", "u2"]
    refs = {u: "привет мир" for u in utts}
    rows = [
        {
            "utt_id": u,
            "input_ids": torch.tensor([3 + i, 4 + i, 5 + i]),
            "codes": torch.randint(0, model.codebook_size, (4 + i, model.n_codebooks)),
            "speaker_id": i % 2,
        }
        for i, u in enumerate(utts)
    ]
    wav_seed = {"n": 0}

    def fake_free_running(model_, rows_, **kw):
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

    def install_stubs(*, payload_factory):
        mod._load_model = lambda p, d: (
            model,
            copy.deepcopy(cfg),
            payload_factory(Path(p)),
        )
        mod.load_codec_items = lambda *a, **k: rows
        mod.enforce_eval_contract = lambda *a, **k: contract
        mod._resolve_speaker_registry = lambda **k: (
            object(),
            "configs/speaker_maps/trackn_v2.json",
        )
        mod._ref_texts_from_manifest = lambda ids, man: dict(refs)
        mod.free_running_asr_eval = fake_free_running
        mod.load_yaml = lambda p: {"infer": dict(contract["infer"])}

    def run_main(ckpt: Path, out: Path, wav: Path) -> tuple[int, str]:
        argv = [
            "74_eval_discrete_lm.py",
            "--ckpt",
            str(ckpt),
            "--codec-tokens-dir",
            str(out / "tokens"),
            "--max-utts",
            "3",
            "--n2-gate",
            "--out",
            str(out),
            "--wav-dir",
            str(wav),
            "--eval-contract",
            str(CONTRACT_PATH),
            "--config",
            "configs/discrete_speech_lm_d1b_443m.yaml",
        ]
        old = sys.argv
        buf = io.StringIO()
        err = ""
        rc = 1
        try:
            sys.argv = argv
            with contextlib.redirect_stdout(buf):
                try:
                    rc = int(mod.main() or 0)
                except SystemExit as exc:
                    err = str(exc)
                    rc = 1 if exc.code not in (0, None) else 0
        finally:
            sys.argv = old
        return rc, err or buf.getvalue()

    with tempfile.TemporaryDirectory(prefix="d1c_r6_") as td_s:
        td = Path(td_s)

        # --- C: rename/re-save attack (no payload step, junk keys for distinct bytes) ---
        install_stubs(
            payload_factory=lambda p: {
                "model": {},
                "cfg": cfg,
                # deliberately NO step
                "hashes": dict(live_hashes),
                "junk": p.name,
            }
        )
        attack_reports = []
        attack_steps = []
        for s in STEPS:
            ck = td / "rename" / f"candidate_step_{s:04d}.pt"
            ck.parent.mkdir(parents=True, exist_ok=True)
            # Byte-distinct files, no payload step.
            torch.save(
                {
                    "model": {},
                    "cfg": cfg,
                    "hashes": dict(live_hashes),
                    "junk": "x" * (s % 17 + 1),
                },
                ck,
            )
            outd = td / f"atk_{s}"
            rc, msg = run_main(ck, outd, td / f"wav_atk_{s}")
            rp = outd / "discrete_eval_report_v2.json"
            if rp.is_file():
                d = json.loads(rp.read_text(encoding="utf-8"))
                attack_reports.append(rp)
                attack_steps.append(d.get("checkpoint", {}).get("step"))
            else:
                attack_steps.append(f"refused:{msg[:60]}")
        rec(
            "C1",
            len(attack_reports) == 0,
            f"rename attack emitted {len(attack_reports)}/5 reports; steps={attack_steps}",
            lines,
        )
        if attack_reports:
            try:
                sel = mod76.select_d1c_formal_reports(attack_reports)
                rec(
                    "C2",
                    False,
                    f"SELECTOR ACCEPTED rename attack selected={sel['selected']['step']}",
                    lines,
                )
            except Exception as exc:  # noqa: BLE001
                rec("C2", True, f"selector refused rename attack: {exc}", lines)
        else:
            rec("C2", True, "no reports to select (evaluator refused)", lines)

        # --- D: legitimate five candidates with payload step + hashes ---
        install_stubs(
            payload_factory=lambda p: {
                "step": int(p.stem.split("_")[-1]),
                "optimizer_step": int(p.stem.split("_")[-1]),
                "cfg": cfg,
                "hashes": dict(live_hashes),
            }
        )
        legit_paths = []
        legit_steps = []
        for s in STEPS:
            ck = td / "legit" / f"candidate_step_{s:04d}.pt"
            ck.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "model": {},
                    "step": s,
                    "optimizer_step": s,
                    "cfg": cfg,
                    "hashes": dict(live_hashes),
                    "marker": f"step-{s}",
                },
                ck,
            )
            outd = td / f"legit_{s}"
            rc, msg = run_main(ck, outd, td / f"wav_legit_{s}")
            rp = outd / "discrete_eval_report_v2.json"
            if not rp.is_file():
                rec("D1", False, f"legit step={s} failed: {msg[:120]}", lines)
                break
            d = json.loads(rp.read_text(encoding="utf-8"))
            # Make eligible for selection.
            d["teacher_forced_metrics"]["tf_eos_final_top1_accuracy"] = 1.0
            d["teacher_forced_metrics"]["tf_eos_early_top1_false_positive_count"] = 0
            for cond in d["conditions"].values():
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
            d["gates"]["text_usage"] = {"pass": True}
            d["gates"]["runaway_or_repetition"] = False
            d["summaries"]["phrase_loop_repetition"] = False
            d["summaries"]["runaway_or_repetition"] = False
            d.pop("report_sha256", None)
            d["report_sha256"] = sha256_json(d)
            validate_formal_report(d)
            rp.write_text(json.dumps(d, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            legit_paths.append(rp)
            legit_steps.append(d["checkpoint"]["step"])
        else:
            rec(
                "D1",
                legit_steps == list(STEPS),
                f"legit main() stamped steps={legit_steps}",
                lines,
            )
            try:
                sel = mod76.select_d1c_formal_reports(legit_paths)
                rec(
                    "D2",
                    sel["selected"]["step"] in STEPS,
                    f"selector ACCEPTS legit set selected_step={sel['selected']['step']}",
                    lines,
                )
                shas = {c["checkpoint_sha256"] for c in sel["candidates"]}
                rec("D3", len(shas) == 5, f"legit distinct sha count={len(shas)}", lines)
            except Exception as exc:  # noqa: BLE001
                rec("D2", False, f"selector refused legit set: {exc}", lines)
                rec("D3", False, traceback.format_exc()[:200], lines)

        # --- E: basename rule still live ---
        if len(legit_paths) == 5:
            bad = json.loads(legit_paths[0].read_text(encoding="utf-8"))
            wrong = td / "best.pt"
            wrong.write_bytes(b"basename-probe")
            bad["checkpoint"]["path"] = str(wrong)
            bad["checkpoint"]["sha256"] = hashlib.sha256(b"basename-probe").hexdigest()
            bad["checkpoint"]["step"] = 250
            bad.pop("report_sha256", None)
            bad["report_sha256"] = sha256_json(bad)
            bp = td / "base_bad.json"
            bp.write_text(json.dumps(bad), encoding="utf-8")
            paths = [bp] + legit_paths[1:]
            import minispeech.eval.discrete_report_schema as schema

            old = schema.validate_formal_report
            schema.validate_formal_report = lambda data, **kwargs: data  # type: ignore[assignment]
            mod76.validate_formal_report = schema.validate_formal_report
            try:
                mod76.select_d1c_formal_reports(paths)
                rec("E1", False, "basename mismatch accepted", lines)
            except ValueError as exc:
                rec("E1", "basename" in str(exc), f"basename refused: {exc}", lines)
            finally:
                schema.validate_formal_report = old
                mod76.validate_formal_report = old

    n_pass = sum(1 for L in lines if " PASS " in L)
    n_fail = sum(1 for L in lines if " FAIL " in L)
    summary = f"SUMMARY pass={n_pass} fail={n_fail}"
    lines.append(summary)
    print(summary)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    LOG.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())

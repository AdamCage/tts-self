#!/usr/bin/env python3
"""REAUDIT_ROUND4 — NEW-surface probe: does `--checkpoint-step` break plan §8.4?

Plan §8.4 requires the D1c formal selector to enforce
"один checkpoint hash/step match на report" (one checkpoint hash/step per report)
and the test-case list requires "checkpoint hash/step mismatch rejected".

Round 4 introduced `--checkpoint-step`, which lets an operator assert an arbitrary
candidate step for any checkpoint file. This probe asks whether the *same physical
checkpoint* can therefore be published five times as five different candidates.
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
            pcm = np.linspace(-0.3, 0.3, 2400, dtype=np.float32)
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

    real_tf = mod.teacher_forced_ce
    mod.teacher_forced_ce = lambda m_, b, d: {
        **real_tf(m_, b, d),
        "tf_eos_final_top1_accuracy": 1.0,
        "tf_eos_early_top1_false_positive_count": 0,
    }
    mod.load_codec_items = lambda *a, **k: rows
    mod.enforce_eval_contract = lambda *a, **k: contract
    mod._resolve_speaker_registry = lambda **k: (object(), "configs/speaker_maps/trackn_v2.json")
    mod._ref_texts_from_manifest = lambda ids, m: dict(refs)
    mod.free_running_asr_eval = fake_fr
    mod.load_yaml = lambda p: {"infer": dict(contract["infer"])}

    with tempfile.TemporaryDirectory() as tmp:
        td = Path(tmp)
        # ONE physical checkpoint, deliberately NOT candidate-named, NO payload step.
        one = td / "best.pt"
        torch.save({"model": model.state_dict(), "cfg": cfg}, one)

        paths, shas, steps = [], [], []
        for s in STEPS:
            outd = td / f"dup_{s}"
            argv = [
                "74_eval_discrete_lm.py",
                "--ckpt", str(one),
                "--codec-tokens-dir", str(td / "tokens"),
                "--max-utts", "3", "--n2-gate",
                "--out", str(outd),
                "--wav-dir", str(td / f"wav_{s}"),
                "--eval-contract", str(REAL_CONTRACT),
                "--config", "configs/discrete_speech_lm_d1b_443m.yaml",
                "--checkpoint-step", str(s),
            ]
            old = sys.argv
            buf = io.StringIO()
            try:
                sys.argv = argv
                with contextlib.redirect_stdout(buf):
                    mod.main()
            except BaseException:  # noqa: BLE001
                print(traceback.format_exc())
            finally:
                sys.argv = old
            rp = outd / "discrete_eval_report_v2.json"
            if rp.is_file():
                d = json.loads(rp.read_text(encoding="utf-8"))
                paths.append(rp)
                shas.append(d["checkpoint"]["sha256"])
                steps.append(d["checkpoint"]["step"])

        rec(
            "N1",
            None,
            f"five reports produced from ONE file: steps={steps} "
            f"distinct_ckpt_sha256={len(set(shas))}",
        )
        if len(paths) == 5:
            try:
                sel = mod76.select_d1c_formal_reports(paths)
                rec(
                    "N2",
                    False,
                    "SELECTOR ACCEPTED five reports that all point at the SAME checkpoint "
                    f"sha256={shas[0][:16]} (selected step={sel['selected']['step']}) — "
                    "plan §8.4 'one checkpoint hash/step match per report' is NOT enforced",
                )
            except Exception as exc:  # noqa: BLE001
                rec("N2", True, f"selector refused duplicate-checkpoint set: {type(exc).__name__}: {exc}")
        else:
            rec("N2", None, f"only {len(paths)} reports emitted; cannot test")

        # Does the selector verify checkpoint.sha256 against checkpoint.path?
        if paths:
            d = json.loads(paths[0].read_text(encoding="utf-8"))
            from minispeech.eval.discrete_report_schema import sha256_json, validate_formal_report

            d["checkpoint"]["sha256"] = "0" * 64
            d.pop("report_sha256", None)
            d["report_sha256"] = sha256_json(d)
            try:
                validate_formal_report(d)
                rec(
                    "N3",
                    False,
                    "schema accepted a checkpoint.sha256 that does not match the file at "
                    "checkpoint.path (no on-disk hash verification)",
                )
            except Exception as exc:  # noqa: BLE001
                rec("N3", True, f"schema verified checkpoint.sha256 against disk: {exc}")

    fails = sum(1 for _, st, _ in RESULTS if st == "FAIL")
    print(f"\nSUMMARY fails={fails} total={len(RESULTS)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

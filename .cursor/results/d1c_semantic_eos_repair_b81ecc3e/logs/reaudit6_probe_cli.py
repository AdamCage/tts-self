#!/usr/bin/env python3
"""REAUDIT_ROUND6 — the same refusals driven through the PRODUCTION CLI.

`python scripts/76_select_discrete_checkpoint.py --d1c-formal --formal-reports ...`
as a real subprocess: exit code, message, and whether `checkpoint_selection.json`
was written. Reports are produced by production `74 main()` with the payload read
back from the real checkpoint bytes.

CPU only. No GPU, no commit.
"""

from __future__ import annotations

import contextlib
import copy
import hashlib
import importlib.util
import io
import json
import os
import subprocess
import sys
import tempfile
from dataclasses import asdict
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "src"))
RESULTS = ROOT / ".cursor" / "results" / "d1c_semantic_eos_repair_b81ecc3e"
LOG = RESULTS / "logs" / "reaudit6_probe_cli.log"
STEPS = (250, 375, 500, 625, 750)
CONTRACT_PATH = ROOT / "configs" / "eval_contracts" / "d1c_semantic_eos_v2_0_0.json"
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


def main() -> int:  # noqa: C901, PLR0915
    from minispeech.data.d1b_oracles import load_eval_contract
    from minispeech.eval.discrete_report_schema import rows_sha256, sha256_json
    from minispeech.frontend.normalize import normalize_for_asr
    from minispeech.models.discrete_speech_lm import (
        DiscreteSpeechLM,
        DiscreteSpeechLMConfig,
    )

    mod = _load_script("74_eval_discrete_lm.py")
    contract = load_eval_contract(CONTRACT_PATH)
    live_hashes = {
        "eval_contract": str(contract["sha256"]),
        "recipe": str(contract["recipe_sha256"]),
        "recipe_source_manifest": str(contract["recipe_source_manifest_sha256"]),
        "run_manifest": "r" * 64,
    }

    cfg_tiny = DiscreteSpeechLMConfig.tiny()
    cfg_tiny.termination_mode = "semantic_eos_v1"
    cfg_tiny.eos_id = cfg_tiny.codebook_size
    cfg_tiny.codec_vocab_size = cfg_tiny.codebook_size
    cfg_tiny.semantic_vocab_size = cfg_tiny.codebook_size + 1
    model = DiscreteSpeechLM(cfg_tiny)
    model_cfg = dict(asdict(cfg_tiny))
    model_cfg["termination"] = {
        "mode": "semantic_eos_v1",
        "eos_id": int(cfg_tiny.eos_id),
        "semantic_vocab_size": int(cfg_tiny.semantic_vocab_size),
        "codec_vocab_size": int(cfg_tiny.codec_vocab_size),
    }
    cfg = {"model": model_cfg, "data": {"pad_token_id": 0}, "infer": dict(contract["infer"])}
    state = {k: v.detach().clone() for k, v in model.state_dict().items()}

    utts = ["u0", "u1", "u2"]
    refs = dict.fromkeys(utts, "привет мир")
    rows = [
        {
            "utt_id": u,
            "input_ids": torch.tensor([3 + i, 4 + i, 5 + i]),
            "codes": torch.randint(0, int(cfg_tiny.codebook_size), (4 + i, 16)),
            "speaker_id": i % 2,
        }
        for i, u in enumerate(utts)
    ]
    wav_seed = {"n": 0}
    hyp_for = {
        "model": "привет мир",
        "copy_synth": "привет мир",
        "wrong_utt": "щщщщщщ щщщ",
        "random_codes": "щщщщщщщщщщ",
        "shuffle_text": "щщщщщщ щщщ",
        "mask_text": "щщщщщщщщщщ",
    }

    def fake_free_running(model_, rows_, **kw):
        condition = kw["condition"]
        out_wav_dir = Path(kw["out_wav_dir"])
        icfg = dict(
            mod.resolve_infer_settings(
                cfg,
                infer_overrides=kw.get("infer_overrides"),
                max_frames_override=kw["max_frames"],
                codebook_size=int((cfg.get("model") or {}).get("codebook_size", 2048)),
            )
        )
        icfg["termination_mode"] = "semantic_eos_v1"
        rows_out, gen_codes, cers, edit_counts = [], [], [], []
        for i, r in enumerate(rows_):
            uid = str(r["utt_id"])
            wav_seed["n"] += 1
            pcm = np.linspace(-0.3, 0.3, 2400 + wav_seed["n"], dtype=np.float32)
            wav = out_wav_dir / f"{i:04d}_{uid}.wav"
            wav.parent.mkdir(parents=True, exist_ok=True)
            sf.write(str(wav), pcm, 24000)
            hyp = hyp_for[condition]
            ref_n, hyp_n = normalize_for_asr(refs[uid]), normalize_for_asr(hyp)
            cer = float(mod.compute_cer([ref_n], [hyp_n]))
            edit_counts.append(mod.edits_from_jiwer_chars(ref_n, hyp_n))
            n_frames = int(r["codes"].shape[0])
            codes_t = torch.randint(0, 2048, (n_frames, 16))
            gen_codes.append(codes_t)
            cers.append(cer)
            rows_out.append(
                {
                    "utt_id": uid, "ordinal": i, "condition": condition,
                    "n_frames": n_frames, "gt_frames": int(r["codes"].shape[0]),
                    "cer": cer, "ref": refs[uid], "hyp": hyp, "wav": str(wav),
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
                "cb0": mod.cb0_stats(gen_codes), "rows": rows_out,
                "edit_counts": edit_counts,
                "infer": {k: v for k, v in icfg.items() if not str(k).startswith("_")},
            }
        )

    def load_from_disk(p, d):
        payload = torch.load(Path(p), map_location="cpu", weights_only=False)
        return model, copy.deepcopy(cfg), (payload if isinstance(payload, dict) else {})

    mod._load_model = load_from_disk
    mod.load_codec_items = lambda *a, **k: rows
    mod.enforce_eval_contract = lambda *a, **k: contract
    mod._resolve_speaker_registry = lambda **k: (object(), "configs/speaker_maps/trackn_v2.json")
    mod._ref_texts_from_manifest = lambda ids, man: dict(refs)
    mod.free_running_asr_eval = fake_free_running
    mod.load_yaml = lambda p: {"infer": dict(contract["infer"])}

    def run_main(ckpt: Path, out: Path, wav: Path):
        argv = [
            "74_eval_discrete_lm.py", "--ckpt", str(ckpt),
            "--codec-tokens-dir", str(out / "tokens"), "--max-utts", "3", "--n2-gate",
            "--out", str(out), "--wav-dir", str(wav),
            "--eval-contract", str(CONTRACT_PATH),
            "--config", "configs/discrete_speech_lm_d1b_443m.yaml",
        ]
        old, buf, err = sys.argv, io.StringIO(), ""
        try:
            sys.argv = argv
            with contextlib.redirect_stdout(buf):
                try:
                    mod.main()
                except SystemExit as exc:
                    err = str(exc)
                except ValueError as exc:
                    err = f"ValueError: {exc}"
        finally:
            sys.argv = old
        return err or buf.getvalue()

    def make_eligible(path: Path) -> dict:
        d = json.loads(path.read_text(encoding="utf-8"))
        d["teacher_forced_metrics"]["tf_eos_final_top1_accuracy"] = 1.0
        d["teacher_forced_metrics"]["tf_eos_early_top1_false_positive_count"] = 0
        for cond in d["conditions"].values():
            for row in cond["rows"]:
                row["cer"] = 0.0
                row["catastrophic"] = False
                row["gt_frames"] = max(1, int(row.get("gt_frames") or 1))
                row["n_frames"] = int(row["gt_frames"])
                gen = row.setdefault("generation", {})
                gen.update(
                    termination_reason="learned_eos",
                    n_decoded_frames=int(row["n_frames"]),
                    early_eos_suppressed_count=0,
                    max_cb0_run=2,
                )
            cond["rows_sha256"] = rows_sha256(cond["rows"])
        d["gates"]["text_usage"] = {"pass": True}
        d["gates"]["runaway_or_repetition"] = False
        d["summaries"]["phrase_loop_repetition"] = False
        d["summaries"]["runaway_or_repetition"] = False
        d.pop("report_sha256", None)
        d["report_sha256"] = sha256_json(d)
        path.write_text(json.dumps(d, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return d

    def reseal(d: dict, path: Path) -> Path:
        d.pop("report_sha256", None)
        d["report_sha256"] = sha256_json(d)
        path.write_text(json.dumps(d, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return path

    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT / "src")
    env["CUDA_VISIBLE_DEVICES"] = ""

    def cli(reports: list[Path], out: Path, extra: list[str] | None = None):
        if out.exists():
            out.unlink()
        cmd = [
            sys.executable, str(ROOT / "scripts" / "76_select_discrete_checkpoint.py"),
            "--d1c-formal", "--out", str(out), "--formal-reports", *[str(p) for p in reports],
        ] + (extra or [])
        proc = subprocess.run(cmd, cwd=str(ROOT), env=env, capture_output=True, text=True)
        return proc.returncode, (proc.stdout + proc.stderr).strip(), out.exists()

    td = Path(tempfile.mkdtemp(prefix="d1c_r6_cli_"))
    try:
        legit, ck_files = [], []
        for s in STEPS:
            ck = td / "legit" / f"candidate_step_{s:04d}.pt"
            ck.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {"model": state, "cfg": cfg, "step": s, "optimizer_step": s,
                 "hashes": dict(live_hashes), "marker": f"step-{s}"},
                ck,
            )
            ck_files.append(ck)
            outd = td / f"legit_{s}"
            msg = run_main(ck, outd, td / f"wav_{s}")
            rp = outd / "discrete_eval_report_v2.json"
            if not rp.is_file():
                rec("K0", False, f"legit step={s} refused by evaluator: {msg[:140]}")
                break
            make_eligible(rp)
            legit.append(rp)
        else:
            rec("K0", len(legit) == 5, "five legit reports emitted by production main()")

        sel_out = td / "checkpoint_selection.json"
        rc, msg, wrote = cli(legit, sel_out)
        sel_step = json.loads(sel_out.read_text())["selected"]["step"] if wrote else None
        rec("K1", rc == 0 and wrote, f"CLI accepts the legit set: rc={rc} wrote={wrote} step={sel_step}")

        # mixed run identity
        mixed = []
        for i, p in enumerate(legit):
            d = json.loads(p.read_text(encoding="utf-8"))
            d["checkpoint"]["hashes"]["run_manifest"] = "abcde"[i] * 64
            mixed.append(reseal(d, td / f"cli_mixed_{i}.json"))
        rc, msg, wrote = cli(mixed, sel_out)
        rec("K2", rc != 0 and not wrote, f"CLI refuses mixed run identity: rc={rc} wrote={wrote}")
        rec("K2a", "mixed checkpoint.hashes" in msg, f"message: {msg.splitlines()[-1][:110]!r}")

        # stripped checkpoint.hashes
        d = json.loads(legit[0].read_text(encoding="utf-8"))
        d["checkpoint"].pop("hashes")
        nh = reseal(d, td / "cli_nohash.json")
        rc, msg, wrote = cli([nh, *legit[1:]], sel_out)
        rec("K3", rc != 0 and not wrote, f"CLI refuses report without checkpoint.hashes: rc={rc} wrote={wrote}")
        rec("K3a", "checkpoint.hashes missing" in msg, f"message: {msg.splitlines()[-1][:110]!r}")

        # basename alias
        d = json.loads(legit[0].read_text(encoding="utf-8"))
        alias = td / "best.pt"
        alias.write_bytes(ck_files[0].read_bytes())
        d["checkpoint"]["path"] = str(alias)
        d["checkpoint"]["sha256"] = hashlib.sha256(alias.read_bytes()).hexdigest()
        ba = reseal(d, td / "cli_base.json")
        rc, msg, wrote = cli([ba, *legit[1:]], sel_out)
        rec("K4", rc != 0 and not wrote, f"CLI refuses best.pt basename alias: rc={rc} wrote={wrote}")
        rec("K4a", "basename" in msg, f"message: {msg.splitlines()[-1][:110]!r}")

        # duplicate sha
        dup = []
        src = json.loads(legit[0].read_text(encoding="utf-8"))
        for i, s in enumerate(STEPS):
            d = copy.deepcopy(src)
            clone = td / "dup" / f"candidate_step_{s:04d}.pt"
            clone.parent.mkdir(parents=True, exist_ok=True)
            clone.write_bytes(ck_files[0].read_bytes())
            d["checkpoint"].update(path=str(clone), step=int(s),
                                   sha256=hashlib.sha256(clone.read_bytes()).hexdigest())
            dup.append(reseal(d, td / f"cli_dup_{i}.json"))
        rc, msg, wrote = cli(dup, sel_out)
        rec("K5", rc != 0 and not wrote, f"CLI refuses duplicate checkpoint.sha256: rc={rc} wrote={wrote}")

        # flat/manual inputs still refused in formal mode
        proc = subprocess.run(
            [sys.executable, str(ROOT / "scripts" / "76_select_discrete_checkpoint.py"),
             "--d1c-formal", "--scored", str(legit[0]), "--out", str(sel_out)],
            cwd=str(ROOT), env=env, capture_output=True, text=True,
        )
        rec("K6", proc.returncode != 0, f"CLI refuses flat/manual inputs in formal mode: rc={proc.returncode}")

        # RESIDUAL at the CLI: five re-saves of one weight state, stamped steps
        sem = []
        for s in STEPS:
            ck = td / "semdup" / f"candidate_step_{s:04d}.pt"
            ck.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {"model": state, "cfg": cfg, "step": s, "optimizer_step": s,
                 "hashes": dict(live_hashes), "pad": "s" * (s % 7 + 1)},
                ck,
            )
            outd = td / f"sem_{s}"
            msg = run_main(ck, outd, td / f"wsem_{s}")
            rp = outd / "discrete_eval_report_v2.json"
            if not rp.is_file():
                rec("K7", None, f"semantic duplicate step={s} refused: {msg[:110]}")
                break
            make_eligible(rp)
            sem.append(rp)
        else:
            rc, msg, wrote = cli(sem, sel_out)
            rec(
                "K7", None,
                "RESIDUAL at production CLI: five re-saves of ONE weight state with "
                f"fabricated payload steps -> rc={rc} wrote={wrote}",
            )
    finally:
        import shutil

        shutil.rmtree(td, ignore_errors=True)

    n_pass = sum(1 for line in LINES if " PASS " in line)
    n_fail = sum(1 for line in LINES if " FAIL " in line)
    summary = f"SUMMARY pass={n_pass} fail={n_fail}"
    LINES.append(summary)
    print(summary, flush=True)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    LOG.write_text("\n".join(LINES) + "\n", encoding="utf-8")
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())

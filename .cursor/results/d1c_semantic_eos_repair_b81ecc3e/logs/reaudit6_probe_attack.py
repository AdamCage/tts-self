#!/usr/bin/env python3
"""REAUDIT_ROUND6 — independent adversarial probe of the D1c round-6 INFRA repair.

Differences from the repair agent's own probe (`repair_round6_probe_rename.py`):

* the checkpoint payload is **read back from the real file bytes** on every
  `_load_model` call (the repair probe injected a payload factory, which is a
  self-fulfilling seam: it can hand `main()` a payload the file does not have);
* the **real, unpatched** `validate_formal_report` runs before every selector
  call (the repair probe monkeypatched it to a pass-through for E1);
* the real `_load_model` is exercised once end-to-end (round-5 LOW #6);
* extra attack shapes the repair probe never tried: wrong payload step,
  placeholder hash values, mixed run identity across candidates,
  hashes-present-but-report-hand-edited, semantic duplicates with stamped steps,
  and the round-4 CLI invent attack re-run.

CPU only. No GPU, no seed invented, no commit, no training.
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
from dataclasses import asdict
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "src"))
RESULTS = ROOT / ".cursor" / "results" / "d1c_semantic_eos_repair_b81ecc3e"
LOG = RESULTS / "logs" / "reaudit6_probe_attack.log"

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


def main() -> int:  # noqa: C901, PLR0912, PLR0915
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

    mod = _load_script("74_eval_discrete_lm.py")
    mod76 = _load_script("76_select_discrete_checkpoint.py")
    real_load_model = mod._load_model
    contract = load_eval_contract(CONTRACT_PATH)
    live_contract_sha = str(contract["sha256"])
    live_hashes = {
        "eval_contract": live_contract_sha,
        "recipe": str(contract["recipe_sha256"]),
        "recipe_source_manifest": str(contract["recipe_source_manifest_sha256"]),
        "run_manifest": "r" * 64,
    }

    # ---------------------------------------------------------------- model
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
    cfg = {
        "model": model_cfg,
        "data": {"pad_token_id": 0},
        "infer": dict(contract["infer"]),
    }
    state = {k: v.detach().clone() for k, v in model.state_dict().items()}

    def save_ckpt(path: Path, *, step: int | None, hashes: dict | None, junk: str = "") -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload: dict = {"model": state, "cfg": cfg}
        if step is not None:
            payload["step"] = int(step)
            payload["optimizer_step"] = int(step)
        if hashes is not None:
            payload["hashes"] = dict(hashes)
        if junk:
            payload["junk"] = junk
        torch.save(payload, path)

    # ------------------------------------------------- F0: real _load_model
    with tempfile.TemporaryDirectory(prefix="d1c_r6_real_") as td0:
        p0 = Path(td0) / "candidate_step_0500.pt"
        save_ckpt(p0, step=500, hashes=live_hashes, junk="real")
        try:
            m0, c0, pay0 = real_load_model(p0, torch.device("cpu"))
            ok = (
                isinstance(pay0, dict)
                and int(pay0.get("step") or 0) == 500
                and isinstance(pay0.get("hashes"), dict)
            )
            rec(
                "F0",
                ok,
                "REAL _load_model returns payload with step=%s hashes_keys=%s "
                "(round-5 LOW #6 now covered live)"
                % (pay0.get("step"), sorted((pay0.get("hashes") or {}).keys())),
            )
            rec(
                "F1",
                mod.resolve_checkpoint_step(
                    ckpt_path=p0, payload=pay0, allowed_steps=STEPS, require=True
                )
                == 500,
                "real payload resolves step=500 on the formal path",
            )
        except Exception as exc:  # noqa: BLE001
            rec("F0", False, f"REAL _load_model failed: {type(exc).__name__}: {exc}")

    # ------------------------------------------------- unit-level refusals
    for fid, kwargs, needle in (
        (
            "A1",
            dict(ckpt_path="runs/x/candidate_step_0625.pt", payload={"model": {}}),
            "payload['step']",
        ),
        (
            "A2",
            dict(ckpt_path="runs/x/candidate_step_0625.pt", payload=None),
            "payload['step']",
        ),
        (
            "A3",
            dict(ckpt_path="runs/x/candidate_step_0625.pt", payload={"step": 250}),
            "mismatch",
        ),
        (
            "A4",
            dict(
                ckpt_path="runs/x/candidate_step_0625.pt",
                payload={"model": {}},
                cli_step=625,
            ),
            "cannot invent",
        ),
    ):
        got, err = None, ""
        try:
            got = mod.resolve_checkpoint_step(allowed_steps=STEPS, require=True, **kwargs)
        except SystemExit as exc:
            err = str(exc)
        rec(fid, got is None and needle in err, f"refused ({needle}): {err[:110]!r}")

    # non-formal must stay permissive (documented contract)
    rec(
        "A5",
        mod.resolve_checkpoint_step(
            ckpt_path="runs/x/candidate_step_0625.pt", payload={}, require=False
        )
        == 625,
        "documented: require=False still derives 625 from filename (non-formal)",
    )

    for fid, payload, expected, needle in (
        ("B1", {"step": 500}, None, "hashes"),
        ("B2", {"step": 500, "hashes": {}}, None, "hashes"),
        (
            "B3",
            {"step": 500, "hashes": {**live_hashes, "run_manifest": "0" * 64}},
            None,
            "hashes",
        ),
        (
            "B4",
            {"step": 500, "hashes": {**live_hashes, "recipe_source_manifest": "  "}},
            None,
            "hashes",
        ),
        (
            "B5",
            {"step": 500, "hashes": {k: v for k, v in live_hashes.items() if k != "recipe"}},
            None,
            "hashes",
        ),
    ):
        err = ""
        try:
            mod.require_checkpoint_payload_hashes(
                payload, expected_eval_contract=live_contract_sha
            )
        except SystemExit as exc:
            err = str(exc)
        rec(fid, needle in err, f"refused: {err[:110]!r}")

    err = ""
    try:
        mod.require_checkpoint_payload_hashes(
            {"step": 500, "hashes": {**live_hashes, "eval_contract": "9" * 64}},
            expected_eval_contract=live_contract_sha,
        )
    except SystemExit as exc:
        err = str(exc)
    rec("B6", "eval_contract mismatch" in err, f"contract mismatch refused: {err[:110]!r}")
    rec(
        "B7",
        mod.require_checkpoint_payload_hashes(
            {"step": 500, "hashes": dict(live_hashes)},
            expected_eval_contract=live_contract_sha,
        )["run_manifest"]
        == live_hashes["run_manifest"],
        "legit hashes accepted (not a blanket refusal)",
    )

    # --------------------------------------------------------- main() rig
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
        return mod.enrich_model_report(
            {
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
        )

    def install_stubs() -> None:
        """Outer deps only. The payload always comes from the REAL file bytes."""

        def load_from_disk(p, d):
            payload = torch.load(Path(p), map_location="cpu", weights_only=False)
            return model, copy.deepcopy(cfg), (payload if isinstance(payload, dict) else {})

        mod._load_model = load_from_disk
        mod.load_codec_items = lambda *a, **k: rows
        mod.enforce_eval_contract = lambda *a, **k: contract
        mod._resolve_speaker_registry = lambda **k: (
            object(),
            "configs/speaker_maps/trackn_v2.json",
        )
        mod._ref_texts_from_manifest = lambda ids, man: dict(refs)
        mod.free_running_asr_eval = fake_free_running
        mod.load_yaml = lambda p: {"infer": dict(contract["infer"])}

    install_stubs()

    def run_main(ckpt: Path, out: Path, wav: Path, *, cli_step: int | None = None):
        argv = [
            "74_eval_discrete_lm.py",
            "--ckpt", str(ckpt),
            "--codec-tokens-dir", str(out / "tokens"),
            "--max-utts", "3",
            "--n2-gate",
            "--out", str(out),
            "--wav-dir", str(wav),
            "--eval-contract", str(CONTRACT_PATH),
            "--config", "configs/discrete_speech_lm_d1b_443m.yaml",
        ]
        if cli_step is not None:
            argv += ["--checkpoint-step", str(cli_step)]
        old, buf, err, rc = sys.argv, io.StringIO(), "", 1
        try:
            sys.argv = argv
            with contextlib.redirect_stdout(buf):
                try:
                    rc = int(mod.main() or 0)
                except SystemExit as exc:
                    err = str(exc)
                    rc = 1 if exc.code not in (0, None) else 0
                except ValueError as exc:
                    err = f"ValueError: {exc}"
                    rc = 1
        finally:
            sys.argv = old
        return rc, (err or buf.getvalue())

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
        validate_formal_report(d)  # REAL schema, no monkeypatch
        path.write_text(json.dumps(d, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return d

    def reseal(d: dict, path: Path) -> Path:
        d.pop("report_sha256", None)
        d["report_sha256"] = sha256_json(d)
        path.write_text(json.dumps(d, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return path

    with tempfile.TemporaryDirectory(prefix="d1c_r6_atk_") as td_s:
        td = Path(td_s)

        # ---- C: rename/re-save, NO payload step (round-5 M2 attack shape) ----
        emitted, detail = [], []
        for s in STEPS:
            ck = td / "rename" / f"candidate_step_{s:04d}.pt"
            save_ckpt(ck, step=None, hashes=live_hashes, junk="x" * (s % 17 + 1))
            outd = td / f"atk_{s}"
            _, msg = run_main(ck, outd, td / f"wav_atk_{s}")
            rp = outd / "discrete_eval_report_v2.json"
            if rp.is_file():
                emitted.append(rp)
            detail.append(msg[:70])
        rec("C1", len(emitted) == 0, f"rename attack (no payload step): {len(emitted)}/5 reports")
        rec(
            "C1a",
            all("payload['step']" in m for m in detail),
            f"all five refusals cite payload['step']: {detail[0]!r}",
        )
        leaked = sorted(p.name for p in td.glob("atk_*/n2_gate.json"))
        rec("C1b", not leaked, f"no n2_gate.json leaked from refused runs: {leaked}")

        # ---- C2: rename attack with a WRONG payload step ----
        emitted2, detail2 = [], []
        for s in STEPS:
            ck = td / "wrongstep" / f"candidate_step_{s:04d}.pt"
            save_ckpt(ck, step=250, hashes=live_hashes, junk="y" * (s % 13 + 1))
            outd = td / f"wrong_{s}"
            _, msg = run_main(ck, outd, td / f"wav_wrong_{s}")
            rp = outd / "discrete_eval_report_v2.json"
            if rp.is_file():
                emitted2.append(int(json.loads(rp.read_text(encoding="utf-8"))["checkpoint"]["step"]))
            detail2.append(msg[:70])
        rec(
            "C2",
            emitted2 == [250],
            f"wrong payload step: only the self-consistent 250 survives; emitted_steps={emitted2}",
        )
        rec(
            "C2a",
            sum(1 for m in detail2 if "mismatch" in m) == 4,
            "4/5 refused with payload-vs-filename mismatch",
        )

        # ---- C3: correct payload step but NO hashes ----
        emitted3, detail3 = [], []
        for s in STEPS:
            ck = td / "nohash" / f"candidate_step_{s:04d}.pt"
            save_ckpt(ck, step=s, hashes=None, junk="z" * (s % 11 + 1))
            outd = td / f"nohash_{s}"
            _, msg = run_main(ck, outd, td / f"wav_nohash_{s}")
            if (outd / "discrete_eval_report_v2.json").is_file():
                emitted3.append(s)
            detail3.append(msg[:80])
        rec("C3", not emitted3, f"payload hashes missing: {len(emitted3)}/5 reports")
        rec("C3a", all("hashes" in m for m in detail3), f"refusal cites hashes: {detail3[0]!r}")

        # ---- C4: hashes present but eval_contract foreign ----
        ck = td / "badhash" / "candidate_step_0500.pt"
        save_ckpt(ck, step=500, hashes={**live_hashes, "eval_contract": "9" * 64})
        outd = td / "badhash_out"
        _, msg = run_main(ck, outd, td / "wav_badhash")
        rec(
            "C4",
            not (outd / "discrete_eval_report_v2.json").is_file()
            and "eval_contract mismatch" in msg,
            f"foreign eval_contract refused: {msg[:100]!r}",
        )

        # ---- C5: placeholder run_manifest ----
        ck = td / "placeholder" / "candidate_step_0500.pt"
        save_ckpt(ck, step=500, hashes={**live_hashes, "run_manifest": "0" * 64})
        outd = td / "placeholder_out"
        _, msg = run_main(ck, outd, td / "wav_placeholder")
        rec(
            "C5",
            not (outd / "discrete_eval_report_v2.json").is_file(),
            f"placeholder run_manifest refused: {msg[:100]!r}",
        )

        # ---- C6: round-4 CLI invent attack, re-run ----
        best = td / "r4" / "best.pt"
        save_ckpt(best, step=None, hashes=live_hashes, junk="r4")
        r4_reports, r4_msgs = [], []
        for s in STEPS:
            outd = td / f"r4_{s}"
            _, msg = run_main(best, outd, td / f"wav_r4_{s}", cli_step=s)
            if (outd / "discrete_eval_report_v2.json").is_file():
                r4_reports.append(s)
            r4_msgs.append(msg[:70])
        rec("C6", not r4_reports, f"round-4 CLI invent attack still closed: {len(r4_reports)}/5")
        rec("C6a", all("cannot invent" in m for m in r4_msgs), f"cites 'cannot invent': {r4_msgs[0]!r}")

        # ---- D: legitimate five candidates ----
        legit_paths, legit_steps, legit_ck = [], [], []
        for s in STEPS:
            ck = td / "legit" / f"candidate_step_{s:04d}.pt"
            save_ckpt(ck, step=s, hashes=live_hashes, junk=f"step-{s}")
            legit_ck.append(ck)
            outd = td / f"legit_{s}"
            _, msg = run_main(ck, outd, td / f"wav_legit_{s}")
            rp = outd / "discrete_eval_report_v2.json"
            if not rp.is_file():
                rec("D1", False, f"legit step={s} refused: {msg[:140]}")
                break
            d = make_eligible(rp)
            legit_paths.append(rp)
            legit_steps.append(int(d["checkpoint"]["step"]))
        else:
            rec("D1", legit_steps == list(STEPS), f"legit main() stamped steps={legit_steps}")
            rec(
                "D1a",
                len({hashlib.sha256(p.read_bytes()).hexdigest() for p in legit_ck}) == 5,
                "five legit candidate files are byte-distinct",
            )
            try:
                sel = mod76.select_d1c_formal_reports(legit_paths)
                rec("D2", sel["selected"]["step"] in STEPS, f"selector ACCEPTS legit set -> step={sel['selected']['step']}")
                rec("D3", len({c["checkpoint_sha256"] for c in sel["candidates"]}) == 5, "5 distinct checkpoint sha256")
            except Exception as exc:  # noqa: BLE001
                rec("D2", False, f"selector refused the LEGIT set: {exc}")

        if len(legit_paths) == 5:
            # ---- E: basename rule through the REAL schema + selector ----
            d = json.loads(legit_paths[0].read_text(encoding="utf-8"))
            alias = td / "best.pt"
            alias.write_bytes(legit_ck[0].read_bytes())
            d["checkpoint"]["path"] = str(alias)
            d["checkpoint"]["sha256"] = hashlib.sha256(alias.read_bytes()).hexdigest()
            bp = reseal(d, td / "e_base.json")
            schema_ok = True
            try:
                validate_formal_report(json.loads(bp.read_text(encoding="utf-8")))
            except Exception as exc:  # noqa: BLE001
                schema_ok = False
                rec("E0", None, f"schema also refused best.pt alias: {exc}")
            if schema_ok:
                rec("E0", None, "schema ACCEPTS a best.pt alias (no candidate_step in name) — selector is the only layer")
            try:
                mod76.select_d1c_formal_reports([bp, *legit_paths[1:]])
                rec("E1", False, "REAL-schema basename attack ACCEPTED by selector")
            except ValueError as exc:
                rec("E1", "basename" in str(exc), f"selector refused basename alias: {exc}")

            # ---- E2: cross-wired basename (step 250 report bound to step-750 file) ----
            d = json.loads(legit_paths[0].read_text(encoding="utf-8"))
            d["checkpoint"]["path"] = str(legit_ck[4])
            d["checkpoint"]["sha256"] = hashlib.sha256(legit_ck[4].read_bytes()).hexdigest()
            xp = reseal(d, td / "e_cross.json")
            try:
                validate_formal_report(json.loads(xp.read_text(encoding="utf-8")))
                rec("E2", False, "schema accepted cross-wired basename")
            except ValueError as exc:
                rec("E2", "!= checkpoint.step" in str(exc), f"schema refused cross-wired basename: {exc}")

            # ---- F: mixed run identity across the five candidates ----
            mixed = []
            for i, p in enumerate(legit_paths):
                d = json.loads(p.read_text(encoding="utf-8"))
                d["checkpoint"]["hashes"]["run_manifest"] = "abcde"[i] * 64
                mixed.append(reseal(d, td / f"f_mixed_{i}.json"))
            for p in mixed:
                validate_formal_report(json.loads(p.read_text(encoding="utf-8")))
            rec("F2", None, "5/5 mixed-identity reports pass the REAL schema — selector is the only layer")
            try:
                mod76.select_d1c_formal_reports(mixed)
                rec("F3", False, "selector ACCEPTED mixed run identity")
            except ValueError as exc:
                rec("F3", "mixed checkpoint.hashes" in str(exc), f"selector refused mixed run identity: {exc}")

            # partial mix: only one candidate foreign
            d = json.loads(legit_paths[2].read_text(encoding="utf-8"))
            d["checkpoint"]["hashes"]["recipe"] = "7" * 64
            pm = reseal(d, td / "f_partial.json")
            try:
                mod76.select_d1c_formal_reports([*legit_paths[:2], pm, *legit_paths[3:]])
                rec("F4", False, "selector ACCEPTED a single foreign run identity")
            except ValueError as exc:
                rec("F4", "mixed checkpoint.hashes" in str(exc), f"selector refused 1-of-5 foreign identity: {str(exc)[:90]}")

            # stripped hashes
            d = json.loads(legit_paths[1].read_text(encoding="utf-8"))
            d["checkpoint"].pop("hashes")
            sp = reseal(d, td / "f_nohash.json")
            try:
                validate_formal_report(json.loads(sp.read_text(encoding="utf-8")))
                rec("F5", False, "schema accepted a report without checkpoint.hashes")
            except ValueError as exc:
                rec("F5", "checkpoint.hashes missing" in str(exc), f"schema refused report w/o hashes: {exc}")

            # ---- G: duplicate sha (round-5 regression) ----
            dup = []
            src = json.loads(legit_paths[0].read_text(encoding="utf-8"))
            for i, s in enumerate(STEPS):
                d = copy.deepcopy(src)
                clone = td / "dup" / f"candidate_step_{s:04d}.pt"
                clone.parent.mkdir(parents=True, exist_ok=True)
                clone.write_bytes(legit_ck[0].read_bytes())
                d["checkpoint"]["path"] = str(clone)
                d["checkpoint"]["step"] = int(s)
                d["checkpoint"]["sha256"] = hashlib.sha256(clone.read_bytes()).hexdigest()
                dup.append(reseal(d, td / f"g_dup_{i}.json"))
            n_schema_ok = 0
            for p in dup:
                try:
                    validate_formal_report(json.loads(p.read_text(encoding="utf-8")))
                    n_schema_ok += 1
                except ValueError:
                    pass
            rec("G1", None, f"{n_schema_ok}/5 duplicate-sha reports pass the REAL schema")
            try:
                mod76.select_d1c_formal_reports(dup)
                rec("G2", False, "selector ACCEPTED duplicate checkpoint.sha256")
            except ValueError as exc:
                rec("G2", "duplicate checkpoint.sha256" in str(exc), f"selector refused duplicate sha: {str(exc)[:90]}")

            # ---- H: forged sha / unreachable path ----
            d = json.loads(legit_paths[3].read_text(encoding="utf-8"))
            d["checkpoint"]["sha256"] = "b" * 64
            hp = reseal(d, td / "h_forged.json")
            try:
                validate_formal_report(json.loads(hp.read_text(encoding="utf-8")))
                rec("H1", False, "schema accepted a forged sha")
            except ValueError as exc:
                rec("H1", "does not match file" in str(exc), f"schema refused forged sha: {str(exc)[:80]}")
            d = json.loads(legit_paths[3].read_text(encoding="utf-8"))
            d["checkpoint"]["path"] = str(td / "gone" / "candidate_step_0625.pt")
            up = reseal(d, td / "h_missing.json")
            schema_let_through = True
            try:
                validate_formal_report(json.loads(up.read_text(encoding="utf-8")))
            except ValueError:
                schema_let_through = False
            try:
                mod76.select_d1c_formal_reports([*legit_paths[:3], up, legit_paths[4]])
                rec("H2", False, "selector ACCEPTED an unreachable checkpoint.path")
            except ValueError as exc:
                rec(
                    "H2",
                    "not readable" in str(exc),
                    f"schema_let_through={schema_let_through}; selector refused unreachable path",
                )

            # ---- I: RESIDUAL — semantic duplicates with stamped steps ----
            sem_paths, sem_ck = [], []
            ok_all = True
            for s in STEPS:
                ck = td / "semdup" / f"candidate_step_{s:04d}.pt"
                # identical weights, only the stamped step + junk differ
                save_ckpt(ck, step=s, hashes=live_hashes, junk="s" * (s % 7 + 1))
                sem_ck.append(ck)
                outd = td / f"sem_{s}"
                _, msg = run_main(ck, outd, td / f"wav_sem_{s}")
                rp = outd / "discrete_eval_report_v2.json"
                if not rp.is_file():
                    ok_all = False
                    rec("I1", None, f"semantic-duplicate step={s} refused: {msg[:90]}")
                    break
                make_eligible(rp)
                sem_paths.append(rp)
            if ok_all and len(sem_paths) == 5:
                w = {
                    hashlib.sha256(
                        b"".join(
                            torch.load(c, map_location="cpu", weights_only=False)["model"][k]
                            .numpy()
                            .tobytes()
                            for k in sorted(state)
                        )
                    ).hexdigest()
                    for c in sem_ck
                }
                try:
                    sel = mod76.select_d1c_formal_reports(sem_paths)
                    rec(
                        "I1",
                        None,
                        "RESIDUAL: five re-saves of ONE weight state with fabricated payload "
                        f"steps ACCEPTED (selected step={sel['selected']['step']}, "
                        f"distinct_weight_digests={len(w)}/5)",
                    )
                except ValueError as exc:
                    rec("I1", None, f"semantic duplicates refused: {str(exc)[:110]}")

            # ---- J: report-level hash/provenance layering ----
            j = []
            for i, p in enumerate(legit_paths):
                d = json.loads(p.read_text(encoding="utf-8"))
                d["checkpoint"]["hashes"]["eval_contract"] = "8" * 64
                j.append(reseal(d, td / f"j_{i}.json"))
            try:
                mod76.select_d1c_formal_reports(j)
                rec(
                    "J1",
                    None,
                    "OBSERVATION: selector does not cross-check checkpoint.hashes.eval_contract "
                    "against provenance.eval_contract_sha256 (consistent forgery accepted)",
                )
            except ValueError as exc:
                rec("J1", None, f"selector refused report-level contract skew: {str(exc)[:100]}")

    n_pass = sum(1 for line in LINES if " PASS " in line)
    n_fail = sum(1 for line in LINES if " FAIL " in line)
    n_info = sum(1 for line in LINES if " INFO " in line)
    summary = f"SUMMARY pass={n_pass} fail={n_fail} info={n_info}"
    LINES.append(summary)
    print(summary, flush=True)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    LOG.write_text("\n".join(LINES) + "\n", encoding="utf-8")
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())

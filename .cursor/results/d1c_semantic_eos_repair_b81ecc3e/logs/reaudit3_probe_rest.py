#!/usr/bin/env python3
"""REAUDIT ROUND 3 — part 2.

Covers, with live execution:
  M1  multi-batch production aggregation arithmetic
  M2  aggregation fail-closed cases
  M3  evaluator -> selector formal interface (plan §8.4) with REAL emitter output
  M4  selector refuses absent text_usage
  M5  runtime recipe_source_manifest verify actually fires
  M6  D1c subset aliases
  H   published DEV32 provenance vs live artifacts
  R2  FAIL #2 contract / recipe binding regression
  R7  FAIL #7 resume hash regression
  RX  residual grads / select8 2+6 / immediate-EOS empty codes
"""
from __future__ import annotations

import copy
import hashlib
import importlib.util
import io
import json
import os
import re
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

from minispeech.data.collate_codec_seq import collate_codec_seq  # noqa: E402
from minispeech.data.d1b_oracles import (  # noqa: E402
    SUBSET_ALIASES,
    enforce_eval_contract,
    eval_contract_content_sha256,
    load_eval_contract,
    load_frozen_utt_ids,
    verify_recipe_source_manifest,
)
from minispeech.eval.discrete_report_schema import (  # noqa: E402
    sha256_json,
    validate_formal_report,
)
from minispeech.models.discrete_speech_lm import (  # noqa: E402
    DiscreteSpeechLM,
    DiscreteSpeechLMConfig,
)
from minispeech.utils import load_yaml  # noqa: E402

CONTRACT_PATH = ROOT / "configs/eval_contracts/d1c_semantic_eos_v2_0_0.json"
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


# =========================================================================
# M1 / M2 — aggregation
# =========================================================================
def probe_aggregation(mod) -> None:
    torch.manual_seed(3)
    m = semantic_model()

    def batch(n_utt: int, base: int):
        return collate_codec_seq(
            [
                {
                    "utt_id": f"x{base}_{i}",
                    "input_ids": torch.tensor([3 + i, 4 + i]),
                    "codes": torch.randint(0, m.codebook_size, (2 + i, m.n_codebooks)),
                    "speaker_id": i % 2,
                }
                for i in range(n_utt)
            ],
            pad_token_id=0,
            speech_eos_id=m.eos_id,
            termination_mode="semantic_eos_v1",
        )

    per = [
        mod.teacher_forced_ce(m, batch(2, 0), torch.device("cpu")),
        mod.teacher_forced_ce(m, batch(3, 1), torch.device("cpu")),
        mod.teacher_forced_ce(m, batch(1, 2), torch.device("cpu")),
    ]
    agg = mod.aggregate_teacher_forced_batches(per)
    exp_acc = sum(float(p["tf_eos_final_top1_accuracy"]) for p in per) / 3
    exp_fp = sum(int(p["tf_eos_early_top1_false_positive_count"]) for p in per)
    rec("M1a", agg["n_batches"] == 3, f"n_batches={agg['n_batches']}")
    rec(
        "M1b",
        abs(float(agg["tf_eos_final_top1_accuracy"]) - exp_acc) < 1e-9,
        f"multi-batch EOS accuracy is the batch mean ({agg['tf_eos_final_top1_accuracy']})",
    )
    rec(
        "M1c",
        int(agg["tf_eos_early_top1_false_positive_count"]) == exp_fp,
        f"early-EOS false positives are summed ({agg['tf_eos_early_top1_false_positive_count']})",
    )
    rec(
        "M1d",
        agg["teacher_forced_ce_source"] == "eval_computed",
        "source=eval_computed for a fully-measured multi-batch run",
    )
    rec(
        "M1e",
        None,
        "CAVEAT: batch-mean is UNWEIGHTED by utterances (batches of 2/3/1 here); "
        "the published accuracy is not the corpus mean when the last batch is partial. "
        "The strict ==1.0 selector predicate is unaffected.",
    )

    # M2 fail-closed
    mixed = [dict(per[0]), dict(per[1])]
    mixed[1].pop("tf_eos_final_top1_accuracy")
    a2 = mod.aggregate_teacher_forced_batches(mixed)
    rec(
        "M2a",
        a2["teacher_forced_ce_source"] == "unavailable"
        and a2["tf_eos_final_top1_accuracy"] == "unavailable",
        "one batch missing EOS accuracy => whole run unavailable (no partial credit)",
    )
    mixed2 = [dict(per[0]), dict(per[1])]
    mixed2[0]["teacher_forced_ce_source"] = "decode_bundle"
    a3 = mod.aggregate_teacher_forced_batches(mixed2)
    rec(
        "M2b",
        a3["teacher_forced_ce_source"] == "unavailable",
        "mixed per-batch sources => unavailable",
    )
    a4 = mod.aggregate_teacher_forced_batches([])
    rec(
        "M2c",
        a4["teacher_forced_ce_source"] == "unavailable",
        "empty batch list => unavailable (not a fabricated 0.0 pass)",
    )
    proj = mod.teacher_forced_metrics_for_emitter(a4)
    rec(
        "M2d",
        proj["teacher_forced_ce_source"] == "unavailable",
        "projection preserves unavailable",
    )


# =========================================================================
# M3 — evaluator -> selector, using the REAL emitter output
# =========================================================================
def probe_selector_interface(mod, m76) -> None:
    torch.manual_seed(5)
    model = semantic_model()
    contract = load_eval_contract(CONTRACT_PATH)

    with tempfile.TemporaryDirectory() as tmp:
        td = Path(tmp)
        wav = td / "u0.wav"
        sf.write(str(wav), np.linspace(-0.2, 0.2, 2400, dtype=np.float32), 24000)
        ckpt = td / "cand_step250.pt"
        torch.save({"model": {}}, ckpt)

        per = [mod.teacher_forced_ce(model, _mk_batch(model), torch.device("cpu"))]
        tf_ce = mod.teacher_forced_metrics_for_emitter(
            mod.aggregate_teacher_forced_batches(per)
        )

        def row(cer):
            return {
                "utt_id": "u0",
                "cer": cer,
                "ref": "привет",
                "hyp": "привет",
                "n_frames": 10,
                "gt_frames": 10,
                "wav": str(wav),
                "stop_reason": "learned_eos",
                "generation": {
                    "termination_reason": "learned_eos",
                    "n_decoded_frames": 10,
                    "early_eos_suppressed_count": 0,
                    "max_cb0_run": 2,
                },
                "generated_codes_sha256": "d" * 64,
            }

        reports = {
            n: {"rows": [row(0.05 if n == "copy_synth" else 0.05)]}
            for n in (
                "model",
                "copy_synth",
                "wrong_utt",
                "random_codes",
                "shuffle_text",
                "mask_text",
            )
        }
        gate = {
            "verdict": "PASS",
            "criteria": {"macro_cer": 0.05, "n_catastrophic": 0},
            "text_usage_acceptance": {"pass": True},
            "stability_acceptance": {
                "phrase_loop_repetition": False,
                "runaway_or_repetition": False,
                "no_runaway": True,
            },
        }
        # provenance EXACTLY as scripts/74 main() builds it (no checkpoint_step key)
        provenance = {
            "git_sha": "deadbeef",
            "git_dirty": True,
            "subset_label": "d1c_select8",
            "subset_split": "select8",
            "subset_items_sha256": "a" * 64,
            "eval_contract_path": str(CONTRACT_PATH),
            "eval_contract_sha256": contract["sha256"],
            "max_frames": 750,
            "teacher_forced_ce": tf_ce,
        }
        out = mod._emit_d1c_v2_from_gate(
            out_dir=td / "emit",
            contract=contract,
            ckpt=str(ckpt),
            provenance=dict(provenance),
            infer=dict(contract["infer"]),
            teacher_forced_ce=tf_ce,
            ordered_utt_ids=["u0"],
            condition_reports=reports,
            gate=gate,
        )
        rec("M3a", out is not None, f"emitter produced {out}")
        report = json.loads(Path(out).read_text(encoding="utf-8"))
        step = report["checkpoint"]["step"]
        rec(
            "M3b",
            step != 0,
            "checkpoint.step stamped by the PRODUCTION provenance shape "
            f"(main() never sets checkpoint_step, evaluate_n2_gate_d1b never sets "
            f"gate['step']) -> step={step}",
        )

        # Five production-shaped reports -> selector
        paths = []
        for i, want_step in enumerate((250, 375, 500, 625, 750)):
            c = td / f"cand_{want_step}.pt"
            torch.save({"model": {}, "step": want_step}, c)
            o = mod._emit_d1c_v2_from_gate(
                out_dir=td / f"emit_{want_step}",
                contract=contract,
                ckpt=str(c),
                provenance=dict(provenance),
                infer=dict(contract["infer"]),
                teacher_forced_ce=tf_ce,
                ordered_utt_ids=["u0"],
                condition_reports=copy.deepcopy(reports),
                gate=copy.deepcopy(gate),
            )
            paths.append(str(o))
        steps = [
            json.loads(Path(p).read_text(encoding="utf-8"))["checkpoint"]["step"]
            for p in paths
        ]
        rec("M3c", None, f"steps stamped on five production-shaped reports: {steps}")
        try:
            m76.select_d1c_formal_reports(paths)
            rec("M3d", True, "selector accepted five production-shaped reports")
        except Exception as exc:  # noqa: BLE001
            rec(
                "M3d",
                False,
                "evaluator -> selector (plan §8.4) STILL cannot run end-to-end: "
                f"{type(exc).__name__}: {exc}",
            )

        src74 = (ROOT / "scripts/74_eval_discrete_lm.py").read_text(encoding="utf-8")
        step_flags = re.findall(r'ap\.add_argument\(\s*"(--[a-z0-9-]*step[a-z0-9-]*)"', src74)
        rec(
            "M3f",
            not step_flags,
            "scripts/74 exposes NO CLI flag to supply the candidate step "
            f"(found {step_flags}); nothing in production sets provenance"
            "['checkpoint_step'] or gate['step']",
        )
        writers = [
            p
            for p in (ROOT / "scripts").glob("*.py")
            if "checkpoint_step" in p.read_text(encoding="utf-8")
            and p.name != "74_eval_discrete_lm.py"
        ]
        rec("M3g", None, f"other scripts referencing checkpoint_step: {[p.name for p in writers]}")

        # Control: same reports with the step patched in by hand -> selector runs
        fixed = []
        for p, want in zip(paths, (250, 375, 500, 625, 750)):
            data = json.loads(Path(p).read_text(encoding="utf-8"))
            data["checkpoint"]["step"] = want
            data["teacher_forced_metrics"]["tf_eos_final_top1_accuracy"] = 1.0
            data["teacher_forced_metrics"]["tf_eos_early_top1_false_positive_count"] = 0
            data.pop("report_sha256", None)
            data["report_sha256"] = sha256_json(data)
            q = Path(p).with_name(f"fixed_{want}.json")
            q.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
            fixed.append(str(q))
        try:
            res = m76.select_d1c_formal_reports(fixed)
            rec(
                "M3e",
                True,
                "CONTROL: identical reports with step hand-patched + EOS==1.0 select "
                f"-> step={res['selected']['step']} (isolates the missing step stamp)",
            )
        except Exception as exc:  # noqa: BLE001
            rec("M3e", False, f"CONTROL failed: {type(exc).__name__}: {exc}")

        # M4 — absent text_usage must refuse
        noeos = []
        for p, want in zip(paths, (250, 375, 500, 625, 750)):
            data = json.loads(Path(p).read_text(encoding="utf-8"))
            data["checkpoint"]["step"] = want
            data["teacher_forced_metrics"]["tf_eos_final_top1_accuracy"] = 1.0
            data["teacher_forced_metrics"]["tf_eos_early_top1_false_positive_count"] = 0
            data["gates"].pop("text_usage", None)
            data.pop("report_sha256", None)
            data["report_sha256"] = sha256_json(data)
            q = Path(p).with_name(f"notu_{want}.json")
            q.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
            noeos.append(str(q))
        try:
            m76.select_d1c_formal_reports(noeos)
            rec("M4a", False, "absent gates.text_usage still SELECTED (vacuous pass)")
        except ValueError as exc:
            rec("M4a", "no eligible" in str(exc), f"absent gates.text_usage refused: {exc}")


def _mk_batch(m):
    return collate_codec_seq(
        [
            {
                "utt_id": "a",
                "input_ids": torch.tensor([3, 4]),
                "codes": torch.randint(0, m.codebook_size, (2, m.n_codebooks)),
                "speaker_id": 0,
            },
            {
                "utt_id": "b",
                "input_ids": torch.tensor([5, 6, 7]),
                "codes": torch.randint(0, m.codebook_size, (4, m.n_codebooks)),
                "speaker_id": 1,
            },
        ],
        pad_token_id=0,
        speech_eos_id=m.eos_id,
        termination_mode="semantic_eos_v1",
    )


# =========================================================================
# M5 — runtime recipe_source_manifest verification
# =========================================================================
def probe_manifest_runtime() -> None:
    contract = load_eval_contract(CONTRACT_PATH)
    manifest = contract["recipe_source_manifest"]
    mism = verify_recipe_source_manifest(manifest, root=ROOT)
    rec("M5a", mism == [], f"manifest verify from repo root: n_locked={len(manifest)} mismatches={mism}")

    cwd = os.getcwd()
    try:
        os.chdir(ROOT)
        c = enforce_eval_contract(
            None,
            contract_path=CONTRACT_PATH,
            require_clean_sha=False,
        )
        rec("M5b", isinstance(c, dict), "enforce_eval_contract passes from repo root")
    except Exception as exc:  # noqa: BLE001
        rec("M5b", False, f"enforce_eval_contract failed from repo root: {exc}")
    finally:
        os.chdir(cwd)

    # Prove the check is genuinely executed at runtime (not dead code):
    with tempfile.TemporaryDirectory() as tmp:
        try:
            os.chdir(tmp)
            enforce_eval_contract(
                None, contract_path=CONTRACT_PATH, require_clean_sha=False
            )
            rec(
                "M5c",
                False,
                "manifest verify did NOT fire from a foreign cwd (check is inert)",
            )
        except RuntimeError as exc:
            fired = "recipe_source_manifest verify failed" in str(exc)
            rec(
                "M5c",
                fired,
                "manifest verify fires at runtime "
                f"(foreign cwd -> {'refused' if fired else str(exc)[:120]})",
            )
        finally:
            os.chdir(cwd)
    rec(
        "M5d",
        None,
        "CAVEAT: verify_recipe_source_manifest() is called with root=cwd, so "
        "enforce_eval_contract hard-fails whenever the job is launched outside the "
        "repo root (fail-closed, but an operator foot-gun).",
    )


# =========================================================================
# M6 — D1c subset aliases
# =========================================================================
def probe_aliases() -> None:
    d1c = {k: v for k, v in SUBSET_ALIASES.items() if k.startswith("d1c_")}
    rec("M6a", set(d1c) >= {"d1c_dev32", "d1c_full32", "d1c_select8", "d1c_gate24"}, f"d1c aliases: {sorted(d1c)}")
    contract = load_eval_contract(CONTRACT_PATH)
    subsets = contract.get("subsets") or {}
    for alias, ckey in (
        ("d1c_dev32", "dev32"),
        ("d1c_full32", "full32"),
        ("d1c_select8", "select8"),
        ("d1c_gate24", "gate24"),
    ):
        try:
            ids, _payload, digest = load_frozen_utt_ids(alias, verify_hash=True)
        except Exception as exc:  # noqa: BLE001
            rec(f"M6-{alias}", False, f"{alias} failed to load: {exc}")
            continue
        declared = (subsets.get(ckey) or {}).get("items_sha256")
        ok = declared is None or declared == digest
        rec(
            f"M6-{alias}",
            ok,
            f"{alias} -> n={len(ids)} items_sha256={digest[:16]} "
            f"contract[{ckey}]={str(declared)[:16]} match={declared == digest}",
        )
    # D1b aliases must be untouched
    for alias, want in (
        ("dev32", "d1b_dev32.json"),
        ("select8", "d1b_confirm32_v2.json"),
        ("gate24", "d1b_confirm32_v2.json"),
        ("full32", "d1b_confirm32_v2.json"),
    ):
        p = SUBSET_ALIASES[alias][0]
        rec(f"M6b-{alias}", want in p, f"D1b alias {alias} -> {p}")


# =========================================================================
# H — published DEV32 provenance vs live artifacts
# =========================================================================
def probe_docs() -> None:
    res = ROOT / ".cursor/results/d1c_semantic_eos_repair_b81ecc3e"
    dev32 = ROOT / "configs/train_subsets/d1c_dev32.json"
    payload = json.loads(dev32.read_text(encoding="utf-8"))
    live = {
        "file_sha256": payload.get("file_sha256"),
        "items_sha256": payload.get("items_sha256"),
    }
    from minispeech.data.d1b_oracles import items_sha256 as _isha

    for split in ("select8", "gate24"):
        blk = payload.get(split) or payload.get("splits", {}).get(split)
        ids = blk.get("utt_ids") if isinstance(blk, dict) else blk
        if ids:
            live[split] = _isha(list(ids))
    rec("H0", None, "live d1c_dev32: " + json.dumps({k: (v or "")[:16] for k, v in live.items()}))

    wave1 = (res / "WAVE1_REPORT.md").read_text(encoding="utf-8")
    hits = {}
    for key, val in live.items():
        if not val:
            continue
        hits[key] = val in wave1
    rec(
        "H1",
        all(hits.values()),
        "WAVE1_REPORT.md contains every live DEV32 digest: "
        + json.dumps(hits),
    )
    stale = ["4f57a222f4738464", "e32d70aebe34284e", "832ce5fafe2cf9ee"]
    found_stale = [s for s in stale if s in wave1]
    rec("H2", not found_stale, f"round-2 stale digests absent from WAVE1_REPORT.md: {found_stale}")

    ov = json.loads((res / "artifacts/oracle_overlap_report.json").read_text(encoding="utf-8"))
    blob = json.dumps(ov)
    ov_hits = {k: (v in blob) for k, v in live.items() if v}
    rec("H3", all(ov_hits.values()), f"oracle_overlap_report.json matches live: {json.dumps(ov_hits)}")
    ov_stale = [s for s in stale if s in blob]
    rec("H4", not ov_stale, f"stale digests absent from oracle_overlap_report.json: {ov_stale}")
    rec(
        "H5",
        "payload-content" in wave1 or "self-referential" in wave1 or "canonic" in wave1.lower(),
        "WAVE1_REPORT.md states the file_sha256 redefinition caveat",
    )
    # contract digests published in the pack
    contract = load_eval_contract(CONTRACT_PATH)
    live_sha = contract["sha256"]
    rec("H6", live_sha in wave1, f"WAVE1_REPORT.md carries live contract sha {live_sha[:16]}")
    hashes = json.loads((res / "artifacts/repair_round3_hashes.json").read_text(encoding="utf-8"))
    rec(
        "H7",
        live_sha in json.dumps(hashes),
        f"repair_round3_hashes.json carries live contract sha {live_sha[:16]}",
    )
    r3 = (res / "REPAIR_ROUND3.md").read_text(encoding="utf-8")
    rec("H8", live_sha in r3, "REPAIR_ROUND3.md quotes the live contract sha")
    rsm = contract.get("recipe_source_manifest_sha256")
    rec(
        "H9",
        (rsm or "") in r3,
        f"REPAIR_ROUND3.md quotes live recipe_source_manifest_sha256 {str(rsm)[:16]}",
    )


# =========================================================================
# R2 / R7 — regression
# =========================================================================
def probe_regressions() -> None:
    contract = load_eval_contract(CONTRACT_PATH)
    raw = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
    body = {
        k: v
        for k, v in raw.items()
        if k not in ("sha256", "recipe_sha256") and not str(k).startswith("_")
    }
    indep = hashlib.sha256(
        json.dumps(body, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    rec(
        "R2a",
        indep == raw.get("sha256"),
        f"independent content recompute {indep[:16]} == declared {str(raw.get('sha256'))[:16]}",
    )
    rec(
        "R2b",
        eval_contract_content_sha256(raw) == raw.get("sha256"),
        "library recompute agrees",
    )
    tampered = copy.deepcopy(raw)
    tampered["infer"]["max_frames"] = 999
    rec(
        "R2c",
        eval_contract_content_sha256(tampered) != raw.get("sha256"),
        "tampering infer.max_frames changes the content sha",
    )
    mism = verify_recipe_source_manifest(raw["recipe_source_manifest"], root=ROOT)
    rec("R2d", mism == [], f"recipe_source_manifest n_locked={len(raw['recipe_source_manifest'])} mismatches={mism}")
    man_sha = hashlib.sha256(
        json.dumps(
            raw["recipe_source_manifest"], ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()
    rec(
        "R2e",
        man_sha == raw.get("recipe_source_manifest_sha256"),
        f"manifest dict sha {man_sha[:16]} == declared {str(raw.get('recipe_source_manifest_sha256'))[:16]}",
    )
    for w in ("w1", "w2"):
        y = load_yaml(ROOT / f"configs/discrete_speech_lm_d1c_semantic_eos_{w}.yaml")
        t = y.get("train") or {}
        rec(
            f"R2f-{w}",
            t.get("eval_contract_hash") == raw["sha256"]
            and t.get("recipe_hash") == raw["recipe_sha256"]
            and t.get("recipe_source_manifest_hash") == raw["recipe_source_manifest_sha256"],
            f"{w} binds contract={str(t.get('eval_contract_hash'))[:16]} "
            f"recipe={str(t.get('recipe_hash'))[:16]} "
            f"manifest={str(t.get('recipe_source_manifest_hash'))[:16]}",
        )
    v201 = json.loads(
        (ROOT / "configs/eval_contracts/d1c_semantic_eos_v2_0_1.json").read_text(encoding="utf-8")
    )
    rec(
        "R2g",
        v201.get("parent_sha256") == raw["sha256"]
        and v201.get("parent_recipe_sha256") == raw["recipe_sha256"],
        f"v2.0.1 parent pointers repointed at the live 2.0.0 digests "
        f"({str(v201.get('parent_sha256'))[:16]})",
    )
    src72 = (ROOT / "scripts/72_train_discrete_lm.py").read_text(encoding="utf-8")
    rec(
        "R2h",
        'tcfg["eval_contract_hash"] = contract.get("sha256")' not in src72,
        "scripts/72 no longer silently overwrites eval_contract_hash",
    )
    rec(
        "R2i",
        "eval_contract_hash mismatch (fail-closed)" in src72
        and "expected_contract_sha256=str(declared_hash)" in src72,
        "scripts/72 still asserts the declared contract hash before stamping",
    )
    rec(
        "R2j",
        None,
        "RESIDUAL (low, unchanged from round 2): scripts/72 still silently overwrites "
        'tcfg["recipe_hash"] / tcfg["recipe_source_manifest_hash"] from the contract '
        "instead of asserting equality with the declared values",
    )

    # FAIL #7 — resume hashes
    from minispeech.train.discrete_lm_trainer import resolve_repro_hashes

    cwd = os.getcwd()
    os.chdir(ROOT)
    try:
        for w in ("w1", "w2"):
            cfg = load_yaml(ROOT / f"configs/discrete_speech_lm_d1c_semantic_eos_{w}.yaml")
            h = resolve_repro_hashes(cfg)
            nones = sorted(k for k, v in h.items() if v is None)
            rec(f"R7a-{w}", nones == [], f"{w} resolve_repro_hashes inert keys={nones} (n={len(h)})")
        cfg = load_yaml(ROOT / "configs/discrete_speech_lm_d1c_semantic_eos_w1.yaml")
        bad = copy.deepcopy(cfg)
        bad["train"]["eval_contract_hash"] = "a" * 64
        try:
            resolve_repro_hashes(bad)
            rec("R7b", False, "stale eval_contract_hash accepted")
        except Exception as exc:  # noqa: BLE001
            rec("R7b", "mismatch" in str(exc), f"stale eval_contract_hash raises: {str(exc)[:110]}")
        # dropping a declared hash must still resolve (backfill), not disable the check
        for key in (
            "run_manifest_hash",
            "recipe_hash",
            "recipe_source_manifest_hash",
            "eval_contract_hash",
        ):
            c2 = copy.deepcopy(cfg)
            c2["train"].pop(key, None)
            h2 = resolve_repro_hashes(c2)
            nones = sorted(k for k, v in h2.items() if v is None)
            rec(f"R7c-{key}", nones == [], f"dropping train.{key} still resolves all keys (None={nones})")
    finally:
        os.chdir(cwd)

    tr = (ROOT / "src/minispeech/train/discrete_lm_trainer.py").read_text(encoding="utf-8")
    rec(
        "R7d",
        "resume fail-closed: D1c requires" in tr,
        "D1c resume fail-closed message still present in the trainer",
    )


# =========================================================================
# RX — prior PASS spot checks
# =========================================================================
def probe_prior() -> None:
    torch.manual_seed(11)
    m = semantic_model()
    b = _mk_batch(m)
    out = m(
        b["input_ids"],
        b["codes"],
        text_mask=b["text_mask"],
        frame_mask=b["codec_frame_mask"],
        semantic_targets=b["semantic_targets"],
        semantic_mask=b["semantic_mask"],
        speaker_id=b["speaker_id"],
        cb0_weight=0.0,
        eos_loss_weight=0.0,
        residual_weights=torch.ones(m.n_codebooks - 1),
    )
    out.loss["total"].backward()
    ps = [p for p in m.heads.residuals.parameters() if p.requires_grad]
    n = sum(1 for p in ps if p.grad is not None and float(p.grad.abs().sum()) > 0)
    rec("RXa", n == len(ps) and bool(ps), f"residual params with grads (cb0_w=0, eos_w=0): {n}/{len(ps)}")

    m2 = semantic_model()

    class AlwaysEOS(torch.nn.Module):
        def forward(self, h):  # noqa: ANN001
            lg = h.new_full((*h.shape[:-1], m2.semantic_vocab_size), -1.0e9)
            lg[..., m2.eos_id] = 1.0e9
            return lg

    m2.heads.semantic = AlwaysEOS()
    gen = m2.generate(
        torch.tensor([[3, 4, 5]]),
        max_frames=4,
        min_frames=0,
        max_cb0_repeat=100,
        temperature=0.0,
        top_k=0,
    )
    codes = gen.codes if hasattr(gen, "codes") else gen
    lengths = list(gen.lengths.tolist()) if hasattr(gen, "lengths") else None
    rec(
        "RXb",
        tuple(codes.shape)[:2] == (1, 0) and lengths == [0],
        f"immediate EOS -> codes.shape={tuple(codes.shape)} lengths={lengths} (no phantom frame)",
    )

    ids8, _, s8 = load_frozen_utt_ids("d1c_select8", verify_hash=True)
    ids24, _, s24 = load_frozen_utt_ids("d1c_gate24", verify_hash=True)
    ids32, _, s32 = load_frozen_utt_ids("d1c_full32", verify_hash=True)
    rus = sum(1 for u in ids8 if "ruslan" in str(u).lower())
    rec("RXc", len(ids8) == 8 and rus == 2, f"select8 composition: {rus} RUSLAN + {8 - rus} CV")
    rec(
        "RXd",
        sorted(ids8 + ids24) == sorted(ids32) and not (set(ids8) & set(ids24)),
        f"select8 | gate24 == full32 and disjoint (|s|={len(ids8)} |g|={len(ids24)} |f|={len(ids32)})",
    )


def main() -> int:
    mod = load_script("74_eval_discrete_lm.py")
    m76 = load_script("76_select_discrete_checkpoint.py")
    for fn, args in (
        (probe_aggregation, (mod,)),
        (probe_selector_interface, (mod, m76)),
        (probe_manifest_runtime, ()),
        (probe_aliases, ()),
        (probe_docs, ()),
        (probe_regressions, ()),
        (probe_prior, ()),
    ):
        try:
            fn(*args)
        except Exception:  # noqa: BLE001
            rec(fn.__name__, False, "probe section raised")
            traceback.print_exc()
    n_fail = sum(1 for _, s, _ in RESULTS if s == "FAIL")
    print(f"\nSUMMARY pass={sum(1 for _, s, _ in RESULTS if s == 'PASS')} fail={n_fail}")
    print("ROUND3_REAUDIT_REST_" + ("PASS" if n_fail == 0 else "FAIL"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

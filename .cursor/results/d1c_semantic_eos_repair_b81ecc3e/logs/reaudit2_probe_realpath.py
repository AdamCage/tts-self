"""REAUDIT ROUND2 probe H — FAIL #4 on the REAL evaluator call path.

Probe D fed the emitter a hand-built teacher-forced dict. This probe reproduces what
scripts/74 main() actually passes:

    scripts/74:1335-1342   totals = {"loss","ce_cb0","ce_residual"}  # EOS TF metrics dropped
    scripts/74:1426        tf_ce = summary.get("loss")               # bare float
    scripts/74:1558        _emit_d1c_v2_from_gate(teacher_forced_ce=tf_ce)   # not wrapped

and what --score-only passes:

    scripts/74:1005        gate["teacher_forced_ce"] = bundle.get("teacher_forced_ce")
    scripts/74:1010        provenance["score_only"] = True
    scripts/74:1045        _emit_d1c_v2_from_gate(teacher_forced_ce=bundle.get(...))  # not wrapped
"""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import inspect
import json
import os
import re
import sys
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

ROOT = Path("/mnt/c/projects/tts-self-d1c")
sys.path.insert(0, str(ROOT / "src"))
os.chdir(ROOT)

out: list[tuple[str, str, str]] = []


def rec(tag, ok, msg):
    out.append((tag, "PASS" if ok else "FAIL", msg))
    print(f"H{tag} {'PASS' if ok else 'FAIL'} {msg}", flush=True)


def info(tag, msg):
    out.append((tag, "INFO", msg))
    print(f"H{tag} INFO {msg}", flush=True)


def load_script(name):
    p = ROOT / "scripts" / name
    spec = importlib.util.spec_from_file_location(name.replace(".py", ""), p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


m74 = load_script("74_eval_discrete_lm.py")

from minispeech.data.collate_codec_seq import collate_codec_seq  # noqa: E402
from minispeech.data.d1b_oracles import load_eval_contract  # noqa: E402
from minispeech.data.discrete_oracles import load_formal_subset  # noqa: E402
from minispeech.models.discrete_speech_lm import (  # noqa: E402
    DiscreteSpeechLM,
    DiscreteSpeechLMConfig,
)

CONDITIONS = ("model", "copy_synth", "wrong_utt", "random_codes", "shuffle_text", "mask_text")
contract = load_eval_contract(ROOT / "configs/eval_contracts/d1c_semantic_eos_v2_0_0.json")
sel = load_formal_subset("select8", contract=contract)
uids = [str(u) for u in sel["utt_ids"]]

# ------------------------------------------------------------------ H1 static: what main() aggregates
src = (ROOT / "scripts/74_eval_discrete_lm.py").read_text(encoding="utf-8").splitlines()
info("1a", f"scripts/74:1335 -> {src[1334].strip()}")
info("1b", f"scripts/74:1338-1340 -> {src[1337].strip()} / {src[1338].strip()} / {src[1339].strip()}")
info("1c", f"scripts/74:1342 -> {src[1341].strip()}")
info("1d", f"scripts/74:1426 -> {src[1425].strip()}")
rec(
    "1e",
    "tf_eos_final_top1_accuracy" in src[1334],
    "main() teacher-forced aggregation includes tf_eos_final_top1_accuracy "
    f"(actual totals init: {src[1334].strip()})",
)
rec(
    "1f",
    not re.match(r'^\s*tf_ce\s*=\s*summary\.get\("loss"\)\s*$', src[1425]),
    f"main() passes a full TF metrics dict to the emitter (actual: {src[1425].strip()})",
)

# ------------------------------------------------------------------ H2 live: reproduce the aggregation
def semantic_model():
    cfg = DiscreteSpeechLMConfig.tiny()
    cfg.termination_mode = "semantic_eos_v1"
    cfg.eos_id = cfg.codebook_size
    cfg.codec_vocab_size = cfg.codebook_size
    cfg.semantic_vocab_size = cfg.codebook_size + 1
    return DiscreteSpeechLM(cfg)


torch.manual_seed(0)
m = semantic_model()
batch = collate_codec_seq(
    [
        {"utt_id": "a", "input_ids": torch.tensor([3, 4]), "codes": torch.randint(0, m.codebook_size, (2, m.n_codebooks)), "speaker_id": 0},
        {"utt_id": "b", "input_ids": torch.tensor([5, 6, 7]), "codes": torch.randint(0, m.codebook_size, (4, m.n_codebooks)), "speaker_id": 1},
    ],
    pad_token_id=0,
    speech_eos_id=m.eos_id,
    termination_mode="semantic_eos_v1",
)
batch["frame_mask"] = batch["codec_frame_mask"]
per_batch = m74.teacher_forced_ce(m, batch, torch.device("cpu"))
info("2a", f"teacher_forced_ce() per-batch keys = {sorted(per_batch)}")

# EXACT replication of scripts/74:1335-1347
totals = {"loss": 0.0, "ce_cb0": 0.0, "ce_residual": 0.0}
n = 0
for _ in range(1):
    mm = per_batch
    for k in totals:
        totals[k] += mm[k]
    n += 1
summary = {k: v / max(n, 1) for k, v in totals.items()}
summary["n_batches"] = n
summary["n_utts"] = 2
summary["note"] = "…"
info("2b", f"aggregated summary keys = {sorted(summary)}")
rec(
    "2c",
    "tf_eos_final_top1_accuracy" in summary,
    "aggregated summary retains the measured EOS TF metrics "
    f"(dropped keys: {sorted(set(per_batch) - set(summary))})",
)
tf_ce = summary.get("loss")  # scripts/74:1426
info("2d", f"tf_ce passed to the emitter = {tf_ce!r} (type={type(tf_ce).__name__})")
rec("2e", isinstance(tf_ce, dict), f"tf_ce is a metrics dict, not a bare scalar (got {type(tf_ce).__name__})")

# ------------------------------------------------------------------ H3 live: does the real call crash?
tmp = Path(tempfile.mkdtemp(prefix="reaudit2real_"))
wav_dir = tmp / "wavs"
wav_dir.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(3)


def rows_for(condition):
    rows = []
    for i, uid in enumerate(uids):
        p = wav_dir / f"{condition}_{i:04d}.wav"
        sf.write(str(p), (rng.standard_normal(2400) * 0.01).astype(np.float32), 24000)
        rows.append(
            {
                "utt_id": uid,
                "ordinal": i,
                "condition": condition,
                "n_frames": 42,
                "gt_frames": 42,
                "cer": 0.06,
                "ref": "привет мир",
                "hyp": "привет мир",
                "wav": str(p),
                "stop_reason": "learned_eos",
                "generation": {
                    "termination_mode": "semantic_eos_v1",
                    "termination_reason": "learned_eos",
                    "n_decoded_frames": 42,
                    "early_eos_suppressed_count": 0,
                    "max_cb0_run": 3,
                },
                "generated_codes_sha256": hashlib.sha256(f"{condition}{i}".encode()).hexdigest(),
            }
        )
    return rows


reports = {c: {"condition": c, "n": len(uids), "rows": rows_for(c), "infer": dict(contract.get("infer") or {})} for c in CONDITIONS}
ckpt = tmp / "cand.pt"
ckpt.write_bytes(b"fake")
gate = {"verdict": "PASS", "criteria": {"macro_cer": 0.06, "n_catastrophic": 0, "max_abs_log_duration_error": 0.0, "mean_abs_log_duration_error": 0.0}}
prov_main = {
    "git_sha": "deadbeef",
    "git_dirty": False,
    "subset_label": "select8",
    "subset_path": "configs/train_subsets/d1c_dev32.json",
    "subset_file_sha256": sel["file_sha256"],
    "checkpoint_step": 750,
}
prov_score = {**prov_main, "score_only": True}


def try_emit(tf, prov, sub):
    try:
        p = m74._emit_d1c_v2_from_gate(
            out_dir=tmp / sub,
            contract=contract,
            ckpt=str(ckpt),
            provenance=dict(prov),
            infer=dict(contract.get("infer") or {}),
            teacher_forced_ce=tf,
            ordered_utt_ids=list(uids),
            condition_reports=copy.deepcopy(reports),
            gate=copy.deepcopy(gate),
        )
        return None, p
    except Exception as e:  # noqa: BLE001
        return f"{type(e).__name__}: {e}", None


err, p = try_emit(tf_ce, prov_main, "mainpath")
rec("3a", err is None, f"MAIN eval path (tf_ce=summary['loss'] float) emits a v2 report -> {err or p}")

err2, p2 = try_emit(tf_ce, prov_score, "scoreonly")
rec("3b", err2 is None, f"--score-only path (bundle tf float + score_only=True) emits a v2 report -> {err2 or p2}")

# control: same rows, but with the metrics the evaluator measured but discarded
tf_full = dict(per_batch)
tf_full["total_ce"] = tf_full["loss"]
tf_full["tf_eos_final_top1_accuracy"] = 1.0
tf_full["tf_eos_early_top1_false_positive_count"] = 0
err3, p3 = try_emit(tf_full, prov_main, "control")
rec("3c", err3 is None, f"CONTROL: same rows + full measured TF dict emits fine -> {err3 or 'OK'} (isolates the defect to TF plumbing)")

# ------------------------------------------------------------------ H4 are the call sites protected?
txt = (ROOT / "scripts/74_eval_discrete_lm.py").read_text(encoding="utf-8")
for lineno in (1039, 1558):
    ctx = "\n".join(src[max(0, lineno - 8) : lineno + 2])
    guarded = "try:" in ctx
    rec(f"4-{lineno}", guarded, f"call site at scripts/74:{lineno} is wrapped in try/except (guarded={guarded})")
info("4c", "unguarded => the ValueError aborts the run AFTER n2_gate.json is written, producing no v2 report")

print("\n=== SUMMARY H ===")
nf = 0
for tag, v, msg in out:
    if v == "FAIL":
        nf += 1
        print(f"FAIL H{tag}: {msg}")
print(f"H_total={len(out)} H_fail={nf}")

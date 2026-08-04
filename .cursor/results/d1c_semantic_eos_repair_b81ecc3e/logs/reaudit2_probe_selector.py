"""REAUDIT ROUND2 probe E — selector §11 eligibility with a GENUINELY ELIGIBLE baseline.

Probe D's baseline carried tf_eos_final_top1_accuracy=0.0 (untrained tiny model), so every
negative case there refused vacuously. Here the baseline is constructed to satisfy §11 in
full; each negative mutation is then attributable to the single field it changes.
"""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path("/mnt/c/projects/tts-self-d1c")
sys.path.insert(0, str(ROOT / "src"))
os.chdir(ROOT)

out: list[tuple[str, str, str]] = []


def rec(tag, ok, msg):
    out.append((tag, "PASS" if ok else "FAIL", msg))
    print(f"E{tag} {'PASS' if ok else 'FAIL'} {msg}", flush=True)


def info(tag, msg):
    out.append((tag, "INFO", msg))
    print(f"E{tag} INFO {msg}", flush=True)


def load_script(name):
    p = ROOT / "scripts" / name
    spec = importlib.util.spec_from_file_location(name.replace(".py", ""), p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


m74 = load_script("74_eval_discrete_lm.py")
m76 = load_script("76_select_discrete_checkpoint.py")

from minispeech.data.d1b_oracles import load_eval_contract  # noqa: E402
from minispeech.data.discrete_oracles import load_formal_subset  # noqa: E402
from minispeech.eval.discrete_report_schema import rows_sha256, sha256_json  # noqa: E402

CONDITIONS = ("model", "copy_synth", "wrong_utt", "random_codes", "shuffle_text", "mask_text")
contract = load_eval_contract(ROOT / "configs/eval_contracts/d1c_semantic_eos_v2_0_0.json")
sel = load_formal_subset("select8", contract=contract)
uids = [str(u) for u in sel["utt_ids"]]

tmp = Path(tempfile.mkdtemp(prefix="reaudit2sel_"))
wav_dir = tmp / "wavs"
wav_dir.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(11)


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
                "n_frames": 42,  # ratio 1.0 -> per-row and mean duration OK
                "gt_frames": 42,
                "cer": 0.06 if condition == "model" else 0.05,
                "ref": "привет мир как дела",
                "hyp": "привет мир как дела",
                "wav": str(p),
                "stop_reason": "learned_eos",
                "generation": {
                    "termination_mode": "semantic_eos_v1",
                    "termination_reason": "learned_eos",
                    "n_decoded_frames": 42,
                    "eos_selected": True,
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
tf = {
    "loss": 2.0,
    "total_ce": 2.0,
    "ce_cb0": 1.0,
    "ce_residual": 0.8,
    "ce_semantic_eos": 0.2,
    "teacher_forced_ce_source": "eval_computed",
    "tf_eos_final_top1_accuracy": 1.0,
    "tf_eos_early_top1_false_positive_count": 0,
}
gate = {"verdict": "PASS", "criteria": {"macro_cer": 0.06, "n_catastrophic": 0, "max_abs_log_duration_error": 0.0, "mean_abs_log_duration_error": 0.0}}
prov = {
    "git_sha": "deadbeef",
    "git_dirty": False,
    "subset_label": "select8",
    "subset_path": "configs/train_subsets/d1c_dev32.json",
    "subset_file_sha256": sel["file_sha256"],
    "checkpoint_step": 750,
}

p = m74._emit_d1c_v2_from_gate(
    out_dir=tmp / "base",
    contract=contract,
    ckpt=str(ckpt),
    provenance=dict(prov),
    infer=dict(contract.get("infer") or {}),
    teacher_forced_ce=dict(tf),
    ordered_utt_ids=list(uids),
    condition_reports=copy.deepcopy(reports),
    gate=copy.deepcopy(gate),
)
BASE = json.loads(Path(p).read_text(encoding="utf-8"))
info("0a", f"baseline emitted; formal_eligible={BASE.get('formal_eligible')} macro rows={len(BASE['conditions']['model']['rows'])}")

counter = [0]


def verdict(mutate, label):
    r = json.loads(json.dumps(BASE))
    mutate(r)
    for c in CONDITIONS:
        r["conditions"][c]["rows_sha256"] = rows_sha256(r["conditions"][c]["rows"])
    counter[0] += 1
    d = tmp / f"s{counter[0]}_{label}"
    d.mkdir(parents=True, exist_ok=True)
    paths = []
    for step in (250, 375, 500, 625, 750):
        x = json.loads(json.dumps(r))
        x["checkpoint"]["step"] = step
        x["checkpoint"]["sha256"] = f"{step:064d}"
        x.pop("report_sha256", None)
        x["report_sha256"] = sha256_json(x)
        q = d / f"rep_{step}.json"
        q.write_text(json.dumps(x, ensure_ascii=False), encoding="utf-8")
        paths.append(q)
    try:
        res = m76.select_d1c_formal_reports(paths)
        return f"SELECTED step={res['selected']['step']}"
    except Exception as e:  # noqa: BLE001
        return f"REFUSED: {str(e)[:100]}"


v = verdict(lambda r: None, "clean")
rec("1", str(v).startswith("SELECTED"), f"§11-clean baseline IS selectable -> {v}")

cases = [
    ("early_eos_suppressed_count=7", lambda r: r["conditions"]["model"]["rows"][0]["generation"].__setitem__("early_eos_suppressed_count", 7)),
    ("max_cb0_run=99", lambda r: r["conditions"]["model"]["rows"][0]["generation"].__setitem__("max_cb0_run", 99)),
    ("max_cb0_run=13 (boundary >12)", lambda r: r["conditions"]["model"]["rows"][0]["generation"].__setitem__("max_cb0_run", 13)),
    ("zero-frame generation", lambda r: (r["conditions"]["model"]["rows"][0].__setitem__("n_frames", 0), r["conditions"]["model"]["rows"][0]["generation"].__setitem__("n_decoded_frames", 0))),
    ("per-row duration ratio 9.5x", lambda r: (r["conditions"]["model"]["rows"][0].__setitem__("n_frames", 399), r["conditions"]["model"]["rows"][0]["generation"].__setitem__("n_decoded_frames", 399))),
    ("mean duration ratio 1.43", lambda r: [(row.__setitem__("n_frames", 60), row["generation"].__setitem__("n_decoded_frames", 60)) for row in r["conditions"]["model"]["rows"]]),
    ("catastrophic row", lambda r: r["conditions"]["model"]["rows"][0].__setitem__("catastrophic", True)),
    ("termination_reason=safety_max_frames", lambda r: r["conditions"]["model"]["rows"][0]["generation"].__setitem__("termination_reason", "safety_max_frames")),
    ("tf_eos_final_top1_accuracy=0.9", lambda r: r["teacher_forced_metrics"].__setitem__("tf_eos_final_top1_accuracy", 0.9)),
    ("tf_eos_early_top1_false_positive_count=3", lambda r: r["teacher_forced_metrics"].__setitem__("tf_eos_early_top1_false_positive_count", 3)),
    ("tf_eos metrics = 'unavailable'", lambda r: (r["teacher_forced_metrics"].__setitem__("tf_eos_final_top1_accuracy", "unavailable"), r["teacher_forced_metrics"].__setitem__("tf_eos_early_top1_false_positive_count", "unavailable"))),
    ("model CER 0.40 vs copy 0.05", lambda r: [row.__setitem__("cer", 0.40) for row in r["conditions"]["model"]["rows"]]),
    ("model CER 0.09 vs copy 0.05 (+0.04 > +0.03)", lambda r: [row.__setitem__("cer", 0.09) for row in r["conditions"]["model"]["rows"]]),
    ("text_usage.pass=False", lambda r: r.setdefault("gates", {}).__setitem__("text_usage", {"pass": False})),
    ("phrase_loop_repetition=True", lambda r: r.setdefault("summaries", {}).__setitem__("phrase_loop_repetition", True)),
    ("runaway_or_repetition=True", lambda r: r.setdefault("gates", {}).__setitem__("runaway_or_repetition", True)),
    ("formal_eligible=False", lambda r: r.__setitem__("formal_eligible", False)),
    ("teacher_forced_ce_source=decode_bundle", lambda r: r["teacher_forced_metrics"].__setitem__("teacher_forced_ce_source", "decode_bundle")),
]
for i, (label, fn) in enumerate(cases, start=2):
    v = verdict(fn, f"c{i}")
    rec(str(i), str(v).startswith("REFUSED"), f"{label} -> {v}")

# boundary that SHOULD still pass
v = verdict(lambda r: r["conditions"]["model"]["rows"][0]["generation"].__setitem__("max_cb0_run", 12), "cb0_12")
rec("20", str(v).startswith("SELECTED"), f"max_cb0_run=12 (at the allowed bound) -> {v}")
v = verdict(lambda r: [row.__setitem__("cer", 0.08) for row in r["conditions"]["model"]["rows"]], "cer_plus3")
rec("21", str(v).startswith("SELECTED"), f"model CER 0.08 vs copy 0.05 (exactly +0.03) -> {v}")

# residual vacuity probes
v = verdict(lambda r: [row.__setitem__("cer", 0.05) for row in r["conditions"]["copy_synth"]["rows"]] and None, "copyok")
info("22", f"copy_synth untouched control -> {v}")
v = verdict(lambda r: r.setdefault("gates", {}).pop("text_usage", None), "notextusage")
info("23", f"text_usage block ABSENT (vacuous pass by design, scripts/76:144-146) -> {v}")
v = verdict(lambda r: [r["conditions"]["copy_synth"]["rows"][i].__setitem__("cer", None) for i in range(len(uids))], "copynone")
info("24", f"copy_synth CERs all None (copy_macro=None -> cer_vs_copy vacuous) -> {v}")

print("\n=== SUMMARY E ===")
nf = 0
for tag, v, msg in out:
    if v == "FAIL":
        nf += 1
        print(f"FAIL E{tag}: {msg}")
print(f"E_total={len(out)} E_fail={nf}")

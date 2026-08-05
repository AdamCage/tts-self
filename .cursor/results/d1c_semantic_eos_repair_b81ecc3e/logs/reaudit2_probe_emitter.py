"""REAUDIT ROUND2 probe D — FAIL #4 (v2 emitter E2E) + selector eligibility (§11).

Drives ``_emit_d1c_v2_from_gate`` with rows shaped EXACTLY as scripts/74
``free_running_asr_eval`` builds them, with real .wav artifacts on disk and a
teacher-forced measurement produced by a real (tiny) semantic-EOS model forward.
"""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

ROOT = Path("/mnt/c/projects/tts-self-d1c")
sys.path.insert(0, str(ROOT / "src"))
import os  # noqa: E402

os.chdir(ROOT)

out: list[tuple[str, str, str]] = []


def rec(tag, ok, msg):
    out.append((tag, "PASS" if ok else "FAIL", msg))
    print(f"D{tag} {'PASS' if ok else 'FAIL'} {msg}", flush=True)


def info(tag, msg):
    out.append((tag, "INFO", msg))
    print(f"D{tag} INFO {msg}", flush=True)


def load_script(name: str):
    p = ROOT / "scripts" / name
    spec = importlib.util.spec_from_file_location(name.replace(".py", ""), p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


m74 = load_script("74_eval_discrete_lm.py")
m76 = load_script("76_select_discrete_checkpoint.py")

from minispeech.data.collate_codec_seq import collate_codec_seq  # noqa: E402
from minispeech.data.d1b_oracles import load_eval_contract  # noqa: E402
from minispeech.data.discrete_oracles import load_formal_subset  # noqa: E402
from minispeech.eval.discrete_report_schema import sha256_json  # noqa: E402
from minispeech.models.discrete_speech_lm import (  # noqa: E402
    DiscreteSpeechLM,
    DiscreteSpeechLMConfig,
)

CONDITIONS = ("model", "copy_synth", "wrong_utt", "random_codes", "shuffle_text", "mask_text")

contract = load_eval_contract(ROOT / "configs/eval_contracts/d1c_semantic_eos_v2_0_0.json")
rec("0a", m74._is_d1c_semantic_contract(contract), "contract 2.0.0 recognised as D1c semantic contract")
sel = load_formal_subset("select8", contract=contract)
uids = [str(u) for u in sel["utt_ids"]]
info("0b", f"select8 n={len(uids)} file_sha256={sel['file_sha256'][:16]}")


# ---------------------------------------------------------------- real TF measurement
def _semantic_model():
    cfg = DiscreteSpeechLMConfig.tiny()
    cfg.termination_mode = "semantic_eos_v1"
    cfg.eos_id = cfg.codebook_size
    cfg.codec_vocab_size = cfg.codebook_size
    cfg.semantic_vocab_size = cfg.codebook_size + 1
    return DiscreteSpeechLM(cfg)


torch.manual_seed(0)
m = _semantic_model()
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
tf_measured = m74.teacher_forced_ce(m, batch, torch.device("cpu"))
info("1a", f"teacher_forced_ce keys={sorted(tf_measured)}")
rec(
    "1b",
    "tf_eos_final_top1_accuracy" in tf_measured and "tf_eos_early_top1_false_positive_count" in tf_measured,
    "teacher_forced_ce returns MEASURED EOS metrics "
    f"(acc={tf_measured.get('tf_eos_final_top1_accuracy')}, fp={tf_measured.get('tf_eos_early_top1_false_positive_count')})",
)
rec(
    "1c",
    tf_measured.get("tf_eos_final_top1_accuracy") != 1.0 or tf_measured.get("tf_eos_early_top1_false_positive_count") != 0,
    "untrained tiny model does NOT report a perfect 1.0/0 EOS result (would indicate a hardcoded default); "
    f"got acc={tf_measured.get('tf_eos_final_top1_accuracy')} fp={tf_measured.get('tf_eos_early_top1_false_positive_count')}",
)

tmp = Path(tempfile.mkdtemp(prefix="reaudit2_"))
wav_dir = tmp / "wavs"
wav_dir.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(7)


def make_wav(i: int, uid: str) -> Path:
    p = wav_dir / f"{i:04d}_{abs(hash(uid)) % 10**8}.wav"
    audio = rng.standard_normal(2400).astype(np.float32) * 0.01
    sf.write(str(p), audio, 24000)
    return p


def evaluator_rows(condition: str, *, mutate=None) -> list[dict]:
    """Replicate scripts/74 free_running_asr_eval row_out (lines ~484-503)."""
    rows = []
    for i, uid in enumerate(uids):
        wav = make_wav(i, uid + condition)
        codes = torch.randint(0, 2048, (40, 16))
        r = {
            "utt_id": uid,
            "ordinal": i,
            "condition": condition,
            "n_frames": 40,
            "gt_frames": 42,
            "cer": 0.08 if condition == "model" else 0.05,
            "ref": "привет мир как дела",
            "hyp": "привет мир как дела",
            "wav": str(wav),
            "stop_reason": "learned_eos",
            "generation": {
                "termination_mode": "semantic_eos_v1",
                "termination_reason": "learned_eos",
                "n_decoded_frames": 40,
                "eos_selected": True,
                "early_eos_suppressed_count": 0,
                "max_cb0_run": 3,
            },
            "generated_codes_sha256": hashlib.sha256(codes.numpy().astype(np.int32).tobytes()).hexdigest(),
        }
        if mutate:
            mutate(r, i)
        rows.append(r)
    return rows


def cond_reports(mutate=None):
    return {
        c: {"condition": c, "n": len(uids), "rows": evaluator_rows(c, mutate=mutate if c == "model" else None), "infer": dict(contract.get("infer") or {})}
        for c in CONDITIONS
    }


gate = {
    "verdict": "PASS",
    "criteria": {"macro_cer": 0.08, "fr_cer": 0.08, "n_catastrophic": 0, "max_abs_log_duration_error": 0.05, "mean_abs_log_duration_error": 0.02},
}
ckpt = tmp / "candidate_step_0750.pt"
ckpt.write_bytes(b"fake-checkpoint-bytes")
prov = {
    "git_sha": "deadbeef",
    "git_dirty": False,
    "subset_label": "select8",
    "subset_path": "configs/train_subsets/d1c_dev32.json",
    "subset_file_sha256": sel["file_sha256"],
    "checkpoint_step": 750,
}


def emit(*, tf, reports, out_sub, provenance=None):
    return m74._emit_d1c_v2_from_gate(
        out_dir=tmp / out_sub,
        contract=contract,
        ckpt=str(ckpt),
        provenance=dict(provenance or prov),
        infer=dict(contract.get("infer") or {}),
        teacher_forced_ce=tf,
        ordered_utt_ids=list(uids),
        condition_reports=reports,
        gate=copy.deepcopy(gate),
    )


# ---------------------------------------------------------------- D2 happy path E2E
report = None
try:
    p = emit(tf=dict(tf_measured), reports=cond_reports(), out_sub="ok")
    report = json.loads(Path(p).read_text(encoding="utf-8"))
    rec("2a", True, f"_emit_d1c_v2_from_gate SUCCEEDED on evaluator-shaped rows -> {Path(p).name}")
except Exception as e:  # noqa: BLE001
    import traceback

    traceback.print_exc()
    rec("2a", False, f"_emit_d1c_v2_from_gate RAISED: {type(e).__name__}: {e}")

if report:
    rows = report["conditions"]["model"]["rows"]
    rec("2b", all(r["ordinal"] == i for i, r in enumerate(rows)), "ordinal stamped 0..n-1 on every emitted row")
    rec("2c", all("catastrophic" in r for r in rows), "catastrophic stamped on every emitted row")
    bad_pcm = [r["utt_id"] for r in rows if r["wav"]["pcm_sha256"] == r["wav"]["file_sha256"]]
    rec("2d", not bad_pcm, f"pcm_sha256 != file_sha256 for all rows (offenders={bad_pcm})")
    # independently verify one pcm hash from the wav samples
    r0 = rows[0]
    audio, _sr = sf.read(r0["wav"]["path"], dtype="float32", always_2d=False)
    indep_pcm = hashlib.sha256(np.asarray(audio, dtype=np.float32).reshape(-1).tobytes()).hexdigest()
    indep_file = hashlib.sha256(Path(r0["wav"]["path"]).read_bytes()).hexdigest()
    rec("2e", r0["wav"]["pcm_sha256"] == indep_pcm, "pcm_sha256 independently reproduces from float32 samples")
    rec("2f", r0["wav"]["file_sha256"] == indep_file, "file_sha256 independently reproduces from container bytes")
    tfm = report["teacher_forced_metrics"]
    info("2g", f"emitted teacher_forced_metrics={json.dumps(tfm, ensure_ascii=False, sort_keys=True)}")
    rec(
        "2h",
        tfm.get("tf_eos_final_top1_accuracy") == tf_measured.get("tf_eos_final_top1_accuracy"),
        "emitted EOS TF accuracy equals the MEASURED value (not a 1.0 default)",
    )
    rec("2i", tfm.get("teacher_forced_ce_source") == "eval_computed", "teacher_forced_ce_source=eval_computed from the real measurement")
    rec("2j", report.get("formal_eligible") is True, "report formal_eligible=True on the fully-measured path")

# ---------------------------------------------------------------- D3 negative: fabricated / missing TF
try:
    emit(tf={"loss": 3.21, "ce_cb0": 2.0, "ce_residual": 1.2}, reports=cond_reports(), out_sub="notf")
    rec("3a", False, "emitter ACCEPTED TF metrics with no measured EOS values (fabrication path open)")
except Exception as e:  # noqa: BLE001
    rec("3a", True, f"emitter refuses unmeasured EOS TF metrics: {str(e)[:120]}")

try:
    tf_hardcoded = dict(tf_measured)
    tf_hardcoded["teacher_forced_ce_source"] = "decode_bundle"
    emit(tf=tf_hardcoded, reports=cond_reports(), out_sub="bundle")
    rec("3b", False, "emitter ACCEPTED teacher_forced_ce_source=decode_bundle for a formal report")
except Exception as e:  # noqa: BLE001
    rec("3b", True, f"emitter refuses decode_bundle provenance: {str(e)[:110]}")

# formal_eligible must not be hardcodable to True by a caller with unmeasured metrics
try:
    m74.emit_d1c_v2_report(
        out_path=tmp / "forced.json",
        checkpoint={"path": str(ckpt), "sha256": "a" * 64, "step": 750, "schema": "minispeech.discrete_lm_checkpoint", "schema_version": "2.0.0"},
        provenance={},
        infer={},
        teacher_forced_metrics={"total_ce": 1.0, "teacher_forced_ce_source": "eval_computed"},
        ordered_utt_ids=list(uids),
        conditions=cond_reports(),
        summaries={},
        gates={},
        formal_eligible=True,
    )
    rec("3c", False, "emit_d1c_v2_report ACCEPTED formal_eligible=True with unmeasured EOS metrics")
except Exception as e:  # noqa: BLE001
    rec("3c", True, f"caller-forced formal_eligible=True still refused: {str(e)[:110]}")

# ---------------------------------------------------------------- D4 artifact refusals
def mut_missing_wav(r, i):
    if i == 0:
        r["wav"] = str(wav_dir / "__does_not_exist__.wav")


try:
    emit(tf=dict(tf_measured), reports=cond_reports(mutate=mut_missing_wav), out_sub="nowav")
    rec("4a", False, "emitter ACCEPTED a row whose wav file is missing")
except Exception as e:  # noqa: BLE001
    rec("4a", True, f"missing wav refused: {str(e)[:110]}")


def mut_placeholder(r, i):
    if i == 0:
        r["wav"] = {"path": r["wav"], "file_sha256": "0" * 64, "pcm_sha256": "0" * 64}
        Path(r["wav"]["path"]).unlink(missing_ok=True)


try:
    emit(tf=dict(tf_measured), reports=cond_reports(mutate=mut_placeholder), out_sub="placeholder")
    rec("4b", False, "emitter ACCEPTED a row with 0*64 placeholder wav hashes and no file")
except Exception as e:  # noqa: BLE001
    rec("4b", True, f"placeholder wav hashes refused: {str(e)[:110]}")


def mut_copied_pcm(r, i):
    p = Path(r["wav"])
    d = hashlib.sha256(p.read_bytes()).hexdigest()
    r["wav"] = {"path": str(p), "file_sha256": d, "pcm_sha256": d}


try:
    p = emit(tf=dict(tf_measured), reports=cond_reports(mutate=mut_copied_pcm), out_sub="copiedpcm")
    rep = json.loads(Path(p).read_text(encoding="utf-8"))
    r0 = rep["conditions"]["model"]["rows"][0]
    rec("4c", r0["wav"]["pcm_sha256"] != r0["wav"]["file_sha256"], "pcm_sha256==file_sha256 shortcut is recomputed from real PCM")
except Exception as e:  # noqa: BLE001
    rec("4c", True, f"copied pcm_sha refused: {str(e)[:110]}")


def mut_no_codes(r, i):
    if i == 0:
        r["generated_codes_sha256"] = "0" * 64


try:
    emit(tf=dict(tf_measured), reports=cond_reports(mutate=mut_no_codes), out_sub="nocodes")
    rec("4d", False, "emitter ACCEPTED a row with placeholder generated_codes_sha256")
except Exception as e:  # noqa: BLE001
    rec("4d", True, f"placeholder generated_codes_sha256 refused: {str(e)[:110]}")

try:
    bad_prov = dict(prov)
    bad_prov["subset_file_sha256"] = "0" * 64
    emit(tf=dict(tf_measured), reports=cond_reports(), out_sub="nosubset", provenance=bad_prov)
    info("4e", "placeholder subset_file_sha256 fell back to the contract entry (accepted)")
except Exception as e:  # noqa: BLE001
    info("4e", f"placeholder subset_file_sha256 -> {str(e)[:110]}")

# ---------------------------------------------------------------- D5 selector eligibility (§11)
def five_reports(base_report: dict, out_dir: Path) -> list[Path]:
    paths = []
    for step in (250, 375, 500, 625, 750):
        r = json.loads(json.dumps(base_report))
        r["checkpoint"]["step"] = step
        r["checkpoint"]["sha256"] = f"{step:064d}"
        r.pop("report_sha256", None)
        r["report_sha256"] = sha256_json(r)
        p = out_dir / f"rep_{step}.json"
        p.write_text(json.dumps(r, ensure_ascii=False), encoding="utf-8")
        paths.append(p)
    return paths


def selector_verdict(mutate_report, label):
    """Mutate a copy of the good report, re-hash consistently, run the selector."""
    if report is None:
        return None
    r = json.loads(json.dumps(report))
    mutate_report(r)
    # re-hash rows + report so refusals cannot be attributed to tamper detection
    from minispeech.eval.discrete_report_schema import rows_sha256

    for c in CONDITIONS:
        r["conditions"][c]["rows_sha256"] = rows_sha256(r["conditions"][c]["rows"])
    d = tmp / f"sel_{label}"
    d.mkdir(parents=True, exist_ok=True)
    paths = five_reports(r, d)
    try:
        res = m76.select_d1c_formal_reports(paths)
        return f"SELECTED step={res['selected']['step']} eligible={res['selected']['gate_eligible']}"
    except Exception as e:  # noqa: BLE001
        return f"REFUSED: {str(e)[:110]}"


if report is not None:
    v = selector_verdict(lambda r: None, "clean")
    rec("5a", str(v).startswith("SELECTED"), f"baseline clean report is selectable -> {v}")

    def m_early(r):
        r["conditions"]["model"]["rows"][0]["generation"]["early_eos_suppressed_count"] = 7

    v = selector_verdict(m_early, "early")
    rec("5b", str(v).startswith("REFUSED"), f"early_eos_suppressed_count=7 -> {v}")

    def m_cb0(r):
        r["conditions"]["model"]["rows"][0]["generation"]["max_cb0_run"] = 99

    v = selector_verdict(m_cb0, "cb0")
    rec("5c", str(v).startswith("REFUSED"), f"max_cb0_run=99 -> {v}")

    def m_zero(r):
        row = r["conditions"]["model"]["rows"][0]
        row["n_frames"] = 0
        row["generation"]["n_decoded_frames"] = 0

    v = selector_verdict(m_zero, "zero")
    rec("5d", str(v).startswith("REFUSED"), f"zero-frame generation -> {v}")

    def m_dur(r):
        row = r["conditions"]["model"]["rows"][0]
        row["n_frames"] = 400
        row["generation"]["n_decoded_frames"] = 400

    v = selector_verdict(m_dur, "dur")
    rec("5e", str(v).startswith("REFUSED"), f"per-row duration ratio ~9.5x -> {v}")

    def m_cat(r):
        r["conditions"]["model"]["rows"][0]["catastrophic"] = True

    v = selector_verdict(m_cat, "cat")
    rec("5f", str(v).startswith("REFUSED"), f"catastrophic row -> {v}")

    def m_stop(r):
        r["conditions"]["model"]["rows"][0]["generation"]["termination_reason"] = "safety_max_frames"

    v = selector_verdict(m_stop, "stop")
    rec("5g", str(v).startswith("REFUSED"), f"termination_reason=safety_max_frames -> {v}")

    def m_eos(r):
        r["teacher_forced_metrics"]["tf_eos_final_top1_accuracy"] = 0.9

    v = selector_verdict(m_eos, "eos")
    rec("5h", str(v).startswith("REFUSED"), f"tf_eos_final_top1_accuracy=0.9 -> {v}")

    def m_eosfp(r):
        r["teacher_forced_metrics"]["tf_eos_early_top1_false_positive_count"] = 3

    v = selector_verdict(m_eosfp, "eosfp")
    rec("5i", str(v).startswith("REFUSED"), f"tf_eos_early_top1_false_positive_count=3 -> {v}")

    def m_cer(r):
        for row in r["conditions"]["model"]["rows"]:
            row["cer"] = 0.40  # copy_synth rows stay at 0.05 -> +0.35 > +0.03

    v = selector_verdict(m_cer, "cer")
    rec("5j", str(v).startswith("REFUSED"), f"model CER 0.40 vs copy 0.05 (>+0.03) -> {v}")

    def m_meandur(r):
        for row in r["conditions"]["model"]["rows"]:
            row["n_frames"] = 60  # 60/42 = 1.43 mean ratio outside [0.90,1.10]
            row["generation"]["n_decoded_frames"] = 60

    v = selector_verdict(m_meandur, "meandur")
    rec("5k", str(v).startswith("REFUSED"), f"mean duration ratio 1.43 -> {v}")

    def m_textusage(r):
        r.setdefault("gates", {})["text_usage"] = {"pass": False}

    v = selector_verdict(m_textusage, "textusage")
    rec("5l", str(v).startswith("REFUSED"), f"text_usage.pass=False -> {v}")

    def m_runaway(r):
        r.setdefault("summaries", {})["phrase_loop_repetition"] = True

    v = selector_verdict(m_runaway, "phraseloop")
    rec("5m", str(v).startswith("REFUSED"), f"phrase_loop_repetition=True -> {v}")

    # tamper detection still live
    def m_tamper(r):
        r["conditions"]["model"]["rows"][0]["cer"] = 0.01

    rr = json.loads(json.dumps(report))
    rr["conditions"]["model"]["rows"][0]["cer"] = 0.01  # do NOT re-hash
    d = tmp / "sel_tamper"
    d.mkdir(parents=True, exist_ok=True)
    ps = []
    for step in (250, 375, 500, 625, 750):
        x = json.loads(json.dumps(rr))
        x["checkpoint"]["step"] = step
        x["checkpoint"]["sha256"] = f"{step:064d}"
        p = d / f"rep_{step}.json"
        p.write_text(json.dumps(x, ensure_ascii=False), encoding="utf-8")
        ps.append(p)
    try:
        m76.select_d1c_formal_reports(ps)
        rec("5n", False, "selector ACCEPTED reports with stale rows_sha256/report_sha256")
    except Exception as e:  # noqa: BLE001
        rec("5n", True, f"stale hashes refused: {str(e)[:100]}")

    rec("5o", str(selector_verdict(lambda r: None, "clean2")).startswith("SELECTED"), "clean report still selectable after all mutations (no side effects)")

# ---------------------------------------------------------------- D6 score-only emitter path
info(
    "6a",
    "score-only sets teacher_forced_ce_source=decode_bundle when the bundle lacks it "
    "(scripts/74:848-850) and emit_d1c_v2_report refuses anything but eval_computed "
    "(scripts/74:117-131) -> --score-only cannot produce a formal v2 report",
)
import subprocess  # noqa: E402

g = subprocess.run(["grep", "-n", "-B3", "_emit_d1c_v2_from_gate(", "scripts/74_eval_discrete_lm.py"], capture_output=True, text=True, cwd=str(ROOT))
info("6b", "call sites (check for try/except wrapping):\n" + g.stdout)

print("\n=== SUMMARY D ===")
nf = 0
for tag, v, msg in out:
    if v == "FAIL":
        nf += 1
        print(f"FAIL D{tag}: {msg}")
print(f"D_total={len(out)} D_fail={nf}")
print(f"tmpdir={tmp}")

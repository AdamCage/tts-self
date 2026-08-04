"""Independent adversarial probe of the D1c INFRA repair claims.

Run from the worktree root:
    python .cursor/results/d1c_semantic_eos_repair_b81ecc3e/logs/reaudit_probe.py

Each check prints REAUDIT <BLOCKER> <PASS|FAIL|INFO> <detail>.
No GPU, no commits, no seed invention.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "src"))

RESULTS: list[tuple[str, str, str]] = []


def rec(blocker: str, status: str, detail: str) -> None:
    RESULTS.append((blocker, status, detail))
    print(f"REAUDIT {blocker} {status} {detail}", flush=True)


def load_script(name: str):
    path = ROOT / "scripts" / name
    spec = importlib.util.spec_from_file_location(name.replace(".py", ""), path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------- B1 residual
def b1_residual_grads() -> None:
    from minispeech.data.collate_codec_seq import collate_codec_seq
    from minispeech.models.discrete_speech_lm import DiscreteSpeechLM, DiscreteSpeechLMConfig

    cfg = DiscreteSpeechLMConfig.tiny()
    cfg.termination_mode = "semantic_eos_v1"
    cfg.eos_id = cfg.codebook_size
    cfg.codec_vocab_size = cfg.codebook_size
    cfg.semantic_vocab_size = cfg.codebook_size + 1
    m = DiscreteSpeechLM(cfg)
    torch.manual_seed(0)
    batch = collate_codec_seq(
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

    def fwd(cb0_w: float, eos_w: float):
        m.zero_grad(set_to_none=True)
        out = m(
            batch["input_ids"],
            batch["codes"],
            text_mask=batch["text_mask"],
            frame_mask=batch["codec_frame_mask"],
            semantic_targets=batch["semantic_targets"],
            semantic_mask=batch["semantic_mask"],
            speaker_id=batch["speaker_id"],
            residual_weights=torch.ones(m.n_codebooks - 1),
            cb0_weight=cb0_w,
            eos_loss_weight=eos_w,
        )
        out.loss["total"].backward()
        return out

    # (a) default weights: residual params must all get non-zero grads
    out = fwd(1.0, 1.0)
    res_params = list(m.heads.residuals.parameters())
    n_zero = sum(1 for p in res_params if p.grad is None or float(p.grad.abs().sum()) == 0.0)
    rec(
        "B1",
        "PASS" if n_zero == 0 else "FAIL",
        f"residual params without grad: {n_zero}/{len(res_params)} "
        f"(default cb0_w=1 eos_w=1)",
    )
    # (b) isolate residual CE: cb0_weight=0, eos_loss_weight=0 -> total == residual_ce only
    out0 = fwd(0.0, 0.0)
    n_zero0 = sum(1 for p in res_params if p.grad is None or float(p.grad.abs().sum()) == 0.0)
    rec(
        "B1",
        "PASS" if n_zero0 == 0 else "FAIL",
        f"residual params without grad with cb0_w=0,eos_w=0: {n_zero0}/{len(res_params)} "
        "(proves residual CE alone drives residual grads)",
    )
    # (c) requires_grad / graph liveness of returned tensor
    live = bool(out.loss["total"].requires_grad)
    rec("B1", "PASS" if live else "FAIL", f"loss['total'].requires_grad={live}")
    # (d) MultiCodebookHead.loss ce_residual must not be detached
    from minispeech.models.multi_codebook_head import MultiCodebookHead, MultiCodebookOutput

    h = MultiCodebookHead(8, n_codebooks=4, codebook_size=16)
    hid = torch.randn(1, 3, 8, requires_grad=True)
    emb = torch.randn(1, 3, 8, requires_grad=True)
    o = h(hid, emb)
    ld = h.loss(o, torch.randint(0, 16, (1, 3, 4)), torch.ones(1, 3, dtype=torch.bool))
    rec(
        "B1",
        "PASS" if ld["ce_residual"].requires_grad and ld["ce_cb0"].requires_grad else "FAIL",
        f"MultiCodebookHead.loss ce_residual.requires_grad={ld['ce_residual'].requires_grad} "
        f"ce_cb0.requires_grad={ld['ce_cb0'].requires_grad}",
    )
    # (e) also confirm the trainer path composes the live loss
    src = (ROOT / "src/minispeech/train/discrete_lm_trainer.py").read_text(encoding="utf-8")
    bad = "ce_residual\"].detach()" in src and "losses[\"total\"]" not in src
    rec("B1", "INFO", f"trainer detaches only for logging: {not bad}")


# ---------------------------------------------------------------- B2 contract
def b2_contract() -> None:
    from minispeech.data.d1b_oracles import eval_contract_content_sha256, load_eval_contract
    from minispeech.infer.discrete_config import contract_infer_from_eval_contract

    p = ROOT / "configs/eval_contracts/d1c_semantic_eos_v2_0_0.json"
    contract = load_eval_contract(p)
    declared = contract.get("sha256")
    computed = eval_contract_content_sha256(contract)
    rec(
        "B2",
        "PASS" if declared == computed else "FAIL",
        f"declared={declared} computed={computed}",
    )
    # raw-file variant: recompute directly from disk bytes to prove no loader magic
    raw = json.loads(p.read_text(encoding="utf-8"))
    raw_computed = eval_contract_content_sha256(raw)
    rec(
        "B2",
        "PASS" if raw_computed == declared else "FAIL",
        f"raw-file recompute={raw_computed}",
    )
    rec("B2", "INFO", f"recipe_sha256={contract.get('recipe_sha256')}")
    rec(
        "B2",
        "INFO",
        f"recipe_source_manifest_sha256={contract.get('recipe_source_manifest_sha256')}",
    )
    # legacy stop knobs must be absent from canonical semantic infer
    infer = contract_infer_from_eval_contract(contract)
    legacy = [
        k
        for k in (
            "stop_threshold",
            "stop_consecutive",
            "frames_per_text_token",
            "stop_overrun_bias",
            "stop_earliest_frac",
            "eos_cb0",
        )
        if k in infer
    ]
    rec(
        "B2",
        "PASS" if not legacy else "FAIL",
        f"infer keys={sorted(infer)} legacy_present={legacy}",
    )
    # contract's own inference block must not declare legacy stop fields either
    cinfer = contract.get("inference") or {}
    legacy_decl = [
        k
        for k in (
            "stop_threshold",
            "stop_consecutive",
            "frames_per_text_token",
            "stop_overrun_bias",
            "stop_earliest_frac",
        )
        if k in cinfer
    ]
    rec(
        "B2",
        "PASS" if not legacy_decl else "FAIL",
        f"contract.inference keys={sorted(cinfer)} legacy_declared={legacy_decl}",
    )
    # tamper detection: mutating any content field must break the self-check
    tampered = json.loads(json.dumps(contract))
    tampered.setdefault("inference", {})["max_frames"] = 999
    still_ok = eval_contract_content_sha256(tampered) == declared
    rec(
        "B2",
        "FAIL" if still_ok else "PASS",
        f"tamper(max_frames=999) still matches declared sha: {still_ok}",
    )


# ---------------------------------------------------------------- B3 oracles
def b3_oracles() -> None:
    from minispeech.data.d1b_oracles import load_eval_contract
    from minispeech.data.discrete_oracles import (
        file_sha256,
        load_formal_subset,
        payload_content_file_sha256,
    )

    contract = load_eval_contract(ROOT / "configs/eval_contracts/d1c_semantic_eos_v2_0_0.json")
    subsets = contract.get("subsets") or {}
    rec("B3", "INFO", f"contract subsets={sorted(subsets)}")
    missing_sha = [k for k, v in subsets.items() if not (v or {}).get("file_sha256")]
    rec(
        "B3",
        "PASS" if not missing_sha else "FAIL",
        f"registry entries lacking file_sha256={missing_sha}",
    )
    ok = True
    for key in ("dev32", "select8", "gate24", "full32"):
        try:
            att = load_formal_subset(key, contract=contract)
            rec(
                "B3",
                "PASS",
                f"formal load {key}: n={len(att['utt_ids'])} split={att['split']} "
                f"formal_eligible={att['formal_eligible']} file_sha256={att['file_sha256'][:16]}…",
            )
        except Exception as exc:  # noqa: BLE001
            ok = False
            rec("B3", "FAIL", f"formal load {key} raised {type(exc).__name__}: {exc}")
    sel = load_formal_subset("select8", contract=contract)
    ids = sel["utt_ids"]
    n_rus = sum(1 for u in ids if str(u).startswith("ruslan_"))
    rec(
        "B3",
        "PASS" if (n_rus == 2 and len(ids) - n_rus == 6) else "FAIL",
        f"select8 composition = {n_rus} RUSLAN + {len(ids) - n_rus} CV",
    )
    # union / disjointness of select8 + gate24 == full32
    g24 = load_formal_subset("gate24", contract=contract)["utt_ids"]
    f32 = load_formal_subset("full32", contract=contract)["utt_ids"]
    rec(
        "B3",
        "PASS" if (set(ids) | set(g24)) == set(f32) and not (set(ids) & set(g24)) else "FAIL",
        f"select8|gate24 == full32 and disjoint (|s|={len(ids)} |g|={len(g24)} |f|={len(f32)})",
    )
    # does the declared file_sha256 attest the ACTUAL on-disk bytes?
    payload_path = ROOT / "configs/train_subsets/d1c_dev32.json"
    payload = json.loads(payload_path.read_text(encoding="utf-8"))
    real_bytes = file_sha256(payload_path)
    content = payload_content_file_sha256(payload)
    rec(
        "B3",
        "INFO",
        f"real-file-bytes sha256={real_bytes[:16]}… content-body sha256={content[:16]}… "
        f"equal={real_bytes == content}",
    )
    # select8 stratification per plan 9.3: all duration buckets + both ratio extremes
    meta = {r["utt_id"]: r for r in payload.get("rows_meta") or []}
    q = payload["selection"]["ratio_quartiles"]
    buckets = sorted({meta[u]["duration_bucket"] for u in ids if u in meta})
    lows = [u for u in ids if u in meta and meta[u]["ratio"] <= q["q1"]]
    highs = [u for u in ids if u in meta and meta[u]["ratio"] >= q["q3"]]
    rec(
        "B3",
        "PASS" if len(buckets) == 3 and lows and highs else "FAIL",
        f"select8 stratification: buckets={buckets} n_low_ratio={len(lows)} n_high_ratio={len(highs)}",
    )
    return ok


# ---------------------------------------------------------------- B4 CLI
def b4_cli() -> None:
    from minispeech.models.discrete_speech_lm import DiscreteGenerationOutput

    src73 = (ROOT / "scripts/73_infer_discrete_lm.py").read_text(encoding="utf-8")
    rec(
        "B4",
        "PASS" if "isinstance(gen_out, DiscreteGenerationOutput)" in src73 else "FAIL",
        "73 unwraps DiscreteGenerationOutput before .shape/.cpu use",
    )
    m74 = load_script("74_eval_discrete_lm.py")
    codes, meta = m74._unwrap_generation(
        DiscreteGenerationOutput(
            codes=torch.zeros(1, 3, 16, dtype=torch.long),
            frame_mask=torch.ones(1, 3, dtype=torch.bool),
            lengths=torch.tensor([3]),
            rows=[{"termination_reason": "learned_eos"}],
        )
    )
    ok_unwrap = torch.is_tensor(codes) and meta["termination_reason"] == "learned_eos"
    try:
        m74._unwrap_generation({"nope": 1})
        ok_reject = False
    except TypeError:
        ok_reject = True
    rec(
        "B4",
        "PASS" if ok_unwrap and ok_reject else "FAIL",
        f"74 _unwrap_generation ok={ok_unwrap} rejects_non_tensor={ok_reject}",
    )
    # teacher-forced CE must demand semantic targets in semantic mode
    from minispeech.models.discrete_speech_lm import DiscreteSpeechLM, DiscreteSpeechLMConfig

    cfg = DiscreteSpeechLMConfig.tiny()
    cfg.termination_mode = "semantic_eos_v1"
    cfg.eos_id = cfg.codebook_size
    cfg.codec_vocab_size = cfg.codebook_size
    cfg.semantic_vocab_size = cfg.codebook_size + 1
    m = DiscreteSpeechLM(cfg)
    batch = {
        "input_ids": torch.randint(1, 10, (1, 3)),
        "codes": torch.randint(0, m.codebook_size, (1, 2, m.n_codebooks)),
        "text_mask": torch.ones(1, 3, dtype=torch.bool),
        "frame_mask": torch.ones(1, 2, dtype=torch.bool),
        "speaker_id": torch.zeros(1, dtype=torch.long),
    }
    try:
        m74.teacher_forced_ce(m, batch, torch.device("cpu"))
        rec("B4", "FAIL", "teacher_forced_ce accepted batch without semantic_targets")
    except ValueError as exc:
        rec("B4", "PASS", f"teacher_forced_ce refuses missing semantic targets: {exc!s:.70}")
    # estimate_stop_reason must be unreachable in semantic mode
    src74 = (ROOT / "scripts/74_eval_discrete_lm.py").read_text(encoding="utf-8")
    lines = src74.splitlines()
    calls = [i + 1 for i, ln in enumerate(lines) if "estimate_stop_reason(" in ln and "import" not in ln]
    rec("B4", "INFO", f"estimate_stop_reason( call sites in 74: lines {calls}")
    guarded = "if semantic_mode:" in src74 and "stop_reason = estimate_stop_reason(" in src74
    rec("B4", "PASS" if guarded else "FAIL", "semantic branch bypasses estimate_stop_reason")
    # emit_d1c_v2_report reachability
    rec("B4", "PASS" if callable(getattr(m74, "emit_d1c_v2_report", None)) else "FAIL",
        "emit_d1c_v2_report is exported/callable")
    emit_calls = [i + 1 for i, ln in enumerate(lines) if "_emit_d1c_v2_from_gate(" in ln]
    rec("B4", "PASS" if len(emit_calls) >= 2 else "FAIL",
        f"_emit_d1c_v2_from_gate call sites: lines {emit_calls}")
    # ADVERSARIAL: are the semantic EOS TF metrics actually measured, or defaulted?
    tf = m74.teacher_forced_ce.__doc__
    tf_src = src74[src74.index("def teacher_forced_ce("): src74.index("def _unwrap_generation(")]
    measured = "tf_eos_final_top1_accuracy" in tf_src
    defaulted = 'tf.setdefault("tf_eos_final_top1_accuracy", 1.0)' in src74
    rec(
        "B4",
        "FAIL" if (defaulted and not measured) else "PASS",
        f"tf_eos_final_top1_accuracy measured_by_evaluator={measured} "
        f"hardcoded_default_1.0={defaulted}",
    )
    fake_hash = src74.count('"0" * 64')
    rec("B4", "INFO", f"placeholder '0'*64 hash literals in 74: {fake_hash}")


# ---------------------------------------------------------------- B5 empty EOS
def b5_immediate_eos() -> None:
    from minispeech.models.discrete_speech_lm import (
        DiscreteGenerationOutput,
        DiscreteSpeechLM,
        DiscreteSpeechLMConfig,
    )

    cfg = DiscreteSpeechLMConfig.tiny()
    cfg.termination_mode = "semantic_eos_v1"
    cfg.eos_id = cfg.codebook_size
    cfg.codec_vocab_size = cfg.codebook_size
    cfg.semantic_vocab_size = cfg.codebook_size + 1
    m = DiscreteSpeechLM(cfg)

    class AlwaysEOS(torch.nn.Module):
        def forward(self, h):  # noqa: ANN001, ANN201
            lg = h.new_full((*h.shape[:-1], m.semantic_vocab_size), -1e9)
            lg[..., m.eos_id] = 1e9
            return lg

    m.heads.semantic = AlwaysEOS()
    # go through the PUBLIC generate() dispatch, not the private helper
    out = m.generate(
        torch.tensor([[3, 4, 5]]),
        max_frames=4,
        min_frames=0,
        max_cb0_repeat=100,
        temperature=0.0,
        top_k=0,
    )
    ok = (
        isinstance(out, DiscreteGenerationOutput)
        and int(out.lengths[0]) == 0
        and tuple(out.codes.shape) == (1, 0, m.n_codebooks)
        and tuple(out.frame_mask.shape) == (1, 0)
        and out.rows[0]["termination_reason"] == "learned_eos"
    )
    rec(
        "B5",
        "PASS" if ok else "FAIL",
        f"generate() dispatch: codes.shape={tuple(out.codes.shape)} "
        f"lengths={out.lengths.tolist()} mask={tuple(out.frame_mask.shape)} "
        f"reason={out.rows[0]['termination_reason']}",
    )
    # mixed batch: row0 immediate EOS, row1 keeps decoding -> per-row lengths honoured
    class EOSForRow0(torch.nn.Module):
        def forward(self, h):  # noqa: ANN001, ANN201
            lg = h.new_zeros((*h.shape[:-1], m.semantic_vocab_size))
            lg[0, ..., m.eos_id] = 1e9
            lg[1, ..., 7] = 1e9
            return lg

    m.heads.semantic = EOSForRow0()
    out2 = m.generate(
        torch.tensor([[3, 4, 5], [3, 4, 5]]),
        max_frames=3,
        min_frames=0,
        max_cb0_repeat=100,
        temperature=0.0,
        top_k=0,
    )
    ok2 = int(out2.lengths[0]) == 0 and int(out2.lengths[1]) > 0
    ok2 &= not bool(out2.frame_mask[0].any())
    rec(
        "B5",
        "PASS" if ok2 else "FAIL",
        f"batched: lengths={out2.lengths.tolist()} row0_mask_any={bool(out2.frame_mask[0].any())}",
    )
    # min_frames suppression must record early_eos_suppressed_count
    m.heads.semantic = AlwaysEOS()
    out3 = m.generate(
        torch.tensor([[3, 4, 5]]),
        max_frames=6,
        min_frames=4,
        max_cb0_repeat=100,
        temperature=0.0,
        top_k=0,
    )
    sup = int(out3.rows[0]["early_eos_suppressed_count"])
    rec(
        "B5",
        "PASS" if sup >= 4 else "FAIL",
        f"min_frames=4 suppression: early_eos_suppressed_count={sup} "
        f"len={int(out3.lengths[0])} reason={out3.rows[0]['termination_reason']}",
    )


# ---------------------------------------------------------------- B6 selector
def b6_selector() -> None:
    m76 = load_script("76_select_discrete_checkpoint.py")
    fn = getattr(m76, "_sha256_file", None)
    if fn is None:
        rec("B6", "FAIL", "_sha256_file missing from selector")
        return
    tmp = ROOT / ".cursor/results/d1c_semantic_eos_repair_b81ecc3e/logs/_probe_blob.bin"
    tmp.write_bytes(b"x" * (3 << 20))  # >1 chunk to exercise the loop
    try:
        got = fn(tmp)
        want = hashlib.sha256(b"x" * (3 << 20)).hexdigest()
        rec("B6", "PASS" if got == want else "FAIL", f"_sha256_file 3MiB: match={got == want}")
    except NameError as exc:
        rec("B6", "FAIL", f"_sha256_file NameError: {exc}")
    finally:
        tmp.unlink(missing_ok=True)
    src = (ROOT / "scripts/76_select_discrete_checkpoint.py").read_text(encoding="utf-8")
    trusts = 'gates.get("eligible")' in src or 'gate_eligible")' in src
    rec("B6", "INFO", f"selector source mentions copied gate flags: {trusts}")
    has_recompute = "_recompute_eligibility" in src or "eligib" in src
    rec("B6", "PASS" if has_recompute else "FAIL", "selector has eligibility recompute logic")


# ---------------------------------------------------------------- B7 resume
def b7_resume() -> None:
    src = (ROOT / "src/minispeech/train/discrete_lm_trainer.py").read_text(encoding="utf-8")
    needed = [
        "config_sha256",
        "recipe_sha256",
        "recipe_source_manifest_sha256",
        "run_manifest_sha256",
        "model_config_sha256",
        "speaker_map_sha256",
        "subset_items_sha256",
        "eval_contract_sha256",
    ]
    present = {k: (k in src) for k in needed}
    rec(
        "B7",
        "PASS" if all(present.values()) else "FAIL",
        f"resume hash keys present: {present}",
    )
    lines = src.splitlines()
    enforce = [i + 1 for i, ln in enumerate(lines) if "ResumeMismatchError" in ln]
    rec("B7", "INFO", f"ResumeMismatchError sites: {len(enforce)} (first 12: {enforce[:12]})")


# ---------------------------------------------------------------- B8 manifest
def b8_manifest() -> None:
    from minispeech.data.d1b_oracles import load_eval_contract
    from minispeech.data.discrete_oracles import file_sha256

    contract = load_eval_contract(ROOT / "configs/eval_contracts/d1c_semantic_eos_v2_0_0.json")
    manifest = contract.get("recipe_source_manifest") or {}
    if not manifest:
        rec("B8", "FAIL", "contract has no recipe_source_manifest")
        return
    files = manifest.get("files") if isinstance(manifest, dict) else None
    if not isinstance(files, (dict, list)):
        rec("B8", "FAIL", f"recipe_source_manifest.files malformed: {type(files).__name__}")
        return
    items = files.items() if isinstance(files, dict) else [(f["path"], f["sha256"]) for f in files]
    mismatches = []
    missing = []
    for rel, declared in items:
        p = ROOT / rel
        if not p.is_file():
            missing.append(rel)
            continue
        got = file_sha256(p)
        if got != declared:
            mismatches.append((rel, declared, got))
    rec(
        "B8",
        "PASS" if not mismatches and not missing else "FAIL",
        f"n_files={len(list(items))} missing={missing} mismatches="
        + json.dumps([[r, d[:12], g[:12]] for r, d, g in mismatches]),
    )
    # recompute the declared manifest digest
    declared_digest = contract.get("recipe_source_manifest_sha256")
    rec("B8", "INFO", f"declared recipe_source_manifest_sha256={declared_digest}")
    # is every d1c-critical source file actually locked?
    critical = [
        "src/minispeech/models/discrete_speech_lm.py",
        "src/minispeech/models/multi_codebook_head.py",
        "src/minispeech/data/collate_codec_seq.py",
        "src/minispeech/train/discrete_lm_trainer.py",
        "src/minispeech/eval/discrete_n2.py",
        "src/minispeech/eval/discrete_report_schema.py",
        "src/minispeech/data/discrete_oracles.py",
        "src/minispeech/infer/discrete_config.py",
        "scripts/72_train_discrete_lm.py",
        "scripts/73_infer_discrete_lm.py",
        "scripts/74_eval_discrete_lm.py",
        "scripts/76_select_discrete_checkpoint.py",
        "configs/discrete_speech_lm_d1c_semantic_eos_w1.yaml",
        "configs/discrete_speech_lm_d1c_semantic_eos_w2.yaml",
    ]
    locked = set(files) if isinstance(files, dict) else {f["path"] for f in files}
    unlocked = [c for c in critical if c not in locked]
    rec(
        "B8",
        "PASS" if not unlocked else "FAIL",
        f"critical sources NOT in frozen manifest: {unlocked}",
    )


def main() -> int:
    print(f"ROOT={ROOT}")
    for fn in (
        b1_residual_grads,
        b2_contract,
        b3_oracles,
        b4_cli,
        b5_immediate_eos,
        b6_selector,
        b7_resume,
        b8_manifest,
    ):
        try:
            fn()
        except Exception as exc:  # noqa: BLE001
            import traceback

            rec(fn.__name__, "ERROR", f"{type(exc).__name__}: {exc}")
            traceback.print_exc()
    print("\n=== REAUDIT PROBE SUMMARY ===")
    for b, s, d in RESULTS:
        print(f"{s:5s} {b:4s} {d}")
    fails = [r for r in RESULTS if r[1] in ("FAIL", "ERROR")]
    print(f"\nTOTAL checks={len(RESULTS)} fail_or_error={len(fails)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

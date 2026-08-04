#!/usr/bin/env python3
"""REAUDIT_ROUND5 — reseal audit + prior-blocker regression spot checks.

Covers FAIL #2 (contract / recipe / source-manifest binding after the round-5
reseal), FAIL #7 (resume hashes), the published-digest audit for the round-4 →
round-5 supersession, and the standing prior-PASS spot checks (residual grads,
select8 composition, immediate EOS, D1c subset aliases).

Every digest is recomputed independently; the round-5 claims are treated as
unverified assertions.
"""
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from minispeech.data.d1b_oracles import (  # noqa: E402
    SUBSET_ALIASES,
    eval_contract_content_sha256,
    verify_recipe_source_manifest,
)
from minispeech.train.discrete_lm_trainer import resolve_repro_hashes  # noqa: E402

CONTRACT_200 = ROOT / "configs/eval_contracts/d1c_semantic_eos_v2_0_0.json"
CONTRACT_201 = ROOT / "configs/eval_contracts/d1c_semantic_eos_v2_0_1.json"
W1 = ROOT / "configs/discrete_speech_lm_d1c_semantic_eos_w1.yaml"
W2 = ROOT / "configs/discrete_speech_lm_d1c_semantic_eos_w2.yaml"
RES = ROOT / ".cursor/results/d1c_semantic_eos_repair_b81ecc3e"

CLAIM_CONTRACT = "cff9e9f5dd11ae66d66e44eddc8768f4989daa8c0339822d77252289a155baad"
CLAIM_MANIFEST = "e505acf99640a3c801f9a9451bc60a19c74d2460413564cb0d68b2ab6fc2b596"
ROUND4_CONTRACT = "9922ce4d75932dcaff255dd20601f731f48c369647677c68249b9ff652d938af"
ROUND4_MANIFEST = "62eeeae2c6360e88923dd6416e8c5a22403d869cbbdf806aabce31e6a512b81d"

ROUND5_EDITED = [
    "scripts/74_eval_discrete_lm.py",
    "scripts/76_select_discrete_checkpoint.py",
    "src/minispeech/eval/discrete_report_schema.py",
]

RESULTS: list[tuple[str, str, str]] = []


def rec(tag: str, ok, msg: str) -> None:
    status = "INFO" if ok is None else ("PASS" if ok else "FAIL")
    RESULTS.append((tag, status, msg))
    print(f"{tag} {status} {msg}", flush=True)


def main() -> int:  # noqa: C901, PLR0912, PLR0915
    # ------------------------------------------------- FAIL #2 : contract seal
    raw = json.loads(CONTRACT_200.read_text(encoding="utf-8"))
    live = eval_contract_content_sha256(raw)
    rec("R2a", live == raw.get("sha256"),
        f"declared sha256 == independent recompute {live[:16]}")
    rec("R2b", live == CLAIM_CONTRACT,
        f"REPAIR_ROUND5 claimed contract_sha256 matches live: "
        f"claim={CLAIM_CONTRACT[:16]} live={live[:16]}")
    rec("R2c", raw.get("recipe_sha256") == live,
        f"recipe_sha256 == contract sha ({str(raw.get('recipe_sha256'))[:16]})")
    rec("R2c2", live != ROUND4_CONTRACT,
        f"round-4 contract digest superseded (old={ROUND4_CONTRACT[:16]})")
    tampered = copy.deepcopy(raw)
    tampered.setdefault("infer", {})["max_frames"] = 9999
    rec("R2d", eval_contract_content_sha256(tampered) != live,
        "tampering infer.max_frames changes the content sha (seal is live)")
    manifest = raw.get("recipe_source_manifest") or {}
    mismatches = verify_recipe_source_manifest(manifest, root=ROOT)
    rec("R2e", bool(manifest) and not mismatches,
        f"recipe_source_manifest n_locked={len(manifest)} mismatches={mismatches[:4]}")
    rec("R2f", raw.get("recipe_source_manifest_sha256") == CLAIM_MANIFEST,
        f"recipe_source_manifest_sha256 live="
        f"{str(raw.get('recipe_source_manifest_sha256'))[:16]} claim={CLAIM_MANIFEST[:16]}")
    rec("R2f2", raw.get("recipe_source_manifest_sha256") != ROUND4_MANIFEST,
        f"round-4 manifest digest superseded (old={ROUND4_MANIFEST[:16]})")
    locked_missing = [e for e in ROUND5_EDITED if e not in manifest]
    rec("R2g", not locked_missing,
        f"round-5 edited sources are inside the locked manifest (missing={locked_missing})")
    for name, path in (("w1", W1), ("w2", W2)):
        y = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        t = y.get("train") or {}
        binds = (
            t.get("eval_contract_hash") == live
            and t.get("recipe_hash") == live
            and t.get("recipe_source_manifest_hash") == raw.get("recipe_source_manifest_sha256")
        )
        rec(f"R2h-{name}", binds,
            f"{name} binds contract={str(t.get('eval_contract_hash'))[:16]} "
            f"recipe={str(t.get('recipe_hash'))[:16]} "
            f"manifest={str(t.get('recipe_source_manifest_hash'))[:16]}")
    if CONTRACT_201.is_file():
        blob = CONTRACT_201.read_text(encoding="utf-8")
        rec("R2i", live in blob,
            f"v2.0.1 parent pointer repointed at the live 2.0.0 digest ({live[:16]})")
        rec("R2j", ROUND4_CONTRACT not in blob,
            "v2.0.1 no longer references the superseded round-4 contract digest")
    src72 = (ROOT / "scripts/72_train_discrete_lm.py").read_text(encoding="utf-8")
    rec("R2k", "eval_contract_hash mismatch" in src72,
        "scripts/72 still asserts declared eval_contract_hash before stamping")

    # ------------------------------------------------- FAIL #7 : resume hashes
    for name, path in (("w1", W1), ("w2", W2)):
        cfg = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        hashes = resolve_repro_hashes(cfg)
        inert = [k for k, v in hashes.items() if v is None]
        rec(f"R7a-{name}", not inert,
            f"{name} resolve_repro_hashes inert keys={inert} (n={len(hashes)})")
    cfg = yaml.safe_load(W1.read_text(encoding="utf-8")) or {}
    bad = copy.deepcopy(cfg)
    bad.setdefault("train", {})["eval_contract_hash"] = "a" * 64
    try:
        resolve_repro_hashes(bad)
        rec("R7b", False, "stale declared eval_contract_hash did NOT raise")
    except Exception as exc:  # noqa: BLE001
        rec("R7b", "mismatch" in str(exc), f"stale eval_contract_hash raises: {str(exc)[:110]}")
    # round-4 digest is now stale: a config still declaring it must be refused
    stale4 = copy.deepcopy(cfg)
    stale4.setdefault("train", {})["eval_contract_hash"] = ROUND4_CONTRACT
    try:
        resolve_repro_hashes(stale4)
        rec("R7b2", False, "config declaring the ROUND-4 contract hash was accepted")
    except Exception as exc:  # noqa: BLE001
        rec("R7b2", "mismatch" in str(exc),
            f"config pinned to the superseded round-4 digest is refused: {str(exc)[:100]}")
    for key in ("run_manifest_hash", "recipe_hash", "recipe_source_manifest_hash",
                "eval_contract_hash"):
        dropped = copy.deepcopy(cfg)
        dropped.get("train", {}).pop(key, None)
        try:
            h = resolve_repro_hashes(dropped)
            nones = [k for k, v in h.items() if v is None]
            rec(f"R7c-{key}", not nones,
                f"dropping train.{key} still resolves all keys (None={nones})")
        except Exception as exc:  # noqa: BLE001
            rec(f"R7c-{key}", False, f"raised: {str(exc)[:110]}")
    trainer_src = (ROOT / "src/minispeech/train/discrete_lm_trainer.py").read_text(encoding="utf-8")
    rec("R7d", "resume fail-closed: D1c requires" in trainer_src,
        "D1c resume fail-closed message still present in the trainer")

    # ---------------------------------------- published-digest supersession
    for doc in ("REPAIR_ROUND5.md", "d1c_reliability_gate.json", "WAVE1_REPORT.md"):
        p = RES / doc
        if not p.is_file():
            rec(f"H-{doc}", None, "absent")
            continue
        text = p.read_text(encoding="utf-8", errors="replace")
        has_live = live[:16] in text or live in text
        has_stale = ROUND4_CONTRACT[:16] in text
        # REPAIR_ROUND5.md may mention the superseded digest as history; require
        # it to be explicitly labelled superseded there.
        labelled = "superseded" in text.lower() if has_stale else True
        rec(f"H-{doc}",
            (has_live or doc == "d1c_reliability_gate.json") and (not has_stale or labelled),
            f"live_digest_present={has_live} round4_digest_present={has_stale} "
            f"labelled_superseded={labelled}")

    # ---------------------------------------------- prior-PASS regression spots
    from minispeech.data.collate_codec_seq import collate_codec_seq
    from minispeech.models.discrete_speech_lm import (
        DiscreteGenerationOutput,
        DiscreteSpeechLM,
        DiscreteSpeechLMConfig,
    )

    mcfg = DiscreteSpeechLMConfig.tiny()
    mcfg.termination_mode = "semantic_eos_v1"
    mcfg.eos_id = mcfg.codebook_size
    mcfg.codec_vocab_size = mcfg.codebook_size
    mcfg.semantic_vocab_size = mcfg.codebook_size + 1
    m = DiscreteSpeechLM(mcfg)
    batch = collate_codec_seq(
        [
            {"utt_id": "a", "input_ids": torch.tensor([3, 4]),
             "codes": torch.randint(0, m.codebook_size, (2, m.n_codebooks)), "speaker_id": 0},
            {"utt_id": "b", "input_ids": torch.tensor([5, 6, 7]),
             "codes": torch.randint(0, m.codebook_size, (4, m.n_codebooks)), "speaker_id": 1},
        ],
        pad_token_id=0,
        speech_eos_id=m.eos_id,
        termination_mode="semantic_eos_v1",
    )
    out = m(
        batch["input_ids"], batch["codes"],
        text_mask=batch.get("text_mask"), frame_mask=batch.get("frame_mask"),
        speaker_id=batch.get("speaker_id"),
        semantic_targets=batch["semantic_targets"], semantic_mask=batch["semantic_mask"],
        compute_loss=True, cb0_weight=0.0, eos_loss_weight=0.0,
    )
    out.loss["total"].backward()
    resid = [
        p for n, p in m.named_parameters()
        if "residual" in n or ("heads" in n and "cb0" not in n and "semantic" not in n)
    ]
    with_grad = [p for p in resid if p.grad is not None and torch.any(p.grad != 0)]
    rec("RXa", bool(resid) and len(with_grad) == len(resid),
        f"residual params with non-zero grads (cb0_w=0, eos_w=0): {len(with_grad)}/{len(resid)}")

    class AlwaysEOS(torch.nn.Module):
        def forward(self, h):  # noqa: ANN001, ANN201
            lg = h.new_full((*h.shape[:-1], m.semantic_vocab_size), -1e9)
            lg[..., m.eos_id] = 1e9
            return lg

    m.heads.semantic = AlwaysEOS()
    gen = m.generate(torch.tensor([[3, 4, 5]]), max_frames=4, min_frames=0,
                     max_cb0_repeat=100, temperature=0.0, top_k=0)
    rec("RXb",
        isinstance(gen, DiscreteGenerationOutput)
        and int(gen.lengths[0]) == 0
        and tuple(gen.codes.shape) == (1, 0, m.n_codebooks),
        f"immediate EOS -> codes.shape={tuple(gen.codes.shape)} "
        f"lengths={gen.lengths.tolist()} reason={gen.rows[0]['termination_reason']}")

    from minispeech.data.d1b_oracles import load_frozen_utt_ids

    sel, _, _ = load_frozen_utt_ids("d1c_select8", split="select8", verify_hash=True)
    g24, _, _ = load_frozen_utt_ids("d1c_gate24", split="gate24", verify_hash=True)
    f32, _, _ = load_frozen_utt_ids("d1c_full32", split="full32", verify_hash=True)
    n_rus = sum(1 for u in sel if "ruslan" in str(u).lower())
    rec("RXc", n_rus == 2 and len(sel) - n_rus == 6,
        f"select8 composition: {n_rus} RUSLAN + {len(sel) - n_rus} CV")
    rec("RXd", set(sel) | set(g24) == set(f32) and not (set(sel) & set(g24)),
        f"select8 | gate24 == full32 and disjoint (|s|={len(sel)} |g|={len(g24)} |f|={len(f32)})")
    rec("RXe", "d1c_select8" in SUBSET_ALIASES, "D1c subset aliases still registered")

    # --------------------------------------- round-5 surface: extra observations
    ev_src = (ROOT / "scripts/74_eval_discrete_lm.py").read_text(encoding="utf-8")
    rec("RY1", None,
        "evaluator reads checkpoint payload['hashes'] (run/recipe/contract binding): "
        f"{'yes' if 'payload' in ev_src and 'hashes' in ev_src.split('def _load_model')[1][:600] else 'no'}")
    sel_src = (ROOT / "scripts/76_select_discrete_checkpoint.py").read_text(encoding="utf-8")
    rec("RY2", None,
        "d1c_formal --promote-best refuses symlinked candidates: "
        f"{'yes' if 'is_symlink' in sel_src.split('if args.d1c_formal:')[1].split('candidates = _load_candidates')[0] else 'no'}")

    fails = sum(1 for _, st, _ in RESULTS if st == "FAIL")
    print(f"\nSUMMARY fails={fails} total={len(RESULTS)}")
    for tag, st, msg in RESULTS:
        if st == "FAIL":
            print(f"  FAILED {tag}: {msg}")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())

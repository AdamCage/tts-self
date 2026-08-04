#!/usr/bin/env python3
"""REAUDIT_ROUND4 — regression spot checks after the round-4 reseal.

Covers: FAIL #2 (contract / recipe / manifest binding), FAIL #7 (resume hashes),
published-digest audit for the round-4 reseal, and the standing prior-PASS spot
checks (residual grads, select8 composition, immediate EOS).
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
    load_eval_contract,
    verify_recipe_source_manifest,
)
from minispeech.train.discrete_lm_trainer import resolve_repro_hashes  # noqa: E402

CONTRACT_200 = ROOT / "configs/eval_contracts/d1c_semantic_eos_v2_0_0.json"
CONTRACT_201 = ROOT / "configs/eval_contracts/d1c_semantic_eos_v2_0_1.json"
W1 = ROOT / "configs/discrete_speech_lm_d1c_semantic_eos_w1.yaml"
W2 = ROOT / "configs/discrete_speech_lm_d1c_semantic_eos_w2.yaml"
RES = ROOT / ".cursor/results/d1c_semantic_eos_repair_b81ecc3e"

CLAIM_CONTRACT = "9922ce4d75932dcaff255dd20601f731f48c369647677c68249b9ff652d938af"
CLAIM_MANIFEST = "62eeeae2c6360e88923dd6416e8c5a22403d869cbbdf806aabce31e6a512b81d"
ROUND3_CONTRACT = "25c370a94f589127913623cc04db61d1fd593d4b4158f692ded9e72408006f82"
ROUND3_MANIFEST = "2d31bb9b0b37d548"

RESULTS: list[tuple[str, str, str]] = []


def rec(tag: str, ok, msg: str) -> None:
    status = "INFO" if ok is None else ("PASS" if ok else "FAIL")
    RESULTS.append((tag, status, msg))
    print(f"{tag} {status} {msg}", flush=True)


def main() -> int:  # noqa: C901, PLR0912, PLR0915
    # ------------------------------------------------- FAIL #2 : contract seal
    raw = json.loads(CONTRACT_200.read_text(encoding="utf-8"))
    live = eval_contract_content_sha256(raw)
    rec("R2a", live == raw.get("sha256"), f"declared sha256 == independent recompute {live[:16]}")
    rec(
        "R2b",
        live == CLAIM_CONTRACT,
        f"REPAIR_ROUND4 claimed contract_sha256 matches live: claim={CLAIM_CONTRACT[:16]} live={live[:16]}",
    )
    rec(
        "R2c",
        raw.get("recipe_sha256") == live,
        f"recipe_sha256 == contract sha ({str(raw.get('recipe_sha256'))[:16]})",
    )
    tampered = copy.deepcopy(raw)
    tampered.setdefault("infer", {})["max_frames"] = 9999
    rec(
        "R2d",
        eval_contract_content_sha256(tampered) != live,
        "tampering infer.max_frames changes the content sha (seal is live)",
    )
    manifest = raw.get("recipe_source_manifest") or {}
    mismatches = verify_recipe_source_manifest(manifest, root=ROOT)
    rec(
        "R2e",
        bool(manifest) and not mismatches,
        f"recipe_source_manifest n_locked={len(manifest)} mismatches={mismatches[:4]}",
    )
    rec(
        "R2f",
        raw.get("recipe_source_manifest_sha256") == CLAIM_MANIFEST,
        f"recipe_source_manifest_sha256 live={str(raw.get('recipe_source_manifest_sha256'))[:16]} "
        f"claim={CLAIM_MANIFEST[:16]}",
    )
    # locked set must actually include the files edited in round 4
    edited = [
        "scripts/74_eval_discrete_lm.py",
        "src/minispeech/eval/discrete_report_schema.py",
        "scripts/76_select_discrete_checkpoint.py",
    ]
    locked_missing = [e for e in edited if e not in manifest]
    rec(
        "R2g",
        not locked_missing,
        f"round-4 edited sources are inside the locked manifest (missing={locked_missing})",
    )
    for name, path in (("w1", W1), ("w2", W2)):
        y = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        t = y.get("train") or {}
        binds = (
            t.get("eval_contract_hash") == live
            and t.get("recipe_hash") == live
            and t.get("recipe_source_manifest_hash") == raw.get("recipe_source_manifest_sha256")
        )
        rec(
            f"R2h-{name}",
            binds,
            f"{name} binds contract={str(t.get('eval_contract_hash'))[:16]} "
            f"recipe={str(t.get('recipe_hash'))[:16]} "
            f"manifest={str(t.get('recipe_source_manifest_hash'))[:16]}",
        )
    if CONTRACT_201.is_file():
        c201 = json.loads(CONTRACT_201.read_text(encoding="utf-8"))
        blob = json.dumps(c201, ensure_ascii=False)
        rec(
            "R2i",
            live in blob,
            f"v2.0.1 parent pointer repointed at the live 2.0.0 digest ({live[:16]})",
        )
        rec(
            "R2j",
            ROUND3_CONTRACT not in blob,
            "v2.0.1 no longer references the superseded round-3 contract digest",
        )
    src72 = (ROOT / "scripts/72_train_discrete_lm.py").read_text(encoding="utf-8")
    rec(
        "R2k",
        "eval_contract_hash mismatch" in src72,
        "scripts/72 still asserts declared eval_contract_hash before stamping",
    )

    # ------------------------------------------------- FAIL #7 : resume hashes
    for name, path in (("w1", W1), ("w2", W2)):
        cfg = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        hashes = resolve_repro_hashes(cfg)
        inert = [k for k, v in hashes.items() if v is None]
        rec(f"R7a-{name}", not inert, f"{name} resolve_repro_hashes inert keys={inert} (n={len(hashes)})")
    cfg = yaml.safe_load(W1.read_text(encoding="utf-8")) or {}
    bad = copy.deepcopy(cfg)
    bad.setdefault("train", {})["eval_contract_hash"] = "a" * 64
    try:
        resolve_repro_hashes(bad)
        rec("R7b", False, "stale declared eval_contract_hash did NOT raise")
    except Exception as exc:  # noqa: BLE001
        rec("R7b", "mismatch" in str(exc), f"stale eval_contract_hash raises: {str(exc)[:110]}")
    for key in ("run_manifest_hash", "recipe_hash", "recipe_source_manifest_hash", "eval_contract_hash"):
        dropped = copy.deepcopy(cfg)
        dropped.get("train", {}).pop(key, None)
        try:
            h = resolve_repro_hashes(dropped)
            nones = [k for k, v in h.items() if v is None]
            rec(f"R7c-{key}", not nones, f"dropping train.{key} still resolves all keys (None={nones})")
        except Exception as exc:  # noqa: BLE001
            rec(f"R7c-{key}", False, f"raised: {str(exc)[:110]}")
    trainer_src = (ROOT / "src/minispeech/train/discrete_lm_trainer.py").read_text(encoding="utf-8")
    rec(
        "R7d",
        "resume fail-closed: D1c requires" in trainer_src,
        "D1c resume fail-closed message still present in the trainer",
    )

    # -------------------------------------------- published-digest audit (docs)
    for doc in ("REPAIR_ROUND4.md", "d1c_reliability_gate.json", "WAVE1_REPORT.md"):
        p = RES / doc
        if not p.is_file():
            rec(f"H-{doc}", None, "absent")
            continue
        text = p.read_text(encoding="utf-8", errors="replace")
        has_live = live[:16] in text or live in text
        has_stale = ROUND3_CONTRACT[:16] in text
        rec(
            f"H-{doc}",
            (has_live or doc == "d1c_reliability_gate.json") and not has_stale,
            f"live_digest_present={has_live} round3_stale_digest_present={has_stale}",
        )

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
    out = m(
        batch["input_ids"],
        batch["codes"],
        text_mask=batch.get("text_mask"),
        frame_mask=batch.get("frame_mask"),
        speaker_id=batch.get("speaker_id"),
        semantic_targets=batch["semantic_targets"],
        semantic_mask=batch["semantic_mask"],
        compute_loss=True,
        cb0_weight=0.0,
        eos_loss_weight=0.0,
    )
    out.loss["total"].backward()
    resid = [
        p
        for n, p in m.named_parameters()
        if "residual" in n or ("heads" in n and "cb0" not in n and "semantic" not in n)
    ]
    with_grad = [p for p in resid if p.grad is not None and torch.any(p.grad != 0)]
    rec(
        "RXa",
        bool(resid) and len(with_grad) == len(resid),
        f"residual params with non-zero grads (cb0_w=0, eos_w=0): {len(with_grad)}/{len(resid)}",
    )

    class AlwaysEOS(torch.nn.Module):
        def forward(self, h):  # noqa: ANN001, ANN201
            lg = h.new_full((*h.shape[:-1], m.semantic_vocab_size), -1e9)
            lg[..., m.eos_id] = 1e9
            return lg

    m.heads.semantic = AlwaysEOS()
    gen = m.generate(
        torch.tensor([[3, 4, 5]]), max_frames=4, min_frames=0, max_cb0_repeat=100,
        temperature=0.0, top_k=0,
    )
    rec(
        "RXb",
        isinstance(gen, DiscreteGenerationOutput)
        and int(gen.lengths[0]) == 0
        and tuple(gen.codes.shape) == (1, 0, m.n_codebooks),
        f"immediate EOS -> codes.shape={tuple(gen.codes.shape)} lengths={gen.lengths.tolist()} "
        f"reason={gen.rows[0]['termination_reason']}",
    )

    from minispeech.data.d1b_oracles import load_frozen_utt_ids

    sel, _, _ = load_frozen_utt_ids("d1c_select8", split="select8", verify_hash=True)
    g24, _, _ = load_frozen_utt_ids("d1c_gate24", split="gate24", verify_hash=True)
    f32, _, _ = load_frozen_utt_ids("d1c_full32", split="full32", verify_hash=True)
    n_rus = sum(1 for u in sel if "ruslan" in str(u).lower())
    n_cv = len(sel) - n_rus
    rec("RXc", n_rus == 2 and n_cv == 6, f"select8 composition: {n_rus} RUSLAN + {n_cv} CV")
    rec(
        "RXd",
        set(sel) | set(g24) == set(f32) and not (set(sel) & set(g24)),
        f"select8 | gate24 == full32 and disjoint (|s|={len(sel)} |g|={len(g24)} |f|={len(f32)})",
    )
    rec("RXe", "d1c_select8" in SUBSET_ALIASES, "D1c subset aliases still registered")

    fails = sum(1 for _, st, _ in RESULTS if st == "FAIL")
    print(f"\nSUMMARY fails={fails} total={len(RESULTS)}")
    for tag, st, msg in RESULTS:
        if st == "FAIL":
            print(f"  FAILED {tag}: {msg}")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())

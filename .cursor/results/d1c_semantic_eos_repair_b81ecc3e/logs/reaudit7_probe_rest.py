#!/usr/bin/env python3
"""REAUDIT_ROUND7 — reseal + prior-blocker regression spot checks."""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "src"))

from minispeech.data.d1b_oracles import (  # noqa: E402
    SUBSET_ALIASES,
    eval_contract_content_sha256,
    load_frozen_utt_ids,
    verify_recipe_source_manifest,
)
from minispeech.data.collate_codec_seq import collate_codec_seq  # noqa: E402
from minispeech.models.discrete_speech_lm import (  # noqa: E402
    DiscreteGenerationOutput,
    DiscreteSpeechLM,
    DiscreteSpeechLMConfig,
)
from minispeech.train.discrete_lm_trainer import resolve_repro_hashes  # noqa: E402

CONTRACT_200 = ROOT / "configs/eval_contracts/d1c_semantic_eos_v2_0_0.json"
CONTRACT_201 = ROOT / "configs/eval_contracts/d1c_semantic_eos_v2_0_1.json"
W1 = ROOT / "configs/discrete_speech_lm_d1c_semantic_eos_w1.yaml"
W2 = ROOT / "configs/discrete_speech_lm_d1c_semantic_eos_w2.yaml"
RES = ROOT / ".cursor/results/d1c_semantic_eos_repair_b81ecc3e"
AUTHORING_WAVE1 = Path(
    "/mnt/c/projects/tts-self/.cursor/results/d1c_semantic_eos_repair_b81ecc3e/WAVE1_REPORT.md"
)
AUTHORING_RES = Path(
    "/mnt/c/projects/tts-self/.cursor/results/d1c_semantic_eos_repair_b81ecc3e"
)

CLAIM_CONTRACT = "181acd0274b66253facb9277a9473336d0c52cf87d5fe23dd39199d31518baad"
CLAIM_MANIFEST = "ebb91df5f1574d11e30017eae870642aae6373777d123741e25add62726551fc"
ROUND6_CONTRACT = "c9fd265ac715b3d600dab1fcdf3d0b941d8db6b6fb3d9c9495728c0c55e124cf"
ROUND6_MANIFEST = "85b896a907769af467f9e76d11b2347849b1c7fb776fcd7a6104d1885877ebbd"
ROUND5_CONTRACT = "cff9e9f5dd11ae66d66e44eddc8768f4989daa8c0339822d77252289a155baad"
ROUND5_MANIFEST = "e505acf99640a3c801f9a9451bc60a19c74d2460413564cb0d68b2ab6fc2b596"

ROUND7_EDITED = [
    "scripts/76_select_discrete_checkpoint.py",
]

RESULTS: list[tuple[str, str, str]] = []
LOG = RES / "logs" / "reaudit7_probe_rest.log"


def rec(tag: str, ok, msg: str) -> None:
    status = "INFO" if ok is None else ("PASS" if ok else "FAIL")
    RESULTS.append((tag, status, msg))
    print(f"{tag} {status} {msg}", flush=True)


def main() -> int:  # noqa: C901, PLR0912, PLR0915
    raw = json.loads(CONTRACT_200.read_text(encoding="utf-8"))
    live = eval_contract_content_sha256(raw)
    rec("R2a", live == raw.get("sha256"),
        f"declared sha256 == independent recompute {live[:16]}")
    rec("R2b", live == CLAIM_CONTRACT,
        f"REPAIR_ROUND7 claimed contract_sha256 matches live: "
        f"claim={CLAIM_CONTRACT[:16]} live={live[:16]}")
    rec("R2c", raw.get("recipe_sha256") == live,
        f"recipe_sha256 == contract sha ({str(raw.get('recipe_sha256'))[:16]})")
    rec("R2c2", live != ROUND6_CONTRACT,
        f"round-6 contract digest superseded (old={ROUND6_CONTRACT[:16]})")
    rec("R2c3", live != ROUND5_CONTRACT,
        f"round-5 contract digest superseded (old={ROUND5_CONTRACT[:16]})")
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
    rec("R2f2", raw.get("recipe_source_manifest_sha256") != ROUND6_MANIFEST,
        f"round-6 manifest digest superseded (old={ROUND6_MANIFEST[:16]})")
    locked_missing = [e for e in ROUND7_EDITED if e not in manifest]
    rec("R2g", not locked_missing,
        f"round-7 edited sources are inside the locked manifest (missing={locked_missing})")
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
        rec("R2j", ROUND6_CONTRACT not in blob and ROUND5_CONTRACT not in blob,
            "v2.0.1 no longer references superseded round-5/6 contract digests")

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
    stale6 = copy.deepcopy(cfg)
    stale6.setdefault("train", {})["eval_contract_hash"] = ROUND6_CONTRACT
    try:
        resolve_repro_hashes(stale6)
        rec("R7b2", False, "config declaring the ROUND-6 contract hash was accepted")
    except Exception as exc:  # noqa: BLE001
        rec("R7b2", "mismatch" in str(exc),
            f"config pinned to the superseded round-6 digest is refused: {str(exc)[:100]}")
    trainer_src = (ROOT / "src/minispeech/train/discrete_lm_trainer.py").read_text(encoding="utf-8")
    rec("R7d", "resume fail-closed: D1c requires" in trainer_src,
        "D1c resume fail-closed message still present in the trainer")

    for label, path in (
        ("WAVE1_REPORT.md", RES / "WAVE1_REPORT.md"),
        ("WAVE1_REPORT.md_authoring", AUTHORING_WAVE1),
    ):
        if not path.is_file():
            rec(f"H-{label}", False, f"absent: {path}")
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        has_live = CLAIM_CONTRACT in text and CLAIM_MANIFEST in text
        publishes_stale6 = ROUND6_CONTRACT[:16] in text and CLAIM_CONTRACT[:16] not in text
        rec(f"H-{label}", has_live and not publishes_stale6,
            f"live={has_live} publishes_stale6_as_current={publishes_stale6}")

    # Residual grads / immediate EOS / select8
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
        and tuple(gen.codes.shape) == (1, 0, m.n_codebooks),
        f"immediate EOS codes.shape={tuple(gen.codes.shape)}")

    sel, _, _ = load_frozen_utt_ids("d1c_select8", split="select8", verify_hash=True)
    g24, _, _ = load_frozen_utt_ids("d1c_gate24", split="gate24", verify_hash=True)
    f32, _, _ = load_frozen_utt_ids("d1c_full32", split="full32", verify_hash=True)
    n_rus = sum(1 for u in sel if str(u).startswith("ruslan_") or "/ruslan" in str(u).lower()
                or str(u).split("_")[0].isdigit() is False and "cv_" not in str(u)
                and "common_voice" not in str(u))
    # Match round-6 heuristic: ids containing 'ruslan' (case-insensitive)
    n_rus = sum(1 for u in sel if "ruslan" in str(u).lower())
    rec("RXc", len(sel) == 8 and n_rus == 2 and (len(sel) - n_rus) == 6,
        f"select8 composition: {n_rus} RUSLAN + {len(sel) - n_rus} CV")
    rec("RXd",
        set(sel).isdisjoint(g24) and set(sel) | set(g24) == set(f32),
        f"select8 | gate24 == full32 and disjoint (|s|={len(sel)} |g|={len(g24)} |f|={len(f32)})")
    rec("RXe", "d1c_select8" in SUBSET_ALIASES, "D1c subset aliases still registered")

    n_pass = sum(1 for _, s, _ in RESULTS if s == "PASS")
    n_fail = sum(1 for _, s, _ in RESULTS if s == "FAIL")
    n_info = sum(1 for _, s, _ in RESULTS if s == "INFO")
    summary = f"SUMMARY pass={n_pass} fail={n_fail} info={n_info} total={len(RESULTS)}"
    print(summary, flush=True)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    LOG.write_text("\n".join(f"{t} {s} {m}" for t, s, m in RESULTS) + "\n" + summary + "\n",
                   encoding="utf-8")
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())

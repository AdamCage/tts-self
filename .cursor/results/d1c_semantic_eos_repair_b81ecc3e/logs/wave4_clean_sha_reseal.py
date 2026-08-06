#!/usr/bin/env python3
"""Wave 4 prep: set repro.clean_sha_available=true; reseal without poisoning recipe YAML digests."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(".").resolve()
sys.path.insert(0, str(ROOT / "src"))

from minispeech.data.d1b_oracles import (  # noqa: E402
    eval_contract_content_sha256,
    recipe_source_file_sha256,
    verify_recipe_source_manifest,
)
from minispeech.train.discrete_lm_trainer import resolve_repro_hashes  # noqa: E402
from minispeech.utils import load_yaml  # noqa: E402

LOCKED_SOURCES = [
    "configs/discrete_speech_lm_d1c_semantic_eos_w1.yaml",
    "configs/discrete_speech_lm_d1c_semantic_eos_w2.yaml",
    "src/minispeech/models/discrete_speech_lm.py",
    "src/minispeech/models/multi_codebook_head.py",
    "src/minispeech/data/collate_codec_seq.py",
    "src/minispeech/data/discrete_oracles.py",
    "src/minispeech/data/d1b_oracles.py",
    "src/minispeech/data/codec_seq_protocol.py",
    "src/minispeech/constants.py",
    "src/minispeech/train/discrete_lm_trainer.py",
    "src/minispeech/infer/discrete_config.py",
    "src/minispeech/eval/discrete_n2.py",
    "src/minispeech/eval/discrete_report_schema.py",
    "src/minispeech/eval/asr.py",
    "scripts/72_train_discrete_lm.py",
    "scripts/73_infer_discrete_lm.py",
    "scripts/74_eval_discrete_lm.py",
    "scripts/76_select_discrete_checkpoint.py",
    "scripts/77_freeze_d1c_oracles.py",
]


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def recipe_projection(contract: dict[str, Any]) -> dict[str, Any]:
    skip = {
        "sha256",
        "recipe_sha256",
        "confirm_subset",
        "confirm_seed_reveal",
        "confirm_train_seed_plaintext",
    }
    return {
        k: v
        for k, v in contract.items()
        if k not in skip and not str(k).startswith("_")
    }


def _update_yaml_train_hashes(
    path: Path,
    *,
    eval_contract_hash: str,
    recipe_hash: str,
    recipe_source_manifest_hash: str,
    run_manifest_hash: str | None,
) -> None:
    """Update only train.* binding fields (stripped from recipe YAML digest)."""
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(data, dict)
    # Drop any accidental top-level binding keys from a prior bad reseal.
    for key in (
        "eval_contract_hash",
        "recipe_sha256",
        "recipe_source_manifest_sha256",
        "recipe_hash",
        "recipe_source_manifest_hash",
        "run_manifest_hash",
    ):
        data.pop(key, None)
    train = data.setdefault("train", {})
    train["eval_contract_hash"] = eval_contract_hash
    train["recipe_hash"] = recipe_hash
    train["recipe_source_manifest_hash"] = recipe_source_manifest_hash
    if run_manifest_hash:
        train["run_manifest_hash"] = run_manifest_hash
    else:
        train.pop("run_manifest_hash", None)
    path.write_text(
        yaml.safe_dump(
            data, allow_unicode=True, sort_keys=False, default_flow_style=None
        ),
        encoding="utf-8",
    )


def main() -> None:
    contract_path = ROOT / "configs/eval_contracts/d1c_semantic_eos_v2_0_0.json"
    w1 = ROOT / "configs/discrete_speech_lm_d1c_semantic_eos_w1.yaml"
    w2 = ROOT / "configs/discrete_speech_lm_d1c_semantic_eos_w2.yaml"
    v201 = ROOT / "configs/eval_contracts/d1c_semantic_eos_v2_0_1.json"
    art = ROOT / ".cursor/results/d1c_semantic_eos_repair_b81ecc3e/artifacts"

    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    repro = dict(contract.get("repro") or {})
    repro["clean_sha_available"] = True
    repro.setdefault("dev_subset_selection_seed", 20260805)
    repro.setdefault("bootstrap_seed", 42)
    repro.setdefault("random_baseline_base_seed", 42001)
    contract["repro"] = repro

    # Manifest from current sources BEFORE yaml rewrite; binding-only yaml
    # edits must not change these digests.
    manifest = {rel: recipe_source_file_sha256(ROOT / rel) for rel in LOCKED_SOURCES}
    contract["recipe_source_manifest"] = manifest
    contract["recipe_source_manifest_sha256"] = hashlib.sha256(
        _canonical(manifest)
    ).hexdigest()
    contract.pop("sha256", None)
    contract["recipe_sha256"] = hashlib.sha256(
        _canonical(recipe_projection(contract))
    ).hexdigest()
    contract["sha256"] = eval_contract_content_sha256(contract)
    # Settle recipe after sha field exclusion consistency.
    contract["recipe_sha256"] = hashlib.sha256(
        _canonical(recipe_projection(contract))
    ).hexdigest()
    contract["sha256"] = eval_contract_content_sha256(contract)
    assert contract["sha256"] == eval_contract_content_sha256(contract)

    live = contract["sha256"]
    recipe = contract["recipe_sha256"]
    srcman = contract["recipe_source_manifest_sha256"]
    write_json(contract_path, contract)
    write_json(art / "d1c_semantic_eos_v2_0_0.json", contract)

    for yml in (w1, w2):
        _update_yaml_train_hashes(
            yml,
            eval_contract_hash=live,
            recipe_hash=recipe,
            recipe_source_manifest_hash=srcman,
            run_manifest_hash=None,
        )
        cfg = load_yaml(yml)
        hashes = resolve_repro_hashes(cfg)
        assert hashes["eval_contract"] == live
        assert hashes["recipe"] == recipe
        assert hashes["recipe_source_manifest"] == srcman
        _update_yaml_train_hashes(
            yml,
            eval_contract_hash=live,
            recipe_hash=recipe,
            recipe_source_manifest_hash=srcman,
            run_manifest_hash=str(hashes["run_manifest"]),
        )

    for rel in (
        "configs/discrete_speech_lm_d1c_semantic_eos_w1.yaml",
        "configs/discrete_speech_lm_d1c_semantic_eos_w2.yaml",
    ):
        got = recipe_source_file_sha256(ROOT / rel)
        assert got == manifest[rel], (rel, got, manifest[rel])

    mismatches = verify_recipe_source_manifest(manifest, root=ROOT)
    assert mismatches == [], mismatches

    if v201.is_file():
        c201 = json.loads(v201.read_text(encoding="utf-8"))
        c201["parent_sha256"] = live
        c201["parent_recipe_sha256"] = recipe
        write_json(v201, c201)

    out = {
        "contract_sha256": live,
        "recipe_sha256": recipe,
        "recipe_source_manifest_sha256": srcman,
        "clean_sha_available": True,
        "w1_yaml_digest": manifest[
            "configs/discrete_speech_lm_d1c_semantic_eos_w1.yaml"
        ],
        "w2_yaml_digest": manifest[
            "configs/discrete_speech_lm_d1c_semantic_eos_w2.yaml"
        ],
        "w1_eval_contract_hash": load_yaml(w1)["train"]["eval_contract_hash"],
        "w2_eval_contract_hash": load_yaml(w2)["train"]["eval_contract_hash"],
        "manifest_mismatches": mismatches,
    }
    write_json(art / "wave4_clean_sha_reseal_hashes.json", out)
    print(json.dumps(out, indent=2))
    assert out["w1_eval_contract_hash"] == live
    assert out["w2_eval_contract_hash"] == live
    print("CLEAN_SHA_RESEAL_OK")


if __name__ == "__main__":
    main()

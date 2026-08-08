#!/usr/bin/env bash
set -euo pipefail
# Reseal D1c contracts against Linux LF working tree (round-7 was CRLF-poisoned).
cd ~/projects/tts-self
source .venv/bin/activate
export HF_HOME="$HOME/hf_cache"
export PYTHONPATH="$PWD/src"
export CUDA_VISIBLE_DEVICES=
RESULTS="$PWD/.cursor/results/d1c_semantic_eos_repair_b81ecc3e"
mkdir -p "$RESULTS/logs" "$RESULTS/artifacts"

python - <<'PY'
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(".").resolve()
sys.path.insert(0, str(ROOT / "src"))

from minispeech.data.d1b_oracles import (
    eval_contract_content_sha256,
    recipe_source_file_sha256,
    verify_recipe_source_manifest,
)
from minispeech.train.discrete_lm_trainer import resolve_repro_hashes
from minispeech.utils import load_yaml

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


def _update_yaml_hashes(path: Path, *, eval_contract: str, recipe: str, manifest: str) -> None:
    text = path.read_text(encoding="utf-8")
    data = yaml.safe_load(text)
    assert isinstance(data, dict)
    data["eval_contract_hash"] = eval_contract
    data["recipe_sha256"] = recipe
    data["recipe_source_manifest_sha256"] = manifest
    # Preserve key order-ish via dump; recipe_source_file_sha256 strips these later.
    path.write_text(
        yaml.safe_dump(data, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )


contract_path = ROOT / "configs/eval_contracts/d1c_semantic_eos_v2_0_0.json"
contract = json.loads(contract_path.read_text(encoding="utf-8"))

manifest = {
    rel: recipe_source_file_sha256(ROOT / rel) for rel in LOCKED_SOURCES
}
# First pass: stamp source digests (yaml hashes will change after we write binding fields).
contract["recipe_source_manifest"] = manifest
# Temporary placeholder digests so projection can be hashed after yaml update loop.

# Iterate: update yaml binding fields from provisional contract digests, recompute.
for _ in range(4):
    manifest = {rel: recipe_source_file_sha256(ROOT / rel) for rel in LOCKED_SOURCES}
    contract["recipe_source_manifest"] = manifest
    contract["recipe_source_manifest_sha256"] = hashlib.sha256(
        _canonical(manifest)
    ).hexdigest()
    proj = recipe_projection(contract)
    recipe = hashlib.sha256(_canonical(proj)).hexdigest()
    contract["recipe_sha256"] = recipe
    # content sha excludes self sha256 field conventionally via helper
    # Write without sha256 then compute
    contract.pop("sha256", None)
    write_json(contract_path, contract)
    content_sha = eval_contract_content_sha256(contract_path)
    contract["sha256"] = content_sha
    write_json(contract_path, contract)

    for yml in (
        ROOT / "configs/discrete_speech_lm_d1c_semantic_eos_w1.yaml",
        ROOT / "configs/discrete_speech_lm_d1c_semantic_eos_w2.yaml",
    ):
        _update_yaml_hashes(
            yml,
            eval_contract=content_sha,
            recipe=recipe,
            manifest=contract["recipe_source_manifest_sha256"],
        )

# Final verify
manifest = {rel: recipe_source_file_sha256(ROOT / rel) for rel in LOCKED_SOURCES}
contract["recipe_source_manifest"] = manifest
contract["recipe_source_manifest_sha256"] = hashlib.sha256(_canonical(manifest)).hexdigest()
contract.pop("sha256", None)
write_json(contract_path, contract)
content_sha = eval_contract_content_sha256(contract_path)
contract["sha256"] = content_sha
contract["recipe_sha256"] = hashlib.sha256(_canonical(recipe_projection(contract))).hexdigest()
# Keep recipe == content sha convention used in round7 if that was the design
# Round7 set both to same value via their script — match that after projection settle.
write_json(contract_path, contract)

# One more yaml stamp + remanifest (yaml strip means source digests for yaml ignore binding fields)
for yml in (
    ROOT / "configs/discrete_speech_lm_d1c_semantic_eos_w1.yaml",
    ROOT / "configs/discrete_speech_lm_d1c_semantic_eos_w2.yaml",
):
    _update_yaml_hashes(
        yml,
        eval_contract=contract["sha256"],
        recipe=contract["recipe_sha256"],
        manifest=contract["recipe_source_manifest_sha256"],
    )

manifest = {rel: recipe_source_file_sha256(ROOT / rel) for rel in LOCKED_SOURCES}
assert verify_recipe_source_manifest(manifest, root=ROOT) == []
contract["recipe_source_manifest"] = manifest
contract["recipe_source_manifest_sha256"] = hashlib.sha256(_canonical(manifest)).hexdigest()
# Recompute recipe/content after manifest finalize
contract.pop("sha256", None)
proj = recipe_projection({**contract, "recipe_source_manifest": manifest})
contract["recipe_source_manifest"] = manifest
contract["recipe_sha256"] = hashlib.sha256(_canonical(proj)).hexdigest()
write_json(contract_path, contract)
contract["sha256"] = eval_contract_content_sha256(contract_path)
# Align recipe_sha256 with content sha if projection equals content-hash inputs
# Re-read round7: both were identical 181acd02... — that happens when sha256 field
# is excluded from both content and recipe projection consistently.
write_json(contract_path, contract)

for yml in (
    ROOT / "configs/discrete_speech_lm_d1c_semantic_eos_w1.yaml",
    ROOT / "configs/discrete_speech_lm_d1c_semantic_eos_w2.yaml",
):
    _update_yaml_hashes(
        yml,
        eval_contract=contract["sha256"],
        recipe=contract["recipe_sha256"],
        manifest=contract["recipe_source_manifest_sha256"],
    )

# Parent 2.0.1
v201 = ROOT / "configs/eval_contracts/d1c_semantic_eos_v2_0_1.json"
if v201.is_file():
    c201 = json.loads(v201.read_text(encoding="utf-8"))
    c201["parent_sha256"] = contract["sha256"]
    # keep other fields; rewrite parent only
    write_json(v201, c201)

mismatches = verify_recipe_source_manifest(contract["recipe_source_manifest"], root=ROOT)
w1 = load_yaml(ROOT / "configs/discrete_speech_lm_d1c_semantic_eos_w1.yaml")
w2 = load_yaml(ROOT / "configs/discrete_speech_lm_d1c_semantic_eos_w2.yaml")

out = {
    "contract_sha256": contract["sha256"],
    "recipe_sha256": contract["recipe_sha256"],
    "recipe_source_manifest_sha256": contract["recipe_source_manifest_sha256"],
    "n_locked": len(LOCKED_SOURCES),
    "manifest_mismatches": mismatches,
    "w1_eval_contract_hash": w1.get("eval_contract_hash"),
    "w2_eval_contract_hash": w2.get("eval_contract_hash"),
    "note": "LF reseal on WSL clean tree after round-7 CRLF seal poison",
}
write_json(ROOT / ".cursor/results/d1c_semantic_eos_repair_b81ecc3e/artifacts/repair_wave3_lf_reseal_hashes.json", out)
print(json.dumps(out, indent=2))
assert mismatches == []
print("LF_RESEAL_OK")
PY

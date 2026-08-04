#!/usr/bin/env python3
"""Repair D1c DEV32 select8 composition + contract hashes (CPU only, no GPU).

Re-splits select8/gate24 from existing sealed full32 inventory using the fixed
2 RUSLAN + 6 CV policy, refreshes file/split hashes, updates contract 2.0.0
registry entries and recipe_source_manifest, and re-stamps sha256 fields.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from minispeech.data.d1b_oracles import eval_contract_content_sha256  # noqa: E402
from minispeech.data.discrete_oracles import (  # noqa: E402
    items_sha256,
    payload_content_file_sha256,
)

# Import freezer helpers without executing main.
import importlib.util

spec = importlib.util.spec_from_file_location(
    "freeze_d1c", ROOT / "scripts" / "77_freeze_d1c_oracles.py"
)
assert spec and spec.loader
mod = importlib.util.module_from_spec(spec)
sys.modules["freeze_d1c"] = mod
spec.loader.exec_module(mod)

EligibleRow = mod.EligibleRow
RUSLAN = mod.RUSLAN
CV = mod.CV
split_dev = mod.split_dev
write_json = mod.write_json


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def _file_sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def recipe_projection(contract: dict[str, Any]) -> dict[str, Any]:
    """Match build_d1c_contract_v2 recipe projection (no confirm reveal fields)."""
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


def main() -> int:
    dev_path = ROOT / "configs" / "train_subsets" / "d1c_dev32.json"
    contract_path = ROOT / "configs" / "eval_contracts" / "d1c_semantic_eos_v2_0_0.json"
    payload = json.loads(dev_path.read_text(encoding="utf-8"))
    rows_meta = payload.get("rows_meta") or []
    edges = (
        float(payload["selection"]["duration_edges_s"]["short_lt"]),
        float(payload["selection"]["duration_edges_s"]["medium_lt"]),
    )
    q1 = float(payload["selection"]["ratio_quartiles"]["q1"])
    q3 = float(payload["selection"]["ratio_quartiles"]["q3"])

    selected: list[Any] = []
    for row in rows_meta:
        corpus = str(row["corpus"])
        selected.append(
            EligibleRow(
                utt_id=str(row["utt_id"]),
                corpus=corpus,
                speaker_id=str(row.get("speaker_id") or ""),
                dur_s=float(row["dur_s"]),
                codec_frames=int(row["codec_frames"]),
                text_token_count=int(row["text_token_count"]),
                text_norm="",
                audio_sha256="",
                wav_path="",
                pcm_sha256="",
            )
        )
    # Preserve original full32 order from payload.
    by_id = {r.utt_id: r for r in selected}
    ordered = [by_id[u] for u in payload["utt_ids"]]
    splits = split_dev(ordered, edges=edges, q1=q1, q3=q3)
    n_ruslan = sum(1 for u in splits["select8"] if str(u).startswith("ruslan_"))
    n_cv = len(splits["select8"]) - n_ruslan
    assert n_ruslan == 2 and n_cv == 6, (n_ruslan, n_cv)

    body = {k: v for k, v in payload.items() if k != "file_sha256"}
    body["splits"] = splits
    body["split_hashes"] = {k: items_sha256(v) for k, v in splits.items()}
    # full32 items hash unchanged if utt_ids unchanged.
    body["items_sha256"] = items_sha256(body["utt_ids"])
    assert body["items_sha256"] == body["split_hashes"]["full32"]

    file_sha = write_json(dev_path, body)
    body["file_sha256"] = file_sha
    # Declared field = content hash of body without the field (freezer semantics).
    content_sha = write_json(dev_path, {k: v for k, v in body.items() if k != "file_sha256"})
    final = {**{k: v for k, v in body.items() if k != "file_sha256"}, "file_sha256": content_sha}
    write_json(dev_path, final)
    assert payload_content_file_sha256(final) == content_sha

    # Mirror under results artifacts.
    art = ROOT / ".cursor" / "results" / "d1c_semantic_eos_repair_b81ecc3e" / "artifacts"
    art.mkdir(parents=True, exist_ok=True)
    write_json(art / "d1c_dev32.json", final)

    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    parent_file_sha = content_sha
    for key, split in (
        ("dev32", "full32"),
        ("select8", "select8"),
        ("gate24", "gate24"),
        ("full32", "full32"),
    ):
        entry = dict(contract["subsets"][key])
        entry["path"] = "configs/train_subsets/d1c_dev32.json"
        entry["file_sha256"] = parent_file_sha
        entry["items_sha256"] = body["split_hashes"][split]
        if key == "dev32":
            entry["split_hashes"] = dict(body["split_hashes"])
            entry["n"] = 32
            entry["split"] = "full32"
        else:
            entry["n"] = {"select8": 8, "gate24": 24, "full32": 32}[key]
            entry["split"] = split
            entry["parent"] = "dev32"
        contract["subsets"][key] = entry

    # Refresh recipe_source_manifest after code repairs.
    manifest = {}
    for rel in list(contract.get("recipe_source_manifest") or {}):
        path = ROOT / rel
        if not path.is_file():
            raise SystemExit(f"missing recipe source: {rel}")
        manifest[rel] = _file_sha(path)
    contract["recipe_source_manifest"] = manifest
    contract["recipe_source_manifest_sha256"] = hashlib.sha256(
        _canonical(manifest)
    ).hexdigest()

    # Stamp recipe + content hashes (exclude sha256 and recipe_sha256).
    recipe_sha = hashlib.sha256(_canonical(recipe_projection(contract))).hexdigest()
    contract["recipe_sha256"] = recipe_sha
    content = eval_contract_content_sha256(contract)
    # After setting recipe_sha256, content sha excludes it — recompute once more.
    contract["sha256"] = content
    # Ensure self-check: content hash of body excluding sha256+recipe_sha256.
    assert eval_contract_content_sha256(contract) == contract["sha256"]
    # For 2.0.0 without confirm fields, recipe projection hash should equal content.
    # Keep recipe_sha256 as projection hash (may equal content).
    contract["recipe_sha256"] = hashlib.sha256(
        _canonical(recipe_projection(contract))
    ).hexdigest()
    # Re-stamp sha256 after recipe field update (sha excludes recipe_sha256).
    contract["sha256"] = eval_contract_content_sha256(contract)

    write_json(contract_path, contract)
    write_json(art / "d1c_semantic_eos_v2_0_0.json", contract)

    report = {
        "dev32_file_sha256": content_sha,
        "select8_items_sha256": body["split_hashes"]["select8"],
        "gate24_items_sha256": body["split_hashes"]["gate24"],
        "full32_items_sha256": body["split_hashes"]["full32"],
        "select8": splits["select8"],
        "select8_composition": {"ruslan": n_ruslan, "cv": n_cv},
        "contract_sha256": contract["sha256"],
        "recipe_sha256": contract["recipe_sha256"],
        "recipe_source_manifest_sha256": contract["recipe_source_manifest_sha256"],
    }
    write_json(art / "repair_oracle_contract_hashes.json", report)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

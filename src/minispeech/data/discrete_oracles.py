"""Generic frozen-oracle loading for formal D1c evaluation.

The v2 path deliberately does not infer an oracle from a filename or a first-N
slice.  A formal caller must bind a subset alias to an active contract and to a
payload that carries both file and ordered-ID hashes.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from minispeech.utils import expand

TRACKN_V2_SPEAKER_SHA256 = "64e7cdc5c3447a43d7484b7538f9fd5df88551650d7b95ddb23110f9c36dca6e"


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def items_sha256(items: Sequence[str]) -> str:
    return hashlib.sha256(canonical_json_bytes([str(x) for x in items])).hexdigest()


def file_sha256(path: str | Path) -> str:
    p = expand(path)
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_json_object(path: str | Path) -> dict[str, Any]:
    p = expand(path)
    if not p.is_file():
        raise FileNotFoundError(f"frozen subset missing: {p}")
    value = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"formal oracle payload must be an object, not a bare list: {p}")
    return value


def _ids(payload: Mapping[str, Any], split: str) -> list[str]:
    if split == "full32":
        raw = payload.get("utt_ids")
    else:
        splits = payload.get("splits")
        raw = splits.get(split) if isinstance(splits, Mapping) else None
    if not isinstance(raw, list) or not raw:
        raise ValueError(f"subset has no nonempty {split} list")
    ids = [str(x) for x in raw]
    if len(ids) != len(set(ids)):
        raise ValueError(f"subset {split} contains duplicate utt_ids")
    return ids


def _contract_subsets(contract: Mapping[str, Any]) -> Mapping[str, Any]:
    subsets = contract.get("subsets")
    if not isinstance(subsets, Mapping):
        raise ValueError("active eval contract has no subsets registry")
    return subsets


def resolve_registered_subset(
    subset: str | Path,
    *,
    contract: Mapping[str, Any],
    split: str | None = None,
    formal: bool = True,
    allow_unfrozen_diagnostic: bool = False,
) -> tuple[Path, str, str, Mapping[str, Any], bool]:
    """Resolve only contract-registered aliases in formal mode.

    Returns ``(path, contract_key, split, registry_entry, formal_eligible)``.
    """
    raw = str(subset).strip()
    registry = _contract_subsets(contract)
    if raw in registry:
        entry = registry[raw]
        if not isinstance(entry, Mapping):
            raise ValueError(f"contract subset {raw!r} is malformed")
        path = entry.get("path")
        if not isinstance(path, str) or not path:
            raise ValueError(f"contract subset {raw!r} lacks path")
        return expand(path), raw, split or str(entry.get("split") or "full32"), entry, True
    if formal or not allow_unfrozen_diagnostic:
        raise ValueError(f"unknown/unregistered formal subset alias: {raw!r}")
    path = expand(raw)
    return path, raw, split or "full32", {}, False


def payload_content_file_sha256(payload: Mapping[str, Any]) -> str:
    """SHA256 of subset JSON body excluding the self-hash ``file_sha256`` field.

    Matches ``scripts/77_freeze_d1c_oracles.write_json`` formatting so the
    freezer-declared digest stays stable after the field is appended.
    """
    body = {k: v for k, v in payload.items() if k != "file_sha256"}
    text = json.dumps(body, ensure_ascii=False, indent=2) + "\n"
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_formal_subset(
    subset: str | Path,
    *,
    contract: Mapping[str, Any],
    split: str | None = None,
    allow_unfrozen_diagnostic: bool = False,
) -> dict[str, Any]:
    """Load and verify a frozen subset, returning an explicit attestation."""
    path, key, split_name, entry, registered = resolve_registered_subset(
        subset,
        contract=contract,
        split=split,
        formal=not allow_unfrozen_diagnostic,
        allow_unfrozen_diagnostic=allow_unfrozen_diagnostic,
    )
    if not registered:
        return load_diagnostic_subset(path, split=split_name)
    payload = load_json_object(path)
    ids = _ids(payload, split_name)
    got_items = items_sha256(ids)
    got_file = payload_content_file_sha256(payload)
    top_items = payload.get("items_sha256")
    split_hashes = payload.get("split_hashes")
    declared_split = split_hashes.get(split_name) if isinstance(split_hashes, Mapping) else None
    # Both top-level and split hash are mandatory even for full32.
    if not isinstance(top_items, str) or not top_items:
        raise ValueError("formal subset payload missing top-level items_sha256")
    if not isinstance(declared_split, str) or not declared_split:
        raise ValueError(f"formal subset payload missing split_hashes.{split_name}")
    if top_items != items_sha256(_ids(payload, "full32")):
        raise ValueError("formal subset top-level items_sha256 stale")
    if declared_split != got_items:
        raise ValueError(f"formal subset split hash stale for {split_name}")
    declared_file = payload.get("file_sha256")
    # File self-hashes are not reliable when stored inside their own bytes; use
    # contract's immutable content hash (body without file_sha256) as authority.
    if not isinstance(declared_file, str) or not declared_file:
        raise ValueError("formal subset payload missing file_sha256")
    if declared_file != got_file:
        raise ValueError(
            f"formal subset declared file_sha256 stale: declared={declared_file} content={got_file}"
        )
    expected_items = entry.get("items_sha256")
    expected_file = entry.get("file_sha256")
    if not isinstance(expected_items, str) or not isinstance(expected_file, str):
        raise ValueError(f"contract registry entry {key!r} missing hashes")
    if expected_items != got_items or expected_file != got_file:
        raise ValueError(f"formal subset {key!r} does not match active contract hashes")
    # select8 composition lock: exactly 2 RUSLAN + 6 Common Voice.
    if split_name == "select8":
        n_ruslan = sum(1 for u in ids if str(u).startswith("ruslan_"))
        n_cv = len(ids) - n_ruslan
        if n_ruslan != 2 or n_cv != 6:
            raise ValueError(
                f"select8 composition must be 2 RUSLAN + 6 CV; got {n_ruslan}+{n_cv}"
            )
    return {
        "path": str(path),
        "contract_key": key,
        "split": split_name,
        "utt_ids": ids,
        "items_sha256": got_items,
        "file_sha256": got_file,
        "payload": payload,
        "formal_eligible": registered,
    }


def load_diagnostic_subset(path: str | Path, *, split: str = "full32") -> dict[str, Any]:
    """Explicitly unfrozen diagnostic loader; never returns formal eligibility."""
    payload = load_json_object(path)
    ids = _ids(payload, split)
    return {
        "path": str(expand(path)),
        "contract_key": None,
        "split": split,
        "utt_ids": ids,
        "items_sha256": items_sha256(ids),
        "file_sha256": file_sha256(path),
        "payload": payload,
        "formal_eligible": False,
    }

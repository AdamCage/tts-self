"""Frozen D1b oracle subset loading (DEV32 / CONFIRM32 / select8 / gate24).

Fail-closed: exact ordered ``utt_id`` lists, items_sha256 verification, and
missing-cache errors. Silent first-N from a sorted index is forbidden for
canonical D1b runs.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from minispeech.utils import expand

PLAN_SLUG = "d1b_n2_reliability"
# Committed canonical contract (``out/.../frozen`` is gitignored; mirrored at runtime).
DEFAULT_EVAL_CONTRACT = "configs/eval_contracts/d1b_n2_reliability.json"
DEFAULT_EVAL_CONTRACT_OUT = f"out/{PLAN_SLUG}/frozen/eval_contract.json"

# Alias → (subset JSON path, split key). ``full32`` on DEV means top-level utt_ids.
SUBSET_ALIASES: dict[str, tuple[str, str]] = {
    "dev32": ("configs/train_subsets/d1b_dev32.json", "full32"),
    "d1b_dev32": ("configs/train_subsets/d1b_dev32.json", "full32"),
    "confirm32": ("configs/train_subsets/d1b_confirm32.json", "full32"),
    "d1b_confirm32": ("configs/train_subsets/d1b_confirm32.json", "full32"),
    "select8": ("configs/train_subsets/d1b_confirm32.json", "select8"),
    "gate24": ("configs/train_subsets/d1b_confirm32.json", "gate24"),
    "full32": ("configs/train_subsets/d1b_confirm32.json", "full32"),
}

# Content sha256 of configs/speaker_maps/trackn_v2.json (SpeakerRegistry.sha256).
TRACKN_V2_SPEAKER_SHA256 = (
    "64e7cdc5c3447a43d7484b7538f9fd5df88551650d7b95ddb23110f9c36dca6e"
)


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def items_sha256(items: Sequence[str]) -> str:
    return hashlib.sha256(_canonical_json(list(items))).hexdigest()


def file_sha256(path: Path | str) -> str:
    p = expand(path)
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_subset_payload(path: Path | str) -> dict[str, Any]:
    p = expand(path)
    if not p.is_file():
        raise FileNotFoundError(f"frozen subset missing: {p}")
    data = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"subset must be a JSON object: {p}")
    return data


def utt_ids_from_payload(
    payload: Mapping[str, Any],
    *,
    split: str | None = None,
) -> list[str]:
    """Return ordered utt_ids for ``split`` (or full list when split is None/full32)."""
    split_key = (split or "").strip() or None
    if split_key in (None, "full32", "all", "full"):
        if "utt_ids" in payload:
            ids = [str(x) for x in payload["utt_ids"]]
        elif "items" in payload:
            ids = [str(x) for x in payload["items"]]
        else:
            raise ValueError("subset payload missing utt_ids/items")
    else:
        splits = payload.get("splits")
        if not isinstance(splits, Mapping) or split_key not in splits:
            raise KeyError(
                f"subset split {split_key!r} not found; "
                f"available={sorted((splits or {}).keys())}"
            )
        ids = [str(x) for x in splits[split_key]]

    if not ids:
        raise ValueError("subset resolved to empty utt_id list")
    return ids


def verify_items_sha256(
    ids: Sequence[str],
    payload: Mapping[str, Any],
    *,
    split: str | None = None,
) -> str:
    """Fail-closed items_sha256 check against frozen payload fields."""
    got = items_sha256(ids)
    split_key = (split or "").strip() or None
    expected: str | None = None
    if split_key in (None, "full32", "all", "full"):
        expected = payload.get("items_sha256")
        # Prefer split_hashes.full32 when present (CONFIRM).
        sh = payload.get("split_hashes")
        if isinstance(sh, Mapping) and sh.get("full32"):
            expected = str(sh["full32"])
        elif expected is None:
            expected = None
    else:
        sh = payload.get("split_hashes")
        if isinstance(sh, Mapping) and split_key in sh:
            expected = str(sh[split_key])
    if expected is not None and got != expected:
        raise ValueError(
            f"items_sha256 mismatch for split={split_key or 'full32'}: "
            f"got={got} expected={expected}"
        )
    return got


def resolve_subset_spec(
    subset: str | Path,
    *,
    split: str | None = None,
) -> tuple[Path, str | None]:
    """Resolve CLI alias or path → (json_path, split)."""
    raw = str(subset).strip()
    key = raw.lower().replace("-", "_")
    if key in SUBSET_ALIASES:
        path_s, default_split = SUBSET_ALIASES[key]
        return expand(path_s), (split or default_split)
    return expand(raw), split


def load_frozen_utt_ids(
    subset: str | Path,
    *,
    split: str | None = None,
    verify_hash: bool = True,
) -> tuple[list[str], dict[str, Any], str]:
    """Load ordered utt_ids from a frozen D1b subset.

    Returns ``(utt_ids, payload, items_sha256)``.
    """
    path, split_key = resolve_subset_spec(subset, split=split)
    payload = load_subset_payload(path)
    ids = utt_ids_from_payload(payload, split=split_key)
    digest = (
        verify_items_sha256(ids, payload, split=split_key)
        if verify_hash
        else items_sha256(ids)
    )
    return ids, payload, digest


def expand_path_template(path: str | Path, mapping: Mapping[str, Any]) -> Path:
    """Expand ``${var}`` / ``$var`` placeholders then resolve the path."""
    s = str(path)
    for key, val in mapping.items():
        s = s.replace(f"${{{key}}}", str(val))
        s = s.replace(f"${key}", str(val))
    if "${" in s:
        raise ValueError(f"unexpanded template variables remain in path: {s!r}")
    return expand(s)


def load_eval_contract(path: str | Path | None = None) -> dict[str, Any]:
    candidates: list[Path] = []
    if path is not None:
        candidates.append(expand(path))
    else:
        candidates.extend(
            [expand(DEFAULT_EVAL_CONTRACT), expand(DEFAULT_EVAL_CONTRACT_OUT)]
        )
    p = next((c for c in candidates if c.is_file()), None)
    if p is None:
        raise FileNotFoundError(
            f"eval_contract missing; tried: {[str(c) for c in candidates]}"
        )
    data = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"eval_contract must be object: {p}")
    return data


def enforce_eval_contract(
    cfg: Mapping[str, Any] | None,
    *,
    contract_path: str | Path | None = None,
    subset_items_sha256: str | None = None,
    subset_label: str | None = None,
    require_clean_sha: bool = False,
) -> dict[str, Any]:
    """Fail-closed checks for canonical D1b train/eval/infer.

    Verifies speaker-map content hash, WAVE3c infer knobs, and optional subset
    hash against the frozen contract.
    """
    from minispeech.infer.discrete_config import (
        contract_infer_from_eval_contract,
        infer_settings_match_wave3c,
        resolve_discrete_infer_settings,
    )

    path = expand(
        contract_path
        or (cfg or {}).get("train", {}).get("eval_contract_path")
        or (cfg or {}).get("eval_contract_path")
        or DEFAULT_EVAL_CONTRACT
    )
    contract = load_eval_contract(path)
    errors: list[str] = []

    sp = contract.get("speaker_map") or {}
    expected_spk = str(sp.get("sha256") or sp.get("content_sha256") or "")
    if expected_spk and expected_spk != TRACKN_V2_SPEAKER_SHA256:
        # Contract may be stale relative to canonical trackn_v2 content hash.
        errors.append(
            f"eval_contract speaker_map.sha256={expected_spk} != "
            f"canonical trackn_v2 {TRACKN_V2_SPEAKER_SHA256}"
        )
    if expected_spk in ("", "pending", "null", "None"):
        errors.append("eval_contract speaker_map.sha256 is pending/empty")

    # Live speaker map content hash must match contract.
    try:
        from minispeech.data.speaker_registry import SpeakerRegistry

        map_path = (
            ((cfg or {}).get("train") or {}).get("speaker_map_path")
            or ((cfg or {}).get("data") or {}).get("speaker_map")
            or ((cfg or {}).get("model") or {}).get("speaker", {}).get("map_path")
            or "configs/speaker_maps/trackn_v2.json"
        )
        reg = SpeakerRegistry.load(map_path)
        if expected_spk and reg.sha256 != expected_spk:
            errors.append(
                f"speaker map content sha256 mismatch: live={reg.sha256} "
                f"contract={expected_spk}"
            )
    except Exception as exc:  # noqa: BLE001 — fail-closed surface
        errors.append(f"speaker map load failed: {exc}")

    try:
        from_contract = contract_infer_from_eval_contract(contract)
        diffs = infer_settings_match_wave3c(from_contract, ignore_max_frames=False)
        if diffs:
            errors.append(f"eval_contract infer knobs != WAVE3c: {diffs}")
        if cfg is not None:
            live = resolve_discrete_infer_settings(cfg)
            live_diffs = [
                k for k in from_contract if k != "max_frames" and live.get(k) != from_contract.get(k)
            ]
            # Allow max_frames runtime clamp; other knobs must match.
            if live_diffs:
                errors.append(f"cfg infer != eval_contract: {live_diffs}")
    except Exception as exc:  # noqa: BLE001
        errors.append(f"infer contract check failed: {exc}")

    subsets = contract.get("subsets") or {}
    if subset_label and subset_items_sha256 and isinstance(subsets, Mapping):
        key = str(subset_label).replace("\\", "/").lower()
        alias_map = {
            "dev32": "dev32",
            "d1b_dev32": "dev32",
            "confirm32": "confirm32",
            "d1b_confirm32": "confirm32",
            "full32": "confirm32",
            "select8": "select8",
            "gate24": "gate24",
        }
        alias = alias_map.get(key)
        if alias is None:
            if "d1b_dev32" in key:
                alias = "dev32"
            elif "d1b_confirm32" in key or "confirm32" in key:
                # Path to confirm32 with optional split elsewhere.
                alias = "confirm32"
            elif key.endswith("select8") or key == "select8":
                alias = "select8"
            elif key.endswith("gate24") or key == "gate24":
                alias = "gate24"
            else:
                alias = key
        # If digest matches a known subset hash, accept even if label is a path.
        by_hash = {
            str(v.get("items_sha256")): name
            for name, v in subsets.items()
            if isinstance(v, Mapping) and v.get("items_sha256")
        }
        if subset_items_sha256 in by_hash:
            alias = by_hash[subset_items_sha256]
        entry = subsets.get(alias) or subsets.get(key)
        if isinstance(entry, Mapping):
            exp = entry.get("items_sha256")
            if exp and str(exp) != subset_items_sha256:
                errors.append(
                    f"subset {alias} items_sha256 mismatch: "
                    f"got={subset_items_sha256} contract={exp}"
                )
        elif alias in ("select8", "gate24", "confirm32", "dev32"):
            errors.append(f"eval_contract.subsets missing entry for {alias}")

    if require_clean_sha:
        repro = contract.get("repro") or {}
        if not bool(repro.get("clean_sha_available")):
            errors.append("eval_contract.repro.clean_sha_available is false")

    if errors:
        raise RuntimeError(
            "D1b eval_contract enforcement failed:\n- " + "\n- ".join(errors)
        )
    return contract


__all__ = [
    "DEFAULT_EVAL_CONTRACT",
    "PLAN_SLUG",
    "SUBSET_ALIASES",
    "TRACKN_V2_SPEAKER_SHA256",
    "enforce_eval_contract",
    "expand_path_template",
    "file_sha256",
    "items_sha256",
    "load_eval_contract",
    "load_frozen_utt_ids",
    "load_subset_payload",
    "resolve_subset_spec",
    "utt_ids_from_payload",
    "verify_items_sha256",
]

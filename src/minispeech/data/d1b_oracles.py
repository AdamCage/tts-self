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
# Canonical CONFIRM aliases point at v2 (burned v1 kept under confirm32_v1).
SUBSET_ALIASES: dict[str, tuple[str, str]] = {
    "dev32": ("configs/train_subsets/d1b_dev32.json", "full32"),
    "d1b_dev32": ("configs/train_subsets/d1b_dev32.json", "full32"),
    "confirm32": ("configs/train_subsets/d1b_confirm32_v2.json", "full32"),
    "d1b_confirm32": ("configs/train_subsets/d1b_confirm32_v2.json", "full32"),
    "confirm32_v2": ("configs/train_subsets/d1b_confirm32_v2.json", "full32"),
    "d1b_confirm32_v2": ("configs/train_subsets/d1b_confirm32_v2.json", "full32"),
    "confirm32_v1": ("configs/train_subsets/d1b_confirm32.json", "full32"),
    "d1b_confirm32_v1": ("configs/train_subsets/d1b_confirm32.json", "full32"),
    "select8": ("configs/train_subsets/d1b_confirm32_v2.json", "select8"),
    "gate24": ("configs/train_subsets/d1b_confirm32_v2.json", "gate24"),
    "full32": ("configs/train_subsets/d1b_confirm32_v2.json", "full32"),
    # D1c semantic-EOS DEV oracle (explicit prefixes — do not override D1b aliases).
    "d1c_dev32": ("configs/train_subsets/d1c_dev32.json", "full32"),
    "d1c_full32": ("configs/train_subsets/d1c_dev32.json", "full32"),
    "d1c_select8": ("configs/train_subsets/d1c_dev32.json", "select8"),
    "d1c_gate24": ("configs/train_subsets/d1c_dev32.json", "gate24"),
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


# Train-config fields that bind *to* contract/manifest digests. Including them in
# the recipe_source_manifest digest of W1/W2 YAMLs creates a cyclic seal
# (YAML declares contract sha → YAML bytes change → manifest changes → contract sha
# changes). Strip these before hashing recipe YAML sources.
_RECIPE_YAML_BINDING_KEYS = (
    "eval_contract_hash",
    "recipe_hash",
    "recipe_sha256",
    "recipe_source_manifest_hash",
    "recipe_source_manifest_sha256",
    "run_manifest_hash",
)


def _is_d1c_recipe_yaml(path: Path | str) -> bool:
    name = Path(path).name.lower()
    return name.startswith("discrete_speech_lm_d1c_semantic_eos_w") and name.endswith(
        (".yaml", ".yml")
    )


def recipe_source_file_sha256(path: Path | str) -> str:
    """SHA256 for ``recipe_source_manifest`` entries.

    Recipe YAMLs (W1/W2) are hashed with train binding-hash fields stripped so
    declaring ``eval_contract_hash`` / recipe / manifest digests cannot invalidate
    the seal. All other locked sources use raw file bytes.
    """
    p = Path(path)
    if not _is_d1c_recipe_yaml(p):
        return file_sha256(p)
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("PyYAML required to hash D1c recipe YAMLs") from exc
    data = yaml.safe_load(p.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        return file_sha256(p)
    body = dict(data)
    train = dict(body.get("train") or {})
    for key in _RECIPE_YAML_BINDING_KEYS:
        train.pop(key, None)
    body["train"] = train
    blob = yaml.safe_dump(body, allow_unicode=True, sort_keys=True)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def verify_recipe_source_manifest(
    manifest: Mapping[str, Any],
    *,
    root: Path | str | None = None,
) -> list[str]:
    """Return mismatch descriptions; empty list means all declared digests match."""
    root_path = Path(root) if root is not None else Path.cwd()
    errors: list[str] = []
    if not isinstance(manifest, Mapping):
        return ["recipe_source_manifest must be an object"]
    for rel, expected in manifest.items():
        path = root_path / str(rel)
        if not path.is_file():
            errors.append(f"missing:{rel}")
            continue
        got = recipe_source_file_sha256(path)
        if str(expected) != got:
            errors.append(f"mismatch:{rel}: declared={expected} got={got}")
    return errors


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


def eval_contract_content_sha256(contract: Mapping[str, Any]) -> str:
    """Content sha256 of eval_contract body.

    Excludes the self-hash field ``sha256``, the recipe projection hash
    ``recipe_sha256`` (stored alongside for recipe lock), and private ``_`` keys.
    """
    body = {
        k: v
        for k, v in contract.items()
        if k not in {"sha256", "recipe_sha256"} and not str(k).startswith("_")
    }
    return hashlib.sha256(_canonical_json(body)).hexdigest()


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
    data = dict(data)
    content_sha = eval_contract_content_sha256(data)
    data["_path"] = str(p)
    data["_content_sha256"] = content_sha
    return data


def resolve_subset_alias(subset_label: str | None) -> str | None:
    """Map a subset CLI label/path to a contract ``subsets`` key, if known."""
    if not subset_label:
        return None
    key = str(subset_label).replace("\\", "/").lower()
    alias_map = {
        "dev32": "dev32",
        "d1b_dev32": "dev32",
        "d1c_dev32": "dev32",
        "d1c_full32": "full32",
        "confirm32": "confirm32",
        "d1b_confirm32": "confirm32",
        "confirm32_v2": "confirm32",
        "d1b_confirm32_v2": "confirm32",
        "full32": "confirm32",
        "select8": "select8",
        "d1c_select8": "select8",
        "gate24": "gate24",
        "d1c_gate24": "gate24",
    }
    alias = alias_map.get(key)
    if alias is not None:
        return alias
    if "d1c_dev32" in key or key.endswith("/d1c_dev32.json"):
        return "dev32"
    if "d1b_dev32" in key or key.endswith("/d1b_dev32.json"):
        return "dev32"
    if "confirm32_v2" in key or "d1b_confirm32_v2" in key:
        return "confirm32"
    if "d1b_confirm32" in key or "/confirm32" in key or key.endswith("confirm32.json"):
        return "confirm32"
    if key.endswith("select8") or key == "select8" or "/select8" in key:
        return "select8"
    if key.endswith("gate24") or key == "gate24" or "/gate24" in key:
        return "gate24"
    return key


def enforce_eval_contract(
    cfg: Mapping[str, Any] | None,
    *,
    contract_path: str | Path | None = None,
    subset_items_sha256: str | None = None,
    subset_label: str | None = None,
    require_clean_sha: bool = False,
    expected_contract_sha256: str | None = None,
    live_infer: Mapping[str, Any] | None = None,
    speaker_map_path: str | Path | None = None,
) -> dict[str, Any]:
    """Fail-closed checks for canonical D1b train/eval/infer.

    Verifies speaker-map content hash, WAVE3c infer knobs (including frozen
    ``max_frames``), optional subset hash, and optional recorded contract hash.
    Rejects contradictory ``subset_label`` vs ``subset_items_sha256``.
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

    content_sha = str(contract.get("_content_sha256") or eval_contract_content_sha256(contract))
    declared = str(contract.get("sha256") or "")
    if declared and declared != content_sha:
        errors.append(
            f"eval_contract.sha256 field stale: field={declared} content={content_sha}"
        )
    if expected_contract_sha256 and str(expected_contract_sha256) != content_sha:
        errors.append(
            f"eval_contract hash mismatch: expected={expected_contract_sha256} "
            f"got={content_sha}"
        )
    contract["sha256"] = content_sha

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

    # Live speaker map content hash must match contract (CLI path takes precedence).
    try:
        from minispeech.data.speaker_registry import SpeakerRegistry

        map_path = (
            speaker_map_path
            or ((cfg or {}).get("train") or {}).get("speaker_map_path")
            or ((cfg or {}).get("data") or {}).get("speaker_map")
            or ((cfg or {}).get("model") or {}).get("speaker", {}).get("map_path")
            or sp.get("path")
            or "configs/speaker_maps/trackn_v2.json"
        )
        reg = SpeakerRegistry.load(map_path)
        if expected_spk and reg.sha256 != expected_spk:
            errors.append(
                f"speaker map content sha256 mismatch: live={reg.sha256} "
                f"contract={expected_spk} path={map_path}"
            )
        contract_map_path = sp.get("path")
        if speaker_map_path is not None and contract_map_path:
            live_norm = str(expand(speaker_map_path)).replace("\\", "/").lower()
            contract_norm = str(expand(contract_map_path)).replace("\\", "/").lower()
            # Path mismatch is OK only when content hash already matches; still
            # surface when the CLI path does not exist / hash failed above.
            if live_norm != contract_norm and expected_spk and reg.sha256 == expected_spk:
                pass  # alternate path with identical content
            elif live_norm != contract_norm and not expected_spk:
                errors.append(
                    f"CLI speaker-map path {speaker_map_path!r} != contract "
                    f"speaker_map.path={contract_map_path!r}"
                )
    except Exception as exc:  # noqa: BLE001 — fail-closed surface
        errors.append(f"speaker map load failed: {exc}")

    try:
        from_contract = contract_infer_from_eval_contract(contract)
        term = contract.get("termination") or {}
        semantic = isinstance(term, Mapping) and term.get("mode") == "semantic_eos_v1"
        if semantic:
            # Canonical semantic-EOS infer must not reintroduce legacy stop knobs.
            legacy_stop_keys = (
                "stop_threshold",
                "stop_consecutive",
                "frames_per_text_token",
                "stop_overrun_bias",
                "stop_earliest_frac",
            )
            bad = [k for k in legacy_stop_keys if k in from_contract]
            if bad:
                errors.append(
                    f"semantic_eos_v1 contract infer still exposes legacy stop keys: {bad}"
                )
            block = contract.get("infer") or {}
            if isinstance(block, Mapping):
                forbidden = set(block.get("forbidden_success_paths") or [])
                missing = [k for k in legacy_stop_keys if k not in forbidden]
                if forbidden and missing:
                    errors.append(
                        "semantic_eos_v1 forbidden_success_paths incomplete: "
                        f"missing {missing}"
                    )
        else:
            diffs = infer_settings_match_wave3c(from_contract, ignore_max_frames=False)
            if diffs:
                errors.append(f"eval_contract infer knobs != WAVE3c: {diffs}")
        live = live_infer
        if live is None and cfg is not None:
            live = resolve_discrete_infer_settings(cfg)
        if live is not None:
            # Frozen max_frames must not be replaced by GT-derived clamps.
            live_diffs = [k for k in from_contract if live.get(k) != from_contract.get(k)]
            if live_diffs:
                errors.append(f"cfg/live infer != eval_contract: {live_diffs}")
    except Exception as exc:  # noqa: BLE001
        errors.append(f"infer contract check failed: {exc}")

    subsets = contract.get("subsets") or {}
    if subset_label and subset_items_sha256 and isinstance(subsets, Mapping):
        alias = resolve_subset_alias(subset_label)
        by_hash = {
            str(v.get("items_sha256")): name
            for name, v in subsets.items()
            if isinstance(v, Mapping) and v.get("items_sha256")
        }
        hash_alias = by_hash.get(str(subset_items_sha256))
        # Fail-closed: label and hash must not point at different known subsets.
        if (
            alias in ("dev32", "confirm32", "select8", "gate24")
            and hash_alias is not None
            and hash_alias != alias
        ):
            errors.append(
                f"subset label/hash contradiction: label={subset_label!r} "
                f"(alias={alias}) but items_sha256 maps to {hash_alias}"
            )
        elif hash_alias is not None and alias not in (
            "dev32",
            "confirm32",
            "select8",
            "gate24",
        ):
            # Unresolved path-like label: bind via hash only when label is not a
            # known contradictory alias.
            alias = hash_alias
        entry = subsets.get(alias) or subsets.get(str(subset_label))
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

    # D1c formal path: verify locked recipe_source_manifest digests at runtime.
    term = contract.get("termination") or {}
    if isinstance(term, Mapping) and term.get("mode") == "semantic_eos_v1":
        manifest = contract.get("recipe_source_manifest") or {}
        if not isinstance(manifest, Mapping) or not manifest:
            errors.append("D1c eval_contract missing recipe_source_manifest")
        else:
            mismatches = verify_recipe_source_manifest(manifest)
            if mismatches:
                errors.append(
                    "recipe_source_manifest verify failed: " + "; ".join(mismatches[:8])
                    + (f" (+{len(mismatches) - 8} more)" if len(mismatches) > 8 else "")
                )

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
    "eval_contract_content_sha256",
    "expand_path_template",
    "file_sha256",
    "items_sha256",
    "load_eval_contract",
    "load_frozen_utt_ids",
    "load_subset_payload",
    "recipe_source_file_sha256",
    "resolve_subset_alias",
    "resolve_subset_spec",
    "utt_ids_from_payload",
    "verify_items_sha256",
    "verify_recipe_source_manifest",
]

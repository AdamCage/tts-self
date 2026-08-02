#!/usr/bin/env python3
"""Version, hash, and verify frozen RU-Critical evaluation sets."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Sequence

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SETS_DIR = ROOT / "configs" / "eval_sets"
# Version-controlled registry (was gitignored out/eval_set_hashes.json).
DEFAULT_HASH_FILE = ROOT / "configs" / "eval_set_hashes.json"
SEMVER_RE = re.compile(r"^\d+\.\d+\.\d+$")


def _resolve(path: str | Path) -> Path:
    candidate = Path(path).expanduser()
    return candidate if candidate.is_absolute() else ROOT / candidate


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _content_array(raw: dict[str, Any], path: Path) -> list[Any]:
    """Return the frozen item array without metadata.

    Phrase sets use ``items``. Legacy ``kind: utt_ids`` sets use ``utt_ids``;
    that array is their item inventory and is intentionally hashed directly.
    """
    items = raw.get("items")
    if isinstance(items, list):
        return items
    utt_ids = raw.get("utt_ids")
    if isinstance(utt_ids, list):
        return utt_ids
    raise ValueError(f"{path}: expected an 'items' or 'utt_ids' array")


def _load_set(path: Path) -> dict[str, Any]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: top-level JSON must be an object")
    set_id = str(raw.get("id") or "").strip()
    if not set_id:
        raise ValueError(f"{path}: missing non-empty id")
    version = str(raw.get("version") or "").strip()
    if not SEMVER_RE.fullmatch(version):
        raise ValueError(f"{path}: version must be MAJOR.MINOR.PATCH, got {version!r}")
    _content_array(raw, path)
    return raw


def describe_set(path: Path) -> dict[str, Any]:
    raw = _load_set(path)
    items = _content_array(raw, path)
    try:
        rel_path = path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        rel_path = str(path.resolve())
    return {
        "id": str(raw["id"]),
        "version": str(raw["version"]),
        "path": rel_path,
        "n": len(items),
        "items_sha256": _sha256_bytes(_canonical_json(items)),
        "file_sha256": _sha256_bytes(path.read_bytes()),
    }


def discover_sets(sets_dir: Path) -> list[Path]:
    paths = sorted(p for p in sets_dir.glob("*.json") if p.is_file())
    if not paths:
        raise FileNotFoundError(f"no evaluation-set JSON files under {sets_dir}")
    return paths


def build_hash_payload(sets_dir: Path) -> dict[str, Any]:
    records: dict[str, dict[str, Any]] = {}
    for path in discover_sets(sets_dir):
        record = describe_set(path)
        set_id = record["id"]
        if set_id in records:
            raise ValueError(f"duplicate evaluation-set id {set_id!r}")
        records[set_id] = {k: v for k, v in record.items() if k != "id"}
    return {
        "schema_version": 1,
        "hash_contract": {
            "items_sha256": (
                "sha256(canonical JSON of items; utt_ids array for kind=utt_ids)"
            ),
            "file_sha256": "sha256(exact UTF-8 file bytes)",
        },
        "sets": records,
    }


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    os.replace(tmp, path)


def hash_sets(sets_dir: Path, hash_file: Path) -> dict[str, Any]:
    payload = build_hash_payload(sets_dir)
    _atomic_write_json(hash_file, payload)
    return payload


def verify_sets(sets_dir: Path, hash_file: Path) -> dict[str, Any]:
    if not hash_file.is_file():
        raise FileNotFoundError(
            f"hash registry not found: {hash_file}; run '54_eval_sets.py hash' first"
        )
    expected = json.loads(hash_file.read_text(encoding="utf-8"))
    expected_sets = expected.get("sets") if isinstance(expected, dict) else None
    if not isinstance(expected_sets, dict):
        raise ValueError(f"{hash_file}: missing object field 'sets'")

    actual = build_hash_payload(sets_dir)["sets"]
    errors: list[str] = []
    if set(actual) != set(expected_sets):
        missing = sorted(set(expected_sets) - set(actual))
        unregistered = sorted(set(actual) - set(expected_sets))
        if missing:
            errors.append(f"missing set files: {missing}")
        if unregistered:
            errors.append(f"unregistered set files: {unregistered}")

    checked: dict[str, Any] = {}
    for set_id in sorted(set(actual) & set(expected_sets)):
        got = actual[set_id]
        want = expected_sets[set_id]
        mismatches = [
            key
            for key in ("version", "n", "items_sha256", "file_sha256")
            if got.get(key) != want.get(key)
        ]
        if mismatches:
            errors.append(f"{set_id}: mismatched {', '.join(mismatches)}")
        checked[set_id] = {
            "version": got["version"],
            "n": got["n"],
            "items_sha256": got["items_sha256"],
            "file_sha256": got["file_sha256"],
            "ok": not mismatches,
        }
    if errors:
        raise ValueError("; ".join(errors))
    return {"ok": True, "n_sets": len(checked), "sets": checked}


def bump_set(
    *,
    set_id: str,
    version: str,
    sets_dir: Path,
    hash_file: Path,
    confirm_content_changed: bool,
) -> dict[str, Any]:
    if not SEMVER_RE.fullmatch(version):
        raise ValueError(f"--to must be MAJOR.MINOR.PATCH, got {version!r}")
    path = sets_dir / f"{set_id}.json"
    if not path.is_file():
        raise FileNotFoundError(f"evaluation set not found: {path}")
    if not hash_file.is_file():
        raise FileNotFoundError(
            f"hash registry not found: {hash_file}; cannot prove whether content changed"
        )

    registry = json.loads(hash_file.read_text(encoding="utf-8"))
    previous = (registry.get("sets") or {}).get(set_id)
    if not isinstance(previous, dict):
        raise ValueError(f"{set_id!r} is absent from {hash_file}; run hash before bump")
    before = describe_set(path)
    content_changed = before["items_sha256"] != previous.get("items_sha256")
    if content_changed and not confirm_content_changed:
        raise ValueError(
            "evaluation-set content changed; rerun with "
            "--confirm-content-changed after auditing every change"
        )

    raw = _load_set(path)
    old_version = str(raw["version"])
    raw["version"] = version
    _atomic_write_json(path, raw)
    payload = hash_sets(sets_dir, hash_file)
    record = payload["sets"][set_id]
    return {
        "ok": True,
        "set": set_id,
        "from": old_version,
        "to": version,
        "content_changed": content_changed,
        "items_sha256": record["items_sha256"],
        "file_sha256": record["file_sha256"],
    }


def _add_paths(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--sets-dir", default=str(DEFAULT_SETS_DIR))
    parser.add_argument("--hash-file", default=str(DEFAULT_HASH_FILE))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p_hash = sub.add_parser("hash", help="Recompute the frozen-set hash registry")
    _add_paths(p_hash)
    p_verify = sub.add_parser("verify", help="Verify versions and frozen hashes")
    _add_paths(p_verify)
    p_bump = sub.add_parser("bump", help="Bump one audited set version and rehash")
    p_bump.add_argument("--set", required=True, dest="set_id")
    p_bump.add_argument("--to", required=True, dest="version")
    p_bump.add_argument("--confirm-content-changed", action="store_true")
    _add_paths(p_bump)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    sets_dir = _resolve(args.sets_dir)
    hash_file = _resolve(args.hash_file)
    try:
        if args.command == "hash":
            payload = hash_sets(sets_dir, hash_file)
            result: dict[str, Any] = {
                "ok": True,
                "hash_file": str(hash_file),
                "n_sets": len(payload["sets"]),
                "sets": payload["sets"],
            }
        elif args.command == "verify":
            result = verify_sets(sets_dir, hash_file)
            result["hash_file"] = str(hash_file)
        else:
            result = bump_set(
                set_id=args.set_id,
                version=args.version,
                sets_dir=sets_dir,
                hash_file=hash_file,
                confirm_content_changed=bool(args.confirm_content_changed),
            )
    except (FileNotFoundError, ValueError, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

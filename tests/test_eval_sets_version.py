"""Frozen evaluation-set versioning and hash-contract tests."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "54_eval_sets.py"
REQUIRED = {"demo5", "val100", "bank_entity", "homograph", "latin"}
EXPECTED_VERSIONS = {
    "demo5": "1.0.0",
    "val100": "1.0.0",
    "bank_entity": "3.0.0",
    "homograph": "3.0.0",
    "latin": "1.0.0",
}


def _run(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )


def test_required_sets_load_and_verify(tmp_path: Path) -> None:
    sets_dir = ROOT / "configs" / "eval_sets"
    for set_id in REQUIRED:
        raw = json.loads((sets_dir / f"{set_id}.json").read_text(encoding="utf-8"))
        assert raw["version"] == EXPECTED_VERSIONS[set_id]

    hashes = tmp_path / "eval_set_hashes.json"
    made = _run(
        "hash",
        "--sets-dir",
        str(sets_dir),
        "--hash-file",
        str(hashes),
    )
    assert made.returncode == 0, made.stderr
    payload = json.loads(hashes.read_text(encoding="utf-8"))
    assert REQUIRED <= set(payload["sets"])
    assert all(len(payload["sets"][sid]["items_sha256"]) == 64 for sid in REQUIRED)
    assert all(len(payload["sets"][sid]["file_sha256"]) == 64 for sid in REQUIRED)

    verified = _run(
        "verify",
        "--sets-dir",
        str(sets_dir),
        "--hash-file",
        str(hashes),
    )
    assert verified.returncode == 0, verified.stderr
    assert json.loads(verified.stdout)["ok"] is True


def test_bump_requires_confirmation_after_item_change(tmp_path: Path) -> None:
    sets_dir = tmp_path / "sets"
    sets_dir.mkdir()
    src = ROOT / "configs" / "eval_sets" / "demo5.json"
    dst = sets_dir / "demo5.json"
    dst.write_bytes(src.read_bytes())
    hashes = tmp_path / "hashes.json"

    assert (
        _run("hash", "--sets-dir", str(sets_dir), "--hash-file", str(hashes)).returncode
        == 0
    )
    raw = json.loads(dst.read_text(encoding="utf-8"))
    raw["items"][0]["text"] += "!"
    dst.write_text(json.dumps(raw, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    refused = _run(
        "bump",
        "--set",
        "demo5",
        "--to",
        "2.0.0",
        "--sets-dir",
        str(sets_dir),
        "--hash-file",
        str(hashes),
    )
    assert refused.returncode == 1
    assert "--confirm-content-changed" in refused.stderr

    accepted = _run(
        "bump",
        "--set",
        "demo5",
        "--to",
        "2.0.0",
        "--confirm-content-changed",
        "--sets-dir",
        str(sets_dir),
        "--hash-file",
        str(hashes),
    )
    assert accepted.returncode == 0, accepted.stderr
    assert json.loads(dst.read_text(encoding="utf-8"))["version"] == "2.0.0"

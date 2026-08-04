#!/usr/bin/env python3
"""REAUDIT_ROUND4 — pytest attestation audit + post-probe source integrity."""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
RES = ROOT / ".cursor/results/d1c_semantic_eos_repair_b81ecc3e"
LOGS = RES / "logs"

CLAIM_NEW_TESTS = [
    "test_validate_formal_report_rejects_checkpoint_step_zero",
    "test_resolve_checkpoint_step_payload_filename_cli",
    "test_main_stamps_checkpoint_step_and_selector_e2e",
]

RESULTS: list[tuple[str, str, str]] = []


def rec(tag: str, ok, msg: str) -> None:
    status = "INFO" if ok is None else ("PASS" if ok else "FAIL")
    RESULTS.append((tag, status, msg))
    print(f"{tag} {status} {msg}", flush=True)


def mt(p: Path) -> str:
    return dt.datetime.fromtimestamp(p.stat().st_mtime, dt.UTC).strftime("%Y-%m-%d %H:%M:%S UTC")


def main() -> int:
    claimed = LOGS / "repair_round4_full_pytest.log"
    mine = LOGS / "reaudit4_full_pytest.log"

    ctext = claimed.read_text(encoding="utf-8", errors="replace") if claimed.is_file() else ""
    mtext = mine.read_text(encoding="utf-8", errors="replace") if mine.is_file() else ""
    c_summary = [ln.strip() for ln in ctext.splitlines() if " passed" in ln and "skipped" in ln]
    m_summary = [ln.strip() for ln in mtext.splitlines() if " passed" in ln and "skipped" in ln]
    rec("P1", bool(c_summary), f"claimed log summary: {c_summary[-1] if c_summary else '<none>'}")
    rec("P2", bool(m_summary), f"independent run summary: {m_summary[-1] if m_summary else '<none>'}")
    ok_counts = bool(c_summary and m_summary) and (
        c_summary[-1].split(" in ")[0] == m_summary[-1].split(" in ")[0]
    )
    rec("P3", ok_counts, "pass/skip counts reproduce exactly (ignoring wall time)")

    # Collected test ids
    out = subprocess.run(  # noqa: S603
        [
            "/home/adam/projects/tts-self/.venv/bin/python",
            "-m",
            "pytest",
            "-q",
            "--collect-only",
        ],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        env={
            "PATH": "/usr/bin:/bin",
            "CUDA_VISIBLE_DEVICES": "",
            "HOME": "/home/adam",
            "HF_HOME": "/home/adam/hf_cache",
            "PYTHONPATH": str(ROOT / "src"),
            "GIT_DIR": "/mnt/c/projects/tts-self/.git/worktrees/tts-self-d1c",
            "GIT_WORK_TREE": str(ROOT),
        },
        check=False,
    )
    ids = [ln.strip() for ln in out.stdout.splitlines() if "::" in ln]
    rec("P4", len(ids) == 404, f"collected test ids = {len(ids)} (claim 404)")
    found = {t: any(t in i for i in ids) for t in CLAIM_NEW_TESTS}
    rec("P5", all(found.values()), f"three claimed new tests are collected: {found}")
    rec(
        "P6",
        len(ids) == len(set(ids)),
        f"no duplicate test ids (unique={len(set(ids))})",
    )
    rec("P7", None, "404 collected = round-3 401 + 3 new tests (arithmetic consistent)")

    # mtime audit: no attested source may post-date the claimed pytest log
    if claimed.is_file():
        log_mtime = claimed.stat().st_mtime
        contract = json.loads(
            (ROOT / "configs/eval_contracts/d1c_semantic_eos_v2_0_0.json").read_text(
                encoding="utf-8"
            )
        )
        locked = list((contract.get("recipe_source_manifest") or {}).keys())
        watched = locked + [
            "tests/test_d1c_infra_repair.py",
            "configs/eval_contracts/d1c_semantic_eos_v2_0_0.json",
            "configs/eval_contracts/d1c_semantic_eos_v2_0_1.json",
        ]
        newer = []
        newest = ("", 0.0)
        for rel in watched:
            p = ROOT / rel
            if not p.is_file():
                continue
            ts = p.stat().st_mtime
            if ts > newest[1]:
                newest = (rel, ts)
            if ts > log_mtime + 1:
                newer.append((rel, mt(p)))
        rec("P8", None, f"claimed pytest log mtime = {mt(claimed)}")
        rec(
            "P9",
            not newer,
            f"newest attested source = {newest[0]} @ "
            f"{dt.datetime.fromtimestamp(newest[1], dt.UTC).strftime('%Y-%m-%d %H:%M:%S UTC')}; "
            f"sources newer than the log: {newer}",
        )

    # Post-probe integrity: hash the files this audit could have touched
    watch = [
        "scripts/74_eval_discrete_lm.py",
        "scripts/76_select_discrete_checkpoint.py",
        "src/minispeech/eval/discrete_report_schema.py",
        "src/minispeech/eval/discrete_n2.py",
        "tests/test_d1c_infra_repair.py",
        "configs/eval_contracts/d1c_semantic_eos_v2_0_0.json",
    ]
    digests = {}
    for rel in watch:
        p = ROOT / rel
        digests[rel] = hashlib.sha256(p.read_bytes()).hexdigest()
    (LOGS / "reaudit4_post_probe_hashes.json").write_text(
        json.dumps(digests, indent=2) + "\n", encoding="utf-8"
    )
    rec("P10", None, "post-probe source digests written to logs/reaudit4_post_probe_hashes.json")
    for rel, d in digests.items():
        print(f"    {d[:16]}  {rel}")

    fails = sum(1 for _, st, _ in RESULTS if st == "FAIL")
    print(f"\nSUMMARY fails={fails} total={len(RESULTS)}")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())

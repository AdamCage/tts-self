#!/usr/bin/env python3
"""REAUDIT_ROUND5 — pytest attestation audit + post-probe source integrity."""
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
    "test_selector_refuses_duplicate_checkpoint_hashes",
    "test_selector_refuses_checkpoint_hash_mismatch",
    "test_validate_formal_report_rejects_checkpoint_hash_mismatch",
]
CLAIM_CHANGED_TESTS = [
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


def main() -> int:  # noqa: C901
    claimed = LOGS / "repair_round5_full_pytest.log"
    mine = LOGS / "reaudit5_full_pytest_tail.log"

    ctext = claimed.read_text(encoding="utf-8", errors="replace") if claimed.is_file() else ""
    mtext = mine.read_text(encoding="utf-8", errors="replace") if mine.is_file() else ""
    c_sum = [ln.strip() for ln in ctext.splitlines() if " passed" in ln and "skipped" in ln]
    m_sum = [ln.strip() for ln in mtext.splitlines() if " passed" in ln and "skipped" in ln]
    rec("P1", bool(c_sum), f"claimed log summary: {c_sum[-1] if c_sum else '<none>'}")
    rec("P2", bool(m_sum), f"independent run summary: {m_sum[-1] if m_sum else '<none>'}")
    rec(
        "P3",
        bool(c_sum and m_sum) and c_sum[-1].split(" in ")[0] == m_sum[-1].split(" in ")[0],
        "pass/skip counts reproduce exactly (ignoring wall time)",
    )
    rec(
        "P3b",
        bool(c_sum) and c_sum[-1].startswith("404 passed, 3 skipped"),
        f"REPAIR_ROUND5 claim '404 passed, 3 skipped' is what the log says: "
        f"{c_sum[-1] if c_sum else '<none>'}",
    )

    env = {
        "PATH": "/usr/bin:/bin",
        "CUDA_VISIBLE_DEVICES": "",
        "HOME": "/home/adam",
        "HF_HOME": "/home/adam/hf_cache",
        "PYTHONPATH": str(ROOT / "src"),
        "GIT_DIR": "/mnt/c/projects/tts-self/.git/worktrees/tts-self-d1c",
        "GIT_WORK_TREE": str(ROOT),
    }
    out = subprocess.run(  # noqa: S603
        ["/home/adam/projects/tts-self/.venv/bin/python", "-m", "pytest", "-q", "--collect-only"],
        cwd=str(ROOT), capture_output=True, text=True, env=env, check=False,
    )
    ids = [ln.strip() for ln in out.stdout.splitlines() if "::" in ln]
    rec("P4", len(ids) == 407, f"collected test ids = {len(ids)} (claim 407)")
    rec("P5", len(ids) == len(set(ids)), f"no duplicate test ids (unique={len(set(ids))})")
    found_new = {t: any(t in i for i in ids) for t in CLAIM_NEW_TESTS}
    rec("P6", all(found_new.values()), f"three claimed NEW tests collected: {found_new}")
    found_chg = {t: any(t in i for i in ids) for t in CLAIM_CHANGED_TESTS}
    rec("P7", all(found_chg.values()), f"two claimed CHANGED tests collected: {found_chg}")
    rec("P8", None, "arithmetic: round-4 404 collected + 3 new = 407; 404 passed + 3 skipped = 407")

    # skip identities
    sk = subprocess.run(  # noqa: S603
        ["/home/adam/projects/tts-self/.venv/bin/python", "-m", "pytest", "-q", "-rs"],
        cwd=str(ROOT), capture_output=True, text=True, env=env, check=False,
    )
    skips = [ln.strip() for ln in sk.stdout.splitlines() if ln.startswith("SKIPPED")]
    rec("P9", len(skips) == 3, f"three skips are declared: {skips}")

    # focused log claim
    foc = LOGS / "repair_round5_focused_pytest.log"
    ftext = foc.read_text(encoding="utf-8", errors="replace") if foc.is_file() else ""
    rec("P10", "28 passed" in ftext, f"focused log claim: {ftext.strip().splitlines()[-1] if ftext else '<none>'}")

    # mtime audit
    if claimed.is_file():
        log_mtime = claimed.stat().st_mtime
        contract = json.loads(
            (ROOT / "configs/eval_contracts/d1c_semantic_eos_v2_0_0.json").read_text("utf-8")
        )
        locked = list((contract.get("recipe_source_manifest") or {}).keys())
        watched = locked + [
            "tests/test_d1c_infra_repair.py",
            "configs/eval_contracts/d1c_semantic_eos_v2_0_0.json",
            "configs/eval_contracts/d1c_semantic_eos_v2_0_1.json",
            "configs/discrete_speech_lm_d1c_semantic_eos_w1.yaml",
            "configs/discrete_speech_lm_d1c_semantic_eos_w2.yaml",
        ]
        newer, newest = [], ("", 0.0)
        for rel in watched:
            p = ROOT / rel
            if not p.is_file():
                continue
            ts = p.stat().st_mtime
            if ts > newest[1]:
                newest = (rel, ts)
            if ts > log_mtime + 1:
                newer.append((rel, mt(p)))
        rec("P11", None, f"claimed pytest log mtime = {mt(claimed)}")
        rec(
            "P12",
            not newer,
            f"newest attested source = {newest[0]} @ "
            f"{dt.datetime.fromtimestamp(newest[1], dt.UTC).strftime('%Y-%m-%d %H:%M:%S UTC')}; "
            f"sources newer than the log: {newer}",
        )

    # post-probe integrity
    watch = [
        "scripts/74_eval_discrete_lm.py",
        "scripts/76_select_discrete_checkpoint.py",
        "src/minispeech/eval/discrete_report_schema.py",
        "src/minispeech/eval/discrete_n2.py",
        "src/minispeech/train/discrete_lm_trainer.py",
        "tests/test_d1c_infra_repair.py",
        "configs/eval_contracts/d1c_semantic_eos_v2_0_0.json",
        "configs/eval_contracts/d1c_semantic_eos_v2_0_1.json",
        "configs/discrete_speech_lm_d1c_semantic_eos_w1.yaml",
        "configs/discrete_speech_lm_d1c_semantic_eos_w2.yaml",
    ]
    digests = {rel: hashlib.sha256((ROOT / rel).read_bytes()).hexdigest() for rel in watch}
    (LOGS / "reaudit5_post_probe_hashes.json").write_text(
        json.dumps(digests, indent=2) + "\n", encoding="utf-8"
    )
    rec("P13", None, "post-probe source digests -> logs/reaudit5_post_probe_hashes.json")
    for rel, d in digests.items():
        print(f"    {d[:16]}  {rel}")

    fails = sum(1 for _, st, _ in RESULTS if st == "FAIL")
    print(f"\nSUMMARY fails={fails} total={len(RESULTS)}")
    for tag, st, msg in RESULTS:
        if st == "FAIL":
            print(f"  FAILED {tag}: {msg}")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())

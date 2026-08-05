#!/usr/bin/env python3
"""REAUDIT_ROUND6 — name the test that catches each round-6 guard mutation.

Same on-disk mutate/restore protocol as ``reaudit6_probe_mutation_full.py`` but
scoped to the D1c-relevant test files so the failing node id can be captured
quickly. Source is restored in ``finally`` and verified by sha256 at exit.
"""

from __future__ import annotations

import hashlib
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
LOG = ROOT / ".cursor/results/d1c_semantic_eos_repair_b81ecc3e/logs/reaudit6_probe_mutation_which.log"
PY = "/home/adam/projects/tts-self/.venv/bin/python"

SRC74 = ROOT / "scripts/74_eval_discrete_lm.py"
SRC76 = ROOT / "scripts/76_select_discrete_checkpoint.py"
SCHEMA = ROOT / "src/minispeech/eval/discrete_report_schema.py"
TOUCHED = [SRC74, SRC76, SCHEMA]

FILES = [
    "tests/test_d1c_infra_repair.py",
    "tests/test_d1c_selector_integration.py",
    "tests/test_discrete_eval_report_schema.py",
    "tests/test_d1c_semantic_eos.py",
    "tests/test_d1c_oracles.py",
    "tests/test_discrete_lm_resume.py",
]

sys.path.insert(0, str(Path(__file__).resolve().parent))
from reaudit6_probe_mutation_full import MUTATIONS  # noqa: E402

LINES: list[str] = []


def rec(tag: str, ok, msg: str) -> None:
    status = "INFO" if ok is None else ("PASS" if ok else "FAIL")
    LINES.append(f"{tag} {status} {msg}")
    print(f"{tag} {status} {msg}", flush=True)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_files() -> tuple[int, list[str], str]:
    env = dict(os.environ)
    env.update(
        CUDA_VISIBLE_DEVICES="", HF_HOME="/home/adam/hf_cache",
        PYTHONPATH=str(ROOT / "src"),
        GIT_DIR="/mnt/c/projects/tts-self/.git/worktrees/tts-self-d1c",
        GIT_WORK_TREE=str(ROOT),
    )
    proc = subprocess.run(
        [PY, "-m", "pytest", "-q", "--no-header", "-p", "no:cacheprovider", *FILES],
        cwd=str(ROOT), env=env, capture_output=True, text=True,
    )
    out = proc.stdout + proc.stderr
    failed = sorted(set(re.findall(r"^FAILED (\S+)", out, flags=re.M)))
    tail = out.strip().splitlines()
    return proc.returncode, failed, (tail[-1] if tail else "")


def main() -> int:
    pre = {p: sha(p) for p in TOUCHED}
    originals = {p: p.read_bytes() for p in TOUCHED}
    bak = Path(tempfile.mkdtemp(prefix="d1c_r6_whichbak_"))
    for p in TOUCHED:
        shutil.copy2(p, bak / p.name)

    rc, failed, line = run_files()
    rec("W0", rc == 0, f"unmutated D1c test files: rc={rc} {line!r}")

    try:
        for tag, path, marker, repl, label in MUTATIONS:
            raw = originals[path]
            crlf = b"\r\n" in raw
            text = raw.decode("utf-8")
            mk = marker.replace("\n", "\r\n") if crlf else marker
            rp = repl.replace("\n", "\r\n") if crlf else repl
            if mk not in text:
                rec(tag, None, f"SKIP marker not found for {label}")
                continue
            try:
                path.write_bytes(text.replace(mk, rp, 1).encode("utf-8"))
                rc, failed, line = run_files()
                rec(tag, rc != 0, f"MUTATION {label} -> caught by {failed or 'NOTHING'}")
            finally:
                path.write_bytes(originals[path])
    finally:
        for p in TOUCHED:
            p.write_bytes(originals[p])

    post = {p: sha(p) for p in TOUCHED}
    intact = all(pre[p] == post[p] for p in TOUCHED)
    rec("W10", intact, "post-mutation source integrity: " + " ".join(
        f"{p.name}={'OK' if pre[p] == post[p] else 'CHANGED'}" for p in TOUCHED))
    shutil.rmtree(bak, ignore_errors=True)

    fails = sum(1 for line_ in LINES if " FAIL " in line_)
    summary = f"SUMMARY fails={fails} total={len(LINES)}"
    LINES.append(summary)
    print(summary, flush=True)
    LOG.write_text("\n".join(LINES) + "\n", encoding="utf-8")
    return 0 if intact else 1


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""REAUDIT_ROUND6 — whole-suite mutation check for the five guards that no test
in ``tests/test_d1c_infra_repair.py`` caught.

The in-memory probe can only reach tests that go through the ``_load_script``
seam. To claim "the 406-test suite does not catch this", the mutation has to be
on disk while the *whole* suite runs. Safety:

* every touched file is backed up to /tmp **and** kept in memory before any write;
* restore happens in ``finally`` after each mutation and again at exit;
* sha256 of every touched file is verified against the pre-run snapshot at the
  end and printed into the log.

No commit, no GPU, no source left mutated.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
LOG = ROOT / ".cursor/results/d1c_semantic_eos_repair_b81ecc3e/logs/reaudit6_probe_mutation_full.log"
PY = "/home/adam/projects/tts-self/.venv/bin/python"

SRC74 = ROOT / "scripts/74_eval_discrete_lm.py"
SRC76 = ROOT / "scripts/76_select_discrete_checkpoint.py"
SCHEMA = ROOT / "src/minispeech/eval/discrete_report_schema.py"
TOUCHED = [SRC74, SRC76, SCHEMA]

LINES: list[str] = []


def rec(tag: str, ok, msg: str) -> None:
    status = "INFO" if ok is None else ("PASS" if ok else "FAIL")
    LINES.append(f"{tag} {status} {msg}")
    print(f"{tag} {status} {msg}", flush=True)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


DROP_SELECTOR_HASHES = (
    '        hashes = ckpt.get("hashes")\n'
    "        if not isinstance(hashes, dict):\n"
    "            raise ValueError(\n"
    '                "D1c formal selection refuses checkpoint without hashes "\n'
    "                f\"(step={step}); payload['hashes'] must bind the candidate \"\n"
    '                "to one sealed training run"\n'
    "            )\n"
    "        identity: dict[str, str] = {}\n"
    "        for key in run_identity_keys:\n"
    "            value = hashes.get(key)\n"
    '            if value is None or not str(value).strip() or str(value) == ("0" * 64):\n'
    "                raise ValueError(\n"
    '                    f"D1c formal selection refuses missing checkpoint.hashes.{key} "\n'
    '                    f"(step={step})"\n'
    "                )\n"
    "            identity[key] = str(value)\n"
)
ADD_SELECTOR_HASHES_PERMISSIVE = (
    '        hashes = ckpt.get("hashes") or {}\n'
    "        identity: dict[str, str] = {\n"
    '            k: str(hashes.get(k) or "") for k in run_identity_keys\n'
    "        }\n"
)
DROP_MIXED_IDENTITY = "    if len(set(run_fingerprints)) != 1:\n"
DROP_EMIT_HASHES = (
    '    raw_hashes = provenance.get("checkpoint_hashes")\n'
    "    if not isinstance(raw_hashes, Mapping):\n"
    "        raise ValueError(\n"
)
ADD_EMIT_HASHES_PERMISSIVE = (
    '    raw_hashes = provenance.get("checkpoint_hashes")\n'
    "    if not isinstance(raw_hashes, Mapping):\n"
    '        raw_hashes = {k: "z" * 64 for k in _REQUIRED_CHECKPOINT_HASH_KEYS}\n'
    "    if False:\n"
    "        raise ValueError(\n"
)
DROP_SCHEMA_HASHES = (
    '    hashes = checkpoint.get("hashes")\n'
    "    if not isinstance(hashes, Mapping):\n"
    '        _fail("checkpoint.hashes missing")\n'
    "    for key in (\n"
    '        "eval_contract",\n'
    '        "recipe",\n'
    '        "recipe_source_manifest",\n'
    '        "run_manifest",\n'
    "    ):\n"
    "        value = hashes.get(key)\n"
    '        if value is None or not str(value).strip() or str(value) == ("0" * 64):\n'
    '            _fail(f"checkpoint.hashes.{key} missing")\n'
)
DROP_SCHEMA_BASENAME = (
    "    name_match = _CANDIDATE_STEP_NAME_RE.search(ckpt_path.name)\n"
    "    if name_match is not None and int(name_match.group(1)) != int(step):\n"
)

MUTATIONS = [
    ("N2", SRC76, DROP_SELECTOR_HASHES, ADD_SELECTOR_HASHES_PERMISSIVE,
     "selector tolerates missing/placeholder checkpoint.hashes"),
    ("N3", SRC76, DROP_MIXED_IDENTITY, "    if False:\n",
     "selector accepts mixed checkpoint.hashes across candidates"),
    ("N7", SRC74, DROP_EMIT_HASHES, ADD_EMIT_HASHES_PERMISSIVE,
     "emitter fabricates checkpoint.hashes when provenance lacks them"),
    ("N8", SCHEMA, DROP_SCHEMA_HASHES, "    pass\n",
     "schema drops the checkpoint.hashes requirement"),
    ("N9", SCHEMA, DROP_SCHEMA_BASENAME,
     "    name_match = _CANDIDATE_STEP_NAME_RE.search(ckpt_path.name)\n    if False:\n",
     "schema drops the cross-wired basename check"),
]


def run_suite() -> tuple[int, str]:
    env = dict(os.environ)
    env.update(
        CUDA_VISIBLE_DEVICES="",
        HF_HOME="/home/adam/hf_cache",
        PYTHONPATH=str(ROOT / "src"),
        GIT_DIR="/mnt/c/projects/tts-self/.git/worktrees/tts-self-d1c",
        GIT_WORK_TREE=str(ROOT),
    )
    proc = subprocess.run(
        [PY, "-m", "pytest", "-q", "--no-header", "-p", "no:cacheprovider"],
        cwd=str(ROOT), env=env, capture_output=True, text=True,
    )
    tail = (proc.stdout + proc.stderr).strip().splitlines()
    return proc.returncode, (tail[-1] if tail else "<no output>")


def main() -> int:
    pre = {p: sha(p) for p in TOUCHED}
    backup_dir = Path(tempfile.mkdtemp(prefix="d1c_r6_mutbak_"))
    originals = {}
    for p in TOUCHED:
        originals[p] = p.read_bytes()
        shutil.copy2(p, backup_dir / p.name)
    rec("N0", None, f"pre-mutation sha256: " + " ".join(f"{p.name}={pre[p][:16]}" for p in TOUCHED))
    rec("N0a", None, f"backups at {backup_dir}")

    rc, line = run_suite()
    rec("N1", rc == 0, f"unmutated whole suite: rc={rc} {line!r}")

    try:
        for tag, path, marker, repl, label in MUTATIONS:
            raw = originals[path]
            crlf = b"\r\n" in raw
            text = raw.decode("utf-8")
            mk = marker.replace("\n", "\r\n") if crlf else marker
            rp = repl.replace("\n", "\r\n") if crlf else repl
            if mk not in text:
                rec(tag, None, f"SKIP marker not found verbatim for {label} (crlf={crlf})")
                continue
            try:
                path.write_bytes(text.replace(mk, rp, 1).encode("utf-8"))
                rc, line = run_suite()
                rec(tag, rc != 0, f"MUTATION {label} -> whole suite rc={rc} {line!r}")
            finally:
                path.write_bytes(originals[path])
    finally:
        for p in TOUCHED:
            p.write_bytes(originals[p])

    post = {p: sha(p) for p in TOUCHED}
    intact = all(pre[p] == post[p] for p in TOUCHED)
    rec("N10", intact, "post-mutation source integrity: " + " ".join(
        f"{p.name}={'OK' if pre[p] == post[p] else 'CHANGED'}" for p in TOUCHED))
    shutil.rmtree(backup_dir, ignore_errors=True)

    fails = sum(1 for line in LINES if " FAIL " in line)
    summary = f"SUMMARY fails={fails} total={len(LINES)}"
    LINES.append(summary)
    print(summary, flush=True)
    LOG.write_text("\n".join(LINES) + "\n", encoding="utf-8")
    return 0 if intact else 1


if __name__ == "__main__":
    raise SystemExit(main())

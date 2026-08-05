#!/usr/bin/env python3
"""REAUDIT_ROUND5 — mutation testing of the FOUR round-5 negative tests.

A green suite proves nothing unless the new tests go red when the fix they
attest is removed. Method (same as reaudit4_probe_mutation): rebuild
``scripts/76`` / ``scripts/74`` / the schema **in memory** with one round-5
guard spliced out, hand the mutated module to the test through its own
``_load_script`` seam, and run the test body. Nothing on disk is modified.

A mutation the suite does NOT catch is a coverage hole.
"""
from __future__ import annotations

import importlib
import importlib.util
import sys
import tempfile
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT))

SRC74 = ROOT / "scripts/74_eval_discrete_lm.py"
SRC76 = ROOT / "scripts/76_select_discrete_checkpoint.py"
SCHEMA = ROOT / "src/minispeech/eval/discrete_report_schema.py"
TEST = ROOT / "tests/test_d1c_infra_repair.py"

T_DUP = "test_selector_refuses_duplicate_checkpoint_hashes"
T_MM = "test_selector_refuses_checkpoint_hash_mismatch"
T_SCHEMA = "test_validate_formal_report_rejects_checkpoint_hash_mismatch"
T_CLI = "test_resolve_checkpoint_step_payload_filename_cli"
T_E2E = "test_main_stamps_checkpoint_step_and_selector_e2e"

RESULTS: list[tuple[str, str, str]] = []


def rec(tag: str, ok, msg: str) -> None:
    status = "INFO" if ok is None else ("PASS" if ok else "FAIL")
    RESULTS.append((tag, status, msg))
    print(f"{tag} {status} {msg}", flush=True)


def build_module(src_text: str, name: str, path: Path) -> types.ModuleType:
    mod = types.ModuleType(name)
    mod.__file__ = str(path)
    exec(compile(src_text, str(path), "exec"), mod.__dict__)  # noqa: S102
    return mod


def load_test_module():
    spec = importlib.util.spec_from_file_location("t_d1c_infra_r5", TEST)
    mod = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(mod)
    return mod


def run_test(testname: str, *, mod74=None, mod76=None) -> tuple[bool, str]:
    T = load_test_module()
    real = T._load_script

    def loader(n):
        if n == "74_eval_discrete_lm.py" and mod74 is not None:
            return mod74
        if n == "76_select_discrete_checkpoint.py" and mod76 is not None:
            return mod76
        return real(n)

    T._load_script = loader
    with tempfile.TemporaryDirectory() as td:
        try:
            fn = getattr(T, testname)
            try:
                fn(Path(td))
            except TypeError as exc:
                if "positional argument" in str(exc):
                    fn()
                else:
                    raise
            return True, "PASSED"
        except BaseException as exc:  # noqa: BLE001
            head = (str(exc).splitlines() or ["<no message>"])[0][:170]
            return False, f"{type(exc).__name__}: {head}"


def cut(text: str, marker: str, replacement: str, label: str) -> str:
    assert marker in text, f"marker not found verbatim for {label}"
    return text.replace(marker, replacement, 1)


DROP_SELECTOR_CALL = "    _require_distinct_checkpoint_identities(reports)\n"

DROP_LIVE_HASH = (
    '        live = _sha256_file(ckpt_path)\n'
    '        claimed = str(ckpt["sha256"])\n'
    "        if live != claimed:\n"
    "            raise ValueError(\n"
    '                f"checkpoint sha256 mismatch: claimed={claimed} live={live} "\n'
    '                f"path={ckpt_path}"\n'
    "            )\n"
)

DROP_BASENAME = (
    "        if ckpt_path.name.lower() != expected_name.lower():\n"
    "            raise ValueError(\n"
    '                f"checkpoint path basename {ckpt_path.name!r} does not match "\n'
    '                f"expected {expected_name!r} for step={step}"\n'
    "            )\n"
)

DROP_DISTINCT = (
    "    if len(set(sha_list)) != len(sha_list):\n"
    "        raise ValueError(\n"
)

DROP_SCHEMA_HASH = (
    "    if ckpt_path.is_file():\n"
    "        live = file_sha256(ckpt_path)\n"
    '        claimed = str(checkpoint["sha256"])\n'
    "        if live != claimed:\n"
    "            _fail(\n"
    '                f"checkpoint.sha256 does not match file at checkpoint.path "\n'
    '                f"(claimed={claimed} live={live})"\n'
    "            )\n"
)

RESTORE_INVENT = "        if inferred is None:\n            if require:\n"


def main() -> int:  # noqa: C901, PLR0915
    s74 = SRC74.read_text(encoding="utf-8")
    s76 = SRC76.read_text(encoding="utf-8")
    sschema = SCHEMA.read_text(encoding="utf-8")

    base76 = build_module(s76, "m76_base", SRC76)
    base74 = build_module(s74, "m74_base", SRC74)

    # ------------------------------------------------------------ Y0 baseline
    for tag, tname in (("Y0a", T_DUP), ("Y0b", T_MM), ("Y0c", T_SCHEMA),
                       ("Y0d", T_CLI), ("Y0e", T_E2E)):
        ok, msg = run_test(tname)
        rec(tag, ok, f"unmutated: {tname} -> {msg}")

    # --------------- Y1: remove the whole distinct-identity gate from selector
    m = build_module(cut(s76, DROP_SELECTOR_CALL, "    pass\n", "selector call"),
                     "m76_nogate", SRC76)
    for tag, tname in (("Y1a", T_DUP), ("Y1b", T_MM), ("Y1c", T_E2E)):
        ok, msg = run_test(tname, mod76=m)
        rec(tag, not ok, f"MUTATION selector skips _require_distinct_checkpoint_identities "
                         f"-> {tname} {msg}")

    # ------------------------- Y2: keep distinct-sha, drop the on-disk verify
    m = build_module(cut(s76, DROP_LIVE_HASH, "        pass\n", "live hash"),
                     "m76_nohash", SRC76)
    ok_dup, msg_dup = run_test(T_DUP, mod76=m)
    ok_mm, msg_mm = run_test(T_MM, mod76=m)
    rec("Y2a", ok_dup, f"MUTATION drop on-disk verify -> {T_DUP} {msg_dup} "
                       "(expected still green: different guard)")
    rec("Y2b", not ok_mm, f"MUTATION drop on-disk verify -> {T_MM} {msg_mm}")

    # ------------------------------------------ Y3: drop the basename rule
    m = build_module(cut(s76, DROP_BASENAME, "        pass\n", "basename"),
                     "m76_noname", SRC76)
    caught = []
    for tname in (T_DUP, T_MM, T_E2E):
        ok, msg = run_test(tname, mod76=m)
        if not ok:
            caught.append(f"{tname}: {msg[:60]}")
    rec("Y3", bool(caught),
        f"MUTATION drop candidate_step_<step>.pt basename rule -> caught by {caught or 'NOTHING'}")

    # ------------------------------------------ Y4: drop the distinct-sha rule
    m = build_module(
        cut(s76, DROP_DISTINCT, "    if False:\n        raise ValueError(\n", "distinct"),
        "m76_nodistinct", SRC76,
    )
    caught = []
    for tname in (T_DUP, T_MM, T_E2E):
        ok, msg = run_test(tname, mod76=m)
        if not ok:
            caught.append(f"{tname}: {msg[:60]}")
    rec("Y4", bool(caught),
        f"MUTATION drop distinct-sha rule -> caught by {caught or 'NOTHING'}")

    # ------------------------------------------ Y5: schema on-disk hash check
    real_schema = importlib.import_module("minispeech.eval.discrete_report_schema")
    mutated = types.ModuleType("minispeech.eval.discrete_report_schema")
    mutated.__file__ = str(SCHEMA)
    exec(compile(cut(sschema, DROP_SCHEMA_HASH, "    pass\n", "schema hash"),  # noqa: S102
                 str(SCHEMA), "exec"), mutated.__dict__)
    sys.modules["minispeech.eval.discrete_report_schema"] = mutated
    try:
        ok, msg = run_test(T_SCHEMA)
        rec("Y5", not ok, f"MUTATION schema skips on-disk checkpoint hash -> {T_SCHEMA} {msg}")
    finally:
        sys.modules["minispeech.eval.discrete_report_schema"] = real_schema
        importlib.reload(real_schema)

    # ------------------------------- Y6: restore round-4 CLI invent behaviour
    m74 = build_module(
        cut(s74, RESTORE_INVENT, "        if inferred is None:\n            if False:\n",
            "invent guard"),
        "m74_invent", SRC74,
    )
    ok, msg = run_test(T_CLI, mod74=m74)
    rec("Y6", not ok, f"MUTATION restore round-4 --checkpoint-step invent -> {T_CLI} {msg}")

    # ------------------ Y7: is the un-bypassed (real schema) attack covered?
    src_test = TEST.read_text(encoding="utf-8")
    e2e_body = src_test.split(f"def {T_E2E}")[1]
    rec("Y7", "validate_formal_report(forged)" in e2e_body,
        "at least one attack test runs the REAL schema before the selector "
        f"({T_E2E})")
    rec("Y8", None,
        f"{T_DUP} / {T_MM} stub validate_formal_report to a pass-through "
        f"(schema-bypassed unit tests): "
        f"{src_test.count('schema.validate_formal_report = lambda data, **kwargs: data')} occurrences")

    fails = sum(1 for _, st, _ in RESULTS if st == "FAIL")
    print(f"\nSUMMARY fails={fails} total={len(RESULTS)}")
    for tag, st, msg in RESULTS:
        if st == "FAIL":
            print(f"  FAILED {tag}: {msg}")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())

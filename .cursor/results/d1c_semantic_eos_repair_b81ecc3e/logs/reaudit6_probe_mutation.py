#!/usr/bin/env python3
"""REAUDIT_ROUND6 — mutation testing of the ROUND-6 guards.

A green suite proves nothing unless the new tests go red when the fix they
attest is removed. Method: rebuild ``scripts/74`` / ``scripts/76`` / the schema
**in memory** with one round-6 guard spliced out, hand the mutated module to the
tests through their own ``_load_script`` seam, and run **every** test in
``tests/test_d1c_infra_repair.py`` (not a hand-picked subset). A mutation caught
by NOTHING is a coverage hole.

Nothing on disk is modified.
"""

from __future__ import annotations

import importlib
import importlib.util
import inspect
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
LOG = ROOT / ".cursor/results/d1c_semantic_eos_repair_b81ecc3e/logs/reaudit6_probe_mutation.log"

T_BASENAME = "test_selector_refuses_checkpoint_basename_mismatch"
T_HASHES = "test_require_checkpoint_payload_hashes_fail_closed"
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
    spec = importlib.util.spec_from_file_location("t_d1c_infra_r6", TEST)
    mod = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(mod)
    return mod


def all_test_names() -> list[str]:
    T = load_test_module()
    return sorted(n for n in dir(T) if n.startswith("test_"))


def run_one(T, testname: str) -> tuple[bool, str]:
    fn = getattr(T, testname)
    params = list(inspect.signature(fn).parameters)
    with tempfile.TemporaryDirectory() as td:
        try:
            fn(Path(td)) if params else fn()
            return True, "PASSED"
        except BaseException as exc:  # noqa: BLE001
            head = (str(exc).splitlines() or ["<no message>"])[0][:120]
            return False, f"{type(exc).__name__}: {head}"


def run_all(names: list[str], *, mod74=None, mod76=None) -> dict[str, tuple[bool, str]]:
    T = load_test_module()
    real = T._load_script

    def loader(n):
        if n == "74_eval_discrete_lm.py" and mod74 is not None:
            return mod74
        if n == "76_select_discrete_checkpoint.py" and mod76 is not None:
            return mod76
        return real(n)

    T._load_script = loader
    return {n: run_one(T, n) for n in names}


def cut(text: str, marker: str, replacement: str, label: str) -> str:
    assert marker in text, f"marker not found verbatim for {label}"
    return text.replace(marker, replacement, 1)


DROP_BASENAME = (
    "        if ckpt_path.name.lower() != expected_name.lower():\n"
    "            raise ValueError(\n"
    '                f"checkpoint path basename {ckpt_path.name!r} does not match "\n'
    '                f"expected {expected_name!r} for step={step}"\n'
    "            )\n"
)

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

DROP_DISTINCT_SHA = "    if len(set(sha_list)) != len(sha_list):\n"

DROP_PAYLOAD_STEP_REQUIRED = "    if require and payload_step is None:\n"

DROP_REQUIRE_HASHES = (
    "    hashes = checkpoint_hashes_from_payload(payload)\n"
    "    if hashes is None:\n"
    "        raise SystemExit(\n"
    "            \"formal D1c requires payload['hashes'] with eval_contract/recipe/\"\n"
    '            "recipe_source_manifest/run_manifest stamped by the trainer"\n'
    "        )\n"
    '    if expected_eval_contract and hashes["eval_contract"] != str(expected_eval_contract):\n'
)
ADD_REQUIRE_HASHES_PERMISSIVE = (
    "    hashes = checkpoint_hashes_from_payload(payload) or {\n"
    '        "eval_contract": str(expected_eval_contract or "x" * 64),\n'
    '        "recipe": "x" * 64,\n'
    '        "recipe_source_manifest": "x" * 64,\n'
    '        "run_manifest": "x" * 64,\n'
    "    }\n"
    "    if False:\n"
)

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


def main() -> int:  # noqa: C901, PLR0915
    s74 = SRC74.read_text(encoding="utf-8")
    s76 = SRC76.read_text(encoding="utf-8")
    sschema = SCHEMA.read_text(encoding="utf-8")
    names = all_test_names()
    rec("M0", None, f"test_d1c_infra_repair.py exposes {len(names)} test functions")

    base = run_all(names)
    base_fail = [n for n, (ok, _) in base.items() if not ok]
    rec("M0a", not base_fail, f"unmutated baseline: {len(names) - len(base_fail)}/{len(names)} green; red={base_fail}")

    def report(tag: str, label: str, res: dict[str, tuple[bool, str]], *, expect_caught=True):
        caught = [n for n, (ok, _) in res.items() if not ok and n not in base_fail]
        ok = bool(caught) if expect_caught else not caught
        rec(tag, ok, f"MUTATION {label} -> caught by {caught or 'NOTHING'}")
        return caught

    # ---- M1: selector basename rule (round-5 Y3 hole; round-6 claims it closed)
    m = build_module(cut(s76, DROP_BASENAME, "        pass\n", "basename"), "m76_noname", SRC76)
    caught = report("M1", "selector drops candidate_step_<step:04d>.pt basename rule", run_all(names, mod76=m))
    rec("M1a", T_BASENAME in caught, f"{T_BASENAME} is the guard's dedicated test (caught={T_BASENAME in caught})")

    # ---- M2: selector drops the checkpoint.hashes presence/placeholder rules
    m = build_module(
        cut(s76, DROP_SELECTOR_HASHES, ADD_SELECTOR_HASHES_PERMISSIVE, "selector hashes"),
        "m76_nohashes", SRC76,
    )
    report("M2", "selector tolerates missing/placeholder checkpoint.hashes", run_all(names, mod76=m))

    # ---- M3: selector drops the one-sealed-run identity comparison
    m = build_module(cut(s76, DROP_MIXED_IDENTITY, "    if False:\n", "mixed identity"),
                     "m76_mixedok", SRC76)
    report("M3", "selector accepts mixed checkpoint.hashes across candidates", run_all(names, mod76=m))

    # ---- M4: selector drops the distinct-sha rule (round-5 regression)
    m = build_module(cut(s76, DROP_DISTINCT_SHA, "    if False:\n", "distinct sha"),
                     "m76_dupok", SRC76)
    report("M4", "selector accepts duplicate checkpoint.sha256 (round-5 guard)", run_all(names, mod76=m))

    # ---- M5: evaluator restores round-5 filename-only step (the round-6 fix itself)
    m = build_module(cut(s74, DROP_PAYLOAD_STEP_REQUIRED, "    if False:\n", "payload step"),
                     "m74_filename", SRC74)
    caught = report("M5", "resolve_checkpoint_step no longer requires payload['step']", run_all(names, mod74=m))
    rec("M5a", T_CLI in caught, f"{T_CLI} catches the rename fix removal (caught={T_CLI in caught})")

    # ---- M6: evaluator tolerates a payload with no hashes
    m = build_module(cut(s74, DROP_REQUIRE_HASHES, ADD_REQUIRE_HASHES_PERMISSIVE, "require hashes"),
                     "m74_nohashes", SRC74)
    caught = report("M6", "require_checkpoint_payload_hashes made permissive", run_all(names, mod74=m))
    rec("M6a", T_HASHES in caught, f"{T_HASHES} catches it (caught={T_HASHES in caught})")

    # ---- M7: emitter stops requiring provenance checkpoint_hashes
    m = build_module(cut(s74, DROP_EMIT_HASHES, ADD_EMIT_HASHES_PERMISSIVE, "emit hashes"),
                     "m74_emitfree", SRC74)
    report("M7", "emitter stamps synthetic checkpoint.hashes when provenance lacks them",
           run_all(names, mod74=m))

    # ---- M8 / M9: schema guards (patch sys.modules so every consumer sees it)
    real_schema = importlib.import_module("minispeech.eval.discrete_report_schema")
    for tag, marker, repl, label in (
        ("M8", DROP_SCHEMA_HASHES, "    pass\n", "schema drops checkpoint.hashes requirement"),
        ("M9", DROP_SCHEMA_BASENAME,
         "    name_match = _CANDIDATE_STEP_NAME_RE.search(ckpt_path.name)\n    if False:\n",
         "schema drops cross-wired basename check"),
    ):
        mutated = types.ModuleType("minispeech.eval.discrete_report_schema")
        mutated.__file__ = str(SCHEMA)
        exec(compile(cut(sschema, marker, repl, label), str(SCHEMA), "exec"), mutated.__dict__)  # noqa: S102
        sys.modules["minispeech.eval.discrete_report_schema"] = mutated
        try:
            m76 = build_module(s76, f"m76_{tag}", SRC76)
            m74 = build_module(s74, f"m74_{tag}", SRC74)
            report(tag, label, run_all(names, mod74=m74, mod76=m76))
        finally:
            sys.modules["minispeech.eval.discrete_report_schema"] = real_schema
            importlib.reload(real_schema)

    # ---- disclosure: schema-bypassing stubs in the shipped tests
    src_test = TEST.read_text(encoding="utf-8")
    rec(
        "M10", None,
        "shipped tests monkeypatch validate_formal_report to a pass-through at "
        f"{src_test.count('validate_formal_report = lambda data, **kwargs: data')} sites; "
        f"real-schema attack coverage present in {T_E2E}: "
        f"{'validate_formal_report(forged)' in src_test}",
    )

    fails = sum(1 for _, st, _ in RESULTS if st == "FAIL")
    summary = f"SUMMARY fails={fails} total={len(RESULTS)}"
    print("\n" + summary, flush=True)
    for tag, st, msg in RESULTS:
        if st == "FAIL":
            print(f"  FAILED {tag}: {msg}", flush=True)
    LOG.write_text(
        "\n".join(f"{t} {s} {m}" for t, s, m in RESULTS) + f"\n{summary}\n", encoding="utf-8"
    )
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())

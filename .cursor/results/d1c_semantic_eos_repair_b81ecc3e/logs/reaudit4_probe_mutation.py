#!/usr/bin/env python3
"""REAUDIT_ROUND4 — mutation testing of the new main()-level regression test.

Round 3 proved (X3/X4) that `test_production_tf_aggregation_to_emitter_survives`
could NOT see a `main()`-level regression. Round 4 claims the new
`test_main_stamps_checkpoint_step_and_selector_e2e` closes that gap.

Method: rebuild `scripts/74` in memory with a defect spliced back in, hand the
mutated module to the test through its own `_load_script` seam, and run the test
body. Nothing on disk is modified. A mutation that the test does NOT catch is a
coverage hole; a mutation it DOES catch proves the test is load-bearing.
"""
from __future__ import annotations

import ast
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
SCHEMA = ROOT / "src/minispeech/eval/discrete_report_schema.py"
TEST = ROOT / "tests/test_d1c_infra_repair.py"
MAIN_TEST = "test_main_stamps_checkpoint_step_and_selector_e2e"
TF_TEST = "test_production_tf_aggregation_to_emitter_survives"

RESULTS: list[tuple[str, str, str]] = []


def rec(tag: str, ok, msg: str) -> None:
    status = "INFO" if ok is None else ("PASS" if ok else "FAIL")
    RESULTS.append((tag, status, msg))
    print(f"{tag} {status} {msg}", flush=True)


def build_module(src_text: str, name: str) -> types.ModuleType:
    mod = types.ModuleType(name)
    mod.__file__ = str(SRC74)
    code = compile(src_text, str(SRC74), "exec")
    exec(code, mod.__dict__)  # noqa: S102 - controlled in-memory mutation probe
    return mod


def load_test_module():
    spec = importlib.util.spec_from_file_location("t_d1c_infra_r4", TEST)
    mod = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(mod)
    return mod


def run_test_with(mod74, testname: str) -> tuple[bool, str]:
    T = load_test_module()
    real_loader = T._load_script
    T._load_script = lambda n: mod74 if n == "74_eval_discrete_lm.py" else real_loader(n)
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
            head = (str(exc).splitlines() or ["<no message>"])[0][:200]
            return False, f"{type(exc).__name__}: {head}"


# ------------------------------------------------------------------ mutations
def drop_step_stamping(src: str) -> str:
    """Remove BOTH provenance['checkpoint_step'] and gate['step'] stamps in main()."""
    out = []
    skip_until_dedent = 0
    lines = src.splitlines(keepends=True)
    i = 0
    n_removed = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        if stripped == "if checkpoint_step is not None:":
            indent = len(line) - len(line.lstrip())
            i += 1
            while i < len(lines):
                nxt = lines[i]
                if nxt.strip() and (len(nxt) - len(nxt.lstrip())) <= indent:
                    break
                i += 1
            n_removed += 1
            continue
        out.append(line)
        i += 1
    assert n_removed >= 2, f"expected >=2 stamp blocks, removed {n_removed}"
    _ = skip_until_dedent
    return "".join(out)


def rewire_main_to_bare_float(src: str) -> str:
    """Reinstate the round-2 `tf_ce = summary.get("loss")` inside main() only."""
    tree = ast.parse(src)
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main")
    target = None
    for node in ast.walk(fn):
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "tf_ce" for t in node.targets
        ):
            target = node
    assert target is not None
    lines = src.splitlines(keepends=True)
    indent = " " * target.col_offset
    lines[target.lineno - 1 : target.end_lineno] = [f'{indent}tf_ce = summary.get("loss")\n']
    return "".join(lines)


def swap_gate_d1c_for_d1b(src: str) -> str:
    """Make main() call evaluate_n2_gate_d1b even for a D1c contract."""
    marker = "                if d1c_contract:\n"
    assert marker in src, "d1c gate branch not found"
    return src.replace(marker, "                if False:\n", 1)


def load_model_discards_payload(src: str) -> str:
    """Make _load_model return an empty payload (round-3 defect shape)."""
    old = "    return model, cfg, payload if isinstance(payload, dict) else {}"
    assert old in src, "_load_model return not found"
    return src.replace(old, "    return model, cfg, {}", 1)


def relax_schema_step_check() -> str:
    """Round-3 schema: step=0 passed because only None/'' were rejected."""
    text = SCHEMA.read_text(encoding="utf-8")
    old = (
        "    if isinstance(step, bool) or not isinstance(step, int) or int(step) <= 0:\n"
        '        _fail("checkpoint.step must be a positive integer (not 0/missing)")\n'
    )
    assert old in text, "schema step guard not found verbatim"
    return text.replace(old, "    pass\n", 1)


def main() -> int:
    src = SRC74.read_text(encoding="utf-8")

    base = build_module(src, "m74_base")
    ok, msg = run_test_with(base, MAIN_TEST)
    rec("X0", ok, f"unmutated source: {MAIN_TEST} -> {msg}")

    ok, msg = run_test_with(build_module(drop_step_stamping(src), "m74_nostep"), MAIN_TEST)
    rec(
        "X1",
        not ok,
        f"MUTATION drop provenance['checkpoint_step'] + gate['step'] in main() -> {msg}",
    )

    ok, msg = run_test_with(
        build_module(rewire_main_to_bare_float(src), "m74_bare"), MAIN_TEST
    )
    rec(
        "X2",
        not ok,
        f"MUTATION round-2 main() bare-float rewire (round-3 X3/X4 blind spot) -> {msg}",
    )

    ok, msg = run_test_with(build_module(swap_gate_d1c_for_d1b(src), "m74_d1b"), MAIN_TEST)
    rec("X3", not ok, f"MUTATION main() uses evaluate_n2_gate_d1b on a D1c contract -> {msg}")

    ok, msg = run_test_with(
        build_module(load_model_discards_payload(src), "m74_nopayload"), MAIN_TEST
    )
    rec(
        "X4",
        None,
        f"MUTATION _load_model discards payload -> {msg} "
        "(INFO: test stubs _load_model, and the candidate_step_XXXX.pt filename is a "
        "second independent source, so this is expected not to be caught here)",
    )

    # Schema mutation needs the real module import path, so patch sys.modules.
    import importlib

    import minispeech.eval.discrete_report_schema as real_schema

    mutated = types.ModuleType("minispeech.eval.discrete_report_schema")
    mutated.__file__ = str(SCHEMA)
    exec(compile(relax_schema_step_check(), str(SCHEMA), "exec"), mutated.__dict__)  # noqa: S102
    sys.modules["minispeech.eval.discrete_report_schema"] = mutated
    try:
        ok, msg = run_test_with(base, "test_validate_formal_report_rejects_checkpoint_step_zero")
        rec("X5", not ok, f"MUTATION round-3 schema (step=0 allowed) -> step-zero test {msg}")
    finally:
        sys.modules["minispeech.eval.discrete_report_schema"] = real_schema
        importlib.reload(real_schema)

    ok, msg = run_test_with(base, TF_TEST)
    rec("X6", ok, f"round-3 helper test still green on unmutated source: {msg}")

    fails = sum(1 for _, st, _ in RESULTS if st == "FAIL")
    print(f"\nSUMMARY fails={fails} total={len(RESULTS)}")
    for tag, st, msg in RESULTS:
        if st == "FAIL":
            print(f"  FAILED {tag}: {msg}")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())

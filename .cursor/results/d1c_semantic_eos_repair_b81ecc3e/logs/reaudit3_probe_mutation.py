#!/usr/bin/env python3
"""REAUDIT ROUND 3 — mutation testing of the new regression test.

Question 1: would ``test_production_tf_aggregation_to_emitter_survives`` have
            failed on the round-2 discarded-metrics bug?
Question 2: does it also cover the *production wiring* in ``main()``?

Method: rebuild scripts/74 in memory with the round-2 code spliced back in, hand
the mutated module to the test via its own ``_load_script`` seam, and run the
test body. No file on disk is modified.
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
TEST = ROOT / "tests/test_d1c_infra_repair.py"
TESTNAME = "test_production_tf_aggregation_to_emitter_survives"

RESULTS: list[tuple[str, str, str]] = []


def rec(tag: str, ok: bool | None, msg: str) -> None:
    status = "INFO" if ok is None else ("PASS" if ok else "FAIL")
    RESULTS.append((tag, status, msg))
    print(f"{tag} {status} {msg}", flush=True)


LEGACY_AGG = '''
def aggregate_teacher_forced_batches(batch_metrics):
    """ROUND-2 CODE: discards every EOS field (the defect under test)."""
    totals = {"loss": 0.0, "ce_cb0": 0.0, "ce_residual": 0.0}
    for m in batch_metrics:
        for k in totals:
            totals[k] += float(m[k])
    n = max(len(batch_metrics), 1)
    return {k: v / n for k, v in totals.items()}
'''

LEGACY_PROJ = '''
def teacher_forced_metrics_for_emitter(summary):
    """ROUND-2 CODE: reduces the summary to a bare float."""
    return summary.get("loss")
'''


def replace_function(src: str, name: str, new_src: str) -> str:
    tree = ast.parse(src)
    node = next(
        n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name
    )
    lines = src.splitlines(keepends=True)
    start = node.lineno - 1
    end = node.end_lineno
    return "".join(lines[:start]) + new_src.lstrip("\n") + "".join(lines[end:])


def rewire_main_to_bare_float(src: str) -> str:
    """Reinstate `tf_ce = summary.get("loss")` inside main() only."""
    tree = ast.parse(src)
    fn = next(
        n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main"
    )
    target = None
    for node in ast.walk(fn):
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "tf_ce" for t in node.targets
        ):
            target = node
    assert target is not None
    lines = src.splitlines(keepends=True)
    indent = " " * (target.col_offset)
    lines[target.lineno - 1 : target.end_lineno] = [
        f'{indent}tf_ce = summary.get("loss")\n'
    ]
    return "".join(lines)


def build_module(src_text: str, name: str) -> types.ModuleType:
    mod = types.ModuleType(name)
    mod.__file__ = str(SRC74)
    code = compile(src_text, str(SRC74), "exec")
    exec(code, mod.__dict__)  # noqa: S102 - controlled mutation probe
    return mod


def load_test_module():
    spec = importlib.util.spec_from_file_location("t_d1c_infra", TEST)
    mod = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(mod)
    return mod


def run_test_with(mod74, label: str) -> tuple[bool, str]:
    T = load_test_module()
    real_loader = T._load_script
    T._load_script = lambda n: mod74 if n == "74_eval_discrete_lm.py" else real_loader(n)
    with tempfile.TemporaryDirectory() as td:
        try:
            getattr(T, TESTNAME)(Path(td))
            return True, "test PASSED"
        except BaseException as exc:  # noqa: BLE001
            return False, f"{type(exc).__name__}: {str(exc).splitlines()[0][:140]}"
        finally:
            T._load_script = real_loader


def main() -> int:
    src = SRC74.read_text(encoding="utf-8")

    ok, msg = run_test_with(build_module(src, "m74_clean"), "clean")
    rec("X0", ok, f"unmutated source: {msg}")

    mutated = replace_function(src, "aggregate_teacher_forced_batches", LEGACY_AGG)
    ok, msg = run_test_with(build_module(mutated, "m74_agg"), "legacy-agg")
    rec(
        "X1",
        not ok,
        "round-2 discarding aggregation spliced back in -> the new test "
        f"{'FAILS as required' if not ok else 'STILL PASSES (test is blind)'}: {msg}",
    )

    mutated2 = replace_function(src, "teacher_forced_metrics_for_emitter", LEGACY_PROJ)
    ok, msg = run_test_with(build_module(mutated2, "m74_proj"), "legacy-proj")
    rec(
        "X2",
        not ok,
        "round-2 bare-float projection spliced back in -> the new test "
        f"{'FAILS as required' if not ok else 'STILL PASSES (test is blind)'}: {msg}",
    )

    mutated3 = rewire_main_to_bare_float(src)
    ok, msg = run_test_with(build_module(mutated3, "m74_main"), "main-rewire")
    rec(
        "X3",
        None,
        "main() rewired back to `tf_ce = summary.get(\"loss\")` (helpers untouched) "
        f"-> the new test {'STILL PASSES' if ok else 'fails'}: {msg}",
    )
    rec(
        "X4",
        not ok,
        "COVERAGE: does the new test protect the production wiring in main()? "
        + ("yes" if not ok else "NO — it only covers the helper chain; a main() "
           "regression would ship green. My AST/main() probes cover this gap."),
    )

    n_fail = sum(1 for _, s, _ in RESULTS if s == "FAIL")
    print(f"\nSUMMARY pass={sum(1 for _, s, _ in RESULTS if s == 'PASS')} fail={n_fail}")
    print("ROUND3_REAUDIT_MUTATION_DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

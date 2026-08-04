from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


def _selector():
    path = Path(__file__).parents[1] / "scripts" / "76_select_discrete_checkpoint.py"
    spec = importlib.util.spec_from_file_location("d1c_selector", path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_d1c_formal_refuses_manual_and_missing_candidates(tmp_path):
    mod = _selector()
    with pytest.raises(ValueError, match="exactly five"):
        mod.select_d1c_formal_reports([tmp_path / "manual.json"])

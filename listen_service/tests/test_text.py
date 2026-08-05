"""Unit tests for listen_service text frontend (no GPU)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

SERVICE_ROOT = Path(__file__).resolve().parents[1]
REPO_SRC = SERVICE_ROOT.parent / "src"
if str(SERVICE_ROOT) not in sys.path:
    sys.path.insert(0, str(SERVICE_ROOT))
if REPO_SRC.is_dir() and str(REPO_SRC) not in sys.path:
    sys.path.insert(0, str(REPO_SRC))

from runtime.text import prepare_text  # noqa: E402


def test_prepare_text_rejects_empty() -> None:
    with pytest.raises(ValueError, match="empty"):
        prepare_text("   ")


def test_prepare_text_returns_nonempty_russian() -> None:
    pytest.importorskip("minispeech.frontend.verbalize")
    out = prepare_text("Привет, мир.")
    assert isinstance(out, str)
    assert out.strip()
    # Stress may or may not add U+0301 depending on ruaccent availability;
    # frontend must at least preserve Cyrillic content.
    assert any("а" <= ch.lower() <= "я" or ch.lower() in "ё" for ch in out.lower())

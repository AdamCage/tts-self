"""WER helpers."""

from __future__ import annotations

from typing import Sequence

from jiwer import cer, wer


def compute_wer(refs: Sequence[str], hyps: Sequence[str]) -> float:
    return float(wer(list(refs), list(hyps)))


def compute_cer(refs: Sequence[str], hyps: Sequence[str]) -> float:
    return float(cer(list(refs), list(hyps)))


def delta_wer(wer_orig: float, wer_codec: float) -> float:
    """Percentage points."""
    return (wer_codec - wer_orig) * 100.0

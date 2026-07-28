"""Paired utterance bootstrap for corpus-level WER/CER deltas."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass
class EditCounts:
    hits: int = 0
    substitutions: int = 0
    deletions: int = 0
    insertions: int = 0

    @property
    def edits(self) -> int:
        return self.substitutions + self.deletions + self.insertions

    @property
    def ref_units(self) -> int:
        return self.hits + self.substitutions + self.deletions


@dataclass
class BootstrapResult:
    delta_pp: float
    ci95_low: float
    ci95_high: float
    p_delta_gt_threshold: float
    metric_orig: float
    metric_codec: float


def corpus_rate(counts: Sequence[EditCounts]) -> float:
    """Corpus-level error rate = sum(edits) / sum(ref_units)."""
    edits = sum(c.edits for c in counts)
    refs = sum(c.ref_units for c in counts)
    if refs <= 0:
        return 0.0
    return edits / refs


def paired_bootstrap_delta(
    orig: Sequence[EditCounts],
    codec: Sequence[EditCounts],
    *,
    n_boot: int = 1000,
    threshold_pp: float = 4.0,
    seed: int = 42,
) -> BootstrapResult:
    """
    Paired bootstrap over utterances for Δ = (codec - orig) in percentage points.
    Resamples the same utterance indices for both conditions.
    """
    if len(orig) != len(codec):
        raise ValueError("orig and codec must have the same number of utterances")
    n = len(orig)
    if n == 0:
        return BootstrapResult(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    rate_o = corpus_rate(orig)
    rate_c = corpus_rate(codec)
    delta_pp = (rate_c - rate_o) * 100.0

    rng = np.random.default_rng(seed)
    deltas = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        o = [orig[i] for i in idx]
        c = [codec[i] for i in idx]
        deltas[b] = (corpus_rate(c) - corpus_rate(o)) * 100.0

    lo, hi = np.percentile(deltas, [2.5, 97.5])
    p_gt = float(np.mean(deltas > threshold_pp))
    return BootstrapResult(
        delta_pp=float(delta_pp),
        ci95_low=float(lo),
        ci95_high=float(hi),
        p_delta_gt_threshold=p_gt,
        metric_orig=float(rate_o),
        metric_codec=float(rate_c),
    )


def edits_from_jiwer_words(ref: str, hyp: str) -> EditCounts:
    from jiwer import process_words

    out = process_words(ref, hyp)
    return EditCounts(
        hits=int(out.hits),
        substitutions=int(out.substitutions),
        deletions=int(out.deletions),
        insertions=int(out.insertions),
    )


def edits_from_jiwer_chars(ref: str, hyp: str) -> EditCounts:
    from jiwer import process_characters

    out = process_characters(ref, hyp)
    return EditCounts(
        hits=int(out.hits),
        substitutions=int(out.substitutions),
        deletions=int(out.deletions),
        insertions=int(out.insertions),
    )

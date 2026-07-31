"""Tests for acoustic stress SA helpers (M05)."""

from __future__ import annotations

import numpy as np

from minispeech.eval.stress_acoustic import (
    StressWeights,
    VowelSpan,
    score_vowels,
)


def test_score_vowels_picks_longest_loudest():
    vowels = [
        VowelSpan("а", 0.0, 0.05, False, duration=0.05, energy=1.0, f0=120.0),
        VowelSpan("о", 0.05, 0.15, True, duration=0.10, energy=3.0, f0=180.0),
        VowelSpan("а", 0.15, 0.18, False, duration=0.03, energy=0.8, f0=110.0),
    ]
    idx = score_vowels(vowels, StressWeights(1.0, 1.0, 1.0))
    assert idx == 1


def test_score_vowels_empty():
    assert score_vowels([]) is None

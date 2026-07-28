"""Optional UTMOS / stress accuracy stubs."""

from __future__ import annotations


def stress_accuracy(refs: list[str], hyps: list[str]) -> float:
    from minispeech.frontend.stress import stress_agreement

    if not refs:
        return 0.0
    return sum(stress_agreement(a, b) for a, b in zip(refs, hyps)) / len(refs)

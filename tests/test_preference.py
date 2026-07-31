"""Tests for preference harness (M08)."""

from __future__ import annotations

from pathlib import Path

from minispeech.eval.preference import binomial_ci95, build_pairs, score_responses


def test_binomial_ci_self():
    lo, hi = binomial_ci95(50, 100)
    assert lo < 0.5 < hi


def test_build_and_score_trap(tmp_path: Path):
    a = tmp_path / "a"
    b = tmp_path / "b"
    a.mkdir()
    b.mkdir()
    for name in ("x.wav", "y.wav", "z.wav"):
        (a / name).write_bytes(b"RIFF")
        (b / name).write_bytes(b"RIFF")
    pairs, key = build_pairs(
        list(a.glob("*.wav")),
        list(b.glob("*.wav")),
        seed=0,
        n_traps=2,
        system_a_name="mean",
        system_b_name="pred",
    )
    assert any(p.trap for p in pairs)
    # All prefer left → depends on left_is_a mapping
    responses = []
    for p in pairs:
        if p.trap:
            responses.append({"pair_id": p.pair_id, "choice": "same"})
        else:
            # Always prefer system A
            choice = "left" if p.left_is_a else "right"
            responses.append({"pair_id": p.pair_id, "choice": choice})
    scored = score_responses(responses, key)
    assert scored["prefer_a"] == scored["n_scored"]
    assert scored["trap_same_rate"] == 1.0

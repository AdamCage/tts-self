"""Stressed char-CTC labels must stay 1:1 with text_ctc bases."""

from __future__ import annotations

from minispeech.constants import STRESS_CHAR
from minispeech.data.forced_align import text_to_ctc_char_labels, transcript_for_ctc
from minispeech.frontend.stress import strip_stress


def test_stressed_labels_match_text_ctc_length() -> None:
    text = "Приве́т, мир!"
    labels, meta = text_to_ctc_char_labels(text, already_stressed=False, keep_stress=True)
    text_ctc = meta["text_ctc"]
    assert len(labels) == len(text_ctc)
    # Reconstruct base string from labels
    bases = []
    for lab in labels:
        b = strip_stress(lab)
        bases.append(" " if b == "|" else b)
    assert "".join(bases) == text_ctc
    assert meta.get("stress_labels") is True
    # At least one label carries combining acute when stress is present upstream
    assert any(STRESS_CHAR in lab for lab in labels) or STRESS_CHAR not in meta["text_stressed"]


def test_keep_stress_false_plain_chars() -> None:
    labels, meta = text_to_ctc_char_labels("мо́ре", already_stressed=True, keep_stress=False)
    assert all(STRESS_CHAR not in lab for lab in labels)
    assert meta.get("stress_labels") is False
    assert len(labels) == len(meta["text_ctc"])


def test_transcript_for_ctc_plain_unchanged() -> None:
    meta = transcript_for_ctc("Да́ча")
    assert STRESS_CHAR not in meta["text_ctc"]
    assert "д" in meta["text_ctc"]

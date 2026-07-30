"""Stress overlay on char phones must preserve span count / bases."""

from __future__ import annotations

from minispeech.constants import STRESS_CHAR
from minispeech.data.forced_align import text_to_ctc_char_labels
from minispeech.data.mel_dataset import overlay_stress_on_char_phones
from minispeech.frontend.stress import strip_stress


def test_overlay_matches_stressed_labels() -> None:
    text = "мо́ре"
    plain, meta = text_to_ctc_char_labels(text, already_stressed=True, keep_stress=False)
    stressed, smeta = text_to_ctc_char_labels(text, already_stressed=True, keep_stress=True)
    assert smeta.get("stress_labels")
    out = overlay_stress_on_char_phones(plain, meta["text_stressed"])
    assert out == stressed
    assert any(STRESS_CHAR in p for p in out)


def test_overlay_preserves_sil_slots() -> None:
    text = "да"
    plain, meta = text_to_ctc_char_labels(text, already_stressed=True, keep_stress=False)
    phones = ["sil", *plain, "sil"]
    out = overlay_stress_on_char_phones(phones, meta["text_stressed"])
    assert out[0] == "sil" and out[-1] == "sil"
    assert [strip_stress(p) for p in out[1:-1]] == plain


def test_overlay_mismatch_returns_input() -> None:
    phones = ["а", "б", "в"]
    out = overlay_stress_on_char_phones(phones, "да")
    assert out == phones

"""Tests for Latin→Cyrillic transliteration (Track N / defect D2)."""

from __future__ import annotations

import re

from minispeech.frontend.normalize import normalize_text
from minispeech.frontend.translit import (
    has_latin,
    spell_latin_letters,
    spoken_latin_token,
    transliterate_latin,
    transliterate_word,
)
from minispeech.frontend.verbalize import normalize_for_tts

_LATIN = re.compile(r"[A-Za-z]")
_UNK = "<unk>"


def test_loanwords_dictionary():
    assert spoken_latin_token("Google") == "гугл"
    assert spoken_latin_token("apple") == "эпл"
    assert spoken_latin_token("Play") == "плей"
    assert "гугл" in transliterate_latin("через Google")
    assert not has_latin(transliterate_latin("Google Play"))


def test_acronyms_letter_by_letter():
    assert spoken_latin_token("TTS") == "тэ-тэ-э́с"
    assert spoken_latin_token("HR") == "эйч-а́р"
    assert spoken_latin_token("SMS") == "эс-эм-э́с"
    assert spoken_latin_token("IT") == "ай-ти́"
    assert spoken_latin_token("QR") == "ку-э́р"
    # Heuristic path (not in idiomatic dict): all-caps ≤ 5
    spelled = spell_latin_letters("BBC")
    assert spelled.startswith("би-би-")
    assert "\u0301" in spelled
    assert spoken_latin_token("BBC") == spelled


def test_mixed_sentence_frontend_all_cyrillic():
    raw = "Установите TTS через Google Play"
    out = normalize_text(raw)
    assert not _LATIN.search(out), out
    assert _UNK not in out
    assert "гугл" in out
    assert "плей" in out
    assert "тэ-тэ-э́с" in out
    # Full TN path used by infer / align also stays Cyrillic.
    tts = normalize_for_tts(raw)
    assert not _LATIN.search(tts), tts
    assert _UNK not in tts


def test_single_letters():
    assert spoken_latin_token("A") == spell_latin_letters("A")
    assert "\u0301" in spoken_latin_token("A")
    assert spoken_latin_token("S") == "э́с"
    out = transliterate_latin("код X")
    assert not has_latin(out)
    assert "икс" in out or "и́кс" in out


def test_fallback_never_leaves_latin():
    # Mixed-case / long token → transliteration, not acronym spelling.
    tok = spoken_latin_token("Bluetooth")
    assert tok == transliterate_word("Bluetooth")
    assert not has_latin(tok)
    assert tok  # non-empty
    weird = transliterate_latin("foo BAR Baz Qqqqq")
    assert not has_latin(weird)
    assert _UNK not in weird
    # Absolute fallback on stray letters
    assert not has_latin(transliterate_latin("a"))

"""Text normalization for Russian TTS frontend.

``normalize_text`` — light TTS cleanup (NFC, abbreviations, whitespace).
``normalize_for_asr`` — shared ASR ref/hyp normalization (separate from TN).
Full spoken-form TN (numbers, dates, money) lives in ``verbalize.py``.
"""

from __future__ import annotations

import re
import unicodedata

_WS = re.compile(r"\s+")
# Longer keys first so "т.д." wins over "д." etc.
_ABBR = {
    "т.е.": "то есть",
    "т.д.": "так далее",
    "т.п.": "тому подобное",
    "и т.д.": "и так далее",
    "и т.п.": "и тому подобное",
    "т.к.": "так как",
    "т.н.": "так называемый",
    "гг.": "годы",
    "стр.": "страница",
    "см.": "смотри",
    "др.": "другие",
    "руб.": "рублей",
    "коп.": "копеек",
    "тыс.": "тысяч",
    "млн.": "миллионов",
    "млн": "миллионов",
    "млрд.": "миллиардов",
    "млрд": "миллиардов",
    "ул.": "улица",
    "пр.": "проспект",
    "обл.": "область",
    "г.": "год",  # ambiguous with "город"; verbalize year patterns first
    # Bank / entity letterisms (spoken letter-by-letter)
    "ИНН": "и эн эн",
    "КПП": "ка пэ пэ",
    "БИК": "бик",
    "ОГРН": "о гэ эр эн",
}


def expand_abbreviations(text: str) -> str:
    """Expand common RU abbreviations (case-insensitive)."""
    if not text:
        return ""
    t = text
    # Sort by length descending so multi-char abbreviations match first.
    for a, b in sorted(_ABBR.items(), key=lambda kv: len(kv[0]), reverse=True):
        t = re.sub(re.escape(a), b, t, flags=re.IGNORECASE)
    return t


def normalize_text(text: str) -> str:
    """Lightweight RU TTS normalization (no digit verbalization).

    Preserves ё. Does not strip stress marks. Numbers/dates are left for
    ``verbalize_text`` / ``normalize_for_tts``.
    """
    if not text:
        return ""
    t = unicodedata.normalize("NFC", text.strip())
    t = expand_abbreviations(t)
    t = _WS.sub(" ", t)
    return t.strip()


def strip_stress(text: str, stress_char: str = "\u0301") -> str:
    return text.replace(stress_char, "").replace("+", "")


def has_stress_marks(text: str, stress_char: str = "\u0301") -> bool:
    return stress_char in text or bool(re.search(r"\+[аеёиоуыэюяАЕЁИОУЫЭЮЯ]", text))


_PUNCT = re.compile(r"[^\w\s]", flags=re.UNICODE)


def normalize_for_asr(text: str) -> str:
    """Shared ASR ref/hyp normalization (NFC, lower, no stress/punct, ё→е)."""
    if not text:
        return ""
    t = unicodedata.normalize("NFC", text.strip())
    t = strip_stress(t)
    t = t.lower().replace("ё", "е")
    t = _PUNCT.sub(" ", t)
    t = _WS.sub(" ", t)
    return t.strip()

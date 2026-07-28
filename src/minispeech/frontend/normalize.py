"""Text normalization for Russian TTS frontend."""

from __future__ import annotations

import re
import unicodedata

_WS = re.compile(r"\s+")
_ABBR = {
    "г.": "год",
    "гг.": "годы",
    "т.е.": "то есть",
    "т.д.": "так далее",
    "т.п.": "тому подобное",
    "др.": "другие",
    "руб.": "рублей",
    "тыс.": "тысяч",
    "млн": "миллионов",
    "млрд": "миллиардов",
}


def normalize_text(text: str) -> str:
    """Lightweight RU normalization. Numbers/dates kept for challenge-set coverage."""
    if not text:
        return ""
    t = unicodedata.normalize("NFC", text.strip())
    t = t.replace("Ё", "Е").replace("ё", "е")  # optional; stress path may restore
    # Restore ё if present as combining — keep original ё from input when present
    t = unicodedata.normalize("NFC", text.strip())
    for a, b in _ABBR.items():
        t = re.sub(re.escape(a), b, t, flags=re.IGNORECASE)
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

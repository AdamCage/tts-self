"""Unified stress-mark format (train == infer).

Canonical form: Unicode combining acute U+0301 after the stressed vowel
(as in RUSLAN Stressed / stilletto/ruslan-stressed).
"""

from __future__ import annotations

import re
import unicodedata
from functools import lru_cache

from minispeech.constants import STRESS_CHAR

_PLUS_STRESS = re.compile(r"\+([аеёиоуыэюяАЕЁИОУЫЭЮЯ])")
_VOWELS = set("аеёиоуыэюяАЕЁИОУЫЭЮЯ")


def to_canonical_stress(text: str) -> str:
    """Convert +Vowel stress notation to combining acute; NFC-normalize."""
    if not text:
        return ""
    t = _PLUS_STRESS.sub(lambda m: m.group(1) + STRESS_CHAR, text)
    return unicodedata.normalize("NFC", t)


def strip_stress(text: str) -> str:
    t = text.replace(STRESS_CHAR, "")
    t = _PLUS_STRESS.sub(r"\1", t)
    return unicodedata.normalize("NFC", t)


def stress_agreement(a: str, b: str) -> float:
    """Fraction of vowel positions where stress presence matches (after strip/align by unstressed)."""
    ua, ub = strip_stress(a), strip_stress(b)
    if ua != ub:
        # Compare on shared unstressed skeleton length of a
        pass
    ca = to_canonical_stress(a)
    cb = to_canonical_stress(b)
    # Build stress masks on vowels of stripped text
    sa, sb = strip_stress(ca), strip_stress(cb)
    if not sa or sa != sb:
        return 0.0 if sa != sb else 1.0

    def mask(stressed: str, plain: str) -> list[bool]:
        out: list[bool] = []
        i = 0
        for ch in plain:
            if ch not in _VOWELS:
                continue
            # look ahead in stressed for vowel + optional stress
            while i < len(stressed) and stressed[i] not in _VOWELS and stressed[i] != STRESS_CHAR:
                i += 1
            if i < len(stressed) and stressed[i] in _VOWELS:
                i += 1
                marked = i < len(stressed) and stressed[i] == STRESS_CHAR
                if marked:
                    i += 1
                out.append(marked)
        return out

    ma, mb = mask(ca, sa), mask(cb, sb)
    n = min(len(ma), len(mb))
    if n == 0:
        return 1.0
    return sum(x == y for x, y in zip(ma[:n], mb[:n])) / n


@lru_cache(maxsize=1)
def _get_accentuator():
    try:
        from ruaccent import RUAccent

        acc = RUAccent()
        acc.load(omograph_model_size="turbo", use_dictionary=True)
        return acc
    except Exception:
        return None


def accentuate(text: str, *, use_model: bool = True) -> str:
    """Add stress marks for inference. Falls back to passthrough if ruaccent missing."""
    plain = strip_stress(text)
    if not use_model:
        return plain
    acc = _get_accentuator()
    if acc is None:
        return plain
    try:
        out = acc.process_all(plain)
        return to_canonical_stress(out)
    except Exception:
        return plain

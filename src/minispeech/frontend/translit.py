"""Latin → Cyrillic spoken forms for Russian TTS (defect D2).

Three mechanisms, applied per Latin token in order:

1. Loanword / idiomatic acronym dictionary (``Google`` → ``гугл``).
2. Heuristic: all-caps length ≤ 5 → letter-by-letter with Russian letter names
   (``TTS`` → ``тэ-тэ-э́с``).
3. Grapheme transliteration fallback so Latin never reaches ``<unk>``.

Call after TN / verbalization and before stress marking.
"""

from __future__ import annotations

import re
import unicodedata

STRESS = "\u0301"
_VOWELS = "аеёиоуыэюяАЕЁИОУЫЭЮЯ"

# Established loanwords and idiomatic acronym readings (lowercase keys).
_LOANWORDS: dict[str, str] = {
    "google": "гугл",
    "apple": "эпл",
    "play": "плей",
    "youtube": "ютуб",
    "facebook": "фейсбук",
    "instagram": "инстаграм",
    "twitter": "твиттер",
    "telegram": "телеграм",
    "whatsapp": "ватсап",
    "paypal": "пэйпал",
    "visa": "виза",
    "excel": "эксель",
    "microsoft": "майкрософт",
    "windows": "виндоус",
    "android": "андроид",
    "iphone": "айфон",
    "ipad": "айпад",
    "macos": "макос",
    "ios": "айос",
    "wifi": "вайфай",
    "wi-fi": "вайфай",
    "email": "имейл",
    "online": "онлайн",
    "offline": "офлайн",
    "app": "эп",
    "pay": "пэй",
    "store": "стор",
    "ok": "окей",
    # Idiomatic acronyms (dictionary beats letter heuristic).
    "tts": "тэ-тэ-э́с",
    "hr": "эйч-а́р",
    "sms": "эс-эм-э́с",
    "it": "ай-ти́",
    "qr": "ку-э́р",
    "pdf": "пи-ди-э́ф",
    "url": "ю-ар-э́л",
    "api": "эй-пи-а́й",
    "pin": "пин",
    "iban": "ай-бан",
    "swift": "свифт",
    "faq": "эф-эй-кью́",
    "otp": "о-тэ-пэ́",
    "pos": "пос",
    "cpu": "си-пи-ю́",
    "gpu": "джи-пи-ю́",
    "usb": "ю-эс-би́",
    "vip": "ви-ай-пи́",
    "ceo": "си-и-о́у",
    "ai": "эй-а́й",
    "ml": "эм-э́л",
}

# Russian spoken names of Latin letters (base form; stress added on last in acronym).
_LETTER_NAMES: dict[str, str] = {
    "A": "эй",
    "B": "би",
    "C": "си",
    "D": "ди",
    "E": "и",
    "F": "эф",
    "G": "джи",
    "H": "эйч",
    "I": "ай",
    "J": "джей",
    "K": "кей",
    "L": "эл",
    "M": "эм",
    "N": "эн",
    "O": "оу",
    "P": "пи",
    "Q": "ку",
    "R": "ар",
    "S": "эс",
    "T": "тэ",
    "U": "ю",
    "V": "ви",
    "W": "дабл-ю",
    "X": "икс",
    "Y": "уай",
    "Z": "зет",
}

# Multi-char digraphs first (lowercase), then single letters.
_DIGRAPHS: list[tuple[str, str]] = [
    ("sch", "ш"),
    ("sh", "ш"),
    ("ch", "ч"),
    ("zh", "ж"),
    ("kh", "х"),
    ("th", "т"),
    ("ph", "ф"),
    ("qu", "кв"),
    ("ck", "к"),
    ("gh", "г"),
    ("ng", "нг"),
    ("ts", "ц"),
    ("ya", "я"),
    ("ye", "е"),
    ("yo", "ё"),
    ("yu", "ю"),
    ("ay", "ей"),
    ("ai", "ай"),
    ("ey", "ей"),
    ("ei", "ей"),
    ("oy", "ой"),
    ("oi", "ой"),
    ("au", "ау"),
    ("aw", "о"),
    ("ow", "оу"),
    ("oo", "у"),
    ("ee", "и"),
    ("ea", "и"),
    ("ie", "и"),
    ("ou", "ау"),
    ("igh", "ай"),
]

_SINGLE: dict[str, str] = {
    "a": "а",
    "b": "б",
    "c": "к",
    "d": "д",
    "e": "е",
    "f": "ф",
    "g": "г",
    "h": "х",
    "i": "и",
    "j": "дж",
    "k": "к",
    "l": "л",
    "m": "м",
    "n": "н",
    "o": "о",
    "p": "п",
    "q": "к",
    "r": "р",
    "s": "с",
    "t": "т",
    "u": "у",
    "v": "в",
    "w": "в",
    "x": "кс",
    "y": "й",
    "z": "з",
}

# Latin letter runs, optionally with internal hyphen/apostrophe (wi-fi, O'Neill).
_LATIN_TOKEN = re.compile(r"[A-Za-z]+(?:['\-][A-Za-z]+)*")
_HAS_LATIN = re.compile(r"[A-Za-z]")
_WS = re.compile(r"\s+")


def _stress_first_vowel(token: str) -> str:
    """Put combining acute on the first vowel if none present."""
    if STRESS in token:
        return token
    for i, ch in enumerate(token):
        if ch in _VOWELS:
            return token[: i + 1] + STRESS + token[i + 1 :]
    return token


def spell_latin_letters(token: str) -> str:
    """Read Latin acronym letter-by-letter with Russian letter names."""
    parts: list[str] = []
    for ch in token:
        if not ch.isalpha():
            continue
        name = _LETTER_NAMES.get(ch.upper())
        if name is None:
            name = transliterate_word(ch)
        parts.append(name)
    if not parts:
        return ""
    parts[-1] = _stress_first_vowel(parts[-1])
    return "-".join(parts)


def transliterate_word(token: str) -> str:
    """Grapheme fallback: approximate Russian reading of a Latin word."""
    s = token.lower()
    out: list[str] = []
    i = 0
    while i < len(s):
        if not s[i].isalpha():
            i += 1
            continue
        matched = False
        for dig, repl in _DIGRAPHS:
            if s.startswith(dig, i):
                out.append(repl)
                i += len(dig)
                matched = True
                break
        if matched:
            continue
        out.append(_SINGLE.get(s[i], "е"))
        i += 1
    return "".join(out) or "е"


def spoken_latin_token(token: str) -> str:
    """Dictionary → acronym heuristic → transliteration for one Latin token."""
    key = token.lower()
    if key in _LOANWORDS:
        return _LOANWORDS[key]

    letters_only = "".join(ch for ch in token if ch.isalpha())
    if (
        letters_only
        and letters_only.isupper()
        and len(letters_only) <= 5
        and letters_only.isalpha()
    ):
        return spell_latin_letters(letters_only)

    return transliterate_word(token)


def transliterate_latin(text: str) -> str:
    """Replace every Latin token with a Cyrillic spoken form.

    Guarantees no ``[A-Za-z]`` remain in the output (NFC-normalized).
    """
    if not text:
        return ""
    t = unicodedata.normalize("NFC", text)

    def _repl(m: re.Match[str]) -> str:
        return spoken_latin_token(m.group(0))

    t = _LATIN_TOKEN.sub(_repl, t)
    # Absolute fallback: any stray Latin codepoint → letter name / translit.
    if _HAS_LATIN.search(t):
        chars: list[str] = []
        for ch in t:
            if "A" <= ch <= "Z" or "a" <= ch <= "z":
                chars.append(spoken_latin_token(ch))
            else:
                chars.append(ch)
        t = "".join(chars)
    t = _WS.sub(" ", t).strip()
    return unicodedata.normalize("NFC", t)


def has_latin(text: str) -> bool:
    """True if any basic Latin letter remains."""
    return bool(text and _HAS_LATIN.search(text))

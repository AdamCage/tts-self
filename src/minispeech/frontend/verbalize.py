"""Russian text normalization (spoken form) for TTS / QC references.

Covers common numbers, ordinals, money, dates, percentages, and time.
ASR-side cleanup stays in ``normalize.normalize_for_asr`` — do not mix.
Banking-grade TN (cards, accounts) is a later Track D extension.
"""

from __future__ import annotations

import re

from minispeech.frontend.normalize import normalize_for_asr, normalize_text

_DIGITS = {
    "0": "ноль",
    "1": "один",
    "2": "два",
    "3": "три",
    "4": "четыре",
    "5": "пять",
    "6": "шесть",
    "7": "семь",
    "8": "восемь",
    "9": "девять",
}

_TEEN = {
    10: "десять",
    11: "одиннадцать",
    12: "двенадцать",
    13: "тринадцать",
    14: "четырнадцать",
    15: "пятнадцать",
    16: "шестнадцать",
    17: "семнадцать",
    18: "восемнадцать",
    19: "девятнадцать",
}

_TENS = {
    2: "двадцать",
    3: "тридцать",
    4: "сорок",
    5: "пятьдесят",
    6: "шестьдесят",
    7: "семьдесят",
    8: "восемьдесят",
    9: "девяносто",
}

_HUNDREDS = {
    1: "сто",
    2: "двести",
    3: "триста",
    4: "четыреста",
    5: "пятьсот",
    6: "шестьсот",
    7: "семьсот",
    8: "восемьсот",
    9: "девятьсот",
}

# Feminine forms used in money / some ordinals
_DIGITS_F = {**_DIGITS, "1": "одна", "2": "две"}

_ORD_UNITS = {
    0: "",
    1: "первый",
    2: "второй",
    3: "третий",
    4: "четвёртый",
    5: "пятый",
    6: "шестой",
    7: "седьмой",
    8: "восьмой",
    9: "девятый",
}

_ORD_TEEN = {
    10: "десятый",
    11: "одиннадцатый",
    12: "двенадцатый",
    13: "тринадцатый",
    14: "четырнадцатый",
    15: "пятнадцатый",
    16: "шестнадцатый",
    17: "семнадцатый",
    18: "восемнадцатый",
    19: "девятнадцатый",
}

_ORD_TENS = {
    2: "двадцатый",
    3: "тридцатый",
    4: "сороковой",
    5: "пятидесятый",
    6: "шестидесятый",
    7: "семидесятый",
    8: "восьмидесятый",
    9: "девяностый",
}

_MONTHS = {
    1: "января",
    2: "февраля",
    3: "марта",
    4: "апреля",
    5: "мая",
    6: "июня",
    7: "июля",
    8: "августа",
    9: "сентября",
    10: "октября",
    11: "ноября",
    12: "декабря",
}

_DATE_DMY = re.compile(r"\b(\d{1,2})[./](\d{1,2})[./](\d{4})\b")
_DATE_MONTH_WORD = re.compile(
    r"\b(\d{1,2})\s+"
    r"(января|февраля|марта|апреля|мая|июня|июля|августа|сентября|октября|ноября|декабря)"
    r"\s+(\d{4})\b",
    flags=re.IGNORECASE,
)
_YEAR_RE = re.compile(r"\b(\d{4})\s*(?:г\.?|год[ауе]?)\b", flags=re.IGNORECASE)
# Allow thin/regular spaces as thousand separators: "15 070 руб."
_MONEY_RE = re.compile(
    r"\b(\d{1,3}(?:[ \u00a0]\d{3})*(?:[.,]\d{1,2})?|\d+[.,]\d{1,2})\s*"
    r"(?:₽|руб\.?|рублей|рубля|рубль)\b",
    flags=re.IGNORECASE,
)
_MONEY_KOP_RE = re.compile(
    r"\b(\d{1,2})\s*(?:коп\.?|копеек|копейки|копейка)\b",
    flags=re.IGNORECASE,
)
_PCT_RE = re.compile(r"\b(\d{1,3}(?:[.,]\d+)?)\s*%")
_TIME_RE = re.compile(r"\b(\d{1,2}):(\d{2})\b")
# Whisper often writes clock times as "16.30" (dot); prefer time over decimal.
_TIME_DOT_RE = re.compile(r"\b(\d{1,2})\.(\d{2})\b")
_ORD_RE = re.compile(r"\b(\d{1,4})-(?:й|ый|ой|ая|ое|ые|е|го|му)\b", flags=re.IGNORECASE)
_DECIMAL_RE = re.compile(r"\b(\d+)[.,](\d+)\b")
_GROUPED_NUM_RE = re.compile(r"\b(\d{1,3}(?:[ \u00a0]\d{3})+)\b")
_ACCOUNT_DIGITS_RE = re.compile(
    r"\b((?:номер\s+)?сч[её]та)\s+(\d{2,20})\b",
    flags=re.IGNORECASE,
)
_NUM_RE = re.compile(r"\b(\d{1,9})\b")
_MONTH_NAME_TO_NUM = {v: k for k, v in _MONTHS.items()}


def _under_100(n: int, *, feminine: bool = False) -> str:
    dig = _DIGITS_F if feminine else _DIGITS
    if n < 10:
        return dig[str(n)]
    if n < 20:
        return _TEEN[n]
    tens, ones = divmod(n, 10)
    parts = [_TENS[tens]]
    if ones:
        parts.append(dig[str(ones)])
    return " ".join(parts)


def verbalize_int(n: int, *, feminine: bool = False) -> str:
    """Verbalize a non-negative integer up to 999_999_999 (rough cardinal)."""
    if n < 0:
        return "минус " + verbalize_int(-n, feminine=feminine)
    if n < 100:
        return _under_100(n, feminine=feminine)
    if n < 1000:
        h, rest = divmod(n, 100)
        parts = [_HUNDREDS[h]]
        if rest:
            parts.append(_under_100(rest, feminine=feminine))
        return " ".join(parts)
    if n < 1_000_000:
        th, rest = divmod(n, 1000)
        if th == 1:
            head = "одна тысяча"
        elif th == 2:
            head = "две тысячи"
        elif th in (3, 4):
            head = f"{_under_100(th)} тысячи"
        else:
            head = f"{verbalize_int(th)} тысяч"
        if rest:
            return f"{head} {verbalize_int(rest, feminine=feminine)}"
        return head
    if n < 1_000_000_000:
        m, rest = divmod(n, 1_000_000)
        if m == 1:
            head = "один миллион"
        elif m in (2, 3, 4):
            head = f"{verbalize_int(m)} миллиона"
        else:
            head = f"{verbalize_int(m)} миллионов"
        if rest:
            return f"{head} {verbalize_int(rest, feminine=feminine)}"
        return head
    return " ".join(_DIGITS.get(ch, ch) for ch in str(n))


def verbalize_ordinal(n: int) -> str:
    """Masculine ordinal (1-й → первый). Good enough for dates/list items."""
    if n <= 0:
        return verbalize_int(n)
    if n < 10:
        return _ORD_UNITS[n]
    if n < 20:
        return _ORD_TEEN[n]
    if n < 100:
        tens, ones = divmod(n, 10)
        if ones == 0:
            return _ORD_TENS[tens]
        return f"{_TENS[tens]} {_ORD_UNITS[ones]}"
    # Fall back: cardinal + "-й" spoken as cardinal for large ordinals
    return verbalize_int(n)


def verbalize_decimal(whole: int, frac: str) -> str:
    frac_words = " ".join(_DIGITS[ch] for ch in frac if ch.isdigit())
    return f"{verbalize_int(whole)} точка {frac_words}"


def _ruble_form(n: int) -> str:
    n_abs = abs(n) % 100
    if 11 <= n_abs <= 14:
        return "рублей"
    last = n_abs % 10
    if last == 1:
        return "рубль"
    if last in (2, 3, 4):
        return "рубля"
    return "рублей"


def _parse_amount_number(amount: str) -> str:
    """Strip thousand separators; keep decimal comma/dot."""
    s = amount.replace("\u00a0", " ").strip()
    if "," in s and "." not in s:
        # "1 234,56" or "1234,56"
        left, right = s.rsplit(",", 1)
        left = left.replace(" ", "")
        return f"{left}.{right}"
    s = s.replace(" ", "")
    return s.replace(",", ".")


def verbalize_digits(n_str: str) -> str:
    """Spell a digit string one-by-one (account / code style)."""
    return " ".join(_DIGITS[ch] for ch in n_str if ch.isdigit())


def verbalize_money(amount: str) -> str:
    amount = _parse_amount_number(amount)
    if "." in amount:
        whole_s, frac_s = amount.split(".", 1)
        whole = int(whole_s)
        kop = int((frac_s + "00")[:2])
        parts = [verbalize_int(whole, feminine=False), _ruble_form(whole)]
        if kop:
            parts.append(verbalize_int(kop, feminine=True))
            parts.append("копеек" if not (kop % 10 == 1 and kop % 100 != 11) else "копейка")
        return " ".join(parts)
    whole = int(amount)
    return f"{verbalize_int(whole)} {_ruble_form(whole)}"


def verbalize_date(d: int, m: int, y: int) -> str:
    month = _MONTHS.get(m, verbalize_int(m))
    return f"{verbalize_ordinal(d)} {month} {verbalize_int(y)} года"


def verbalize_time(h: int, mi: int) -> str:
    return f"{verbalize_int(h)} {verbalize_int(mi, feminine=True)}"


def verbalize_text(text: str) -> str:
    """Apply abbreviation expansion + spoken-form number/date/money TN."""
    t = normalize_text(text)

    t = _DATE_DMY.sub(
        lambda m: verbalize_date(int(m.group(1)), int(m.group(2)), int(m.group(3))), t
    )

    def _date_month_word(m: re.Match[str]) -> str:
        month_num = _MONTH_NAME_TO_NUM.get(m.group(2).lower())
        if month_num is None:
            return m.group(0)
        return verbalize_date(int(m.group(1)), month_num, int(m.group(3)))

    t = _DATE_MONTH_WORD.sub(_date_month_word, t)
    t = _YEAR_RE.sub(lambda m: f"{verbalize_int(int(m.group(1)))} год", t)
    t = _ACCOUNT_DIGITS_RE.sub(
        lambda m: f"{m.group(1)} {verbalize_digits(m.group(2))}", t
    )
    t = _MONEY_RE.sub(lambda m: verbalize_money(m.group(1)), t)
    t = _MONEY_KOP_RE.sub(
        lambda m: (
            f"{verbalize_int(int(m.group(1)), feminine=True)} "
            f"{'копейка' if int(m.group(1)) % 10 == 1 and int(m.group(1)) % 100 != 11 else 'копеек'}"
        ),
        t,
    )
    t = _PCT_RE.sub(lambda m: f"{_pct(m.group(1))} процентов", t)
    t = _TIME_RE.sub(lambda m: verbalize_time(int(m.group(1)), int(m.group(2))), t)

    def _time_dot(m: re.Match[str]) -> str:
        h, mi = int(m.group(1)), int(m.group(2))
        if 0 <= h <= 23 and 0 <= mi <= 59:
            return verbalize_time(h, mi)
        return m.group(0)

    t = _TIME_DOT_RE.sub(_time_dot, t)
    t = _ORD_RE.sub(lambda m: verbalize_ordinal(int(m.group(1))), t)
    t = _DECIMAL_RE.sub(lambda m: verbalize_decimal(int(m.group(1)), m.group(2)), t)
    t = _GROUPED_NUM_RE.sub(lambda m: verbalize_int(int(m.group(1).replace(" ", "").replace("\u00a0", ""))), t)
    t = _NUM_RE.sub(lambda m: verbalize_int(int(m.group(1))), t)
    return t


def _pct(raw: str) -> str:
    raw = raw.replace(",", ".")
    if "." in raw:
        w, f = raw.split(".", 1)
        return verbalize_decimal(int(w), f)
    return verbalize_int(int(raw))


def normalize_for_tts(text: str) -> str:
    """Full TN for synthesis / Qwen input prep (spoken form, stress applied later)."""
    return verbalize_text(text)


def normalize_verbalized_for_asr(text: str) -> str:
    """Verbalize digits/money/dates then apply shared ASR normalization.

    Use for CER/WER when Whisper (or similar) emits Arabic digits while the
    TTS reference / prompt is in spoken-word form (bank amounts, dates, etc.).
    Idempotent on already-verbalized Russian.
    """
    return normalize_for_asr(verbalize_text(text))

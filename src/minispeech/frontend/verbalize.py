"""Russian text normalization (spoken form) for TTS / QC references.

Covers common numbers, ordinals, money, dates, percentages, and time.

Date / time / decimal conventions (R3):
- Dates: neuter nominative day by default (``первое февраля``); genitive day
  after ``с``/``до``/``от``/``по``; year as genitive ordinal (``… двадцатого года``).
- Time: digit-style clock (no ``час(ов)/минут``); leading zero minutes keep
  ``ноль``; ``HH:00`` → ``ноль ноль``; genitive after the same range prepositions.
- Decimals: ``целых/десятых/сотых`` with agreement. Comma or bare numeric dot
  uses that reading; ``точка`` only after version-like context
  (``версия``/``python``/``build``/…). Sentence periods are not matched.

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
_GEN_UNITS = {
    0: "нуля",
    1: "одного",
    2: "двух",
    3: "трёх",
    4: "четырёх",
    5: "пяти",
    6: "шести",
    7: "семи",
    8: "восьми",
    9: "девяти",
}
_GEN_TEEN = {
    10: "десяти",
    11: "одиннадцати",
    12: "двенадцати",
    13: "тринадцати",
    14: "четырнадцати",
    15: "пятнадцати",
    16: "шестнадцати",
    17: "семнадцати",
    18: "восемнадцати",
    19: "девятнадцати",
}
_GEN_TENS = {
    2: "двадцати",
    3: "тридцати",
    4: "сорока",
    5: "пятидесяти",
    6: "шестидесяти",
    7: "семидесяти",
    8: "восьмидесяти",
    9: "девяноста",
}
_GEN_HUNDREDS = {
    1: "ста",
    2: "двухсот",
    3: "трёхсот",
    4: "четырёхсот",
    5: "пятисот",
    6: "шестисот",
    7: "семисот",
    8: "восьмисот",
    9: "девятисот",
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
_ORD_HUNDREDS = {
    1: "сотый",
    2: "двухсотый",
    3: "трёхсотый",
    4: "четырёхсотый",
    5: "пятисотый",
    6: "шестисотый",
    7: "семисотый",
    8: "восьмисотый",
    9: "девятисотый",
}
_ORD_GEN_UNITS = {
    1: "первого",
    2: "второго",
    3: "третьего",
    4: "четвёртого",
    5: "пятого",
    6: "шестого",
    7: "седьмого",
    8: "восьмого",
    9: "девятого",
}
_ORD_GEN_TEEN = {n: f"{word[:-2]}ого" for n, word in _ORD_TEEN.items()}
_ORD_GEN_TEEN[13] = "тринадцатого"
_ORD_GEN_TENS = {n: f"{word[:-2]}ого" for n, word in _ORD_TENS.items()}
_ORD_GEN_HUNDREDS = {n: f"{word[:-2]}ого" for n, word in _ORD_HUNDREDS.items()}

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
    r"\b(\d{1,3}(?:[ \u00a0]\d{3})+(?:[.,]\d{1,2})?|\d+(?:[.,]\d{1,2})?)\s*"
    r"(?:₽|руб\.?|рублей|рубля|рубль)\b",
    flags=re.IGNORECASE,
)
_MONEY_KOP_RE = re.compile(
    r"\b(\d{1,2})\s*(?:коп\.?|копеек|копейки|копейка)\b",
    flags=re.IGNORECASE,
)
_PCT_RE = re.compile(r"\b(\d{1,3}(?:[.,]\d+)?)\s*%")
_TIME_RANGE_RE = re.compile(
    r"\b(с|от)\s+(\d{1,2}):(\d{2})\s+до\s+(\d{1,2}):(\d{2})\b",
    flags=re.IGNORECASE,
)
_TIME_RANGE_DOT_RE = re.compile(
    r"\b(с|от)\s+(\d{1,2})\.(\d{2})\s+до\s+(\d{1,2})\.(\d{2})\b",
    flags=re.IGNORECASE,
)
# Preposition + clock; genitive after с/до/от/по, nominative after в/к/после.
_TIME_PREP_RE = re.compile(
    r"\b(в|к|до|после|с|от|по)\s+(\d{1,2}):(\d{2})\b",
    flags=re.IGNORECASE,
)
_TIME_RE = re.compile(r"\b(\d{1,2}):(\d{2})\b")
# Dot clock times are ambiguous with decimals; require a time preposition.
_TIME_DOT_RE = re.compile(
    r"\b(в|к|до|после|с|от|по)\s+(\d{1,2})\.(\d{2})\b",
    flags=re.IGNORECASE,
)
_ORD_RE = re.compile(
    r"\b(\d{1,4})-(й|ый|ой|ая|ое|ые|е|го|му)\b",
    flags=re.IGNORECASE,
)
# Decimal comma/dot between digits. Sentence periods stay outside ``\b...\b``.
# Soft/version "точка" reading is selected by preceding context (see ``_VERSION_CTX_RE``).
_DECIMAL_RE = re.compile(r"\b(\d+)([.,])(\d+)\b")
_VERSION_CTX_RE = re.compile(
    r"(?i)\b(верси[яиейю]|version|python|windows|android|ios|linux|build|v)\s*$"
)
_GROUPED_NUM_RE = re.compile(r"\b(\d{1,3}(?:[ \u00a0]\d{3})+)\b")
_ACCOUNT_DIGITS_RE = re.compile(
    r"\b((?:номер\s+)?сч[её]та)\s+(\d{2,20})\b",
    flags=re.IGNORECASE,
)
_PHONE_RE = re.compile(
    r"(?<!\d)(\+7|8)[\s-]*\(?(\d{3})\)?[\s-]*(\d{3})[\s-]*(\d{2})[\s-]*(\d{2})(?!\d)"
)
_PHONE4_RE = re.compile(
    r"(?<!\d)(\+7|8)[\s-]*\(?(\d{4})\)?[\s-]*(\d{2})[\s-]*(\d{2})[\s-]*(\d{2})(?!\d)"
)
_EXPLICIT_RANGE_RE = re.compile(
    r"\b(с|от)\s+(\d{1,4})\s+до\s+(\d{1,4})\b",
    flags=re.IGNORECASE,
)
_BARE_RANGE_RE = re.compile(r"(?<![\d-])(\d{1,4})\s*[–—-]\s*(\d{1,4})(?![\d-])")
_LONG_CODE_RE = re.compile(r"(?<!\d)(\d(?:[ -]?\d){8,19})(?!\d)")
_UNIT_RE = re.compile(
    r"\b(\d{1,9})\s*"
    r"(км/ч\.?|квт[·⋅*]?ч\.?|м[²2]\.?|кг\.?|км\.?|см\.?|мм\.?|"
    r"гб\.?|мб\.?|тб\.?|гц\.?|мин\.?|мес\.?|сут\.?|дн\.?|ч\.?|"
    r"г\.?|л\.?|м\.?|в\.?|°\s*си(?:\u0301)?\.?)"
    r"(?=$|[\s,;:!?)}\]])",
    flags=re.IGNORECASE,
)
_NUM_RE = re.compile(r"\b(\d{1,9})\b")
_MONTH_NAME_TO_NUM = {v: k for k, v in _MONTHS.items()}
_DATE_GENITIVE_PREPS = frozenset({"с", "до", "от", "по"})
_TIME_GENITIVE_PREPS = frozenset({"с", "до", "от", "по"})
_FRAC_SCALE = {
    1: ("десятая", "десятых", "десятых"),
    2: ("сотая", "сотых", "сотых"),
    3: ("тысячная", "тысячных", "тысячных"),
    4: ("десятитысячная", "десятитысячных", "десятитысячных"),
}


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
    """Masculine nominative ordinal (``1-й`` → ``первый``)."""
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
    if n < 1000:
        hundreds, rest = divmod(n, 100)
        if not rest:
            return _ORD_HUNDREDS[hundreds]
        return f"{_HUNDREDS[hundreds]} {verbalize_ordinal(rest)}"
    if n < 1_000_000:
        rest = n % 1000
        if rest:
            return f"{verbalize_int(n - rest)} {verbalize_ordinal(rest)}"
    return verbalize_int(n)


def verbalize_ordinal_genitive(n: int) -> str:
    """Masculine genitive ordinal (``1-го`` → ``первого``)."""
    if n <= 0:
        return verbalize_int(n)
    if n < 10:
        return _ORD_GEN_UNITS[n]
    if n < 20:
        return _ORD_GEN_TEEN[n]
    if n < 100:
        tens, ones = divmod(n, 10)
        if not ones:
            return _ORD_GEN_TENS[tens]
        return f"{_TENS[tens]} {_ORD_GEN_UNITS[ones]}"
    if n < 1000:
        hundreds, rest = divmod(n, 100)
        if not rest:
            return _ORD_GEN_HUNDREDS[hundreds]
        return f"{_HUNDREDS[hundreds]} {verbalize_ordinal_genitive(rest)}"
    if n < 1_000_000:
        rest = n % 1000
        if rest:
            return f"{verbalize_int(n - rest)} {verbalize_ordinal_genitive(rest)}"
    return verbalize_int(n)


def _to_neuter_nominative_ordinal(masc: str) -> str:
    """Masculine nominative ordinal word → neuter (``первый`` → ``первое``)."""
    if masc.endswith("ий"):
        return masc[:-2] + "ье"
    if masc.endswith(("ый", "ой")):
        return masc[:-2] + "ое"
    return masc


def verbalize_ordinal_neuter(n: int) -> str:
    """Neuter nominative ordinal for calendar days (``1`` → ``первое``)."""
    words = verbalize_ordinal(n).split()
    if not words:
        return verbalize_ordinal(n)
    words[-1] = _to_neuter_nominative_ordinal(words[-1])
    return " ".join(words)


def verbalize_year_genitive(y: int) -> str:
    """Year as genitive ordinal (``2020`` → ``две тысячи двадцатого``)."""
    return verbalize_ordinal_genitive(y)


def _case_before(prefix: str, preps: frozenset[str]) -> str:
    """Return ``genitive`` if ``prefix`` ends with a governing preposition."""
    m = re.search(r"([А-Яа-яЁёA-Za-z]+)\s*$", prefix)
    if m and m.group(1).lower() in preps:
        return "genitive"
    return "nominative"


def _plural_adj_forms(n: int, forms: tuple[str, str, str]) -> str:
    """1 / 2–4 / 5+ adjective agreement (целая/целых, десятая/десятых)."""
    return _plural_form(n, forms)


def _verbalize_decimal_whole(n: int) -> str:
    """Whole part before «целая/целых»: 1→одна, else masculine cardinal."""
    if n == 1:
        return "одна"
    return verbalize_int(n, feminine=False)


def verbalize_decimal(whole: int, frac: str) -> str:
    """Russian decimal fraction: ``0,25`` → ``ноль целых двадцать пять сотых``.

    Uses written fractional length for the scale (1→десятых, 2→сотых, …).
    Trailing zeros in the fraction are stripped (``2,50`` → десятых).
    Software/version contexts keep digit-style ``точка`` via ``verbalize_text``.
    """
    frac_digits = "".join(ch for ch in frac if ch.isdigit())
    if not frac_digits or set(frac_digits) <= {"0"}:
        return _verbalize_decimal_whole(whole)

    frac_digits = frac_digits.rstrip("0") or frac_digits
    places = len(frac_digits)
    if places not in _FRAC_SCALE:
        # Longer fractions: fall back to digit-by-digit after «точка».
        frac_words = " ".join(_DIGITS[ch] for ch in frac_digits)
        return f"{verbalize_int(whole)} точка {frac_words}"

    numer = int(frac_digits)
    whole_words = _verbalize_decimal_whole(whole)
    whole_unit = _plural_adj_forms(whole, ("целая", "целых", "целых"))
    frac_words = verbalize_int(numer, feminine=True)
    frac_unit = _plural_adj_forms(numer, _FRAC_SCALE[places])
    return f"{whole_words} {whole_unit} {frac_words} {frac_unit}"


def verbalize_decimal_point(whole: int, frac: str) -> str:
    """Software/version reading: ``2.5`` → ``два точка пять`` (digit frac)."""
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


def verbalize_genitive(n: int) -> str:
    """Genitive cardinal used after Russian range prepositions."""
    if n < 0:
        return "минус " + verbalize_genitive(-n)
    if n < 10:
        return _GEN_UNITS[n]
    if n < 20:
        return _GEN_TEEN[n]
    if n < 100:
        tens, ones = divmod(n, 10)
        return " ".join(
            part for part in (_GEN_TENS[tens], _GEN_UNITS[ones] if ones else "") if part
        )
    if n < 1000:
        hundreds, rest = divmod(n, 100)
        return " ".join(
            part
            for part in (_GEN_HUNDREDS[hundreds], verbalize_genitive(rest) if rest else "")
            if part
        )
    return verbalize_int(n)


def _plural_form(n: int, forms: tuple[str, str, str]) -> str:
    value = abs(n) % 100
    if 11 <= value <= 14:
        return forms[2]
    last = value % 10
    if last == 1:
        return forms[0]
    if 2 <= last <= 4:
        return forms[1]
    return forms[2]


def _number_with_unit(
    n: int,
    forms: tuple[str, str, str],
    *,
    feminine: bool = False,
) -> str:
    return f"{verbalize_int(n, feminine=feminine)} {_plural_form(n, forms)}"


def verbalize_phone(match: re.Match[str]) -> str:
    """Speak Russian phone prefixes and the remaining number in groups."""
    prefix = "плюс семь" if match.group(1) == "+7" else "восемь"
    groups = []
    for raw in match.groups()[1:]:
        groups.append(
            verbalize_digits(raw)
            if len(raw) > 3 or (len(raw) > 1 and raw.startswith("0"))
            else verbalize_int(int(raw))
        )
    return " ".join([prefix, *groups])


def verbalize_unit(n: int, raw_unit: str) -> str:
    unit = raw_unit.lower().replace(" ", "").rstrip(".")
    if unit == "км/ч":
        return f"{_number_with_unit(n, ('километр', 'километра', 'километров'))} в час"
    if unit in {"м²", "м2"}:
        noun = _plural_form(n, ("метр", "метра", "метров"))
        adjective = "квадратный" if abs(n) % 10 == 1 and abs(n) % 100 != 11 else "квадратных"
        return f"{verbalize_int(n)} {adjective} {noun}"
    if unit.startswith("°си"):
        return _number_with_unit(
            n,
            ("градус Цельсия", "градуса Цельсия", "градусов Цельсия"),
        )
    specs: dict[str, tuple[tuple[str, str, str], bool]] = {
        "кг": (("килограмм", "килограмма", "килограммов"), False),
        "км": (("километр", "километра", "километров"), False),
        "г": (("грамм", "грамма", "граммов"), False),
        "л": (("литр", "литра", "литров"), False),
        "м": (("метр", "метра", "метров"), False),
        "см": (("сантиметр", "сантиметра", "сантиметров"), False),
        "мм": (("миллиметр", "миллиметра", "миллиметров"), False),
        "в": (("вольт", "вольта", "вольт"), False),
        "гц": (("герц", "герца", "герц"), False),
        "гб": (("гигабайт", "гигабайта", "гигабайт"), False),
        "мб": (("мегабайт", "мегабайта", "мегабайт"), False),
        "тб": (("терабайт", "терабайта", "терабайт"), False),
        "квт·ч": (("киловатт-час", "киловатт-часа", "киловатт-часов"), False),
        "квт⋅ч": (("киловатт-час", "киловатт-часа", "киловатт-часов"), False),
        "квт*ч": (("киловатт-час", "киловатт-часа", "киловатт-часов"), False),
        "квтч": (("киловатт-час", "киловатт-часа", "киловатт-часов"), False),
        "мин": (("минута", "минуты", "минут"), True),
        "ч": (("час", "часа", "часов"), False),
        "дн": (("день", "дня", "дней"), False),
        "сут": (("сутки", "суток", "суток"), True),
        "мес": (("месяц", "месяца", "месяцев"), False),
    }
    forms, feminine = specs[unit]
    return _number_with_unit(n, forms, feminine=feminine)


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


def verbalize_date(d: int, m: int, y: int, *, case: str = "nominative") -> str:
    """Calendar date in spoken Russian.

    Default day is neuter nominative (``первое февраля``). After prepositions
    ``с``/``до``/``от``/``по`` the day is genitive (``тридцать первого``).
    The year is always a genitive ordinal (``две тысячи двадцатого года``).
    """
    month = _MONTHS.get(m, verbalize_int(m))
    day = (
        verbalize_ordinal_genitive(d)
        if case == "genitive"
        else verbalize_ordinal_neuter(d)
    )
    return f"{day} {month} {verbalize_year_genitive(y)} года"


def _verbalize_clock_minutes(mi: int, *, case: str = "nominative") -> str:
    """Minutes for digit-style clock reading (leading zero kept as ``ноль``)."""
    if mi == 0:
        return "ноль ноль"
    if mi < 10:
        # Digit-style uses masculine ones (``ноль один``, not ``ноль одна``).
        ones = verbalize_genitive(mi) if case == "genitive" else verbalize_int(mi)
        return f"ноль {ones}"
    if case == "genitive":
        return verbalize_genitive(mi)
    return verbalize_int(mi)


def verbalize_time(h: int, mi: int, *, case: str = "nominative") -> str:
    """Digit-style clock reading (canonical form for TTS).

    Examples: ``09:05`` → ``девять ноль пять``; ``14:00`` → ``четырнадцать ноль ноль``.
    Does **not** insert ``час(ов)`` / ``минут`` — those are a separate style.
    After range prepositions ``с``/``до``/``от``/``по``, both hour and minute
    parts use genitive (``с девяти ноль пяти``); the leading ``ноль`` stays
    undeclined.
    """
    hour = verbalize_genitive(h) if case == "genitive" else verbalize_int(h)
    return f"{hour} {_verbalize_clock_minutes(mi, case=case)}"


def verbalize_text(text: str) -> str:
    """Apply abbreviation expansion + spoken-form number/date/money TN."""
    t = normalize_text(text)
    t = _PHONE4_RE.sub(verbalize_phone, t)
    t = _PHONE_RE.sub(verbalize_phone, t)

    def _date_dmy(m: re.Match[str]) -> str:
        case = _case_before(t[: m.start()], _DATE_GENITIVE_PREPS)
        return verbalize_date(int(m.group(1)), int(m.group(2)), int(m.group(3)), case=case)

    t = _DATE_DMY.sub(_date_dmy, t)

    def _date_month_word(m: re.Match[str]) -> str:
        month_num = _MONTH_NAME_TO_NUM.get(m.group(2).lower())
        if month_num is None:
            return m.group(0)
        case = _case_before(t[: m.start()], _DATE_GENITIVE_PREPS)
        return verbalize_date(int(m.group(1)), month_num, int(m.group(3)), case=case)

    t = _DATE_MONTH_WORD.sub(_date_month_word, t)
    t = _YEAR_RE.sub(lambda m: f"{verbalize_int(int(m.group(1)))} год", t)
    t = _ACCOUNT_DIGITS_RE.sub(
        lambda m: f"{m.group(1)} {verbalize_digits(m.group(2))}", t
    )
    t = _MONEY_RE.sub(lambda m: verbalize_money(m.group(1)), t)
    t = _MONEY_KOP_RE.sub(
        lambda m: (
            f"{verbalize_int(int(m.group(1)), feminine=True)} "
            f"{_plural_form(int(m.group(1)), ('копейка', 'копейки', 'копеек'))}"
        ),
        t,
    )
    t = _PCT_RE.sub(lambda m: f"{_pct(m.group(1))} процентов", t)

    def _time_range(m: re.Match[str]) -> str:
        return (
            f"{m.group(1).lower()} "
            f"{verbalize_time(int(m.group(2)), int(m.group(3)), case='genitive')} "
            f"до {verbalize_time(int(m.group(4)), int(m.group(5)), case='genitive')}"
        )

    t = _TIME_RANGE_RE.sub(_time_range, t)
    t = _TIME_RANGE_DOT_RE.sub(_time_range, t)

    def _time_prep(m: re.Match[str]) -> str:
        prep = m.group(1).lower()
        h, mi = int(m.group(2)), int(m.group(3))
        if not (0 <= h <= 23 and 0 <= mi <= 59):
            return m.group(0)
        case = "genitive" if prep in _TIME_GENITIVE_PREPS else "nominative"
        return f"{prep} {verbalize_time(h, mi, case=case)}"

    t = _TIME_PREP_RE.sub(_time_prep, t)
    t = _TIME_RE.sub(lambda m: verbalize_time(int(m.group(1)), int(m.group(2))), t)
    t = _TIME_DOT_RE.sub(_time_prep, t)
    t = _EXPLICIT_RANGE_RE.sub(
        lambda m: (
            f"{m.group(1).lower()} {verbalize_genitive(int(m.group(2)))} "
            f"до {verbalize_genitive(int(m.group(3)))}"
        ),
        t,
    )
    t = _BARE_RANGE_RE.sub(
        lambda m: (
            f"от {verbalize_genitive(int(m.group(1)))} "
            f"до {verbalize_genitive(int(m.group(2)))}"
        ),
        t,
    )
    t = _UNIT_RE.sub(lambda m: verbalize_unit(int(m.group(1)), m.group(2)), t)
    t = _LONG_CODE_RE.sub(lambda m: verbalize_digits(m.group(1)), t)
    t = _ORD_RE.sub(
        lambda m: (
            verbalize_ordinal_genitive(int(m.group(1)))
            if m.group(2).lower() == "го"
            else verbalize_ordinal(int(m.group(1)))
        ),
        t,
    )

    def _decimal(m: re.Match[str]) -> str:
        whole, _sep, frac = int(m.group(1)), m.group(2), m.group(3)
        prefix = t[: m.start()]
        # Soft/version contexts keep «точка»; numeric decimals use целых/десятых.
        if _VERSION_CTX_RE.search(prefix):
            return verbalize_decimal_point(whole, frac)
        return verbalize_decimal(whole, frac)

    t = _DECIMAL_RE.sub(_decimal, t)
    t = _GROUPED_NUM_RE.sub(
        lambda m: verbalize_int(
            int(m.group(1).replace(" ", "").replace("\u00a0", ""))
        ),
        t,
    )
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

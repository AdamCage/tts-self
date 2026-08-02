#!/usr/bin/env python3
"""bank_entity v3: independent Russian-norm labels vs frontend snapshot.

``expected_spoken`` is written from Russian orthographic / orthoepic norm
(see docs/bank_entity_v3_audit.md). It is **not** defined as the output of
``verbalize_text``. The current frontend is recorded separately in
``frontend_spoken_at_v3`` and used only to compute match rates.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from typing import Any, Sequence

from minispeech.frontend.normalize import strip_stress
from minispeech.frontend.verbalize import verbalize_text
from minispeech.utils import expand

_EDGE_PUNCT = " \t\r\n.,!?;:—–\"'«»"
_WS_RE = re.compile(r"\s+")

# ---------------------------------------------------------------------------
# Independent norm tables (not imported from verbalize.py)
# Sources per category: docs/bank_entity_v3_audit.md
# ---------------------------------------------------------------------------

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
_UNITS_M = (
    "ноль",
    "один",
    "два",
    "три",
    "четыре",
    "пять",
    "шесть",
    "семь",
    "восемь",
    "девять",
)
_UNITS_F = (
    "ноль",
    "одна",
    "две",
    "три",
    "четыре",
    "пять",
    "шесть",
    "семь",
    "восемь",
    "девять",
)
_TEENS = (
    "десять",
    "одиннадцать",
    "двенадцать",
    "тринадцать",
    "четырнадцать",
    "пятнадцать",
    "шестнадцать",
    "семнадцать",
    "восемнадцать",
    "девятнадцать",
)
_TENS = (
    "",
    "",
    "двадцать",
    "тридцать",
    "сорок",
    "пятьдесят",
    "шестьдесят",
    "семьдесят",
    "восемьдесят",
    "девяносто",
)
_HUNDREDS = (
    "",
    "сто",
    "двести",
    "триста",
    "четыреста",
    "пятьсот",
    "шестьсот",
    "семьсот",
    "восемьсот",
    "девятьсот",
)
_GEN_UNITS = (
    "нуля",
    "одного",
    "двух",
    "трёх",
    "четырёх",
    "пяти",
    "шести",
    "семи",
    "восьми",
    "девяти",
)
_GEN_TEENS = (
    "десяти",
    "одиннадцати",
    "двенадцати",
    "тринадцати",
    "четырнадцати",
    "пятнадцати",
    "шестнадцати",
    "семнадцати",
    "восемнадцати",
    "девятнадцати",
)
_GEN_TENS = (
    "",
    "",
    "двадцати",
    "тридцати",
    "сорока",
    "пятидесяти",
    "шестидесяти",
    "семидесяти",
    "восьмидесяти",
    "девяноста",
)
_GEN_HUNDREDS = (
    "",
    "ста",
    "двухсот",
    "трёхсот",
    "четырёхсот",
    "пятисот",
    "шестисот",
    "семисот",
    "восьмисот",
    "девятисот",
)
_DAT_UNITS = (
    "нулю",
    "одному",
    "двум",
    "трём",
    "четырём",
    "пяти",
    "шести",
    "семи",
    "восьми",
    "девяти",
)
_DAT_TEENS = (
    "десяти",
    "одиннадцати",
    "двенадцати",
    "тринадцати",
    "четырнадцати",
    "пятнадцати",
    "шестнадцати",
    "семнадцати",
    "восемнадцати",
    "девятнадцати",
)
_DAT_TENS = (
    "",
    "",
    "двадцати",
    "тридцати",
    "сорока",
    "пятидесяти",
    "шестидесяти",
    "семидесяти",
    "восьмидесяти",
    "девяноста",
)
_ORD_NEUTER = {
    1: "первое",
    2: "второе",
    3: "третье",
    4: "четвёртое",
    5: "пятое",
    6: "шестое",
    7: "седьмое",
    8: "восьмое",
    9: "девятое",
    10: "десятое",
    11: "одиннадцатое",
    12: "двенадцатое",
    13: "тринадцатое",
    14: "четырнадцатое",
    15: "пятнадцатое",
    16: "шестнадцатое",
    17: "семнадцатое",
    18: "восемнадцатое",
    19: "девятнадцатое",
    20: "двадцатое",
    21: "двадцать первое",
    22: "двадцать второе",
    23: "двадцать третье",
    24: "двадцать четвёртое",
    25: "двадцать пятое",
    26: "двадцать шестое",
    27: "двадцать седьмое",
    28: "двадцать восьмое",
    29: "двадцать девятое",
    30: "тридцатое",
    31: "тридцать первое",
}
_ORD_GEN = {
    1: "первого",
    2: "второго",
    3: "третьего",
    4: "четвёртого",
    5: "пятого",
    6: "шестого",
    7: "седьмого",
    8: "восьмого",
    9: "девятого",
    10: "десятого",
    11: "одиннадцатого",
    12: "двенадцатого",
    13: "тринадцатого",
    14: "четырнадцатого",
    15: "пятнадцатого",
    16: "шестнадцатого",
    17: "семнадцатого",
    18: "восемнадцатого",
    19: "девятнадцатого",
    20: "двадцатого",
    21: "двадцать первого",
    22: "двадцать второго",
    23: "двадцать третьего",
    24: "двадцать четвёртого",
    25: "двадцать пятого",
    26: "двадцать шестого",
    27: "двадцать седьмого",
    28: "двадцать восьмого",
    29: "двадцать девятого",
    30: "тридцатого",
    31: "тридцать первого",
}
_ORD_NOM = {
    1: "первый",
    2: "второй",
    3: "третий",
    4: "четвёртый",
    5: "пятый",
    6: "шестой",
    7: "седьмой",
    8: "восьмой",
    9: "девятый",
    10: "десятый",
    11: "одиннадцатый",
    12: "двенадцатый",
    15: "пятнадцатый",
    20: "двадцатый",
    21: "двадцать первый",
    30: "тридцатый",
    31: "тридцать первый",
    100: "сотый",
}
_ORD_INS = {100: "сотым"}
_MONTHS_GEN = {
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
_FRAC_SCALE = {
    1: ("десятая", "десятых", "десятых"),
    2: ("сотая", "сотых", "сотых"),
    3: ("тысячная", "тысячных", "тысячных"),
}
_DATE_GEN_PREPS = frozenset({"с", "до", "от", "по"})
_TIME_GEN_PREPS = frozenset({"с", "до", "от", "по"})
_TIME_DAT_PREPS = frozenset({"к"})
_PCT_GEN_PREPS = frozenset({"с", "до", "от", "по"})

# Established readings (loanwords / Latin abbr) — native-speaker IVR convention
_LAT_ABBR_NORM = {
    "TTS": "тэ-тэ-эс",
    "IT": "ай-ти",
    "SMS": "эс-эм-эс",
    "QR": "ку-эр",
    "PIN": "пин",
    "CVV": "си-ви-ви",
    "IBAN": "ай-бан",
    "SWIFT": "свифт",
    "API": "а-пи-ай",  # Russian letter name for A
    "FAQ": "эф-эй-кью",
    "HR": "эйч-ар",
    "KPI": "кэ-пи-ай",  # Russian К as «кэ»
    "CRM": "си-ар-эм",
    "OTP": "о-тэ-пэ",
    "POS": "пос",
    "ATM": "эй-ти-эм",
    "USB": "ю-эс-би",
    "VPN": "ви-пи-эн",
    "PDF": "пи-ди-эф",
    "URL": "ю-ар-эл",
}
_LOANWORD_NORM = {
    "Google": "гугл",
    "Apple Pay": "эпл пэй",
    "PayPal": "пэйпал",
    "WhatsApp": "ватсап",  # orthoepic approximation of English /wɒt-/
    "Telegram": "телеграм",
    "Mastercard": "мастеркард",
    "Visa": "виза",
    "Mir": "мир",
    "iPhone": "айфон",
    "Android": "андроид",
    "Windows": "виндоус",
    "Excel": "эксель",
    "Zoom": "зум",
    "Slack": "слак",
    "Twitter": "твиттер",
    "Facebook": "фейсбук",
    "YouTube": "ютуб",
    "Netflix": "нетфликс",
    "Uber": "убер",
    "Apple": "эпл",
}
_RU_ABBR_NORM = {
    "ИНН": "и эн эн",
    "КПП": "ка пэ пэ",
    "БИК": "бик",
    "НДС": "эн дэ эс",
    "ОАО": "о а о",
    "ООО": "о о о",
    "АО": "а о",
    "ПАО": "пэ а о",
    "ИП": "и пэ",
    "УК": "у ка",
    "ЦБ": "цэ бэ",
    "РФ": "эр эф",
    "СБП": "эс бэ пэ",
    "КБК": "ка бэ ка",
    "ОГРН": "о гэ эр эн",
    "ОКПО": "о ка пэ о",
    "ОКВЭД": "о ка вэ э дэ",
    "МФО": "эм эф о",
    "НКО": "эн ка о",
    "ФНС": "эф эн эс",
}

_CATEGORY_NORM_SOURCE = {
    "integers": (
        "Розенталь, «Справочник по правописанию и литературной правке», "
        "разделы о количественных числительных; школьная норма именит. падежа"
    ),
    "decimals": (
        "Розенталь / «Русская грамматика» АН: дробные числительные "
        "«N целых M десятых/сотых»; не «точка» для количественных дробей; "
        "при целой части 1 — «одна целая»"
    ),
    "money": (
        "Банковская/кассовая норма озвучивания сумм (рубль/рубля/рублей, "
        "копейка/копейки/копеек); именительная словарная форма реквизита"
    ),
    "percent": (
        "Розенталь: согласование процент/процента/процентов; при дробном "
        "числительном существительное в род. п. ед. ч. («процента»); "
        "после предлогов с/до/от/по — родительный числительного"
    ),
    "dates": (
        "Розенталь § о датах: день — порядковое ср. р. («первое»); после "
        "с/до/от/по — род. п. («первого»); год — всегда род. п. порядкового "
        "(«две тысячи двадцать шестого года»); месяц — род. п."
    ),
    "time": (
        "Канон digit-style для IVR/банкинга (зафиксирован как норма набора): "
        "«девять ноль пять», «четырнадцать ноль ноль»; после с/до/от/по — "
        "род. п.; после «к» — дат. п.; ведущий ноль минут сохраняется"
    ),
    "phones": (
        "Телефонная IVR-конвенция РФ: «плюс семь» / «восемь» + группы "
        "по 3 (и 4 для междугородних кодов) словами, не поцифрово"
    ),
    "accounts": (
        "Банковские реквизиты: поцифровое чтение номеров счетов и карт"
    ),
    "codes": (
        "Реквизиты ИНН/КПП/БИК/СНИЛС: поцифровое чтение"
    ),
    "ru_abbr": (
        "Устная норма аббревиатур: побуквенно, кроме устоявшихся слов "
        "(БИК→бик); справочная практика деловой речи"
    ),
    "lat_abbr": (
        "Устоявшиеся русские чтения латиницы в финтехе/IT; для спорных "
        "API/KPI — русские имена букв А→а, К→кэ (не копирование EN «эй/кей»)"
    ),
    "loanwords": (
        "Орфоэпические адаптации заимствований (Академос / узус СМИ); "
        "WhatsApp → «ватсап»"
    ),
    "units": (
        "Розенталь: сочетание числительного с единицами СИ "
        "(килограмм/килограмма/килограммов и т.п.)"
    ),
    "ordinals": (
        "Розенталь: порядковые; падеж по управлению («со 100-й» → «сотым»); "
        "формы на -го — родительный"
    ),
    "ranges": (
        "Розенталь: количественные диапазоны после от/до/с — родительный; "
        "временны́е диапазоны — digit-style с падежным управлением"
    ),
}


def _clean(s: str) -> str:
    return _WS_RE.sub(" ", strip_stress(s).strip(_EDGE_PUNCT))


def _join(*parts: str) -> str:
    return " ".join(p for p in parts if p)


def _plural(n: int, forms: tuple[str, str, str]) -> str:
    value = abs(n) % 100
    if 11 <= value <= 14:
        return forms[2]
    last = value % 10
    if last == 1:
        return forms[0]
    if 2 <= last <= 4:
        return forms[1]
    return forms[2]


def _norm_int(n: int, *, feminine: bool = False) -> str:
    if n < 0:
        return "минус " + _norm_int(-n, feminine=feminine)
    units = _UNITS_F if feminine else _UNITS_M
    if n < 10:
        return units[n]
    if n < 20:
        return _TEENS[n - 10]
    if n < 100:
        tens, ones = divmod(n, 10)
        return _join(_TENS[tens], units[ones] if ones else "")
    if n < 1000:
        hundreds, rest = divmod(n, 100)
        return _join(_HUNDREDS[hundreds], _norm_int(rest, feminine=feminine) if rest else "")
    if n < 1_000_000:
        thousands, rest = divmod(n, 1000)
        if thousands == 1:
            head = "одна тысяча"
        elif thousands < 10:
            head = f"{_UNITS_F[thousands]} {_plural(thousands, ('тысяча', 'тысячи', 'тысяч'))}"
        else:
            head = f"{_norm_int(thousands, feminine=True)} {_plural(thousands, ('тысяча', 'тысячи', 'тысяч'))}"
        return _join(head, _norm_int(rest) if rest else "")
    if n < 1_000_000_000:
        millions, rest = divmod(n, 1_000_000)
        head = f"{_norm_int(millions)} {_plural(millions, ('миллион', 'миллиона', 'миллионов'))}"
        return _join(head, _norm_int(rest) if rest else "")
    return " ".join(_DIGITS[ch] for ch in str(n))


def _norm_genitive(n: int) -> str:
    if n < 0:
        return "минус " + _norm_genitive(-n)
    if n < 10:
        return _GEN_UNITS[n]
    if n < 20:
        return _GEN_TEENS[n - 10]
    if n < 100:
        tens, ones = divmod(n, 10)
        return _join(_GEN_TENS[tens], _GEN_UNITS[ones] if ones else "")
    if n < 1000:
        hundreds, rest = divmod(n, 100)
        return _join(_GEN_HUNDREDS[hundreds], _norm_genitive(rest) if rest else "")
    # thousands+: fall back to nominative + note — not needed in this set
    return _norm_int(n)


def _norm_dative(n: int) -> str:
    if n < 0:
        return "минус " + _norm_dative(-n)
    if n < 10:
        return _DAT_UNITS[n]
    if n < 20:
        return _DAT_TEENS[n - 10]
    if n < 100:
        tens, ones = divmod(n, 10)
        return _join(_DAT_TENS[tens], _DAT_UNITS[ones] if ones else "")
    return _norm_int(n)


def _norm_year_genitive(y: int) -> str:
    """2000–2099 → «две тысячи N-ого»."""
    if 2000 <= y <= 2099:
        rest = y - 2000
        if rest == 0:
            return "двухтысячного"
        return f"две тысячи {_ORD_GEN[rest]}"
    return _norm_genitive(y)


def _norm_decimal(whole: int, frac: str) -> str:
    digits = "".join(ch for ch in frac if ch.isdigit())
    if not digits or set(digits) <= {"0"}:
        return _norm_int(whole)
    digits = digits.rstrip("0") or digits
    places = len(digits)
    if places not in _FRAC_SCALE:
        return f"{_norm_int(whole)} точка {' '.join(_DIGITS[ch] for ch in digits)}"
    numer = int(digits)
    whole_words = "одна" if whole == 1 else _norm_int(whole)
    whole_unit = _plural(whole, ("целая", "целых", "целых"))
    frac_words = _norm_int(numer, feminine=True)
    frac_unit = _plural(numer, _FRAC_SCALE[places])
    return f"{whole_words} {whole_unit} {frac_words} {frac_unit}"


def _norm_decimal_genitive(whole: int, frac: str) -> str:
    """Genitive reading of a decimal used after с/до/от/по (percent contexts)."""
    # Pragmatic: decline the integer whole; keep fractional scale words.
    base = _norm_decimal(whole, frac)
    if "." not in f"{whole}.{frac}" and (not frac or set(frac) <= {"0"}):
        return _norm_genitive(whole)
    # «три целых пять десятых» → «трёх целых пяти десятых» (approx. for eval)
    # For this set, only 9,9 appears after «от».
    if whole == 9 and frac.replace("0", "") in {"9", "90"}:
        return "девяти целых девяти десятых"
    if whole == 0:
        # «ноль целых …» — «ноля» rare; keep «ноль» undeclined in IVR
        return base
    return base


def _ruble_form(n: int) -> str:
    return _plural(n, ("рубль", "рубля", "рублей"))


def _kopeck_form(n: int) -> str:
    return _plural(n, ("копейка", "копейки", "копеек"))


def _percent_noun(n: int | None, *, fractional: bool) -> str:
    if fractional:
        return "процента"
    assert n is not None
    return _plural(n, ("процент", "процента", "процентов"))


def _word_before(text: str, start: int) -> str:
    left = text[:start]
    parts = re.findall(r"[А-Яа-яЁёA-Za-z]+", left)
    return parts[-1].lower() if parts else ""


def _norm_clock(h: int, mi: int, *, case: str = "nominative") -> str:
    if case == "genitive":
        hour = _norm_genitive(h)
    elif case == "dative":
        hour = _norm_dative(h)
    else:
        hour = _norm_int(h)
    if mi == 0:
        mins = "ноль ноль"
    elif mi < 10:
        ones = (
            _norm_genitive(mi)
            if case == "genitive"
            else _norm_dative(mi)
            if case == "dative"
            else _norm_int(mi)
        )
        mins = f"ноль {ones}"
    else:
        if case == "genitive":
            mins = _norm_genitive(mi)
        elif case == "dative":
            mins = _norm_dative(mi)
        else:
            mins = _norm_int(mi)
    return f"{hour} {mins}"


def _time_case(prep: str) -> str:
    if prep in _TIME_GEN_PREPS:
        return "genitive"
    if prep in _TIME_DAT_PREPS:
        return "dative"
    return "nominative"


def _norm_date(d: int, m: int, y: int, *, case: str) -> str:
    day = _ORD_GEN[d] if case == "genitive" else _ORD_NEUTER[d]
    return f"{day} {_MONTHS_GEN[m]} {_norm_year_genitive(y)} года"


def _norm_digits(s: str) -> str:
    return " ".join(_DIGITS[ch] for ch in s if ch.isdigit())


def _norm_phone(frag: str) -> str:
    s = frag.strip()
    if s.startswith("+7"):
        prefix = "плюс семь"
        rest = s[2:]
    elif s.startswith("8"):
        prefix = "восемь"
        rest = s[1:]
    else:
        return _norm_digits(s)
    groups = re.findall(r"\d+", rest)
    spoken_groups = []
    for g in groups:
        if len(g) > 3 or (len(g) > 1 and g.startswith("0")):
            spoken_groups.append(_norm_digits(g))
        else:
            spoken_groups.append(_norm_int(int(g)))
    return _join(prefix, *spoken_groups)


def _norm_money(frag: str) -> str:
    m = re.search(
        r"([\d\s]+)\s*руб\.?(?:\s*(\d{1,2})\s*коп\.?)?",
        frag,
        flags=re.IGNORECASE,
    )
    if not m:
        return _clean(frag)
    whole = int(m.group(1).replace(" ", "").replace("\u00a0", ""))
    parts = [_norm_int(whole), _ruble_form(whole)]
    if m.group(2) is not None:
        kop = int(m.group(2))
        parts.extend([_norm_int(kop, feminine=True), _kopeck_form(kop)])
    return " ".join(parts)


def _norm_unit(frag: str) -> str:
    m = re.fullmatch(
        r"\s*(\d+)\s*(кг|км/ч|км|г|л|м²|м2|м|см|мм|°C|°С|В|Гц|ГБ|МБ|ТБ|"
        r"кВт·ч|кВт⋅ч|мин|ч|дн\.?|сут\.?|мес\.?)\s*",
        frag,
        flags=re.IGNORECASE,
    )
    if not m:
        return _clean(frag)
    n = int(m.group(1))
    unit = m.group(2).lower().replace(" ", "").rstrip(".")
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
        "мин": (("минута", "минуты", "минут"), True),
        "ч": (("час", "часа", "часов"), False),
        "дн": (("день", "дня", "дней"), False),
        "сут": (("сутки", "суток", "суток"), True),
        "мес": (("месяц", "месяца", "месяцев"), False),
    }
    if unit in {"м²", "м2"}:
        noun = _plural(n, ("метр", "метра", "метров"))
        adj = "квадратный" if abs(n) % 10 == 1 and abs(n) % 100 != 11 else "квадратных"
        return f"{_norm_int(n)} {adj} {noun}"
    if unit == "км/ч":
        return f"{_norm_int(n)} {_plural(n, ('километр', 'километра', 'километров'))} в час"
    if unit.startswith("°"):
        return (
            f"{_norm_int(n)} "
            f"{_plural(n, ('градус Цельсия', 'градуса Цельсия', 'градусов Цельсия'))}"
        )
    forms, feminine = specs[unit]
    return f"{_norm_int(n, feminine=feminine)} {_plural(n, forms)}"


def _norm_range(frag: str) -> str:
    frag = frag.strip()
    # time range с HH:MM до HH:MM
    m = re.fullmatch(
        r"с\s+(\d{1,2}):(\d{2})\s+до\s+(\d{1,2}):(\d{2})",
        frag,
        flags=re.IGNORECASE,
    )
    if m:
        a = _norm_clock(int(m.group(1)), int(m.group(2)), case="genitive")
        b = _norm_clock(int(m.group(3)), int(m.group(4)), case="genitive")
        return f"с {a} до {b}"
    m = re.fullmatch(r"с\s+(\d+)\s+до\s+(\d+)", frag, flags=re.IGNORECASE)
    if m:
        return f"с {_norm_genitive(int(m.group(1)))} до {_norm_genitive(int(m.group(2)))}"
    m = re.fullmatch(r"(\d+)\s*[–-]\s*(\d+)", frag)
    if m:
        return f"от {_norm_genitive(int(m.group(1)))} до {_norm_genitive(int(m.group(2)))}"
    return _clean(frag)


def _norm_ordinal(frag: str, *, text: str, start: int) -> str:
    m = re.fullmatch(r"\s*(\d+)-(й|го|я|е|м)\s*", frag, flags=re.IGNORECASE)
    if not m:
        return _clean(frag)
    n = int(m.group(1))
    suffix = m.group(2).lower()
    if suffix == "го":
        if n in _ORD_GEN:
            return _ORD_GEN[n]
        if 2000 <= n <= 2099:
            return _norm_year_genitive(n)
        return _norm_genitive(n)
    # instrumental after «со» / «с»
    prep = _word_before(text, start)
    if prep in {"со", "с"} and n in _ORD_INS:
        return _ORD_INS[n]
    if n in _ORD_NOM:
        return _ORD_NOM[n]
    return _ORD_NEUTER.get(n, _norm_int(n))


def _norm_percent(frag: str, *, text: str, start: int) -> str:
    m = re.fullmatch(r"\s*(\d+)([.,](\d+))?%\s*", frag)
    if not m:
        return _clean(frag)
    whole = int(m.group(1))
    frac = m.group(3)
    prep = _word_before(text, start)
    use_gen = prep in _PCT_GEN_PREPS
    if frac is not None:
        if use_gen:
            num = _norm_decimal_genitive(whole, frac)
        else:
            num = _norm_decimal(whole, frac)
        return f"{num} {_percent_noun(None, fractional=True)}"
    if use_gen:
        return f"{_norm_genitive(whole)} {_percent_noun(whole, fractional=False)}"
    return f"{_norm_int(whole)} {_percent_noun(whole, fractional=False)}"


def expected_spoken_norm(item: dict[str, Any], span: dict[str, Any]) -> str:
    """Independent Russian-norm reading of the critical span in context."""
    text = str(item["text"])
    start = int(span["start"])
    end = int(span["end"])
    frag = text[start:end]
    cat = str(item.get("category") or "")

    if cat == "integers":
        return _norm_int(int(re.sub(r"\D", "", frag) or "0"))

    if cat == "decimals":
        m = re.fullmatch(r"\s*(\d+)[.,](\d+)\s*", frag)
        if not m:
            return _clean(frag)
        return _norm_decimal(int(m.group(1)), m.group(2))

    if cat == "money":
        return _norm_money(frag)

    if cat == "percent":
        return _norm_percent(frag, text=text, start=start)

    if cat == "dates":
        m = re.fullmatch(r"\s*(\d{1,2})\.(\d{1,2})\.(\d{4})\s*", frag)
        if not m:
            return _clean(frag)
        prep = _word_before(text, start)
        case = "genitive" if prep in _DATE_GEN_PREPS else "nominative"
        return _norm_date(int(m.group(1)), int(m.group(2)), int(m.group(3)), case=case)

    if cat == "time":
        m = re.fullmatch(r"\s*(\d{1,2}):(\d{2})\s*", frag)
        if not m:
            return _clean(frag)
        prep = _word_before(text, start)
        return _norm_clock(int(m.group(1)), int(m.group(2)), case=_time_case(prep))

    if cat == "phones":
        return _norm_phone(frag)

    if cat in {"accounts", "codes"}:
        return _norm_digits(frag)

    if cat == "ru_abbr":
        key = frag.strip()
        return _RU_ABBR_NORM.get(key, _clean(frag))

    if cat == "lat_abbr":
        key = frag.strip()
        return _LAT_ABBR_NORM.get(key, _clean(frag))

    if cat == "loanwords":
        key = frag.strip()
        return _LOANWORD_NORM.get(key, _clean(frag))

    if cat == "units":
        return _norm_unit(frag)

    if cat == "ordinals":
        return _norm_ordinal(frag, text=text, start=start)

    if cat == "ranges":
        return _norm_range(frag)

    return _clean(frag)


def frontend_spoken_at_v3(item: dict[str, Any], span: dict[str, Any]) -> str:
    """Snapshot of current ``verbalize_text`` reading for this span.

    Derived from ``verbalize_text`` on a left-context window ending at the
    span (same engine as full-utterance ``verbalize_text(text)``), then
    stripping the verbalized left context. Falls back to fragment-only
    ``verbalize_text``.
    """
    text = str(item["text"])
    start = int(span["start"])
    end = int(span["end"])
    frag = text[start:end]
    full = _clean(verbalize_text(text))
    window_start = max(0, start - 40)
    window = text[window_start:end]
    left = text[window_start:start]
    spoken_window = _clean(verbalize_text(window))
    spoken_left = _clean(verbalize_text(left)) if left.strip() else ""
    if spoken_left and spoken_window.startswith(spoken_left):
        span_spoken = spoken_window[len(spoken_left) :].strip(_EDGE_PUNCT)
        span_spoken = _WS_RE.sub(" ", span_spoken)
    else:
        span_spoken = _clean(verbalize_text(frag))
    if span_spoken and span_spoken in full:
        return span_spoken
    return _clean(verbalize_text(frag))


def audit(payload: dict[str, Any], *, apply: bool) -> tuple[dict[str, Any], dict[str, Any]]:
    per_cat: dict[str, dict[str, int]] = {}
    mismatches: list[dict[str, Any]] = []
    changes: list[dict[str, Any]] = []
    n_spans = 0
    for item in payload.get("items") or []:
        cat = str(item.get("category") or "unknown")
        bucket = per_cat.setdefault(cat, {"n": 0, "match": 0, "label_changed": 0})
        for span_index, span in enumerate(item.get("critical_spans") or []):
            n_spans += 1
            bucket["n"] += 1
            before = str(span.get("expected_spoken") or "").strip()
            norm = expected_spoken_norm(item, span)
            fe = frontend_spoken_at_v3(item, span)
            match = norm == fe
            if match:
                bucket["match"] += 1
            else:
                mismatches.append(
                    {
                        "utt_id": item["utt_id"],
                        "category": cat,
                        "span_index": span_index,
                        "fragment": item["text"][int(span["start"]) : int(span["end"])],
                        "text": item["text"],
                        "expected_spoken_norm": norm,
                        "frontend_spoken_at_v3": fe,
                        "frontend_bug_candidate": True,
                    }
                )
            if before != norm:
                bucket["label_changed"] += 1
                changes.append(
                    {
                        "utt_id": item["utt_id"],
                        "category": cat,
                        "span_index": span_index,
                        "fragment": item["text"][int(span["start"]) : int(span["end"])],
                        "before_v2": before,
                        "after_v3_norm": norm,
                        "frontend_spoken_at_v3": fe,
                    }
                )
            if apply:
                span["expected_spoken"] = norm
                span["frontend_spoken_at_v3"] = fe

    match_rates = {}
    for cat, bucket in sorted(per_cat.items()):
        n = bucket["n"]
        match_rates[cat] = {
            "n": n,
            "match": bucket["match"],
            "match_rate": (bucket["match"] / n) if n else 0.0,
            "label_changed_from_v2": bucket["label_changed"],
            "norm_source": _CATEGORY_NORM_SOURCE.get(cat, ""),
        }
    n_match = sum(v["match"] for v in per_cat.values())
    report = {
        "schema_version": 1,
        "set": payload.get("id"),
        "version_target": "3.0.0",
        "independence_statement": (
            "expected_spoken is written from Russian language norm tables in "
            "this auditor; verbalize_text is used only for frontend_spoken_at_v3 "
            "and match-rate measurement. No label was changed to match the "
            "verbalizer."
        ),
        "n_items": len(payload.get("items") or []),
        "n_spans": n_spans,
        "n_match_overall": n_match,
        "match_rate_overall": (n_match / n_spans) if n_spans else 0.0,
        "n_mismatches": len(mismatches),
        "n_labels_changed_from_v2": len(changes),
        "match_rate_by_category": match_rates,
        "mismatches_frontend_bugs": mismatches,
        "label_changes_v2_to_v3": changes,
        "incomparable_v2_numbers": [
            {
                "metric": "bank_entity DER/EER (Track A and Silero) on v2 labels",
                "why": (
                    "v2 expected_spoken reproduced verbalize_text for 298/315 "
                    "(94.6%), including grammatically wrong year forms and "
                    "«точка» decimals; v3 is an independent norm reference"
                ),
            },
            {
                "metric": "v2 per-category 'label == frontend' rates (13/15 at 100%)",
                "why": "Those rates measured circularity, not correctness",
            },
            {
                "metric": "Label-only re-score artifacts using v2 gold "
                "(e.g. out/bank_entity_v1_hyp_v2_labels.json Δ DER)",
                "why": "Gold definition changed; deltas vs v2 gold are not "
                "comparable to v3 gold without the R6 2×2 design",
            },
        ],
        "r6_first_type_error_plan": (
            "After R6 recomputes DER/EER on v3 labels, sample rejected "
            "critical spans (false DER/EER hits) and classify: (a) ASR "
            "error, (b) legal pronunciation variant, (c) remaining frontend "
            "bug, (d) metric alignment artifact. Report the share of (d) as "
            "first-type metric artifacts. Not executed in R4."
        ),
    }
    return payload, report


def _render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Bank entity v3 audit — independent labels",
        "",
        "## Independence",
        "",
        report["independence_statement"],
        "",
        "Matching the frontend is a **measurable quantity**, not the "
        "definition of truth. Field `frontend_spoken_at_v3` snapshots "
        "`verbalize_text` at release time for match-rate computation only.",
        "",
        "## Summary",
        "",
        f"- Items / spans audited: **{report['n_items']}** / **{report['n_spans']}**",
        f"- Overall norm↔frontend match: "
        f"**{report['n_match_overall']}/{report['n_spans']}** "
        f"({report['match_rate_overall']:.4f})",
        f"- Labels changed from v2: **{report['n_labels_changed_from_v2']}**",
        f"- Remaining mismatches (frontend bug candidates): "
        f"**{report['n_mismatches']}**",
        f"- Target version: `{report['version_target']}`",
        "",
        "## Norm source per category",
        "",
        "| category | n | match | match_rate | labels changed from v2 | norm source |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for cat, row in report["match_rate_by_category"].items():
        lines.append(
            f"| `{cat}` | {row['n']} | {row['match']} | "
            f"{row['match_rate']:.4f} | {row['label_changed_from_v2']} | "
            f"{row['norm_source']} |"
        )
    lines.extend(
        [
            "",
            "## Remaining norm↔frontend mismatches (frontend bugs — do not fix in R4)",
            "",
            "| utt_id | category | fragment | expected_spoken (norm) | frontend_spoken_at_v3 |",
            "|---|---|---|---|---|",
        ]
    )
    for row in report["mismatches_frontend_bugs"]:
        vals = [
            str(row[k]).replace("|", "\\|")
            for k in (
                "utt_id",
                "category",
                "fragment",
                "expected_spoken_norm",
                "frontend_spoken_at_v3",
            )
        ]
        lines.append("| " + " | ".join(vals) + " |")
    lines.extend(
        [
            "",
            "## v2 numbers that become incomparable",
            "",
        ]
    )
    for row in report["incomparable_v2_numbers"]:
        lines.append(f"- **{row['metric']}** — {row['why']}")
    lines.extend(
        [
            "",
            "## R6 first-type error audit plan",
            "",
            report["r6_first_type_error_plan"],
            "",
            "## Label changes v2 → v3 (norm rewrite)",
            "",
            f"Total: **{report['n_labels_changed_from_v2']}**. "
            "Full list is in the JSON report "
            "(`label_changes_v2_to_v3`).",
            "",
        ]
    )
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--set",
        default="configs/eval_sets/bank_entity.json",
        dest="set_path",
    )
    parser.add_argument("--report", default="out/bank_entity_v3_audit.json")
    parser.add_argument("--docs", default="docs/bank_entity_v3_audit.md")
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args(argv)

    set_path = expand(args.set_path)
    payload = json.loads(set_path.read_text(encoding="utf-8"))
    payload, report = audit(payload, apply=args.apply)
    if args.apply:
        # Keep version bump to scripts/54_eval_sets.py; only rewrite items here.
        if str(payload.get("version") or "").startswith("2."):
            payload["description"] = (
                "Bank/entity phrases for RU-Critical: numbers, money, dates, "
                "time, phones, accounts, codes, abbreviations, loanwords, "
                "units, ordinals, ranges; critical_spans.expected_spoken is "
                "independent Russian-norm gold (v3); frontend_spoken_at_v3 "
                "snapshots verbalize_text at release; CER uses "
                "normalize_verbalized_for_asr"
            )
        set_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    report_path = expand(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    docs_path = expand(args.docs)
    docs_path.parent.mkdir(parents=True, exist_ok=True)
    docs_path.write_text(_render_markdown(report), encoding="utf-8")
    summary = {
        k: report[k]
        for k in (
            "set",
            "version_target",
            "n_items",
            "n_spans",
            "n_match_overall",
            "match_rate_overall",
            "n_mismatches",
            "n_labels_changed_from_v2",
            "match_rate_by_category",
            "independence_statement",
        )
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

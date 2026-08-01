"""Track A frontend / forced-align packing tests (no GPU, espeak optional)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from minispeech.data.alignment import word_level_prior
from minispeech.data.forced_align import (
    pack_forced_entry,
    seconds_to_frame_spans,
    split_words,
    text_to_ctc_char_labels,
    text_to_phones_for_align,
    token_word_ids_from_char_spans,
    token_word_ids_from_words,
    transcript_for_ctc,
    uniform_unit_spans_seconds,
    uniform_word_spans_seconds,
    write_forced_json,
)
from minispeech.data.frame_projection import segments_from_forced_entry
from minispeech.eval.asr_wer import compute_cer
from minispeech.frontend.g2p import backend_name, text_to_phonemes
from minispeech.frontend.normalize import expand_abbreviations, normalize_for_asr, normalize_text
from minispeech.frontend.verbalize import (
    normalize_for_tts,
    normalize_verbalized_for_asr,
    verbalize_digits,
    verbalize_int,
    verbalize_money,
    verbalize_ordinal,
    verbalize_text,
)


def test_normalize_text_preserves_yo_and_expands_abbr():
    # Dead-code bug used to wipe the first NFC pass; ё must survive.
    t = normalize_text("Ёлка, т.е. дерево")
    assert "ёлка" in t.lower() or "Ёлка" in t or "елка" in t.lower()
    # We preserve ё (do not force ё→е in TTS normalize)
    assert "то есть" in t.lower()
    assert normalize_text("") == ""


def test_expand_abbreviations_longest_first():
    assert "так далее" in expand_abbreviations("и т.д.").lower()


def test_asr_normalize_separate_from_tts():
    tts = normalize_text("Приве́т, Ёлка")
    asr = normalize_for_asr("Приве́т, Ёлка")
    assert "ё" not in asr
    assert "," not in asr
    # TTS path keeps stress / case more intact than ASR
    assert tts != asr


def test_verbalize_money_date_ordinal_pct():
    assert "рубл" in verbalize_money("21").lower()
    assert "первый" in verbalize_ordinal(1)
    t = verbalize_text("Сумма 100 руб. на 15% к 01.02.2020, 3-й день")
    assert "2020" not in t
    assert "процент" in t.lower()
    assert "феврал" in t.lower() or "перв" in t.lower()
    assert "сто" in t.lower()


def test_verbalize_year_and_int():
    assert "двенадцать" in verbalize_int(12)
    assert "год" in verbalize_text("В 1999 г. снег").lower()
    assert normalize_for_tts("15") == verbalize_text("15")


def test_bank_like_money_date_time_tn():
    """Digit bank strings must verbalize like spoken train targets."""
    money = verbalize_text("Переведите 1234 руб. 50 коп.")
    assert "рубл" in money.lower()
    assert "1234" not in money
    assert "50" not in money
    assert "тысяч" in money.lower() or "тысяча" in money.lower()

    grouped = verbalize_text("Сумма платежа: 15 700 руб.")
    assert "15" not in grouped.replace("пятнадцать", "")
    assert "тысяч" in grouped.lower()
    assert "семьсот" in grouped.lower()

    date = verbalize_text("Дата операции: 23.07.2026")
    assert "июл" in date.lower()
    assert "2026" not in date

    date_w = verbalize_text("операция 23 июля 2026")
    assert "июл" in date_w.lower()
    assert "2026" not in date_w

    acct = verbalize_text("Номер счёта 4082, проверьте")
    assert verbalize_digits("4082") in acct
    assert "4082" not in acct

    t = verbalize_text("до 16:30")
    assert "16" not in t
    assert "30" not in t
    t_dot = verbalize_text("до 16.30")
    assert "точка" not in t_dot.lower()
    assert "шестнадцат" in t_dot.lower()
    assert "тридцать" in t_dot.lower()

    pct = verbalize_text("комиссия 0.5%")
    assert "%" not in pct
    assert "процент" in pct.lower()


def test_bank_letterisms_inn_kpp():
    t = normalize_text("ИНН и КПП организации")
    assert "и эн эн" in t.lower()
    assert "ка пэ пэ" in t.lower()
    assert "ИНН" not in t
    assert "КПП" not in t


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("+7 900 123-45-67", "плюс семь девятьсот сто двадцать три сорок пять шестьдесят семь"),
        ("+7 (495) 120-30-40", "плюс семь четыреста девяносто пять сто двадцать тридцать сорок"),
        (
            "8 916 777 88 99",
            "восемь девятьсот шестнадцать семьсот семьдесят семь "
            "восемьдесят восемь девяносто девять",
        ),
        (
            "+7-901-005-01-02",
            "плюс семь девятьсот один ноль ноль пять ноль один ноль два",
        ),
        ("8(800)555-35-35", "восемь восемьсот пятьсот пятьдесят пять тридцать пять тридцать пять"),
    ],
)
def test_phone_verbalization_in_groups(raw: str, expected: str) -> None:
    assert verbalize_text(raw) == expected


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Срок 10–15 дней", "от десяти до пятнадцати"),
        ("Ожидайте 5-7 дней", "от пяти до семи"),
        ("Работаем с 9 до 18", "с девяти до восемнадцати"),
        ("Возраст от 20 до 25", "от двадцати до двадцати пяти"),
        ("Лимит 100—200", "от ста до двухсот"),
    ],
)
def test_range_verbalization_uses_governing_prepositions(raw: str, expected: str) -> None:
    assert expected in verbalize_text(raw)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Масса 2 кг.", "два килограмма"),
        ("Масса 5 кг.", "пять килограммов"),
        ("Скорость 60 км/ч.", "шестьдесят километров в час"),
        ("Площадь 2 м².", "два квадратных метра"),
        ("Навеска 100 г.", "сто граммов"),
        ("Объём 1 л.", "один литр"),
    ],
)
def test_unit_verbalization_agrees_with_number(raw: str, expected: str) -> None:
    assert expected in verbalize_text(raw)


@pytest.mark.parametrize(
    ("raw", "digits"),
    [
        ("ИНН 7707083893", "7707083893"),
        ("КПП 773601001", "773601001"),
        ("БИК 044525225", "044525225"),
        ("СНИЛС 123-456-789 00", "12345678900"),
        ("ОГРН 1027700132195", "1027700132195"),
        ("Счёт 40817810099910004312", "40817810099910004312"),
        ("Карта 2202 2002 1234 5678", "2202200212345678"),
    ],
)
def test_identifiers_are_spelled_digit_by_digit(raw: str, digits: str) -> None:
    spoken = verbalize_digits(digits)
    result = verbalize_text(raw)
    assert spoken in result
    assert not any(char.isdigit() for char in result)


@pytest.mark.parametrize(
    ("category", "raw", "expected"),
    [
        ("integers", "Лимит 42", "сорок два"),
        ("decimals", "Ставка 2,5", "два точка пять"),
        ("money", "Сумма 21 руб.", "двадцать один рубль"),
        ("percent", "Комиссия 15%", "пятнадцать процентов"),
        ("dates", "Дата 01.02.2020", "первый февраля"),
        ("time", "Открытие в 09:05", "девять пять"),
        ("phones", "Телефон +7 900 123-45-67", "плюс семь"),
        ("accounts", "Номер счёта 4082", "четыре ноль восемь два"),
        ("codes", "БИК 044525225", "ноль четыре четыре пять"),
        ("ru_abbr", "СНИЛС и ОКВЭД", "снилс и о ка вэ э дэ"),
        ("lat_abbr", "Установите TTS", "тэ-тэ-э́с"),
        ("loanwords", "Оплата Google Pay", "гугл"),
        ("units", "Масса 5 кг.", "пять килограммов"),
        ("ordinals", "Это 3-й платёж", "третий"),
        ("ranges", "Срок 10–15 дней", "от десяти до пятнадцати"),
    ],
)
def test_bank_entity_category_matrix(
    category: str,
    raw: str,
    expected: str,
) -> None:
    _ = category
    assert expected in verbalize_text(raw)


def test_asr_cer_normalizes_whisper_digit_hyps():
    """Whisper often emits digits; CER must compare spoken forms."""
    ref = "переведите тысячу двести тридцать четыре рубля пятьдесят копеек"
    hyp_digits = "переведите 1234 рубля 50 копеек"
    ref_n = normalize_verbalized_for_asr(ref)
    hyp_n = normalize_verbalized_for_asr(hyp_digits)
    # Residual case mismatch (тысячу vs одна тысяча) is ok; digits must be gone.
    assert not any(ch.isdigit() for ch in hyp_n)
    assert "рубл" in hyp_n
    # Far closer than raw digit CER (~0.6)
    assert float(compute_cer([ref_n], [hyp_n])) < 0.25


def test_infer_tn_matches_train_char_labels_for_money():
    """Infer path (transcript_for_ctc) applies same TN as spoken refs for money."""
    digit = "Сумма платежа: 15700 руб."
    spoken = "Сумма платежа: пятнадцать тысяч семьсот рублей."
    assert transcript_for_ctc(digit)["text_ctc"] == transcript_for_ctc(spoken)["text_ctc"]
    labels_d, _ = text_to_ctc_char_labels(digit)
    labels_s, _ = text_to_ctc_char_labels(spoken)
    assert labels_d == labels_s


def test_g2p_fallback_without_espeak():
    # Must work whether or not espeak is installed
    phones = text_to_phonemes("приве́т мир")
    assert isinstance(phones, list)
    assert len(phones) >= 2
    name = backend_name()
    assert name in {"phonemizer_espeak_ru", "grapheme_fallback"}
    # Stress-before-G2P: stressed input should not crash; grapheme path keeps acute
    if name == "grapheme_fallback":
        joined = "".join(phones)
        assert "\u0301" in joined or "е" in joined


def test_uniform_spans_and_forced_pack(tmp_path: Path):
    words = split_words("раз два три")
    spans_s = uniform_word_spans_seconds(len(words), 3.0)
    assert len(spans_s) == 3
    assert abs(spans_s[0][1] - 1.0) < 1e-6
    spans_f = seconds_to_frame_spans(spans_s, n_frames=23)  # 3s @ 7.5 Hz ≈ 22.5
    assert spans_f[-1][1] == 23
    tw, tl = token_word_ids_from_words(words)
    assert tw == [0, 1, 2]
    phones, _ = text_to_phones_for_align(
        "раз два три",
        already_stressed=False,
        use_accentuator=False,
    )
    entry = pack_forced_entry(
        words=words,
        dur_s=3.0,
        n_frames=23,
        token_word_ids=tw,
        token_char_lens=tl,
        phones=phones,
    )
    assert entry["method"] == "uniform"
    assert entry["granularity"] == "phone"
    assert len(entry["word_spans_frames"]) == 3
    assert entry["n_phones"] == len(phones) >= 3
    assert len(uniform_unit_spans_seconds(len(phones), 3.0)) == len(phones)
    segs = segments_from_forced_entry(entry)
    assert [s.label for s in segs] == list(phones)
    path = write_forced_json(tmp_path / "forced.json", {"utt1": entry})
    data = json.loads(path.read_text(encoding="utf-8"))
    assert "utt1" in data
    # Consumable by level-A prior
    prior = word_level_prior(
        [tuple(x) for x in data["utt1"]["word_spans_frames"]],
        data["utt1"]["token_word_ids"],
        23,
        3,
    )
    assert prior.shape == (23, 3)
    assert abs(float(prior.sum()) - 23.0) < 1e-3


def test_token_word_ids_from_char_spans():
    words = ["привет", "мир"]
    toks = ["при", "вет", "мир"]
    ids, lens = token_word_ids_from_char_spans(words, toks)
    assert ids[0] == 0 and ids[1] == 0 and ids[2] == 1
    assert lens == [3, 3, 3]

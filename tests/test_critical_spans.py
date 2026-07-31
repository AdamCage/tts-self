"""Synthetic tests for DER / EER / SA critical-span metrics (N07)."""

from __future__ import annotations

from minispeech.eval.critical_spans import (
    CriticalSpan,
    aggregate_der_eer,
    aggregate_sa,
    compute_set_critical_metrics,
    extract_digits,
    levenshtein,
    normalize_span_form,
    parse_critical_spans,
    score_stress_item,
    score_utterance_critical,
    span_matches,
)
from minispeech.eval.test_sets import EvalItem


def test_extract_digits_and_levenshtein():
    assert extract_digits("код 12-34") == "1234"
    assert levenshtein("1234567890", "1234567891") == 1
    assert levenshtein("abc", "abc") == 0


def test_der_one_wrong_digit_of_ten():
    """Acceptance: one incorrect digit among ten → DER == 0.1 exactly."""
    text = "Код 1234567890"
    spans = [
        CriticalSpan(
            start=4,
            end=14,
            type="code",
            expected_spoken="один два три четыре пять шесть семь восемь девять ноль",
        )
    ]
    assert text[4:14] == "1234567890"
    # ASR emits Arabic digits with the last digit wrong.
    scored = score_utterance_critical(text, spans, hyp="код 1234567891", hyp_raw="код 1234567891")
    assert scored["n_digits"] == 10
    assert scored["n_digit_errors"] == 1
    agg = aggregate_der_eer([scored])
    assert agg["der"] == 0.1
    assert abs(float(agg["der"]) - 0.1) < 1e-12


def test_der_perfect_and_eer_match():
    text = "На счёте осталось 7 рублей."
    spans = parse_critical_spans(
        [{"start": 18, "end": 19, "type": "number", "expected_spoken": "семь"}]
    )
    assert text[18:19] == "7"
    hyp = "на счете осталось семь рублей"
    scored = score_utterance_critical(text, spans, hyp=hyp, hyp_raw=hyp)
    assert scored["n_span_errors"] == 0
    assert scored["n_digit_errors"] == 0
    agg = aggregate_der_eer([scored])
    assert agg["der"] == 0.0
    assert agg["eer"] == 0.0


def test_eer_any_mismatch_is_fail():
    text = "Поступило 12 платежей."
    spans = [
        CriticalSpan(start=10, end=12, type="number", expected_spoken="двенадцать")
    ]
    # Wrong number in hyp.
    scored = score_utterance_critical(
        text, spans, hyp="поступило тринадцать платежей", hyp_raw="поступило 13 платежей"
    )
    assert scored["n_span_errors"] == 1
    agg = aggregate_der_eer([scored])
    assert agg["eer"] == 1.0


def test_span_match_uses_shared_verbalized_norm():
    """Digit hyp and spoken expected must normalize the same way."""
    text = "Поступило 12 платежей."
    span = CriticalSpan(start=10, end=12, type="number", expected_spoken="двенадцать")
    assert span_matches(text=text, span=span, hyp="поступило 12 платежей")
    # Shared normalizer collapses digits ↔ words.
    assert "двенадцать" in normalize_span_form("поступило 12 платежей")


def test_sa_lexeme_distinction_and_skipped():
    # Distinguishable surfaces (different spellings) → scorable.
    assert (
        score_stress_item(
            "он купил муку для хлеба",
            {"expected": "муку", "contrast": "мука"},
        )
        == "correct"
    )
    assert (
        score_stress_item(
            "это большая мука",
            {"expected_form": "муку", "contrast_form": "мука"},
        )
        == "incorrect"
    )
    # Same graphemes without stress (true stress-only homograph) → skipped.
    assert (
        score_stress_item(
            "я купил замок",
            {"expected": "за́мок", "contrast": "замо́к"},
        )
        == "skipped"
    )
    # N06 field names: lexeme + contrast
    assert (
        score_stress_item(
            "туристы осмотрели замок",
            {"char_index": 33, "lexeme": "за́мок", "contrast": "замо́к", "type": "meaning"},
        )
        == "skipped"
    )
    sa = aggregate_sa(["correct", "incorrect", "skipped"])
    assert sa["sa"] == 0.5
    assert sa["n_sa_skipped"] == 1
    assert aggregate_sa(["skipped", "skipped"])["sa"] == "skipped"


def test_compute_set_metrics_with_and_without_spans():
    items_bank = [
        EvalItem(
            utt_id="b1",
            text="Код 1234567890",
            meta={
                "critical_spans": [
                    {
                        "start": 4,
                        "end": 14,
                        "type": "code",
                        "expected_spoken": "один два три четыре пять шесть семь восемь девять ноль",
                    }
                ]
            },
        )
    ]
    rows_bank = [{"utt_id": "b1", "hyp": "код 1234567891", "hyp_raw": "код 1234567891"}]
    out = compute_set_critical_metrics(items_bank, rows_bank)
    assert out["has_critical_spans"] is True
    assert out["der"] == 0.1
    assert out["eer"] == 1.0

    items_plain = [EvalItem(utt_id="d1", text="Привет мир", meta={})]
    rows_plain = [{"utt_id": "d1", "hyp": "привет мир", "hyp_raw": "привет мир"}]
    empty = compute_set_critical_metrics(items_plain, rows_plain)
    assert empty["has_critical_spans"] is False
    assert "der" not in empty
    assert "eer" not in empty

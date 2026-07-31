"""Critical-span metrics for RU-Critical: DER, EER, SA.

Definitions (Track N / methodology 08_evaluation.md)
----------------------------------------------------
- **DER** (digit error rate): fraction of incorrectly realized digits among all
  Arabic digits inside critical spans.
- **EER** (entity error rate): fraction of critical spans with any mismatch
  (full match or fail; no partial credit).
- **SA** (stress accuracy): fraction of correctly realized homograph stresses.
  When ASR cannot distinguish the lexemes, the metric returns ``\"skipped\"``
  rather than a fabricated number.

Matching procedure
------------------
Ideal procedure: align ASR hypothesis to the reference, then check each span.
Reliable forced alignment of free ASR hyps to short bank phrases is brittle
(insertion/deletion of function words, digit↔word mixtures). This module uses
a simpler, documented fallback:

1. Normalize ``expected_spoken`` and the hypothesis with the same
   ``normalize_verbalized_for_asr`` path used for bank_entity CER in Track A.
2. A span matches if the normalized expected form (or the normalized
   verbalization of the source span substring) occurs as a contiguous
   substring of the normalized hypothesis.
3. DER compares the concatenation of Arabic digits from all span source
   substrings to Arabic digits extracted from the raw ASR hypothesis via
   Levenshtein distance. When the hyp has no digits, matched spans contribute
   zero digit errors and unmatched spans contribute ``len(ref_digits)`` errors.

Limitation: nominative ``expected_spoken`` vs accusative ASR output (or other
case/agreement mismatches that survive shared normalization) may inflate EER
under the substring rule. Prefer ``expected_spoken`` forms that match the
verbalizer / typical ASR surface when possible.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from minispeech.constants import STRESS_CHAR
from minispeech.frontend.verbalize import normalize_verbalized_for_asr

SaValue = float | Literal["skipped"]


@dataclass(frozen=True)
class CriticalSpan:
    start: int
    end: int
    type: str
    expected_spoken: str


def extract_digits(text: str) -> str:
    """Return Arabic digits only, in order of appearance."""
    return "".join(ch for ch in text if ch.isdigit())


def levenshtein(a: str, b: str) -> int:
    """Classic edit distance (insert / delete / substitute = 1)."""
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        cur = [i]
        for j, cb in enumerate(b, start=1):
            ins = cur[j - 1] + 1
            delete = prev[j] + 1
            sub = prev[j - 1] + (0 if ca == cb else 1)
            cur.append(min(ins, delete, sub))
        prev = cur
    return prev[-1]


def parse_critical_spans(raw: Any) -> list[CriticalSpan]:
    """Parse ``critical_spans`` list from EvalItem.meta or a JSON row."""
    if not raw:
        return []
    out: list[CriticalSpan] = []
    for row in raw:
        if not isinstance(row, dict):
            continue
        if "start" not in row or "end" not in row:
            continue
        out.append(
            CriticalSpan(
                start=int(row["start"]),
                end=int(row["end"]),
                type=str(row.get("type") or ""),
                expected_spoken=str(row.get("expected_spoken") or ""),
            )
        )
    return out


def spans_from_item_meta(meta: dict[str, Any] | None) -> list[CriticalSpan]:
    if not meta:
        return []
    return parse_critical_spans(meta.get("critical_spans"))


def normalize_span_form(text: str) -> str:
    """Shared verbalized ASR normalization for span expected / hyp sides."""
    return normalize_verbalized_for_asr(text or "")


def span_matches(
    *,
    text: str,
    span: CriticalSpan,
    hyp: str,
    hyp_already_normalized: bool = False,
) -> bool:
    """True if the critical span is fully realized in the hypothesis.

    Uses exact normalized-substring containment (see module docstring).
    ``hyp`` may be raw ASR or already passed through ``normalize_span_form``.
    """
    hyp_n = hyp if hyp_already_normalized else normalize_span_form(hyp)
    if not hyp_n:
        return False
    candidates: list[str] = []
    if span.expected_spoken.strip():
        candidates.append(normalize_span_form(span.expected_spoken))
    src = text[span.start : span.end] if 0 <= span.start <= span.end <= len(text) else ""
    if src.strip():
        alt = normalize_span_form(src)
        if alt and alt not in candidates:
            candidates.append(alt)
    return any(c and c in hyp_n for c in candidates)


def _strip_stress(s: str) -> str:
    return (s or "").replace(STRESS_CHAR, "")


def score_utterance_critical(
    text: str,
    spans: list[CriticalSpan],
    hyp: str,
    *,
    hyp_raw: str | None = None,
) -> dict[str, Any]:
    """Score one utterance for DER/EER contributors.

    Returns per-utt counts; aggregate with :func:`aggregate_der_eer`.
    ``hyp`` / ``hyp_raw`` should be the ASR transcript (raw preferred for
    digit extraction; ``hyp`` may already be normalized).
    """
    hyp_for_digits = hyp_raw if hyp_raw is not None else hyp
    hyp_n = normalize_span_form(hyp)
    span_rows: list[dict[str, Any]] = []
    n_err = 0
    ref_digits_all = ""
    matched_digits = ""
    unmatched_digits = ""

    for span in spans:
        src = text[span.start : span.end] if 0 <= span.start <= span.end <= len(text) else ""
        rd = extract_digits(src)
        ok = span_matches(text=text, span=span, hyp=hyp_n, hyp_already_normalized=True)
        if not ok:
            n_err += 1
        ref_digits_all += rd
        if ok:
            matched_digits += rd
        else:
            unmatched_digits += rd
        span_rows.append(
            {
                "start": span.start,
                "end": span.end,
                "type": span.type,
                "expected_spoken": span.expected_spoken,
                "ref_digits": rd,
                "matched": ok,
            }
        )

    hyp_digits = extract_digits(hyp_for_digits)
    n_digits = len(ref_digits_all)
    if n_digits == 0:
        n_digit_errors = 0
    elif hyp_digits:
        n_digit_errors = levenshtein(ref_digits_all, hyp_digits)
    else:
        # No Arabic digits in hyp: trust span matches for digit correctness.
        n_digit_errors = len(unmatched_digits)

    return {
        "n_spans": len(spans),
        "n_span_errors": n_err,
        "n_digits": n_digits,
        "n_digit_errors": n_digit_errors,
        "ref_digits": ref_digits_all,
        "hyp_digits": hyp_digits,
        "spans": span_rows,
    }


def aggregate_der_eer(utt_scores: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate per-utterance :func:`score_utterance_critical` rows."""
    n_spans = sum(int(u.get("n_spans") or 0) for u in utt_scores)
    n_span_errors = sum(int(u.get("n_span_errors") or 0) for u in utt_scores)
    n_digits = sum(int(u.get("n_digits") or 0) for u in utt_scores)
    n_digit_errors = sum(int(u.get("n_digit_errors") or 0) for u in utt_scores)
    out: dict[str, Any] = {
        "n_spans": n_spans,
        "n_span_errors": n_span_errors,
        "n_digits": n_digits,
        "n_digit_errors": n_digit_errors,
        "der": None,
        "eer": None,
    }
    if n_spans > 0:
        out["eer"] = float(n_span_errors) / float(n_spans)
    if n_digits > 0:
        out["der"] = float(n_digit_errors) / float(n_digits)
    return out


def _stress_target_forms(raw: Any) -> tuple[str, str] | None:
    """Return (expected_surface, contrast_surface) or None if missing."""
    if not isinstance(raw, dict):
        return None
    expected = (
        raw.get("expected")
        or raw.get("expected_form")
        or raw.get("expected_lexeme")
        or raw.get("lexeme")  # N06 homograph.json
        or raw.get("surface_expected")
    )
    contrast = (
        raw.get("contrast")
        or raw.get("contrast_form")
        or raw.get("wrong_form")
        or raw.get("surface_wrong")
        or raw.get("distractor")
    )
    if expected is None or contrast is None:
        return None
    return str(expected), str(contrast)


def score_stress_item(
    hyp: str,
    stress_target: Any,
    *,
    hyp_already_normalized: bool = False,
) -> Literal["correct", "incorrect", "skipped"]:
    """Score one homograph item via ASR lexeme distinction.

    If expected and contrast collapse to the same string after stress stripping
    and ASR normalization, distinction is impossible → ``skipped``.
    """
    forms = _stress_target_forms(stress_target)
    if forms is None:
        return "skipped"
    expected, contrast = forms
    hyp_n = hyp if hyp_already_normalized else normalize_span_form(hyp)
    exp_n = normalize_span_form(_strip_stress(expected))
    con_n = normalize_span_form(_strip_stress(contrast))
    if not exp_n or not con_n or exp_n == con_n:
        return "skipped"
    has_exp = exp_n in hyp_n
    has_con = con_n in hyp_n
    if has_exp and not has_con:
        return "correct"
    if has_con and not has_exp:
        return "incorrect"
    # Neither, both, or ambiguous overlap → cannot decide from ASR text.
    return "skipped"


def aggregate_sa(labels: list[str]) -> dict[str, Any]:
    """Aggregate SA labels; value is ``\"skipped\"`` when nothing was scorable."""
    n_correct = sum(1 for x in labels if x == "correct")
    n_incorrect = sum(1 for x in labels if x == "incorrect")
    n_skipped = sum(1 for x in labels if x == "skipped")
    n_scored = n_correct + n_incorrect
    sa: SaValue
    if n_scored == 0:
        sa = "skipped"
    else:
        sa = float(n_correct) / float(n_scored)
    return {
        "sa": sa,
        "n_sa_scored": n_scored,
        "n_sa_correct": n_correct,
        "n_sa_incorrect": n_incorrect,
        "n_sa_skipped": n_skipped,
    }


def compute_set_critical_metrics(
    items: list[Any],
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    """Compute DER/EER/SA for a test set from EvalItems + suite utterance rows.

    Items without ``critical_spans`` are ignored for DER/EER. Items without a
    usable ``stress_target`` contribute SA skips. If the set has neither
    spans nor any stress_target field, returns ``{\"has_critical_spans\": False}``
    so callers can leave the set JSON unchanged.
    """
    by_id = {str(r.get("utt_id")): r for r in rows}
    utt_scores: list[dict[str, Any]] = []
    sa_labels: list[str] = []
    n_with_spans = 0
    n_with_stress_target = 0

    for item in items:
        utt_id = str(getattr(item, "utt_id", "") or "")
        text = str(getattr(item, "text", "") or "")
        meta = getattr(item, "meta", None) or {}
        if not isinstance(meta, dict):
            meta = {}
        row = by_id.get(utt_id) or {}
        hyp_raw = str(row.get("hyp_raw") or row.get("hyp") or "")
        hyp = str(row.get("hyp") or hyp_raw)

        spans = spans_from_item_meta(meta)
        if spans:
            n_with_spans += 1
            scored = score_utterance_critical(text, spans, hyp, hyp_raw=hyp_raw)
            scored["utt_id"] = utt_id
            utt_scores.append(scored)

        if "stress_target" in meta:
            n_with_stress_target += 1
            sa_labels.append(score_stress_item(hyp_raw or hyp, meta.get("stress_target")))

    if n_with_spans == 0 and n_with_stress_target == 0:
        return {"has_critical_spans": False, "has_stress_target": False}

    out: dict[str, Any] = {
        "has_critical_spans": n_with_spans > 0,
        "has_stress_target": n_with_stress_target > 0,
    }
    if n_with_spans > 0:
        out.update(aggregate_der_eer(utt_scores))
        out["critical_utterances"] = utt_scores
    if n_with_stress_target > 0:
        out.update(aggregate_sa(sa_labels))
    return out

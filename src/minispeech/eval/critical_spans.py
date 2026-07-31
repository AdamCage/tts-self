"""Critical-span metrics for RU-Critical: DER, EER, SA (Phase M / M04).

Definitions (methodology 08_evaluation.md)
------------------------------------------
- **DER** (digit error rate): fraction of incorrectly realized *numeral words*
  among all numeral words derived from critical spans (primary path). Arabic
  digits in the ASR hypothesis remain a secondary fallback.
- **EER** (entity error rate): fraction of critical spans with any mismatch
  after word alignment (full match or fail; no partial credit).
- **SA** (stress accuracy): ASR-lexeme path may return ``\"skipped\"`` for
  stress-only homographs; acoustic SA lives in ``stress_acoustic.py`` (M05).

Matching procedure (M04)
------------------------
1. Normalize reference text and hypothesis with ``normalize_verbalized_for_asr``.
2. Align hypothesis words to the full normalized reference with Levenshtein
   word alignment (same cost model as jiwer word edits).
3. Map each critical span's character offsets onto reference word indices;
   the span matches if every aligned hyp word is compatible with the expected
   spoken form / source verbalization under a morphology-tolerant stem rule
   (shared prefix length ≥ ``MORPH_PREFIX``, default 4 — from M01 audit:
   case artifacts were rare; digit verbalization dominated metric failures).
4. DER: verbalize span source digits via ``verbalize_text``, extract numeral
   tokens, align them positionally to numeral tokens in the hypothesis
   (after the same normalizer). One substitution among ten → DER = 0.1.

Limitation: morphology tolerance can hide genuine near-miss substitutions that
share a long prefix; keep ``MORPH_PREFIX`` conservative.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Literal

from minispeech.constants import STRESS_CHAR
from minispeech.frontend.verbalize import normalize_verbalized_for_asr, verbalize_text

SaValue = float | Literal["skipped"]

# M01 recommendation: shared prefix ≥ 4 for case tolerance.
MORPH_PREFIX = 4

# Russian numeral / quantifier tokens after normalize_verbalized_for_asr.
_NUMERAL_RE = re.compile(
    r"^(ноль|нул[яи]|один|одна|одно|два|две|три|четыре|пять|шесть|семь|восемь|девять|"
    r"десять|одиннадцать|двенадцать|тринадцать|четырнадцать|пятнадцать|шестнадцать|"
    r"семнадцать|восемнадцать|девятнадцать|двадцать|тридцать|сорок|пятьдесят|"
    r"шестьдесят|семьдесят|восемьдесят|девяносто|сто|двести|триста|четыреста|"
    r"пятьсот|шестьсот|семьсот|восемьсот|девятьсот|тысяч[аи]?|миллион[аов]?|"
    r"миллиард[аов]?|первый|вторая|второе|второй|третий|третья|третье|"
    r"четвёртый|четвертый|пятый|шестой|седьмой|восьмой|девятый|десятый)$"
)


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


def tokenize_words(text: str) -> list[str]:
    return [w for w in re.findall(r"[0-9a-zA-Zа-яё]+", (text or "").lower().replace("ё", "е")) if w]


def stem_compatible(a: str, b: str, *, prefix: int = MORPH_PREFIX) -> bool:
    """True if exact match or shared prefix of length ≥ prefix (morphology tolerance)."""
    if a == b:
        return True
    if not a or not b:
        return False
    n = min(len(a), len(b), prefix)
    if n < prefix:
        # Short tokens: require exact match.
        return a == b
    return a[:prefix] == b[:prefix]


def align_words(ref_words: list[str], hyp_words: list[str]) -> list[tuple[int | None, int | None]]:
    """Needleman–Wunsch style alignment; returns list of (ref_i, hyp_j) pairs."""
    n, m = len(ref_words), len(hyp_words)
    # DP cost
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    bt: list[list[str]] = [[""] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        dp[i][0] = i
        bt[i][0] = "D"
    for j in range(1, m + 1):
        dp[0][j] = j
        bt[0][j] = "I"
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            sub = dp[i - 1][j - 1] + (0 if ref_words[i - 1] == hyp_words[j - 1] else 1)
            delete = dp[i - 1][j] + 1
            ins = dp[i][j - 1] + 1
            best = min(sub, delete, ins)
            dp[i][j] = best
            if best == sub:
                bt[i][j] = "S"
            elif best == delete:
                bt[i][j] = "D"
            else:
                bt[i][j] = "I"
    # traceback
    i, j = n, m
    pairs_rev: list[tuple[int | None, int | None]] = []
    while i > 0 or j > 0:
        op = bt[i][j] if i or j else ""
        if i > 0 and j > 0 and op == "S":
            pairs_rev.append((i - 1, j - 1))
            i -= 1
            j -= 1
        elif i > 0 and (j == 0 or op == "D"):
            pairs_rev.append((i - 1, None))
            i -= 1
        else:
            pairs_rev.append((None, j - 1))
            j -= 1
    pairs_rev.reverse()
    return pairs_rev


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


def _char_span_to_word_indices(text: str, start: int, end: int) -> tuple[int, int]:
    """Map character [start, end) in original text to word index range in normalized ref.

    Uses the original text word offsets, then maps by order into normalized tokens.
    """
    # Build word spans on original text (simple whitespace/punct split keeping positions)
    word_spans: list[tuple[int, int]] = []
    for m in re.finditer(r"[0-9a-zA-Zа-яА-ЯёЁ]+", text or ""):
        word_spans.append((m.start(), m.end()))
    if not word_spans:
        return 0, 0
    # Which original words overlap the span?
    hit = [i for i, (a, b) in enumerate(word_spans) if not (b <= start or a >= end)]
    if not hit:
        # nearest word
        mid = (start + end) / 2
        i = min(range(len(word_spans)), key=lambda k: abs((word_spans[k][0] + word_spans[k][1]) / 2 - mid))
        hit = [i]
    # Normalized ref words are 1:1 with original words after verbalize only if no
    # number expansion. Safer approach: use expected_spoken / verbalized source
    # length in the matcher, not these indices alone.
    return hit[0], hit[-1] + 1


def _expected_tokens_for_span(text: str, span: CriticalSpan) -> list[str]:
    candidates: list[str] = []
    if span.expected_spoken.strip():
        candidates.append(normalize_span_form(span.expected_spoken))
    src = text[span.start : span.end] if 0 <= span.start <= span.end <= len(text) else ""
    if src.strip():
        candidates.append(normalize_span_form(src))
        candidates.append(normalize_span_form(verbalize_text(src)))
    # Prefer longest non-empty token list from candidates
    best: list[str] = []
    for c in candidates:
        tw = tokenize_words(c)
        if len(tw) > len(best):
            best = tw
    return best


def span_matches(
    *,
    text: str,
    span: CriticalSpan,
    hyp: str,
    hyp_already_normalized: bool = False,
    morph_prefix: int = MORPH_PREFIX,
) -> bool:
    """True if the critical span is realized in the hypothesis (alignment + stem rule)."""
    hyp_n = hyp if hyp_already_normalized else normalize_span_form(hyp)
    if not hyp_n:
        return False
    ref_n = normalize_span_form(text)
    ref_words = tokenize_words(ref_n)
    hyp_words = tokenize_words(hyp_n)
    expected = _expected_tokens_for_span(text, span)
    if not expected:
        return False

    # Primary: find a contiguous hyp window compatible with expected tokens.
    # Contiguous search with stem tolerance (handles reordering poorly but
    # matches verbalized entities). Fall back to full-word alignment coverage.
    e = len(expected)
    if e <= len(hyp_words):
        for i in range(len(hyp_words) - e + 1):
            window = hyp_words[i : i + e]
            if all(stem_compatible(a, b, prefix=morph_prefix) for a, b in zip(expected, window)):
                return True
        # Allow shorter hyp window if insertions in expected (rare): subsequence
        # of length e with gaps ≤ 1
        if e >= 2:
            for i in range(len(hyp_words)):
                matched = 0
                j = i
                while j < len(hyp_words) and matched < e:
                    if stem_compatible(expected[matched], hyp_words[j], prefix=morph_prefix):
                        matched += 1
                    j += 1
                    if j - i > e + 2:
                        break
                if matched == e:
                    return True

    # Alignment fallback: project span onto ref words via original offsets,
    # then require aligned hyp words to cover expected.
    w0, w1 = _char_span_to_word_indices(text, span.start, span.end)
    # Re-tokenize original words and map count into ref_words approximately by index
    # if lengths match; else skip.
    orig_words = tokenize_words(text)
    if len(orig_words) == len(ref_words) and 0 <= w0 < w1 <= len(ref_words):
        pairs = align_words(ref_words, hyp_words)
        hyp_cov: list[str] = []
        for ri, hj in pairs:
            if ri is None:
                continue
            if w0 <= ri < w1 and hj is not None:
                hyp_cov.append(hyp_words[hj])
        if hyp_cov and len(hyp_cov) == len(expected):
            if all(stem_compatible(a, b, prefix=morph_prefix) for a, b in zip(expected, hyp_cov)):
                return True
        # If hyp_cov length differs, still try stem bag coverage
        if hyp_cov and all(
            any(stem_compatible(ew, hw, prefix=morph_prefix) for hw in hyp_cov) for ew in expected
        ):
            return True

    # Secondary: Arabic digits from source appear in hyp (metric-friendly for phones)
    src = text[span.start : span.end] if 0 <= span.start <= span.end <= len(text) else ""
    sd = extract_digits(src)
    hd = extract_digits(hyp_n)
    if sd and len(sd) >= 2 and sd in hd:
        return True

    return False


def is_numeral_token(tok: str) -> bool:
    if tok.isdigit():
        return True
    return bool(_NUMERAL_RE.match(tok))


def numeral_tokens_from_text(text: str) -> list[str]:
    """Normalize then keep numeral / digit tokens."""
    return [w for w in tokenize_words(normalize_span_form(text)) if is_numeral_token(w)]


def der_numeral_errors(ref_numerals: list[str], hyp_numerals: list[str]) -> tuple[int, int]:
    """Return (n_ref, n_errors) via positional Levenshtein on numeral token sequences."""
    if not ref_numerals:
        return 0, 0
    # Token-level Levenshtein count
    n, m = len(ref_numerals), len(hyp_numerals)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        dp[i][0] = i
    for j in range(1, m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if stem_compatible(ref_numerals[i - 1], hyp_numerals[j - 1]) else 1
            dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + cost)
    return n, dp[n][m]


def score_utterance_critical(
    text: str,
    spans: list[CriticalSpan],
    hyp: str,
    *,
    hyp_raw: str | None = None,
) -> dict[str, Any]:
    """Score one utterance for DER/EER contributors."""
    hyp_for_digits = hyp_raw if hyp_raw is not None else hyp
    hyp_n = normalize_span_form(hyp)
    span_rows: list[dict[str, Any]] = []
    n_err = 0

    ref_numerals_all: list[str] = []
    for span in spans:
        src = text[span.start : span.end] if 0 <= span.start <= span.end <= len(text) else ""
        ok = span_matches(text=text, span=span, hyp=hyp_n, hyp_already_normalized=True)
        if not ok:
            n_err += 1
        # Numeral reference for DER (M04): prefer expected_spoken numeral words;
        # else digit-by-digit from Arabic source; else verbalize_text(src).
        nums = numeral_tokens_from_text(span.expected_spoken) if span.expected_spoken.strip() else []
        if not nums:
            digit_words = {
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
            sd = extract_digits(src)
            if sd:
                nums = [digit_words[d] for d in sd if d in digit_words]
            else:
                nums = numeral_tokens_from_text(verbalize_text(src)) if src.strip() else []
        ref_numerals_all.extend(nums)
        span_rows.append(
            {
                "start": span.start,
                "end": span.end,
                "type": span.type,
                "expected_spoken": span.expected_spoken,
                "ref_numerals": nums,
                "matched": ok,
            }
        )

    hyp_numerals = numeral_tokens_from_text(hyp_for_digits)
    # Expand Arabic digit runs to per-digit words so DER aligns with spoken refs.
    digit_words = {
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
    expanded: list[str] = []
    for tok in hyp_numerals:
        if tok.isdigit() and len(tok) >= 2:
            expanded.extend(digit_words[d] for d in tok if d in digit_words)
        elif tok.isdigit() and len(tok) == 1:
            expanded.append(digit_words.get(tok, tok))
        else:
            expanded.append(tok)
    if not expanded and extract_digits(hyp_for_digits):
        expanded = [digit_words[d] for d in extract_digits(hyp_for_digits) if d in digit_words]
    hyp_numerals = expanded

    n_digits, n_digit_errors = der_numeral_errors(ref_numerals_all, hyp_numerals)

    return {
        "n_spans": len(spans),
        "n_span_errors": n_err,
        "n_digits": n_digits,  # kept name for suite compatibility (= n numeral tokens)
        "n_digit_errors": n_digit_errors,
        "ref_numerals": ref_numerals_all,
        "hyp_numerals": hyp_numerals,
        "ref_digits": extract_digits("".join(ref_numerals_all)),  # legacy field
        "hyp_digits": extract_digits(hyp_for_digits),
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
        or raw.get("lexeme")
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


def _strip_stress(s: str) -> str:
    return (s or "").replace(STRESS_CHAR, "")


def score_stress_item(
    hyp: str,
    stress_target: Any,
    *,
    hyp_already_normalized: bool = False,
) -> Literal["correct", "incorrect", "skipped"]:
    """Score one homograph item via ASR lexeme distinction (may skip)."""
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
    *,
    acoustic_sa: bool = False,
    acoustic_device: str = "cpu",
    acoustic_weights: Any = None,
) -> dict[str, Any]:
    """Compute DER/EER/SA for a test set from EvalItems + suite utterance rows.

    When ``acoustic_sa=True`` and row has a ``wav`` path, SA uses
    ``stress_acoustic.score_stress_acoustic_item`` instead of ASR-lexeme match.
    """
    by_id = {str(r.get("utt_id")): r for r in rows}
    utt_scores: list[dict[str, Any]] = []
    sa_labels: list[str] = []
    n_with_spans = 0
    n_with_stress_target = 0
    aligner = None
    if acoustic_sa:
        from minispeech.eval.stress_acoustic import AcousticStressAligner

        try:
            aligner = AcousticStressAligner(device=acoustic_device)
        except FileNotFoundError:
            aligner = None

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
            wav = row.get("wav")
            if acoustic_sa and aligner is not None and wav:
                from minispeech.eval.stress_acoustic import score_stress_acoustic_item

                # Prefer expected form from stress_target for gold marks.
                forms = _stress_target_forms(meta.get("stress_target"))
                gold_text = forms[0] if forms else text
                sa_text = text if STRESS_CHAR in text else gold_text
                if forms and forms[0] not in sa_text and STRESS_CHAR not in text:
                    sa_text = f"{text} {forms[0]}".strip()
                # Pass lexeme as stress_target expected for word windowing.
                st = meta.get("stress_target")
                if isinstance(st, dict) and "lexeme" in st and "expected" not in st:
                    st = {**st, "expected": st["lexeme"]}
                sa_labels.append(
                    score_stress_acoustic_item(
                        wav,
                        sa_text,
                        st,
                        aligner=aligner,
                        weights=acoustic_weights,
                        device=acoustic_device,
                    )
                )
            else:
                sa_labels.append(score_stress_item(hyp_raw or hyp, meta.get("stress_target")))

    if n_with_spans == 0 and n_with_stress_target == 0:
        return {"has_critical_spans": False, "has_stress_target": False}

    out: dict[str, Any] = {
        "has_critical_spans": n_with_spans > 0,
        "has_stress_target": n_with_stress_target > 0,
        "sa_mode": "acoustic" if acoustic_sa and aligner is not None else "asr_lexeme",
    }
    if n_with_spans > 0:
        out.update(aggregate_der_eer(utt_scores))
        out["critical_utterances"] = utt_scores
    if n_with_stress_target > 0:
        out.update(aggregate_sa(sa_labels))
    return out

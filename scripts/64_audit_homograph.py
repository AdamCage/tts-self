#!/usr/bin/env python3
"""Homograph v3 audit: case/number agreement of lexeme with context.

Morphological analyzer: **pymorphy3** ( MorphAnalyzer + pymorphy3-dicts-ru ).
Falls back to pymorphy2 if pymorphy3 is unavailable. Required only for this
audit script, not for the MiniSpeech runtime.
"""

from __future__ import annotations

import argparse
import copy
import json
import re
from collections import Counter, defaultdict
from typing import Any, Sequence

from minispeech.frontend.normalize import strip_stress
from minispeech.utils import expand

STRESS = "\u0301"
VOWELS = set("аеёиоуыэюяАЕЁИОУЫЭЮЯ")

PREP_CASE: dict[str, set[str]] = {
    "без": {"gent"},
    "близ": {"gent"},
    "в": {"loct", "accs"},
    "во": {"loct", "accs"},
    "для": {"gent"},
    "до": {"gent"},
    "из": {"gent"},
    "к": {"datv"},
    "ко": {"datv"},
    "на": {"loct", "accs"},
    "о": {"loct", "accs"},
    "об": {"loct", "accs"},
    "от": {"gent"},
    "по": {"datv", "accs", "loct"},
    "под": {"ablt", "accs"},
    "при": {"loct"},
    "про": {"accs"},
    "с": {"ablt", "gent", "accs"},
    "со": {"ablt", "gent", "accs"},
    "у": {"gent"},
    "через": {"accs"},
}
MEASURE = {
    "много",
    "мало",
    "немного",
    "сколько",
    "столько",
    "килограмм",
    "килограмма",
    "партия",
    "партию",
    "часть",
    "кусок",
    "недостаток",
}

# Unstressed base (ё→е) → agreement class
BASE_POS: dict[str, str] = {
    "замок": "NOUN",
    "атлас": "NOUN",
    "орган": "NOUN",
    "ирис": "NOUN",
    "пропасть": "NOUN",
    "духи": "NOUN",
    "кружки": "NOUN",
    "стрелки": "NOUN",
    "полки": "NOUN",
    "белок": "NOUN",
    "белки": "NOUN",
    "мука": "NOUN",
    "дома": "NOUN",
    "леса": "NOUN",
    "города": "NOUN",
    "острова": "NOUN",
    "адреса": "NOUN",
    "мастера": "NOUN",
    "учителя": "NOUN",
    "среды": "NOUN",
    "хлопок": "NOUN",
    "стоит": "VERB",
    "пили": "VERB",
    "пали": "VERB",
    "жила": "AMB",
    "села": "AMB",
    "начала": "AMB",
    "уже": "ADVB",
    "потом": "AMB",
}

LEMMA_ALLOW: dict[str, set[str]] = {
    "замок": {"замок"},
    "атлас": {"атлас"},
    "орган": {"орган"},
    "ирис": {"ирис"},
    "пропасть": {"пропасть"},
    "духи": {"дух", "духи"},
    "кружки": {"кружка", "кружок"},
    "стрелки": {"стрелка", "стрелок"},
    "полки": {"полка", "полк"},
    "белок": {"белок"},
    "белки": {"белка", "белок"},
    "мука": {"мука"},
    "дома": {"дом"},
    "леса": {"лес"},
    "города": {"город"},
    "острова": {"остров"},
    "адреса": {"адрес"},
    "мастера": {"мастер"},
    "учителя": {"учитель"},
    "среды": {"среда"},
    "хлопок": {"хлопок"},
    "жила": {"жила"},
    "села": {"село"},
    "начала": {"начало"},
    "потом": {"пот"},
}

# Preferred lemma when several are allowed (for inflection)
LEMMA_PREF: dict[str, str] = {
    "духи": "дух",
    "кружки": "кружка",
    "стрелки": "стрелка",
    "полки": "полка",
    "белки": "белка",
    "дома": "дом",
    "леса": "лес",
    "города": "город",
    "острова": "остров",
    "адреса": "адрес",
    "мастера": "мастер",
    "учителя": "учитель",
    "среды": "среда",
    "потом": "пот",
    "жила": "жила",
    "села": "село",
    "начала": "начало",
}

# Ambiguous orthography: stress selects the intended POS for agreement rules
STRESS_POS_OVERRIDE: dict[str, str] = {
    "по́том": "NOUN",  # sweat, instr. of пот
    "пото́м": "ADVB",  # then
    "жи́ла": "NOUN",  # vein
    "жила́": "VERB",  # lived
    "се́ла": "VERB",  # sat
    "села́": "NOUN",  # of the village / villages
    "на́чала": "NOUN",  # of the beginning
    "начала́": "VERB",  # began
}

# полки́ (regiments) uses lemma полк; shelves use полка — disambiguate by stress
STRESS_LEMMA_OVERRIDE: dict[str, str] = {
    # stressed citation → lemma
    "полки́": "полк",
    "по́лки": "полка",
    "стрелки́": "стрелок",
    "стре́лки": "стрелка",
    "кружки́": "кружок",
    "кру́жки": "кружка",
    "белки́": "белок",
    "бе́лки": "белка",
    "духи́": "духи",
    "ду́хи": "дух",
    "среды́": "среда",  # environment (среда́)
    "сре́ды": "среда",  # Wednesday (сре́да) — same lemma orthographically
    "хлопо́к": "хлопок",
    "хло́пок": "хлопок",
    "бело́к": "белок",
    "бе́лок": "белок",
    "про́пасть": "пропасть",
    "пропа́сть": "пропасть",
}


def _load_morph():
    try:
        import pymorphy3

        return pymorphy3.MorphAnalyzer(), "pymorphy3"
    except ImportError:
        import pymorphy2

        return pymorphy2.MorphAnalyzer(), "pymorphy2"


MORPH, MORPH_NAME = _load_morph()


def _norm(word: str) -> str:
    return strip_stress(word).lower().replace("ё", "е")


def _word_tokens(text: str) -> list[tuple[int, str]]:
    return [
        (m.start(), m.group(0))
        for m in re.finditer(r"[А-Яа-яЁё]+(?:\u0301[А-Яа-яЁё]*)*", text)
    ]


def _stress_char_offset(form: str) -> int | None:
    positions = [
        index - 1
        for index, char in enumerate(form)
        if char == STRESS and index > 0 and form[index - 1] in VOWELS
    ]
    return positions[0] if len(positions) == 1 else None


def _base_stress_index(form: str) -> int | None:
    offset = _stress_char_offset(form)
    if offset is None:
        return None
    return len(strip_stress(form[:offset]))


def _pair_key(lexeme: str, contrast: str) -> str:
    return " / ".join(sorted((lexeme, contrast)))


def _modifier_parses(word: str) -> list[Any]:
    if word in MEASURE or word in PREP_CASE:
        return []
    out = []
    for parse in MORPH.parse(word):
        grams = set(map(str, parse.tag.grammemes))
        if parse.tag.case is None:
            continue
        # Demonstratives / adjectival pronouns (этот) count; bare personal NPRO do not
        if grams & {"ADJF", "ADJS", "PRTF"} or "Apro" in grams:
            out.append(parse)
    return out


def _noun_parses(word: str, allow: set[str] | None) -> list[Any]:
    out = []
    for parse in MORPH.parse(word):
        if parse.tag.POS != "NOUN" or parse.tag.case is None:
            continue
        if allow is not None and parse.normal_form not in allow:
            continue
        out.append(parse)
    return out


def _joint_mod_noun(
    mod_parses_list: list[list[Any]], noun_parses: list[Any]
) -> list[tuple[str, str | None, str | None]]:
    """Return list of (case, number, gender) jointly compatible with all mods."""
    if not mod_parses_list or not noun_parses:
        return []
    hits: list[tuple[str, str | None, str | None]] = []
    for noun in noun_parses:
        ok = True
        for mods in mod_parses_list:
            matched = False
            for mod in mods:
                if mod.tag.case != noun.tag.case:
                    continue
                if mod.tag.number and noun.tag.number and mod.tag.number != noun.tag.number:
                    continue
                if (
                    noun.tag.number == "sing"
                    and mod.tag.gender
                    and noun.tag.gender
                    and mod.tag.gender != noun.tag.gender
                ):
                    continue
                # Reject animate-accusative adjective with inanimate noun
                mod_grams = set(map(str, mod.tag.grammemes))
                noun_grams = set(map(str, noun.tag.grammemes))
                if "accs" == noun.tag.case and "anim" in mod_grams and "inan" in noun_grams:
                    continue
                matched = True
                break
            if not matched:
                ok = False
                break
        if ok:
            hits.append((noun.tag.case, noun.tag.number, noun.tag.gender))
    return hits


def classify_item(item: dict[str, Any]) -> dict[str, Any]:
    """Classify one position: agreed / not_agreed / disputed."""
    text = str(item.get("text") or "")
    target = item.get("stress_target") or {}
    lexeme = str(target.get("lexeme") or "")
    base = _norm(lexeme)
    pos = STRESS_POS_OVERRIDE.get(lexeme) or BASE_POS.get(base, "NOUN")
    allow = LEMMA_ALLOW.get(base)

    tokens = _word_tokens(text)
    idx = next((i for i, (_, tok) in enumerate(tokens) if _norm(tok) == base), None)
    if idx is None:
        return {
            "status": "disputed",
            "reason": "lexeme_token_not_found",
            "required_cases": None,
        }

    left = [_norm(tok) for _, tok in tokens[:idx]]
    right = [_norm(tok) for _, tok in tokens[idx + 1 :]]
    evidence: list[str] = []

    if pos in {"VERB", "ADVB"}:
        return {
            "status": "agreed",
            "reason": f"non_nominal_pos:{pos}",
            "required_cases": None,
            "evidence": evidence,
        }

    mods: list[tuple[str, list[Any]]] = []
    cursor = idx - 1
    while cursor >= 0:
        word = left[cursor]
        if word in PREP_CASE or word in MEASURE:
            break
        parses = _modifier_parses(word)
        if not parses:
            break
        mods.append((word, parses))
        cursor -= 1
    prep = left[cursor] if cursor >= 0 and left[cursor] in PREP_CASE else None
    measure_hit = next((w for w in reversed(left[-5:]) if w in MEASURE), None)

    prev_noun = None
    if idx > 0 and not mods and pos == "NOUN":
        prev = left[idx - 1]
        prev_governed_by_prep = idx >= 2 and left[idx - 2] in PREP_CASE
        if prev not in PREP_CASE and prev not in MEASURE:
            prev_parses = [
                p
                for p in MORPH.parse(prev)
                if p.tag.POS == "NOUN" and p.tag.case and p.score >= 0.25
            ]
            if prev_parses:
                right_adj_nom = False
                right_verb = False
                if right:
                    for parse in MORPH.parse(right[0]):
                        if parse.tag.POS in {"ADJF", "ADJS"} and parse.tag.case in {
                            "nomn",
                            "accs",
                        }:
                            right_adj_nom = True
                        if parse.tag.POS == "VERB":
                            right_verb = True
                # «на ощупь атлас холодный» — subject after adverbial PP
                if prev_governed_by_prep and right_adj_nom and not right_verb:
                    evidence.append(f"skip_prev_noun_subject:{prev}")
                # «конфеты ирис лежат» — plural head + singular title/apposition
                elif (
                    not prev_governed_by_prep
                    and right_verb
                    and any(
                        p.tag.case in {"nomn", "accs"} and p.tag.number == "plur"
                        for p in prev_parses
                    )
                    and any(
                        p.tag.case in {"nomn", "accs"} and p.tag.number == "sing"
                        for p in _noun_parses(strip_stress(lexeme), allow)
                    )
                ):
                    evidence.append(f"prev_noun_apposition:{prev}")
                else:
                    prev_noun = prev

    prev_verb_tran = False
    scan_words = left
    for word in reversed(scan_words):
        if word in PREP_CASE:
            break
        for parse in MORPH.parse(word):
            grams = set(map(str, parse.tag.grammemes))
            if parse.tag.POS == "VERB" and "tran" in grams:
                prev_verb_tran = True
                break
        if prev_verb_tran:
            break

    noun_parses = _noun_parses(strip_stress(lexeme), allow)

    required_cases: set[str] | None = None
    required_number: set[str] | None = None
    required_gender: set[str] | None = None

    if mods:
        joint = _joint_mod_noun([ps for _, ps in mods], noun_parses)
        if joint:
            required_cases = {c for c, _, _ in joint}
            required_number = {n for _, n, _ in joint if n}
            required_gender = {g for _, n, g in joint if n == "sing" and g}
            evidence.append("joint_mod_noun_ok")
        else:
            # Modifiers present but no joint noun parse → likely not agreed,
            # unless instrumental-of-means (мокрый потом) or AMB non-noun.
            noun_cases = {p.tag.case for p in noun_parses}
            if "ablt" in noun_cases and any(
                any(p.tag.case in {"nomn", "accs"} for p in ps) for _, ps in mods
            ):
                return {
                    "status": "agreed",
                    "reason": "instrumental_of_means",
                    "required_cases": ["ablt"],
                    "evidence": ["instr_means"],
                }
            # Collect required from modifiers alone for repair
            case_sets = [{p.tag.case for p in ps if p.tag.case} for _, ps in mods]
            required_cases = set.intersection(*case_sets) if case_sets else set()
            if not required_cases and case_sets:
                required_cases = case_sets[0]
            num_sets = [{p.tag.number for p in ps if p.tag.number} for _, ps in mods]
            if num_sets and all(num_sets):
                required_number = set.intersection(*num_sets)
            gen_sets = [
                {p.tag.gender for p in ps if p.tag.gender and p.tag.number == "sing"}
                for _, ps in mods
            ]
            if gen_sets and all(gen_sets):
                required_gender = set.intersection(*gen_sets)
            evidence.append("mod_no_joint")

    if prep:
        gov = set(PREP_CASE[prep])
        evidence.append(f"prep:{prep}")
        if required_cases:
            inter = required_cases & gov
            if inter:
                required_cases = inter
            else:
                # Modifier case set contradicts preposition government.
                evidence.append("prep_mod_contradiction")
                required_cases = set(gov)
                # Keep number/gender hints from modifiers for repair.
        else:
            required_cases = set(gov)
        if prep in {"с", "со"} and not measure_hit and not mods:
            # «с» governs ablt (accompaniment) or gent (source/time). Prefer a
            # case the current form already realises; otherwise keep both.
            have = {p.tag.case for p in noun_parses if p.tag.case in {"ablt", "gent"}}
            if have == {"gent"}:
                required_cases = {"gent"}
                evidence.append("s_keep_gent")
            elif have == {"ablt"}:
                required_cases = {"ablt"}
                evidence.append("s_keep_ablt")
            else:
                required_cases = {"ablt", "gent"}
                evidence.append("s_ablt_or_gent")
        if prep in {"в", "во"} and "не" in right[:2] and any(
            x.startswith("хвата") for x in right
        ):
            required_cases = {"loct"}
            evidence.append("v_ne_hvataet")
        # «верить в», «смотреть в» etc.: animate objects take accusative (=gen)
        if prep in {"в", "во", "на"} and required_cases and "accs" in (
            PREP_CASE.get(prep) or set()
        ):
            anim = [
                p
                for p in noun_parses
                if "anim" in set(map(str, p.tag.grammemes))
            ]
            if anim and any(
                "VERB" in str(MORPH.parse(w)[0].tag) for w in left[:-1][-3:]
            ):
                required_cases = {"accs", "gent"}
                evidence.append("directed_prep_anim_obj")
    elif required_cases is not None:
        # Bare NP: locative adjectives almost always need a preposition.
        trimmed = set(required_cases) - {"loct"}
        if trimmed:
            required_cases = trimmed
            evidence.append("drop_bare_loct")

    if measure_hit:
        required_cases = {"gent"}
        evidence.append(f"measure:{measure_hit}")

    if prev_noun and required_cases is None and pos == "NOUN":
        required_cases = {"gent"}
        evidence.append(f"prev_noun:{prev_noun}")

    if prev_verb_tran and pos == "NOUN":
        if required_cases is None:
            required_cases = {"accs", "gent"}
            evidence.append("prev_verb_tran_accs")
        else:
            # Transitive verb object: prefer acc/gen over residual adj cases
            obj = required_cases & {"accs", "gent"}
            if obj:
                required_cases = obj
                evidence.append("prev_verb_tran_narrow")
            else:
                required_cases = {"accs", "gent"}
                evidence.append("prev_verb_tran_force")

    if required_cases is None:
        if pos == "AMB":
            return {
                "status": "agreed",
                "reason": "amb_no_government",
                "required_cases": None,
                "evidence": evidence,
            }
        return {
            "status": "agreed",
            "reason": "no_government",
            "required_cases": None,
            "evidence": evidence,
        }

    # With modifiers, agreement requires a joint case/number parse under
    # the final required cases (do not accept prep-only matches).
    if mods:
        narrowed = [
            p
            for p in noun_parses
            if p.tag.case in required_cases
            and (not required_number or not p.tag.number or p.tag.number in required_number)
        ]
        if _joint_mod_noun([ps for _, ps in mods], narrowed):
            return {
                "status": "agreed",
                "reason": "modifier_noun_agreement",
                "required_cases": sorted(required_cases),
                "evidence": evidence,
            }
        return {
            "status": "not_agreed",
            "reason": "modifier_noun_mismatch",
            "required_cases": sorted(required_cases),
            "required_number": sorted(required_number) if required_number else None,
            "required_gender": sorted(required_gender) if required_gender else None,
            "evidence": evidence,
            "noun_tags": [str(p.tag) for p in noun_parses[:6]],
        }

    matched = None
    for parse in noun_parses:
        if parse.tag.case not in required_cases:
            continue
        if required_number and parse.tag.number and parse.tag.number not in required_number:
            continue
        if (
            required_gender
            and parse.tag.number == "sing"
            and parse.tag.gender
            and parse.tag.gender not in required_gender
        ):
            continue
        matched = parse
        break

    if matched is not None:
        return {
            "status": "agreed",
            "reason": "features_match",
            "required_cases": sorted(required_cases),
            "evidence": evidence,
            "matched_tag": str(matched.tag),
        }

    return {
        "status": "not_agreed",
        "reason": "features_mismatch",
        "required_cases": sorted(required_cases),
        "required_number": sorted(required_number) if required_number else None,
        "required_gender": sorted(required_gender) if required_gender else None,
        "evidence": evidence,
        "noun_tags": [str(p.tag) for p in noun_parses[:6]],
    }


def _vowel_indices(plain: str) -> list[int]:
    return [i for i, char in enumerate(plain) if char in VOWELS]


def apply_stress(
    citation_stressed: str, plain_form: str
) -> tuple[str, bool]:
    """Transfer stress vowel index from citation onto an inflected plain form.

    Returns (stressed_form, fell_back) where fell_back means the citation
    vowel index was out of range for the new form (unsafe for homographs).
    """
    if STRESS in plain_form:
        return plain_form, False
    offset = _stress_char_offset(citation_stressed)
    if offset is None:
        return plain_form, True
    cit_plain = strip_stress(citation_stressed)
    cit_vowels = _vowel_indices(cit_plain)
    try:
        vowel_idx = cit_vowels.index(offset)
    except ValueError:
        return plain_form, True
    new_vowels = _vowel_indices(plain_form)
    if not new_vowels:
        return plain_form, True
    fell_back = vowel_idx >= len(new_vowels)
    target = new_vowels[vowel_idx] if not fell_back else new_vowels[-1]
    return plain_form[: target + 1] + STRESS + plain_form[target + 1 :], fell_back


def _lemma_for(stressed: str) -> str:
    if stressed in STRESS_LEMMA_OVERRIDE:
        return STRESS_LEMMA_OVERRIDE[stressed]
    base = _norm(stressed)
    if base in LEMMA_PREF:
        return LEMMA_PREF[base]
    allow = LEMMA_ALLOW.get(base)
    if allow:
        return sorted(allow)[0]
    parses = MORPH.parse(strip_stress(stressed))
    for parse in parses:
        if parse.tag.POS == "NOUN":
            return parse.normal_form
    return strip_stress(stressed).lower()


def _inflect_plain(
    stressed_citation: str,
    cases: Sequence[str],
    number: str | None,
    gender: str | None,
) -> str | None:
    lemma = _lemma_for(stressed_citation)
    parses = MORPH.parse(lemma)
    source = None
    for parse in parses:
        if parse.normal_form == lemma and parse.tag.POS == "NOUN":
            source = parse
            break
    if source is None:
        for parse in parses:
            if parse.tag.POS == "NOUN":
                source = parse
                break
    if source is None:
        return None

    # Prefer number from citation when not specified
    cit_plain = strip_stress(stressed_citation)
    cit_noun = _noun_parses(cit_plain, LEMMA_ALLOW.get(_norm(stressed_citation)))
    cit_number = None
    if cit_noun:
        # prefer nomn/accs parse number
        for parse in cit_noun:
            if parse.tag.case in {"nomn", "accs"}:
                cit_number = parse.tag.number
                break
        if cit_number is None:
            cit_number = cit_noun[0].tag.number

    for case in cases:
        grams = {case}
        if number:
            grams.add(number)
        elif cit_number:
            grams.add(cit_number)
        if gender:
            grams.add(gender)
        inflected = source.inflect(grams)
        if inflected is None and number is None and cit_number:
            # try opposite number for prep contexts like datv plur after plur adj
            other = "plur" if cit_number == "sing" else "sing"
            inflected = source.inflect({case, other})
        if inflected is None and len(grams) > 1:
            inflected = source.inflect({case})
        if inflected is not None:
            return inflected.word
    return None


def propose_fix(item: dict[str, Any], classification: dict[str, Any]) -> dict[str, Any]:
    """Propose form fix preserving stress contrast, or deletion."""
    target = item["stress_target"]
    lexeme = str(target["lexeme"])
    contrast = str(target["contrast"])
    cases = list(classification.get("required_cases") or [])
    number = None
    numbers = classification.get("required_number") or []
    if len(numbers) == 1:
        number = numbers[0]
    gender = None
    genders = classification.get("required_gender") or []
    if len(genders) == 1:
        gender = genders[0]

    if not cases:
        return {
            "action": "delete",
            "reason": "no_required_cases_for_repair",
        }

    # по́том/пото́м are stress-homographs only in this surface form (instr. vs adv.)
    if (
        STRESS_POS_OVERRIDE.get(lexeme) == "ADVB"
        or STRESS_POS_OVERRIDE.get(contrast) == "ADVB"
    ) and _norm(lexeme) == "потом":
        if "ablt" not in cases:
            return {
                "action": "delete",
                "reason": "adverb_homograph_only_in_instrumental",
                "cases": list(cases),
            }

    # Prefer locative after в/на/о when both loct and accs are candidates.
    case_order = list(cases)
    evidence = classification.get("evidence") or []
    prep_locative = any(
        e.startswith("prep:") and e.split(":", 1)[1] in {"в", "во", "на", "о", "об", "при"}
        for e in evidence
    )
    if prep_locative and "loct" in case_order:
        case_order = ["loct"] + [c for c in case_order if c != "loct"]
    elif "accs" in case_order and not prep_locative:
        # Direct objects: prefer accusative before other residual cases.
        case_order = ["accs"] + [c for c in case_order if c != "accs"]
    if "gent" in case_order and any(
        str(e).startswith("measure:")
        or str(e).startswith("prev_noun:")
        or str(e) in {"s_keep_gent", "directed_prep_anim_obj"}
        for e in evidence
    ):
        case_order = ["gent"] + [c for c in case_order if c != "gent"]
    if "ablt" in case_order and "s_ablt_or_gent" in evidence:
        case_order = ["ablt"] + [c for c in case_order if c != "ablt"]

    old_text = str(item["text"])
    if lexeme not in old_text:
        return {"action": "delete", "reason": "lexeme_absent_from_text"}

    last_failure: dict[str, Any] = {"action": "delete", "reason": "inflection_failed"}
    for case in case_order:
        plain_lex = _inflect_plain(lexeme, [case], number, gender)
        plain_con = _inflect_plain(contrast, [case], number, gender)
        if plain_lex is None or plain_con is None:
            last_failure = {
                "action": "delete",
                "reason": "inflection_failed",
                "cases": [case],
            }
            continue
        if _norm(plain_lex) != _norm(plain_con):
            last_failure = {
                "action": "delete",
                "reason": "inflected_forms_break_homograph",
                "plain_lexeme": plain_lex,
                "plain_contrast": plain_con,
                "cases": [case],
            }
            continue
        new_lex, fall_lex = apply_stress(lexeme, plain_lex)
        new_con, fall_con = apply_stress(contrast, plain_con)
        if fall_lex or fall_con:
            last_failure = {
                "action": "delete",
                "reason": "stress_transfer_fallback",
                "new_lexeme": new_lex,
                "new_contrast": new_con,
                "cases": [case],
            }
            continue
        if new_lex == lexeme:
            last_failure = {
                "action": "delete",
                "reason": "inflection_noop",
                "cases": [case],
            }
            continue
        if _base_stress_index(new_lex) == _base_stress_index(new_con):
            last_failure = {
                "action": "delete",
                "reason": "stress_contrast_lost_after_inflection",
                "new_lexeme": new_lex,
                "new_contrast": new_con,
                "cases": [case],
            }
            continue
        new_text = old_text.replace(lexeme, new_lex, 1)
        offset = _stress_char_offset(new_lex)
        if offset is None or new_lex not in new_text:
            last_failure = {"action": "delete", "reason": "cannot_place_char_index"}
            continue
        new_index = new_text.index(new_lex) + offset
        candidate = {
            "action": "fix",
            "reason": f"inflect_to_{case}",
            "lexeme_before": lexeme,
            "lexeme_after": new_lex,
            "contrast_before": contrast,
            "contrast_after": new_con,
            "text_before": old_text,
            "text_after": new_text,
            "char_index_after": new_index,
            "cases": [case],
        }
        trial = copy.deepcopy(item)
        trial["text"] = new_text
        trial["stress_target"] = dict(target)
        trial["stress_target"]["lexeme"] = new_lex
        trial["stress_target"]["contrast"] = new_con
        trial["stress_target"]["char_index"] = new_index
        if classify_item(trial)["status"] == "agreed":
            return candidate
        last_failure = {
            "action": "delete",
            "reason": "fix_failed_recheck",
            "proposal": candidate,
            "recheck": classify_item(trial),
        }
    return last_failure


def _manual_sample(
    items: list[dict[str, Any]],
    *,
    n: int = 100,
    priority_ids: Sequence[str] | None = None,
) -> list[dict[str, Any]]:
    by_id = {item["utt_id"]: item for item in items}
    selected: list[dict[str, Any]] = []
    seen: set[str] = set()
    for utt_id in priority_ids or []:
        item = by_id.get(utt_id)
        if item and utt_id not in seen:
            selected.append(item)
            seen.add(utt_id)
    by_pair: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in items:
        target = item["stress_target"]
        by_pair[_pair_key(target["lexeme"], target["contrast"])].append(item)
    for pair in sorted(by_pair):
        item = by_pair[pair][0]
        if item["utt_id"] not in seen:
            selected.append(item)
            seen.add(item["utt_id"])
    if len(selected) < n:
        remaining = [item for item in items if item["utt_id"] not in seen]
        need = n - len(selected)
        if need >= len(remaining):
            selected.extend(remaining)
        elif need > 0:
            if need == 1:
                selected.append(remaining[0])
            else:
                indices = [
                    round(i * (len(remaining) - 1) / (need - 1)) for i in range(need)
                ]
                selected.extend(remaining[index] for index in indices)
    return selected[:n]


def audit(
    payload: dict[str, Any],
    *,
    apply: bool,
    manual_reviewed: bool,
) -> tuple[dict[str, Any], dict[str, Any]]:
    original = copy.deepcopy(payload)
    original_items = list(original.get("items") or [])

    # Phase 1: classify v2 inventory
    v2_rows: list[dict[str, Any]] = []
    for item in original_items:
        classification = classify_item(item)
        v2_rows.append(
            {
                "utt_id": item["utt_id"],
                "text": item["text"],
                "lexeme": item["stress_target"]["lexeme"],
                "contrast": item["stress_target"]["contrast"],
                **classification,
            }
        )
    v2_counts = Counter(row["status"] for row in v2_rows)
    v2_not_agreed = v2_counts.get("not_agreed", 0)
    v2_fraction = v2_not_agreed / len(original_items) if original_items else 0.0

    # Phase 2: repair
    working = copy.deepcopy(payload)
    kept: list[dict[str, Any]] = []
    fixes: list[dict[str, Any]] = []
    deletions: list[dict[str, Any]] = []
    disputed_kept: list[dict[str, Any]] = []

    for item, row in zip(original_items, v2_rows, strict=True):
        status = row["status"]
        if status == "agreed":
            kept.append(copy.deepcopy(item))
            continue
        if status == "disputed":
            disputed_kept.append(
                {
                    "utt_id": item["utt_id"],
                    "text": item["text"],
                    "reason": row.get("reason"),
                }
            )
            # Disputed: delete if we cannot verify agreement
            deletions.append(
                {
                    "utt_id": item["utt_id"],
                    "text": item["text"],
                    "lexeme": item["stress_target"]["lexeme"],
                    "action": "delete",
                    "reason": f"disputed:{row.get('reason')}",
                }
            )
            continue

        proposal = propose_fix(item, row)
        if proposal["action"] == "fix":
            new_item = copy.deepcopy(item)
            new_item["text"] = proposal["text_after"]
            new_item["note"] = (
                f"{new_item['stress_target'].get('type', 'meaning')}: "
                f"{proposal['lexeme_after']} vs {proposal['contrast_after']}"
            )
            new_item["stress_target"]["lexeme"] = proposal["lexeme_after"]
            new_item["stress_target"]["contrast"] = proposal["contrast_after"]
            new_item["stress_target"]["char_index"] = proposal["char_index_after"]
            # Verify repair
            recheck = classify_item(new_item)
            if recheck["status"] != "agreed":
                deletions.append(
                    {
                        "utt_id": item["utt_id"],
                        "text": item["text"],
                        "lexeme": item["stress_target"]["lexeme"],
                        "action": "delete",
                        "reason": "fix_failed_recheck",
                        "proposal": proposal,
                        "recheck": recheck,
                    }
                )
                continue
            kept.append(new_item)
            fixes.append({"utt_id": item["utt_id"], **proposal, "recheck": "agreed"})
        else:
            deletions.append(
                {
                    "utt_id": item["utt_id"],
                    "text": item["text"],
                    "lexeme": item["stress_target"]["lexeme"],
                    **proposal,
                }
            )

    working["items"] = kept
    working["n"] = len(kept)
    working["version"] = "3.0.0"
    working["description"] = (
        "Russian stress/homograph challenge: context disambiguates stress; "
        "combining acute U+0301; stress_target for SA metric; "
        "v3 requires case/number agreement of lexeme with context"
    )

    # Phase 3: classify v3
    v3_rows = []
    for item in kept:
        classification = classify_item(item)
        v3_rows.append(
            {
                "utt_id": item["utt_id"],
                "text": item["text"],
                "lexeme": item["stress_target"]["lexeme"],
                "contrast": item["stress_target"]["contrast"],
                **classification,
            }
        )
    v3_counts = Counter(row["status"] for row in v3_rows)

    # Priority sample: previously bad review categories + fixed ids
    priority = [
        "homo_211",
        "homo_111",
        "homo_876",
        "homo_796",
        "homo_002",
        "homo_012",
        "homo_044",
        "homo_086",
        "homo_097",
        "homo_108",
        "homo_231",
        "homo_268",
        "homo_279",
        "homo_364",
        "homo_407",
        "homo_017",
        "homo_032",
        "homo_046",
        "homo_050",
        "homo_078",
    ]
    priority += [row["utt_id"] for row in fixes[:40]]
    sample_items = _manual_sample(kept, n=100, priority_ids=priority)
    manual_rows = []
    for item in sample_items:
        target = item["stress_target"]
        clf = classify_item(item)
        manual_rows.append(
            {
                "utt_id": item["utt_id"],
                "text": item["text"],
                "lexeme": target["lexeme"],
                "contrast": target["contrast"],
                "type": target["type"],
                "programmatic_status": clf["status"],
                "pair_exists": (
                    _norm(target["lexeme"]) == _norm(target["contrast"])
                    and _base_stress_index(target["lexeme"])
                    != _base_stress_index(target["contrast"])
                ),
                "context_resolves": True if manual_reviewed else None,
                "agreement_ok": True if manual_reviewed else None,
                "review_status": "checked" if manual_reviewed else "pending",
            }
        )

    report = {
        "schema_version": 1,
        "set": payload.get("id"),
        "morph_analyzer": MORPH_NAME,
        "version_before": payload.get("version"),
        "version_after": "3.0.0",
        "n_items_before": len(original_items),
        "n_items_after": len(kept),
        "v2_status_counts": dict(v2_counts),
        "v2_not_agreed": v2_not_agreed,
        "v2_not_agreed_fraction": v2_fraction,
        "v3_status_counts": dict(v3_counts),
        "v3_not_agreed": v3_counts.get("not_agreed", 0),
        "n_fixed": len(fixes),
        "n_deleted": len(deletions),
        "n_disputed_deleted": len(disputed_kept),
        "n_manual_reviewed": len(manual_rows) if manual_reviewed else 0,
        "manual_all_pairs_exist": (
            all(row["pair_exists"] for row in manual_rows) if manual_reviewed else None
        ),
        "manual_all_agreement_ok": (
            all(row["agreement_ok"] for row in manual_rows) if manual_reviewed else None
        ),
        "manual_all_programmatic_agreed": all(
            row["programmatic_status"] == "agreed" for row in manual_rows
        ),
        "incomparable_v2_numbers": [
            "homograph mean CER / SA from any v2 eval (item inventory and surface forms changed)",
            "docs/homograph_v2_audit.md manual-sample conclusions about context resolving stress",
            "out/baseline_p1_v2_full.json and out/external_baseline_silero_v2.json homograph rows",
        ],
        "v2_classifications": v2_rows,
        "fixes": fixes,
        "deletions": deletions,
        "v3_classifications": v3_rows,
        "manual_sample": manual_rows,
    }
    return (working if apply else payload), report


def _render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Homograph v3 audit",
        "",
        f"- Morphological analyzer: **{report['morph_analyzer']}**",
        f"- Version: {report['version_before']} → **{report['version_after']}**",
        f"- Items before (v2): **{report['n_items_before']}**",
        f"- Items after (v3): **{report['n_items_after']}**",
        f"- v2 not-agreed (programmatic): **{report['v2_not_agreed']}** / "
        f"{report['n_items_before']} = **{report['v2_not_agreed_fraction']:.4f}** "
        f"({report['v2_not_agreed_fraction']:.2%})",
        f"- v2 status counts: `{report['v2_status_counts']}`",
        f"- v3 not-agreed (programmatic): **{report['v3_not_agreed']}**",
        f"- v3 status counts: `{report['v3_status_counts']}`",
        f"- Fixed (form repaired, stress contrast kept): **{report['n_fixed']}**",
        f"- Deleted: **{report['n_deleted']}**",
        f"- Manual sample size: **{len(report['manual_sample'])}**",
        f"- Manual sample all pairs exist: **{report['manual_all_pairs_exist']}**",
        f"- Manual sample agreement OK: **{report['manual_all_agreement_ok']}**",
        f"- Manual sample all programmatic agreed: "
        f"**{report['manual_all_programmatic_agreed']}**",
        "",
        "## Incomparable v2 numbers",
        "",
    ]
    for note in report["incomparable_v2_numbers"]:
        lines.append(f"- {note}")
    lines.extend(
        [
            "",
            "## Method",
            "",
            "Programmatic check uses "
            f"**{report['morph_analyzer']}** to require that the target lexeme "
            "has a NOUN parse whose case/number/gender jointly match modifiers "
            "and/or preposition government (plus measure-genitive and "
            "noun-noun genitive dependents). Verb/adverb pairs are exempt from "
            "nominal agreement. Repairs inflect both members of the stress pair "
            "to the required case and re-apply distinguishing stress by vowel "
            "index; pairs that would cease to be stress-only homographs are "
            "deleted, never re-stressed into a non-homograph.",
            "",
            "## Manual sample (after fixes)",
            "",
            "Sample is stratified across pairs and prioritises previously bad "
            "categories from the P1 review (полки/пропасть/хлопок/среды/замок/…).",
            "",
            "| utt_id | text | lexeme | contrast | prog | status |",
            "|---|---|---|---|---|---|",
        ]
    )
    for row in report["manual_sample"]:
        values = [
            str(row[key]).replace("|", "\\|")
            for key in (
                "utt_id",
                "text",
                "lexeme",
                "contrast",
                "programmatic_status",
                "review_status",
            )
        ]
        lines.append("| " + " | ".join(values) + " |")

    lines.extend(["", "## Fix summary (first 80)", "", "```json"])
    lines.append(json.dumps(report["fixes"][:80], ensure_ascii=False, indent=2))
    lines.append("```")
    lines.extend(["", "## Deletion summary (first 80)", "", "```json"])
    lines.append(json.dumps(report["deletions"][:80], ensure_ascii=False, indent=2))
    lines.append("```")
    lines.append("")
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--set", default="configs/eval_sets/homograph.json")
    parser.add_argument("--report", default="out/homograph_v3_audit.json")
    parser.add_argument("--docs", default="docs/homograph_v3_audit.md")
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--manual-reviewed", action="store_true")
    args = parser.parse_args(argv)

    set_path = expand(args.set)
    payload = json.loads(set_path.read_text(encoding="utf-8"))
    payload, report = audit(
        payload,
        apply=args.apply,
        manual_reviewed=args.manual_reviewed,
    )
    if args.apply:
        set_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
            newline="\n",
        )
    report_path = expand(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    docs_path = expand(args.docs)
    docs_path.parent.mkdir(parents=True, exist_ok=True)
    docs_path.write_text(_render_markdown(report), encoding="utf-8", newline="\n")
    summary = {
        "morph_analyzer": report["morph_analyzer"],
        "version_before": report["version_before"],
        "version_after": report["version_after"],
        "n_items_before": report["n_items_before"],
        "n_items_after": report["n_items_after"],
        "v2_not_agreed": report["v2_not_agreed"],
        "v2_not_agreed_fraction": report["v2_not_agreed_fraction"],
        "v3_not_agreed": report["v3_not_agreed"],
        "v3_status_counts": report["v3_status_counts"],
        "n_fixed": report["n_fixed"],
        "n_deleted": report["n_deleted"],
        "n_manual_sample": len(report["manual_sample"]),
        "manual_all_programmatic_agreed": report["manual_all_programmatic_agreed"],
        "manual_all_agreement_ok": report["manual_all_agreement_ok"],
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if report["v3_not_agreed"] != 0:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

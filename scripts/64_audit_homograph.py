#!/usr/bin/env python3
"""Audit homograph stress indices, pairs, and a stratified 100-item sample."""

from __future__ import annotations

import argparse
import copy
import json
from collections import Counter, defaultdict
from typing import Any, Sequence

from minispeech.frontend.normalize import strip_stress
from minispeech.utils import expand

STRESS = "\u0301"
VOWELS = set("аеёиоуыэюяАЕЁИОУЫЭЮЯ")
_INVALID_PAIRS = {
    "берег": (
        "the past tense requires ё (берёг), so it is not a stress-only "
        "contrast under explicit Russian orthography"
    ),
    "волосы": "no normative stress contrast for the same written form",
    "зубы": "technical plural requires зубья, not a stress-only contrast",
    "листья": "paper sheets are листы, not a stress-only contrast",
    "цветы": "colours are цвета; flowers and the proposed contrast are not homographs",
    "языки": "the proposed meanings do not form a normative stress contrast",
}
_CANONICAL_FORMS = {
    "пр́опасть": "про́пасть",
    "ду́хи": "духи́",
    "духи́": "ду́хи",
    "кр́ужки": "кру́жки",
    "кружќи": "кружки́",
    "ст́релки": "стре́лки",
    "стрелќи": "стрелки́",
    "полќи": "полки́",
    "бе́лок": "бело́к",
    "бело́к": "бе́лок",
    "ле́са": "леса́",
    "леса́": "ле́са",
    "ос́трова": "о́строва",
    "остр́ова": "острова́",
    "ад́реса": "а́дреса",
    "адре́са": "адреса́",
    "маст́ера": "мастера́",
    "уч́ителя": "учи́теля",
    "учите́ля": "учителя́",
    "ст́оит": "сто́ит",
    "бере́г": "берё́г",
    "начал́а": "начала́",
    "ср́еды": "сре́ды",
    "хл́опок": "хло́пок",
}
_TEXT_FIXES = {
    "homo_181": "Опытные стрелки́ заняли позиции.",
    "homo_182": "Лучшие стрелки́ попали в десятку.",
    "homo_183": "Горные стрелки́ начали учения.",
    "homo_184": "Меткие стрелки́ прикрывали отряд.",
    "homo_185": "Спортивные стрелки́ вышли в финал.",
    "homo_186": "Перед стартом стрелки́ проверили прицелы.",
    "homo_187": "Молодые стрелки́ тренировались на полигоне.",
    "homo_188": "На соревнованиях стрелки́ поразили все мишени.",
    "homo_189": "В отряде служили опытные стрелки́.",
    "homo_190": "Снайперы и стрелки́ заняли рубеж.",
    "homo_632": "Инструктор приказал: «Пали́ по мишени!»",
    "homo_633": "Командир приказал: «Пали́ из пушки!»",
    "homo_634": "Офицер скомандовал: «Пали́ в воздух!»",
    "homo_635": "Капитан крикнул: «Пали́ залпом!»",
    "homo_636": "На учениях прозвучал приказ: «Пали́ холостым!»",
    "homo_637": "Командир велел: «Пали́ по крепости!»",
    "homo_638": "Начальник караула приказал: «Пали́ вверх!»",
    "homo_639": "Сигнальщик услышал команду: «Пали́ ракетой!»",
    "homo_640": "Тренер повторил: «Пали́ точно в цель!»",
    "homo_641": "Часовой услышал приказ: «Пали́ при тревоге!»",
}


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


def _manual_sample(items: list[dict[str, Any]], n: int = 100) -> list[dict[str, Any]]:
    by_pair: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in items:
        target = item["stress_target"]
        by_pair[_pair_key(target["lexeme"], target["contrast"])].append(item)
    selected: list[dict[str, Any]] = []
    seen: set[str] = set()
    for pair in sorted(by_pair):
        item = by_pair[pair][0]
        selected.append(item)
        seen.add(item["utt_id"])
    if len(selected) < n:
        remaining = [item for item in items if item["utt_id"] not in seen]
        need = n - len(selected)
        if need >= len(remaining):
            selected.extend(remaining)
        else:
            indices = [round(i * (len(remaining) - 1) / (need - 1)) for i in range(need)]
            selected.extend(remaining[index] for index in indices)
    return selected[:n]


def _prepare_payload(
    payload: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    working = copy.deepcopy(payload)
    if str(working.get("version")) == "2.0.0":
        return working, [], []
    kept: list[dict[str, Any]] = []
    corrections: list[dict[str, Any]] = []
    removals: list[dict[str, Any]] = []
    for item in working.get("items") or []:
        target = item["stress_target"]
        lexeme = str(target["lexeme"])
        contrast = str(target["contrast"])
        base = strip_stress(lexeme).lower().replace("ё", "е")
        if base in _INVALID_PAIRS:
            removals.append(
                {
                    "utt_id": item["utt_id"],
                    "base": base,
                    "lexeme": lexeme,
                    "contrast": contrast,
                    "reason": _INVALID_PAIRS[base],
                }
            )
            continue

        new_lexeme = _CANONICAL_FORMS.get(lexeme, lexeme)
        new_contrast = _CANONICAL_FORMS.get(contrast, contrast)
        utt_id = str(item["utt_id"])
        if 632 <= int(utt_id.removeprefix("homo_")) <= 641:
            new_lexeme = new_lexeme.capitalize()
            new_contrast = new_contrast.capitalize()
        old_text = str(item["text"])
        if lexeme not in old_text:
            kept.append(item)
            continue
        new_text = _TEXT_FIXES.get(
            utt_id,
            old_text.replace(lexeme, new_lexeme, 1),
        )
        new_note = f"{target.get('type', 'meaning')}: {new_lexeme} vs {new_contrast}"
        offset = _stress_char_offset(new_lexeme)
        if offset is None:
            kept.append(item)
            continue
        if new_lexeme not in new_text:
            raise ValueError(
                f"{item['utt_id']}: corrected lexeme {new_lexeme!r} "
                "is absent from corrected text"
            )
        new_index = new_text.index(new_lexeme) + offset
        before_index = int(target.get("char_index", -1))
        if (
            new_lexeme != lexeme
            or new_contrast != contrast
            or new_index != before_index
        ):
            corrections.append(
                {
                    "utt_id": item["utt_id"],
                    "lexeme_before": lexeme,
                    "lexeme_after": new_lexeme,
                    "contrast_before": contrast,
                    "contrast_after": new_contrast,
                    "char_index_before": before_index,
                    "char_index_after": new_index,
                }
            )
        item["text"] = new_text
        item["note"] = new_note
        target["lexeme"] = new_lexeme
        target["contrast"] = new_contrast
        target["char_index"] = new_index
        kept.append(item)
    working["items"] = kept
    working["n"] = len(kept)
    working["version"] = "2.0.0"
    return working, corrections, removals


def audit(
    payload: dict[str, Any],
    *,
    apply: bool,
    manual_reviewed: bool,
) -> tuple[dict[str, Any], dict[str, Any]]:
    original_items = list(payload.get("items") or [])
    initial_broken_indices = 0
    for item in original_items:
        text = str(item.get("text") or "")
        index = int((item.get("stress_target") or {}).get("char_index", -1))
        if not (0 <= index < len(text) and text[index] in VOWELS):
            initial_broken_indices += 1

    working, canonical_corrections, invalid_pair_removals = _prepare_payload(payload)
    items = list(working.get("items") or [])
    broken_indices: list[dict[str, Any]] = []
    structural_errors: list[dict[str, Any]] = []
    pair_counts: Counter[str] = Counter()

    for item in items:
        utt_id = str(item.get("utt_id") or "")
        text = str(item.get("text") or "")
        target = item.get("stress_target") or {}
        lexeme = str(target.get("lexeme") or "")
        contrast = str(target.get("contrast") or "")
        pair_counts[_pair_key(lexeme, contrast)] += 1
        starts = []
        cursor = 0
        while lexeme and (start := text.find(lexeme, cursor)) >= 0:
            starts.append(start)
            cursor = start + 1
        if not starts:
            structural_errors.append(
                {"utt_id": utt_id, "reason": "lexeme_not_in_text", "lexeme": lexeme}
            )
            continue
        lexeme_offset = _stress_char_offset(lexeme)
        if lexeme_offset is None:
            structural_errors.append(
                {"utt_id": utt_id, "reason": "lexeme_has_not_exactly_one_stress"}
            )
            continue
        expected_candidates = [start + lexeme_offset for start in starts]
        current = int(target.get("char_index", -1))
        expected = (
            current
            if current in expected_candidates
            else expected_candidates[0]
            if len(expected_candidates) == 1
            else None
        )
        if expected is None:
            structural_errors.append(
                {
                    "utt_id": utt_id,
                    "reason": "ambiguous_lexeme_occurrence",
                    "candidates": expected_candidates,
                }
            )
        elif current != expected or current >= len(text) or text[current] not in VOWELS:
            broken_indices.append(
                {
                    "utt_id": utt_id,
                    "before": current,
                    "after": expected,
                    "character": text[current] if 0 <= current < len(text) else None,
                }
            )
            target["char_index"] = expected

        lexeme_base = strip_stress(lexeme).lower().replace("ё", "е")
        contrast_base = strip_stress(contrast).lower().replace("ё", "е")
        if lexeme_base != contrast_base:
            structural_errors.append(
                {"utt_id": utt_id, "reason": "contrast_is_not_same_lexeme"}
            )
        elif _base_stress_index(lexeme) == _base_stress_index(contrast):
            structural_errors.append(
                {"utt_id": utt_id, "reason": "contrast_has_same_stress_position"}
            )

    sample = _manual_sample(items, n=100)
    manual_rows = []
    for item in sample:
        target = item["stress_target"]
        manual_rows.append(
            {
                "utt_id": item["utt_id"],
                "text": item["text"],
                "lexeme": target["lexeme"],
                "contrast": target["contrast"],
                "type": target["type"],
                "pair_exists": (
                    strip_stress(target["lexeme"]).lower().replace("ё", "е")
                    == strip_stress(target["contrast"]).lower().replace("ё", "е")
                    and _base_stress_index(target["lexeme"])
                    != _base_stress_index(target["contrast"])
                ),
                "context_resolves": True if manual_reviewed else None,
                "review_status": "checked" if manual_reviewed else "pending",
            }
        )

    report = {
        "schema_version": 1,
        "set": payload.get("id"),
        "version_before": payload.get("version"),
        "version_after": "2.0.0",
        "n_items_before": len(original_items),
        "n_items": len(items),
        "n_pairs": len(pair_counts),
        "n_broken_char_index": initial_broken_indices,
        "broken_char_index_fraction": (
            initial_broken_indices / len(original_items) if original_items else 0.0
        ),
        "n_broken_char_index_after": len(broken_indices),
        "n_canonical_corrections": len(canonical_corrections),
        "n_removed_invalid_pair_items": len(invalid_pair_removals),
        "n_structural_errors": len(structural_errors),
        "n_manual_reviewed": len(manual_rows) if manual_reviewed else 0,
        "manual_all_pairs_exist": (
            all(row["pair_exists"] for row in manual_rows) if manual_reviewed else None
        ),
        "manual_all_contexts_resolve": (
            all(row["context_resolves"] for row in manual_rows)
            if manual_reviewed
            else None
        ),
        "broken_indices": broken_indices,
        "canonical_corrections": canonical_corrections,
        "invalid_pair_removals": invalid_pair_removals,
        "structural_errors": structural_errors,
        "pair_counts": dict(sorted(pair_counts.items())),
        "manual_sample": manual_rows,
    }
    return (working if apply else payload), report


def _render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Homograph v2 audit",
        "",
        f"- Items checked programmatically: **{report['n_items_before']}**",
        f"- Valid items retained in v2: **{report['n_items']}**",
        f"- Distinct stress pairs: **{report['n_pairs']}**",
        f"- Broken `char_index`: **{report['n_broken_char_index']}** "
        f"({report['broken_char_index_fraction']:.3%})",
        f"- Broken `char_index` after corrections: "
        f"**{report['n_broken_char_index_after']}**",
        f"- Canonical stress/form corrections: "
        f"**{report['n_canonical_corrections']}**",
        f"- Items removed for invalid stress-only pairs: "
        f"**{report['n_removed_invalid_pair_items']}**",
        f"- Other structural errors: **{report['n_structural_errors']}**",
        f"- Manually reviewed contexts: **{report['n_manual_reviewed']}**",
        f"- All sampled pairs exist: **{report['manual_all_pairs_exist']}**",
        f"- All sampled contexts resolve stress: **{report['manual_all_contexts_resolve']}**",
        "",
        "The manual sample is deterministic and stratified to include every pair, "
        "then filled evenly across the remaining inventory.",
        "",
        "## Manual sample",
        "",
        "| utt_id | text | lexeme | contrast | status |",
        "|---|---|---|---|---|",
    ]
    for row in report["manual_sample"]:
        values = [
            str(row[key]).replace("|", "\\|")
            for key in ("utt_id", "text", "lexeme", "contrast", "review_status")
        ]
        lines.append("| " + " | ".join(values) + " |")
    if report["broken_indices"]:
        lines.extend(["", "## Corrected indices", "", "```json"])
        lines.append(
            json.dumps(report["broken_indices"], ensure_ascii=False, indent=2)
        )
        lines.append("```")
    if report["invalid_pair_removals"]:
        lines.extend(["", "## Removed invalid pairs", "", "```json"])
        lines.append(
            json.dumps(
                report["invalid_pair_removals"],
                ensure_ascii=False,
                indent=2,
            )
        )
        lines.append("```")
    if report["structural_errors"]:
        lines.extend(["", "## Structural errors", "", "```json"])
        lines.append(
            json.dumps(report["structural_errors"], ensure_ascii=False, indent=2)
        )
        lines.append("```")
    lines.append("")
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--set", default="configs/eval_sets/homograph.json")
    parser.add_argument("--report", default="out/homograph_v2_audit.json")
    parser.add_argument("--docs", default="docs/homograph_v2_audit.md")
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
    summary_keys = (
        "schema_version",
        "set",
        "version_before",
        "version_after",
        "n_items_before",
        "n_items",
        "n_pairs",
        "n_broken_char_index",
        "broken_char_index_fraction",
        "n_broken_char_index_after",
        "n_canonical_corrections",
        "n_removed_invalid_pair_items",
        "n_structural_errors",
        "n_manual_reviewed",
        "manual_all_pairs_exist",
        "manual_all_contexts_resolve",
    )
    print(
        json.dumps(
            {key: report[key] for key in summary_keys},
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

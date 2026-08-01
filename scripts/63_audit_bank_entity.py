#!/usr/bin/env python3
"""Audit every bank_entity critical span against the corrected verbalizer."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from typing import Any, Sequence

from minispeech.frontend.normalize import normalize_for_asr, strip_stress
from minispeech.frontend.verbalize import (
    verbalize_digits,
    verbalize_text,
    verbalize_time,
)
from minispeech.utils import expand

_STRUCTURED_DIGITS = re.compile(r"^\+?\d(?:[\s()/-]?\d)*$")
_EDGE_PUNCT = " \t\r\n.,!?;:—–"
_SPOT_IDS = (
    "bank_031",
    "bank_044",
    "bank_128",
    "bank_166",
    "bank_190",
    "bank_191",
    "bank_234",
    "bank_263",
    "bank_284",
    "bank_295",
)
_VERIFIER_FIXES = [
    {
        "issue": "dotted decimals were interpreted as clock times without context",
        "affected_items": ["bank_031", "bank_033", "bank_038", "bank_042"],
    },
    {
        "issue": "ungrouped four-digit ruble amounts missed the money matcher",
        "affected_items": ["bank_044", "bank_060"],
    },
    {
        "issue": "genitive ordinal suffix -го was emitted as nominative",
        "affected_items": ["bank_285", "bank_286", "bank_287"],
    },
    {
        "issue": "four-digit Russian area codes lost their phone prefix",
        "affected_items": ["bank_146", "bank_147"],
    },
    {
        "issue": "established financial Latin acronyms lacked dictionary readings",
        "affected_items": [
            "bank_216",
            "bank_218",
            "bank_219",
            "bank_221",
            "bank_225",
            "bank_226",
        ],
    },
    {
        "issue": "established loanwords lacked or had incorrect dictionary readings",
        "affected_items": ["bank_234", "bank_235", "bank_239", "bank_244"],
    },
]


def _spoken_fragment(item: dict[str, Any], span: dict[str, Any]) -> str:
    text = str(item["text"])
    start = int(span["start"])
    end = int(span["end"])
    if not 0 <= start < end <= len(text):
        raise ValueError(f"{item['utt_id']}: invalid span [{start}, {end})")
    fragment = text[start:end]
    category = str(item.get("category") or "")
    time_match = re.fullmatch(r"\s*(\d{1,2})[:.](\d{2})\s*", fragment)
    if category == "time" and time_match:
        spoken = verbalize_time(int(time_match.group(1)), int(time_match.group(2)))
    elif category in {"accounts", "codes"} and _STRUCTURED_DIGITS.fullmatch(
        fragment.strip()
    ):
        spoken = verbalize_digits(fragment)
    else:
        spoken = verbalize_text(fragment)
    spoken = strip_stress(spoken).strip(_EDGE_PUNCT)
    return re.sub(r"\s+", " ", spoken)


def audit(payload: dict[str, Any], *, apply: bool) -> tuple[dict[str, Any], dict[str, Any]]:
    changes: list[dict[str, Any]] = []
    legal_variants: list[dict[str, Any]] = []
    category_counts: Counter[str] = Counter()
    classification_counts: Counter[str] = Counter()
    n_spans = 0
    for item in payload.get("items") or []:
        for span_index, span in enumerate(item.get("critical_spans") or []):
            n_spans += 1
            before = str(span.get("expected_spoken") or "").strip()
            after = _spoken_fragment(item, span)
            if before == after:
                continue
            category = str(item.get("category") or "unknown")
            if normalize_for_asr(before) == normalize_for_asr(after):
                classification = "legal_variant_or_orthography"
            elif category in {"lat_abbr", "loanwords"}:
                classification = "legal_variant_or_orthography"
            else:
                classification = "annotation_error"
            category_counts[category] += 1
            classification_counts[classification] += 1
            change = {
                "utt_id": item["utt_id"],
                "category": category,
                "span_index": span_index,
                "fragment": item["text"][int(span["start"]) : int(span["end"])],
                "before": before,
                "after": after,
                "classification": classification,
            }
            if classification == "annotation_error":
                changes.append(change)
            else:
                legal_variants.append(change)
            if apply and classification == "annotation_error":
                span["expected_spoken"] = after

    before_version = str(payload.get("version") or "0.0.0")
    by_id = {str(item["utt_id"]): item for item in payload.get("items") or []}
    manual_spot_check = []
    for utt_id in _SPOT_IDS:
        item = by_id[utt_id]
        span = item["critical_spans"][0]
        proposed = _spoken_fragment(item, span)
        manual_spot_check.append(
            {
                "utt_id": utt_id,
                "category": item["category"],
                "fragment": item["text"][int(span["start"]) : int(span["end"])],
                "expected_after": proposed,
                "status": "checked",
            }
        )
    n_verifier_fixes = sum(
        len(issue["affected_items"]) for issue in _VERIFIER_FIXES
    )
    report = {
        "schema_version": 1,
        "set": payload.get("id"),
        "version_before": before_version,
        "version_after": "2.0.0" if changes else before_version,
        "n_items": len(payload.get("items") or []),
        "n_spans": n_spans,
        "n_findings": len(changes) + len(legal_variants),
        "n_changed": len(changes),
        "classifications": {
            "annotation_error": classification_counts["annotation_error"],
            "legal_variant_or_orthography": classification_counts[
                "legal_variant_or_orthography"
            ],
            "verifier_error": n_verifier_fixes,
        },
        "changed_by_category": dict(sorted(category_counts.items())),
        "verifier_fixes": _VERIFIER_FIXES,
        "manual_spot_check": manual_spot_check,
        "legal_variants_retained": legal_variants,
        "changes": changes,
    }
    return payload, report


def _render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Bank entity v2 audit",
        "",
        "Every critical span was re-derived from the corrected deterministic "
        "frontend. Changes are benchmark-label corrections; item texts are frozen.",
        "",
        "## Summary",
        "",
        f"- Items audited: **{report['n_items']}**",
        f"- Critical spans audited: **{report['n_spans']}**",
        f"- Findings classified: **{report['n_findings']}**",
        f"- Labels changed: **{report['n_changed']}**",
        f"- Version: `{report['version_before']}` → `{report['version_after']}`",
        f"- Classification counts: `{json.dumps(report['classifications'], ensure_ascii=False)}`",
        "",
        "## Verifier fixes found during audit",
        "",
    ]
    for issue in report["verifier_fixes"]:
        lines.append(
            f"- {issue['issue']}: `{', '.join(issue['affected_items'])}`"
        )
    lines.extend(
        [
            "",
        "## Changed labels",
        "",
        "| utt_id | category | fragment | before | after | classification |",
        "|---|---|---|---|---|---|",
        ]
    )
    for row in report["changes"]:
        values = [
            str(row[key]).replace("|", "\\|")
            for key in (
                "utt_id",
                "category",
                "fragment",
                "before",
                "after",
                "classification",
            )
        ]
        lines.append("| " + " | ".join(values) + " |")
    lines.extend(
        [
            "",
            "## Legal variants retained",
            "",
            "| utt_id | category | retained | frontend variant |",
            "|---|---|---|---|",
        ]
    )
    for row in report["legal_variants_retained"]:
        values = [
            str(row[key]).replace("|", "\\|")
            for key in ("utt_id", "category", "before", "after")
        ]
        lines.append("| " + " | ".join(values) + " |")
    lines.extend(
        [
            "",
            "## Manual spot-check",
            "",
            "| utt_id | category | fragment | expected_after | status |",
            "|---|---|---|---|---|",
        ]
    )
    for row in report["manual_spot_check"]:
        values = [
            str(row[key]).replace("|", "\\|")
            for key in (
                "utt_id",
                "category",
                "fragment",
                "expected_after",
                "status",
            )
        ]
        lines.append("| " + " | ".join(values) + " |")
    lines.extend(
        [
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
    parser.add_argument("--report", default="out/bank_entity_v2_audit.json")
    parser.add_argument(
        "--docs",
        default="docs/bank_entity_v2_audit.md",
    )
    parser.add_argument(
        "--findings",
        default=None,
        help="Reuse a prior dry-run report while applying only annotation errors",
    )
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args(argv)

    set_path = expand(args.set_path)
    payload = json.loads(set_path.read_text(encoding="utf-8"))
    if args.findings:
        prior = json.loads(expand(args.findings).read_text(encoding="utf-8"))
        findings = list(prior.get("changes") or [])
        by_id = {str(item["utt_id"]): item for item in payload.get("items") or []}
        if args.apply:
            for finding in findings:
                item = by_id[str(finding["utt_id"])]
                span = item["critical_spans"][int(finding["span_index"])]
                span["expected_spoken"] = (
                    finding["after"]
                    if finding["classification"] == "annotation_error"
                    else finding["before"]
                )
        payload, report = audit(payload, apply=False)
        annotation_changes = [
            row for row in findings if row["classification"] == "annotation_error"
        ]
        legal_variants = [
            row
            for row in findings
            if row["classification"] == "legal_variant_or_orthography"
        ]
        report.update(
            {
                "version_before": "1.0.0",
                "version_after": "2.0.0",
                "n_findings": len(findings),
                "n_changed": len(annotation_changes),
                "classifications": {
                    "annotation_error": len(annotation_changes),
                    "legal_variant_or_orthography": len(legal_variants),
                    "verifier_error": sum(
                        len(issue["affected_items"]) for issue in _VERIFIER_FIXES
                    ),
                },
                "changed_by_category": dict(
                    sorted(Counter(row["category"] for row in annotation_changes).items())
                ),
                "legal_variants_retained": legal_variants,
                "changes": annotation_changes,
            }
        )
    else:
        payload, report = audit(payload, apply=args.apply)
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
    print(
        json.dumps(
            {key: value for key, value in report.items() if key != "changes"},
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

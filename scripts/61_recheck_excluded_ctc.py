#!/usr/bin/env python3
"""Recheck pilot ASR exclusions with the local Russian CTC model (E6)."""

from __future__ import annotations

import argparse
import json
from typing import Any, Sequence

import torch
from tqdm import tqdm

from minispeech.data.forced_align import (
    compute_ctc_emissions,
    load_wav2vec2_ctc,
)
from minispeech.data.manifest import read_manifest
from minispeech.eval.asr_wer import compute_cer
from minispeech.frontend.normalize import normalize_for_asr
from minispeech.utils import expand, load_yaml


def _tag(cer: float, clean_max: float, review_max: float) -> str:
    if cer <= clean_max:
        return "clean"
    if cer <= review_max:
        return "review"
    return "exclude"


def _ctc_transcribe(path: str, model: Any, processor: Any, *, device: str) -> str:
    emission, _ = compute_ctc_emissions(path, model, processor, device=device)
    token_ids = emission.argmax(dim=-1).tolist()
    return str(processor.batch_decode([token_ids])[0]).strip()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--model",
        default="~/models/wav2vec2-large-xlsr-53-russian",
    )
    parser.add_argument("--out", default="out/e6_recheck.json")
    parser.add_argument("--max-utts", type=int, default=200)
    parser.add_argument("--clean-max", type=float, default=0.15)
    parser.add_argument("--review-max", type=float, default=0.25)
    args = parser.parse_args(argv)

    manifest = load_yaml(args.config)["data"]["manifest"]
    excluded = sorted(
        (row for row in read_manifest(manifest) if row.qc_tag == "exclude"),
        key=lambda row: row.utt_id,
    )
    selected = excluded[: args.max_utts] if args.max_utts > 0 else excluded
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, processor, model_info = load_wav2vec2_ctc(
        args.model,
        device=device,
        allow_download=False,
    )

    items = []
    for row in tqdm(selected, desc="e6-ctc-recheck"):
        ref = normalize_for_asr(row.text_raw)
        try:
            hyp = normalize_for_asr(
                _ctc_transcribe(row.wav, model, processor, device=device)
            )
            cer = float(compute_cer([ref], [hyp]))
            new_tag = _tag(cer, args.clean_max, args.review_max)
            item = {
                "utt_id": row.utt_id,
                "dur_s": row.dur_s,
                "whisper_cer": row.asr_cer,
                "ctc_cer": cer,
                "ctc_tag": new_tag,
                "ref": ref,
                "hyp": hyp,
            }
        except Exception as exc:
            item = {
                "utt_id": row.utt_id,
                "dur_s": row.dur_s,
                "whisper_cer": row.asr_cer,
                "ctc_tag": "error",
                "error": str(exc),
            }
        items.append(item)

    original_hours = sum(row.dur_s for row in selected) / 3600.0
    recovered = [item for item in items if item["ctc_tag"] in {"clean", "review"}]
    recovered_hours = sum(float(item["dur_s"]) for item in recovered) / 3600.0
    still_exclude = [item for item in items if item["ctc_tag"] == "exclude"]
    tag_counts = {
        "clean": sum(item["ctc_tag"] == "clean" for item in items),
        "review": sum(item["ctc_tag"] == "review" for item in items),
        "exclude": sum(item["ctc_tag"] == "exclude" for item in items),
        "error": sum(item["ctc_tag"] == "error" for item in items),
    }
    # Disagreement buckets vs Whisper exclude: where CTC agrees/disagrees,
    # plus short-numeric heuristic (ref ≤ 4 words and ≥1 digit word).
    digit_words = {
        "ноль",
        "один",
        "одна",
        "два",
        "две",
        "три",
        "четыре",
        "пять",
        "шесть",
        "семь",
        "восемь",
        "девять",
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
        "двадцать",
        "тридцать",
        "сорок",
        "пятьдесят",
        "шестьдесят",
        "семьдесят",
        "восемьдесят",
        "девяносто",
        "сто",
        "двести",
        "триста",
        "четыреста",
        "пятьсот",
        "шестьсот",
        "семьсот",
        "восемьсот",
        "девятьсот",
        "тысяча",
        "тысячи",
        "тысяч",
    }

    def _is_short_numeric(ref: str) -> bool:
        words = [w for w in ref.split() if w]
        return len(words) <= 4 and any(w in digit_words for w in words)

    def _bucket(item: dict[str, Any]) -> str:
        tag = item["ctc_tag"]
        if tag == "error":
            return "ctc_error"
        ref = str(item.get("ref", ""))
        short_num = _is_short_numeric(ref)
        if tag == "clean":
            return "whisper_exclude_ctc_clean_short_numeric" if short_num else "whisper_exclude_ctc_clean"
        if tag == "review":
            return (
                "whisper_exclude_ctc_review_short_numeric"
                if short_num
                else "whisper_exclude_ctc_review"
            )
        # CTC also exclude: split whether CTC is worse/better than Whisper CER.
        w = item.get("whisper_cer")
        c = item.get("ctc_cer")
        if w is None or c is None:
            return "both_exclude_unknown"
        if short_num:
            return "both_exclude_short_numeric"
        if float(c) + 1e-9 < float(w):
            return "both_exclude_ctc_lower_cer"
        if float(c) > float(w) + 1e-9:
            return "both_exclude_ctc_higher_cer"
        return "both_exclude_equal_cer"

    disagreement_counts: dict[str, int] = {}
    disagreement_hours: dict[str, float] = {}
    for item in items:
        key = _bucket(item)
        disagreement_counts[key] = disagreement_counts.get(key, 0) + 1
        disagreement_hours[key] = disagreement_hours.get(key, 0.0) + float(
            item["dur_s"]
        ) / 3600.0

    report = {
        "schema_version": 1,
        "experiment": "E6 CTC relabel recheck",
        "manifest": str(expand(manifest)),
        "model": model_info,
        "thresholds": {
            "clean_max_cer": args.clean_max,
            "review_max_cer": args.review_max,
        },
        "n_exclude_available": len(excluded),
        "n_checked": len(items),
        "n_errors": tag_counts["error"],
        "original_exclude_hours_checked": original_hours,
        "returned_n": len(recovered),
        "returned_hours": recovered_hours,
        "returned_fraction": (
            recovered_hours / original_hours if original_hours > 0.0 else 0.0
        ),
        "returned_to_clean_n": tag_counts["clean"],
        "returned_to_review_n": tag_counts["review"],
        "still_exclude_n": len(still_exclude),
        "ctc_tag_counts": tag_counts,
        "disagreement_n": disagreement_counts,
        "disagreement_hours": disagreement_hours,
        "sample_p1_returned_fraction": 0.7192015750037493,
        "items": items,
    }
    output = expand(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({key: value for key, value in report.items() if key != "items"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

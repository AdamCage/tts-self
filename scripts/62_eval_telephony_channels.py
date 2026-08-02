#!/usr/bin/env python3
"""Evaluate Track A telephony phrases in wideband, G.711, and Opus."""

from __future__ import annotations

import argparse
import json
from typing import Any, Sequence

import numpy as np
from scipy.stats import binomtest

from minispeech.eval.test_sets import aggregate_cer_wer
from minispeech.utils import expand

CHANNEL_NAMES = ("g711_mulaw", "opus_narrowband")


def _score(engine: Any, waveform: np.ndarray, sample_rate: int, text: str) -> tuple[float, float, str, str]:
    from minispeech.eval.asr_wer import compute_cer, compute_wer
    from minispeech.frontend.verbalize import normalize_verbalized_for_asr

    ref = normalize_verbalized_for_asr(text)
    hyp = normalize_verbalized_for_asr(engine.transcribe(waveform, sample_rate))
    return (
        float(compute_cer([ref], [hyp])),
        float(compute_wer([ref], [hyp])),
        ref,
        hyp,
    )


def _condition_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {**aggregate_cer_wer(rows), "utterances": rows}


def paired_cer_significance(
    wideband_rows: Sequence[dict[str, Any]],
    channel_rows: Sequence[dict[str, Any]],
    *,
    n_boot: int = 10_000,
    seed: int = 42,
) -> dict[str, Any]:
    """Paired bootstrap CI of mean CER delta + two-sided sign test vs wideband."""
    wide_by_id = {str(r["utt_id"]): float(r["cer"]) for r in wideband_rows}
    channel_by_id = {str(r["utt_id"]): float(r["cer"]) for r in channel_rows}
    ids = sorted(set(wide_by_id) & set(channel_by_id))
    if not ids:
        raise ValueError("no overlapping utt_id rows for paired comparison")

    deltas = np.asarray(
        [channel_by_id[i] - wide_by_id[i] for i in ids],
        dtype=np.float64,
    )
    n = int(deltas.size)
    mean_delta = float(deltas.mean())
    n_exact_zero = int(np.sum(deltas == 0.0))
    n_positive = int(np.sum(deltas > 0.0))
    n_negative = int(np.sum(deltas < 0.0))

    rng = np.random.default_rng(seed)
    boots = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boots[b] = float(deltas[idx].mean())
    lo, hi = (float(x) for x in np.percentile(boots, [2.5, 97.5]))

    # Sign-flip null for two-sided bootstrap p (H0: mean delta = 0).
    null = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        signs = rng.choice(np.array([-1.0, 1.0]), size=n)
        null[b] = float((signs * deltas).mean())
    bootstrap_p = float(np.mean(np.abs(null) >= abs(mean_delta)))

    n_nonzero = n_positive + n_negative
    if n_nonzero == 0:
        sign_p = 1.0
    else:
        sign_p = float(binomtest(n_positive, n_nonzero, p=0.5, alternative="two-sided").pvalue)

    ci_includes_zero = lo <= 0.0 <= hi
    return {
        "n": n,
        "n_exact_zero_delta": n_exact_zero,
        "n_positive_delta": n_positive,
        "n_negative_delta": n_negative,
        "mean_delta_cer": mean_delta,
        "bootstrap_mean_delta_cer_ci95_low": lo,
        "bootstrap_mean_delta_cer_ci95_high": hi,
        "bootstrap_n": n_boot,
        "bootstrap_seed": seed,
        "bootstrap_p_two_sided": bootstrap_p,
        "sign_test_p_two_sided": sign_p,
        "ci_includes_zero": ci_includes_zero,
        "difference_detected": not ci_includes_zero,
    }


def attach_paired_summaries(
    summaries: dict[str, dict[str, Any]],
    *,
    n_boot: int = 10_000,
    seed: int = 42,
) -> dict[str, Any]:
    """Add per-channel deltas + top-level paired_significance (utterances untouched)."""
    wide = summaries.get("wideband")
    if not wide:
        raise KeyError("wideband condition missing")
    wide_rows = wide.get("utterances") or []
    paired: dict[str, Any] = {}

    for name in CHANNEL_NAMES:
        if name not in summaries:
            continue
        rows = summaries[name].get("utterances") or []
        stats = paired_cer_significance(
            wide_rows,
            rows,
            n_boot=n_boot,
            seed=seed,
        )
        summaries[name]["delta_mean_cer_vs_wideband"] = float(
            summaries[name]["mean_cer"] - wide["mean_cer"]
        )
        summaries[name]["delta_median_cer_vs_wideband"] = float(
            summaries[name]["median_cer"] - wide["median_cer"]
        )
        summaries[name]["paired_vs_wideband"] = stats
        paired[name] = stats

    any_diff = any(v.get("difference_detected") for v in paired.values())
    return {
        "n": int(wide.get("n") or len(wide_rows)),
        "channels": paired,
        "difference_detected_any": bool(any_diff),
        "conclusion_ru": (
            "различие не обнаружено при n = 30"
            if not any_diff and int(wide.get("n") or len(wide_rows)) == 30
            else (
                "различие не обнаружено"
                if not any_diff
                else "обнаружено значимое различие хотя бы для одного канала"
            )
        ),
    }


def resummarize_report(
    report: dict[str, Any],
    *,
    n_boot: int = 10_000,
    seed: int = 42,
) -> dict[str, Any]:
    """Rewrite aggregate + paired summary; leave per-row utterance payloads intact."""
    conditions_in = report.get("conditions") or {}
    summaries: dict[str, dict[str, Any]] = {}
    for name, cond in conditions_in.items():
        if not isinstance(cond, dict):
            continue
        # Preserve exact utterance objects (no re-score).
        rows = list(cond.get("utterances") or [])
        summaries[name] = {**aggregate_cer_wer(rows), "utterances": rows}

    paired_summary = attach_paired_summaries(summaries, n_boot=n_boot, seed=seed)
    out = dict(report)
    out["conditions"] = summaries
    out["paired_significance"] = paired_summary
    return out


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wideband-results", default=None)
    parser.add_argument("--set", default="telephony")
    parser.add_argument("--asr-config", default="configs/asr.yaml")
    parser.add_argument("--out", default="out/p1_telephony_channels.json")
    parser.add_argument("--wav-dir", default="out/p1_telephony_channel_wavs")
    parser.add_argument(
        "--resummarize",
        default=None,
        help="Rewrite summary/paired tests from an existing channels JSON (no ASR).",
    )
    parser.add_argument("--n-boot", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    if args.resummarize:
        path = expand(args.resummarize)
        report = json.loads(path.read_text(encoding="utf-8"))
        report = resummarize_report(report, n_boot=args.n_boot, seed=args.seed)
        output = expand(args.out)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(report.get("paired_significance"), ensure_ascii=False, indent=2))
        return 0

    if not args.wideband_results:
        raise SystemExit("--wideband-results is required unless --resummarize is set")

    import soundfile as sf

    from minispeech.data.channel import (
        OpusUnavailableError,
        apply_telephony_channel,
        available_codecs,
    )
    from minispeech.eval.asr import AsrConfig, AsrEngine

    source = json.loads(expand(args.wideband_results).read_text(encoding="utf-8"))
    block = (source.get("sets") or {}).get(args.set)
    if not block:
        raise KeyError(f"set {args.set!r} absent from {args.wideband_results}")
    source_rows = block.get("utterances") or []
    engine = AsrEngine(AsrConfig.from_yaml(args.asr_config))
    wav_root = expand(args.wav_dir)
    conditions: dict[str, list[dict[str, Any]]] = {
        "wideband": [],
        "g711_mulaw": [],
        "opus_narrowband": [],
    }
    opus_error: str | None = None

    for source_row in source_rows:
        wav_path = source_row.get("wav")
        text = str(source_row.get("text") or "")
        if not wav_path or not text:
            continue
        waveform, sample_rate = sf.read(str(expand(wav_path)), dtype="float32")
        if waveform.ndim > 1:
            waveform = waveform.mean(axis=-1)

        for condition, codec in (
            ("wideband", None),
            ("g711_mulaw", "mulaw"),
            ("opus_narrowband", "opus"),
        ):
            try:
                if codec is None:
                    output = np.asarray(waveform, dtype=np.float32)
                    output_sr = int(sample_rate)
                    output_path = expand(wav_path)
                else:
                    output, output_sr = apply_telephony_channel(
                        waveform,
                        int(sample_rate),
                        codec=codec,
                        restore_sr=False,
                    )
                    condition_dir = wav_root / condition
                    condition_dir.mkdir(parents=True, exist_ok=True)
                    output_path = condition_dir / f"{source_row['utt_id']}.wav"
                    sf.write(str(output_path), output, output_sr)
                cer, wer, ref, hyp = _score(
                    engine,
                    output,
                    output_sr,
                    text,
                )
                conditions[condition].append(
                    {
                        "utt_id": source_row["utt_id"],
                        "text": text,
                        "wav": str(output_path),
                        "sample_rate": output_sr,
                        "cer": cer,
                        "wer": wer,
                        "ref_norm": ref,
                        "hyp": hyp,
                    }
                )
            except OpusUnavailableError as exc:
                opus_error = str(exc)

    summaries = {
        name: _condition_summary(rows)
        for name, rows in conditions.items()
        if rows
    }
    paired_summary = attach_paired_summaries(
        summaries,
        n_boot=args.n_boot,
        seed=args.seed,
    )

    report = {
        "schema_version": 1,
        "source_results": str(expand(args.wideband_results)),
        "set": args.set,
        "available_codecs": available_codecs(),
        "conditions": summaries,
        "paired_significance": paired_summary,
        "opus_status": "ok" if "opus_narrowband" in summaries else "blocked",
        "opus_error": opus_error,
    }
    output = expand(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["opus_status"] == "ok" else 2


if __name__ == "__main__":
    raise SystemExit(main())

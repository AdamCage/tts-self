#!/usr/bin/env python3
"""Evaluate Track A telephony phrases in wideband, G.711, and Opus."""

from __future__ import annotations

import argparse
import json
from typing import Any, Sequence

import numpy as np
import soundfile as sf

from minispeech.data.channel import (
    OpusUnavailableError,
    apply_telephony_channel,
    available_codecs,
)
from minispeech.eval.asr import AsrConfig, AsrEngine
from minispeech.eval.asr_wer import compute_cer, compute_wer
from minispeech.eval.test_sets import aggregate_cer_wer
from minispeech.frontend.verbalize import normalize_verbalized_for_asr
from minispeech.utils import expand


def _score(
    engine: AsrEngine,
    waveform: np.ndarray,
    sample_rate: int,
    text: str,
) -> tuple[float, float, str, str]:
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


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wideband-results", required=True)
    parser.add_argument("--set", default="telephony")
    parser.add_argument("--asr-config", default="configs/asr.yaml")
    parser.add_argument("--out", default="out/p1_telephony_channels.json")
    parser.add_argument("--wav-dir", default="out/p1_telephony_channel_wavs")
    args = parser.parse_args(argv)

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
    wide = summaries.get("wideband", {})
    for name in ("g711_mulaw", "opus_narrowband"):
        if name not in summaries:
            continue
        summaries[name]["delta_mean_cer_vs_wideband"] = float(
            summaries[name]["mean_cer"] - wide["mean_cer"]
        )
        summaries[name]["delta_median_cer_vs_wideband"] = float(
            summaries[name]["median_cer"] - wide["median_cer"]
        )

    report = {
        "schema_version": 1,
        "source_results": str(expand(args.wideband_results)),
        "set": args.set,
        "available_codecs": available_codecs(),
        "conditions": summaries,
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

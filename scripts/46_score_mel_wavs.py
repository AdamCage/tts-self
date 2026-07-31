#!/usr/bin/env python3
"""Score a mel demo wav dir (meta.json) with ASR → intelligibility JSON."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import soundfile as sf

from minispeech.eval.asr import AsrConfig, AsrEngine
from minispeech.eval.asr_wer import compute_cer, compute_wer
from minispeech.frontend.normalize import normalize_for_asr
from minispeech.infer.vocoder import vocoder_status
from minispeech.utils import expand


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--wav-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--asr-config", default="configs/asr.yaml")
    ap.add_argument("--label", default="mel_free_fixed")
    ap.add_argument("--before-cer", type=float, default=None)
    ap.add_argument("--oracle-cer", type=float, default=None)
    args = ap.parse_args()

    wav_dir = expand(args.wav_dir)
    meta = json.loads((wav_dir / "meta.json").read_text(encoding="utf-8"))
    engine = AsrEngine(AsrConfig.from_yaml(args.asr_config))
    rows = []
    for u in meta["utterances"]:
        wav_path = Path(u["wav"])
        x, sr = sf.read(str(wav_path), dtype="float32")
        if getattr(x, "ndim", 1) > 1:
            x = x.mean(axis=-1)
        hyp_raw = engine.transcribe(x, sr=sr)
        ref = normalize_for_asr(u["text"])
        hyp = normalize_for_asr(hyp_raw or "")
        peak = float(np.max(np.abs(x))) if len(x) else 0.0
        rms = float(np.sqrt(np.mean(np.square(x)))) if len(x) else 0.0
        flag = "ok"
        if peak < 1e-3 or rms < 1e-4:
            flag = "NEAR_SILENT"
        elif peak < 0.05 or rms < 0.005:
            flag = "WEAK"
        row = {
            "text": u["text"],
            "wav": str(wav_path),
            "cer": float(compute_cer([ref], [hyp])),
            "wer": float(compute_wer([ref], [hyp])),
            "hyp": hyp,
            "hyp_raw": hyp_raw or "",
            "n_frames": u.get("n_frames"),
            "n_sil": u.get("n_sil"),
            "n_phones": u.get("n_phones"),
            "sum_dur_s": u.get("sum_dur_s"),
            "sanity": {
                "path": str(wav_path),
                "peak": peak,
                "rms": rms,
                "flag": flag,
                "sr": int(sr),
            },
        }
        rows.append(row)
        print(f"cer={row['cer']:.3f} hyp={row['hyp'][:70]!r}", flush=True)

    cers = [r["cer"] for r in rows]
    wers = [r["wer"] for r in rows]
    out = {
        "label": args.label,
        "ckpt": meta.get("ckpt"),
        "duration_mode": meta.get("duration_mode"),
        "sil_policy": meta.get("sil_policy"),
        "forced_path": meta.get("forced_path"),
        "vocoder": vocoder_status(None),
        "packing": meta.get("gap_mode")
        or meta.get("sil_policy")
        or "mean_stats durations",
        "phrases": [r["text"] for r in rows],
        "mean_cer": float(np.mean(cers)) if cers else None,
        "mean_wer": float(np.mean(wers)) if wers else None,
        "n": len(rows),
        "n_near_silent": sum(1 for r in rows if r["sanity"]["flag"] == "NEAR_SILENT"),
        "utterances": rows,
    }
    if args.before_cer is not None:
        out["before_mean_cer"] = float(args.before_cer)
    if args.oracle_cer is not None:
        out["oracle_mean_cer_s1d"] = float(args.oracle_cer)
    out_path = expand(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"MEAN_CER={out['mean_cer']} MEAN_WER={out['mean_wer']} -> {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

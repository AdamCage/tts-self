#!/usr/bin/env python3
"""ASR-verify manifest texts; write asr_cer and drop high-CER rows optionally."""

from __future__ import annotations

import argparse
import json

import numpy as np
import soundfile as sf
import torch
from tqdm import tqdm

from minispeech.data.manifest import read_manifest, write_manifest
from minispeech.eval.asr_wer import compute_cer
from minispeech.frontend.stress import strip_stress
from minispeech.utils import expand, load_yaml


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/data_ruslan.yaml")
    ap.add_argument("--model", default="openai/whisper-large-v3")
    ap.add_argument("--max-utts", type=int, default=None)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    cfg = load_yaml(args.config)["data"]
    rows = read_manifest(cfg["manifest"])
    if args.max_utts:
        rows = rows[: args.max_utts]

    import whisper

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = whisper.load_model(args.model.replace("openai/", ""), device=device)

    kept = []
    for r in tqdm(rows, desc="asr-verify"):
        audio, sr = sf.read(r.wav, dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=-1)
        if sr != 16000:
            import torchaudio
            import torch as T

            audio = (
                torchaudio.functional.resample(T.from_numpy(audio).unsqueeze(0), sr, 16000)
                .squeeze(0)
                .numpy()
            )
        result = model.transcribe(audio, language="ru", fp16=device == "cuda")
        hyp = result.get("text", "").strip()
        ref = strip_stress(r.text_raw)
        try:
            r.asr_cer = compute_cer([ref], [hyp])
        except Exception:
            r.asr_cer = 1.0
        if r.asr_cer <= cfg.get("asr_cer_reject", 0.25):
            kept.append(r)

    out = expand(cfg["manifest"])
    write_manifest(out, kept)
    summary = {
        "before": len(rows),
        "after": len(kept),
        "mean_cer_kept": float(np.mean([r.asr_cer for r in kept if r.asr_cer is not None]) if kept else -1),
    }
    print(json.dumps(summary, indent=2))
    out.with_suffix(".asr.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

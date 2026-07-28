#!/usr/bin/env python3
"""Non-destructive corpus QC via ASR + DSP → corpus_qc.jsonl (resume-safe)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from tqdm import tqdm

from minispeech.data.manifest import read_manifest
from minispeech.eval.asr import AsrConfig, AsrEngine
from minispeech.eval.asr_wer import compute_cer, compute_wer
from minispeech.eval.bootstrap import edits_from_jiwer_chars, edits_from_jiwer_words
from minispeech.frontend.normalize import has_stress_marks, normalize_for_asr
from minispeech.utils import expand, load_yaml


def _energy_vad(audio: np.ndarray, sr: int, frame_ms: float = 20.0, thr_db: float = -40.0):
    frame = max(1, int(sr * frame_ms / 1000))
    n = len(audio) // frame
    if n == 0:
        return 0.0, 0.0, 1.0
    frames = audio[: n * frame].reshape(n, frame)
    rms = np.sqrt(np.mean(frames**2, axis=1) + 1e-12)
    db = 20 * np.log10(rms + 1e-12)
    active = db > thr_db
    if not active.any():
        return len(audio) / sr, 0.0, 1.0
    first = int(np.argmax(active))
    last = int(n - 1 - np.argmax(active[::-1]))
    lead = first * frame / sr
    trail = (n - 1 - last) * frame / sr
    silence_ratio = 1.0 - float(active.mean())
    return lead, trail, silence_ratio


def _dsp_features(audio: np.ndarray, sr: int) -> dict:
    if audio.ndim > 1:
        audio = audio.mean(axis=-1)
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    rms = float(np.sqrt(np.mean(audio**2))) if audio.size else 0.0
    clip_frac = float(np.mean(np.abs(audio) > 0.99)) if audio.size else 0.0
    lead, trail, sil = _energy_vad(audio, sr)
    return {
        "dur_s": float(len(audio) / max(sr, 1)),
        "rms": rms,
        "peak": peak,
        "clip_frac": clip_frac,
        "lead_sil_s": lead,
        "trail_sil_s": trail,
        "silence_ratio": sil,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/data_ruslan.yaml")
    ap.add_argument("--asr-config", default="configs/asr.yaml")
    ap.add_argument("--out", default=None, help="Sidecar jsonl path")
    ap.add_argument("--max-utts", type=int, default=None)
    ap.add_argument("--flush-every", type=int, default=20)
    ap.add_argument("--backend", default=None, help="Override asr.backend")
    args = ap.parse_args()
    cfg = load_yaml(args.config)["data"]
    rows = read_manifest(cfg["manifest"])
    if args.max_utts:
        rows = rows[: args.max_utts]

    asr_cfg = AsrConfig.from_yaml(args.asr_config)
    if args.backend:
        asr_cfg.backend = args.backend
    elif asr_cfg.backend == "openai_whisper":
        # corpus QC prefers faster_whisper when available
        try:
            import faster_whisper  # noqa: F401

            asr_cfg.backend = "faster_whisper"
        except Exception:
            pass

    out_path = expand(args.out or str(Path(cfg.get("ruslan_dir", "~/datasets/tts/ruslan")).expanduser() / "corpus_qc.jsonl"))
    out_path.parent.mkdir(parents=True, exist_ok=True)

    done: set[str] = set()
    if out_path.exists():
        with open(out_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    done.add(json.loads(line)["utt_id"])
                except Exception:
                    continue
        print(f"Resume: {len(done)} already in {out_path}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    asr = AsrEngine(asr_cfg, device=device)

    pending = [r for r in rows if r.utt_id not in done]
    buf: list[str] = []
    with open(out_path, "a", encoding="utf-8") as fout:
        for r in tqdm(pending, desc="corpus-qc"):
            try:
                audio, sr = sf.read(r.wav, dtype="float32")
            except Exception as e:
                rec = {"utt_id": r.utt_id, "error": str(e)}
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                continue
            if audio.ndim > 1:
                audio = audio.mean(axis=-1)
            dsp = _dsp_features(audio, sr)
            ref_n = normalize_for_asr(r.text_raw)
            hyp = normalize_for_asr(asr.transcribe(audio, sr))
            try:
                cer = compute_cer([ref_n], [hyp])
                wer = compute_wer([ref_n], [hyp])
            except Exception:
                cer, wer = 1.0, 1.0
            ew = edits_from_jiwer_words(ref_n, hyp)
            ec = edits_from_jiwer_chars(ref_n, hyp)
            chars_per_sec = len(ref_n.replace(" ", "")) / max(dsp["dur_s"], 1e-6)
            rec = {
                "utt_id": r.utt_id,
                "wav": r.wav,
                "asr_cer": cer,
                "asr_wer": wer,
                "hyp": hyp,
                "ref_norm": ref_n,
                "chars_per_sec": chars_per_sec,
                "has_digits": any(c.isdigit() for c in r.text_raw),
                "has_latin": bool(__import__("re").search(r"[A-Za-z]", r.text_raw)),
                "has_stress_marks": has_stress_marks(r.text_stressed or r.text_raw),
                "edits_words": {
                    "sub": ew.substitutions,
                    "del": ew.deletions,
                    "ins": ew.insertions,
                    "hits": ew.hits,
                },
                "edits_chars": {
                    "sub": ec.substitutions,
                    "del": ec.deletions,
                    "ins": ec.insertions,
                    "hits": ec.hits,
                },
                **dsp,
            }
            line = json.dumps(rec, ensure_ascii=False)
            fout.write(line + "\n")
            buf.append(line)
            if len(buf) >= args.flush_every:
                fout.flush()
                buf.clear()
        fout.flush()

    print(json.dumps({"qc_path": str(out_path), "new": len(pending), "total_done": len(done) + len(pending)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

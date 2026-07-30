#!/usr/bin/env python3
"""Four-point mel-domain / vocoder probe (roadmap T03).

On n val utterances, compare:
  A) real wav @ 22050
  B) mel@11025 (mel80) → Griffin-Lim
  C) mel@11025 → HiFi-GAN
  D) mel@fmax_vocoder (default 8000) → HiFi-GAN

Does **not** mutate MEL80 or shared mel helpers — variant D uses a local
librosa call. ASR scoring uses ``normalize_verbalized_for_asr``.

Example (WSL)::

    python scripts/34_vocoder_domain_probe.py --n 40 \\
      --out out/vocoder_domain_probe.json
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import torch

from minispeech.constants import MEL80
from minispeech.data.manifest import filter_split, read_manifest
from minispeech.data.mel import load_wav_mono, mel_backend_name, waveform_to_logmel
from minispeech.eval.asr import AsrConfig, AsrEngine
from minispeech.eval.asr_wer import compute_cer, compute_wer
from minispeech.eval.optional_metrics import score_utmos_v2
from minispeech.frontend.verbalize import normalize_verbalized_for_asr
from minispeech.infer.vocoder import griffin_lim, mel_to_wav, vocoder_status
from minispeech.utils import expand, load_yaml


def _git_sha() -> str | None:
    try:
        out = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True)
        return out.strip()
    except Exception:
        return None


def _agg(rows: list[dict[str, Any]], key: str) -> dict[str, float | None]:
    vals = [float(r[key]) for r in rows if r.get(key) is not None]
    if not vals:
        return {"mean": None, "median": None, "n": 0}
    arr = np.asarray(vals, dtype=np.float64)
    return {
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "n": int(arr.size),
    }


def _logmel_fmax(
    wav: np.ndarray,
    *,
    sr: int,
    fmax: float,
    n_fft: int = 1024,
    hop_length: int = 256,
    win_length: int = 1024,
    n_mels: int = 80,
    fmin: float = 0.0,
) -> torch.Tensor:
    """Local mel @ custom fmax — does not touch project MEL80 helpers."""
    import librosa

    mel = librosa.feature.melspectrogram(
        y=np.asarray(wav, dtype=np.float32),
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        n_mels=n_mels,
        fmin=fmin,
        fmax=float(fmax),
        power=1.0,
    )
    logmel = np.log(np.maximum(mel, 1e-5)).T
    return torch.from_numpy(logmel.astype(np.float32))


def _asr_score(engine: AsrEngine, wav: np.ndarray, sr: int, text: str) -> dict[str, Any]:
    hyp_raw = engine.transcribe(wav, sr=sr) or ""
    ref = normalize_verbalized_for_asr(text)
    hyp = normalize_verbalized_for_asr(hyp_raw)
    return {
        "cer": float(compute_cer([ref], [hyp])),
        "wer": float(compute_wer([ref], [hyp])),
        "hyp": hyp,
        "hyp_raw": hyp_raw,
        "ref_norm": ref,
    }


def _utmos(wav: np.ndarray, sr: int) -> float | None:
    res = score_utmos_v2(wav, sr)
    if res.get("skipped"):
        return None
    return res.get("value")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default="configs/data_ruslan_local.yaml")
    ap.add_argument("--asr-config", default="configs/asr.yaml")
    ap.add_argument("--split", default="val", choices=["train", "val", "test"])
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--vocoder-ckpt", default="~/models/hifigan/generator.pt")
    ap.add_argument("--fmax-vocoder", type=float, default=8000.0)
    ap.add_argument("--out", default="out/vocoder_domain_probe.json")
    ap.add_argument("--wav-dir", default="out/vocoder_domain_probe_wavs")
    ap.add_argument("--save-wav-n", type=int, default=5)
    ap.add_argument("--skip-utmos", action="store_true")
    args = ap.parse_args()

    dcfg = load_yaml(args.data).get("data", {})
    manifest = expand(dcfg["manifest"])
    rows = filter_split(read_manifest(manifest), args.split)
    excl = set(dcfg.get("exclude_tags", ("exclude",)))
    if excl:
        rows = [r for r in rows if (r.qc_tag or "clean") not in excl]

    # Prefer utterances that also have mel cache (same contour as script 30).
    mel_dir = expand(dcfg.get("mel_dir") or "")
    mel_ids: set[str] = set()
    if mel_dir and (mel_dir / "mels_index.json").is_file():
        idx = json.loads((mel_dir / "mels_index.json").read_text(encoding="utf-8"))
        mel_ids = set(idx.keys())
    forced_path = expand(dcfg.get("forced_path") or "")
    forced_ids: set[str] = set()
    if forced_path.is_file():
        forced_ids = set(json.loads(forced_path.read_text(encoding="utf-8")).keys())

    candidates = []
    for u in rows:
        if mel_ids and u.utt_id not in mel_ids:
            continue
        if forced_ids and u.utt_id not in forced_ids:
            continue
        candidates.append(u)
    if not candidates:
        candidates = list(rows)
    candidates = candidates[: args.n]
    if len(candidates) < args.n:
        print(f"warn: only {len(candidates)} candidates (requested {args.n})", flush=True)

    vocoder_ckpt = expand(args.vocoder_ckpt)
    engine = AsrEngine(AsrConfig.from_yaml(args.asr_config))
    sr_tgt = int(MEL80.sample_rate)
    wav_dir = expand(args.wav_dir)
    wav_dir.mkdir(parents=True, exist_ok=True)

    variants = ("A_real", "B_mel11025_griffin", "C_mel11025_hifigan", "D_mel8000_hifigan")
    per_variant: dict[str, list[dict[str, Any]]] = {v: [] for v in variants}

    for i, u in enumerate(candidates):
        text = u.text_stressed or u.text_raw
        wav, sr = load_wav_mono(u.wav, target_sr=sr_tgt)
        # A: real
        score_a = _asr_score(engine, wav, sr_tgt, text)
        utmos_a = None if args.skip_utmos else _utmos(wav, sr_tgt)
        row_a = {"utt_id": u.utt_id, "text": text, **score_a, "utmos": utmos_a}
        per_variant["A_real"].append(row_a)

        # B/C: project mel80 (fmax=11025)
        mel11025 = waveform_to_logmel(wav, target="mel80", sample_rate=sr_tgt)
        gl = griffin_lim(mel11025, target="mel80")
        score_b = _asr_score(engine, gl, sr_tgt, text)
        utmos_b = None if args.skip_utmos else _utmos(gl, sr_tgt)
        per_variant["B_mel11025_griffin"].append(
            {"utt_id": u.utt_id, "text": text, **score_b, "utmos": utmos_b}
        )

        neural = mel_to_wav(mel11025, prefer="neural", vocoder_ckpt=vocoder_ckpt, target="mel80")
        if neural.wav is None:
            per_variant["C_mel11025_hifigan"].append(
                {"utt_id": u.utt_id, "error": neural.note, "cer": None, "wer": None, "utmos": None}
            )
            wav_c = None
        else:
            score_c = _asr_score(engine, neural.wav, neural.sample_rate, text)
            utmos_c = None if args.skip_utmos else _utmos(neural.wav, neural.sample_rate)
            per_variant["C_mel11025_hifigan"].append(
                {
                    "utt_id": u.utt_id,
                    "text": text,
                    **score_c,
                    "utmos": utmos_c,
                    "vocoder_backend": neural.backend,
                }
            )
            wav_c = neural.wav

        # D: local mel @ vocoder fmax
        mel8000 = _logmel_fmax(wav, sr=sr_tgt, fmax=float(args.fmax_vocoder))
        neural_d = mel_to_wav(mel8000, prefer="neural", vocoder_ckpt=vocoder_ckpt, target="mel80")
        if neural_d.wav is None:
            per_variant["D_mel8000_hifigan"].append(
                {"utt_id": u.utt_id, "error": neural_d.note, "cer": None, "wer": None, "utmos": None}
            )
            wav_d = None
        else:
            score_d = _asr_score(engine, neural_d.wav, neural_d.sample_rate, text)
            utmos_d = None if args.skip_utmos else _utmos(neural_d.wav, neural_d.sample_rate)
            per_variant["D_mel8000_hifigan"].append(
                {
                    "utt_id": u.utt_id,
                    "text": text,
                    **score_d,
                    "utmos": utmos_d,
                    "vocoder_backend": neural_d.backend,
                    "fmax": float(args.fmax_vocoder),
                }
            )
            wav_d = neural_d.wav

        if i < args.save_wav_n:
            sf.write(str(wav_dir / f"{i+1:02d}_{u.utt_id}_B_griffin.wav"), gl, sr_tgt)
            if wav_c is not None:
                sf.write(str(wav_dir / f"{i+1:02d}_{u.utt_id}_C_hifigan11025.wav"), wav_c, sr_tgt)
            if wav_d is not None:
                sf.write(str(wav_dir / f"{i+1:02d}_{u.utt_id}_D_hifigan{int(args.fmax_vocoder)}.wav"), wav_d, sr_tgt)

        print(
            f"[{i+1}/{len(candidates)}] {u.utt_id} "
            f"A_cer={row_a['cer']:.3f} B_cer={score_b['cer']:.3f} "
            f"C_cer={per_variant['C_mel11025_hifigan'][-1].get('cer')} "
            f"D_cer={per_variant['D_mel8000_hifigan'][-1].get('cer')}",
            flush=True,
        )

    aggregates = {}
    for name, rows_v in per_variant.items():
        aggregates[name] = {
            "cer": _agg(rows_v, "cer"),
            "wer": _agg(rows_v, "wer"),
            "utmos": _agg(rows_v, "utmos"),
        }

    payload = {
        "meta": {
            "git_sha": _git_sha(),
            "mel_backend_name": mel_backend_name(),
            "vocoder_ckpt": str(vocoder_ckpt),
            "vocoder_status": vocoder_status(vocoder_ckpt),
            "split": args.split,
            "n_requested": args.n,
            "n_scored": len(candidates),
            "fmax_project_mel80": float(MEL80.sample_rate) / 2.0,
            "fmax_vocoder_probe": float(args.fmax_vocoder),
            "asr_norm": "normalize_verbalized_for_asr",
            "self_check": (
                "B_mel11025_griffin mean CER should be near 0.027 "
                "(GT mel → griffin topline on train n=20); val may differ slightly."
            ),
        },
        "aggregates": aggregates,
        "utterances": per_variant,
    }
    out_path = expand(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"wrote": str(out_path), "aggregates": aggregates}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

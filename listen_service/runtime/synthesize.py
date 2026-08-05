"""Timed free-text synthesize → wav for the listen UI."""

from __future__ import annotations

import re
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from runtime.bundle import ListenBundle
from runtime.text import prepare_text

SERVICE_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = SERVICE_ROOT / "out"


def _sync(device: torch.device) -> None:
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(device)


def _device_meta(device: torch.device) -> dict[str, Any]:
    out: dict[str, Any] = {
        "device": str(device),
        "cuda_available": bool(torch.cuda.is_available()),
        "gpu_name": None,
    }
    if device.type == "cuda" and torch.cuda.is_available():
        out["gpu_name"] = torch.cuda.get_device_name(device)
    return out


def _vram_meta(device: torch.device) -> dict[str, Any]:
    if device.type != "cuda" or not torch.cuda.is_available():
        return {}
    return {
        "peak_allocated_mb": round(torch.cuda.max_memory_allocated(device) / (1024**2), 2),
        "peak_reserved_mb": round(torch.cuda.max_memory_reserved(device) / (1024**2), 2),
    }


def _ascii_slug(text: str, n: int = 40) -> str:
    s = re.sub(r"\s+", "_", text.strip().lower())
    s = re.sub(r"[^a-z0-9_\-]+", "", s)
    return (s[:n] or "utt").rstrip("_")


def tokenize(bundle: ListenBundle, stressed: str) -> torch.Tensor:
    enc = bundle.tokenizer(
        stressed,
        add_special_tokens=False,
        return_tensors="pt",
    )
    ids = enc["input_ids"][0].long()
    if ids.numel() == 0:
        raise ValueError("tokenizer produced empty input_ids")
    return ids


def synthesize(
    bundle: ListenBundle,
    text: str,
    *,
    speaker_id: int | None = None,
    save_wav: bool = True,
    out_dir: Path | None = None,
) -> tuple[np.ndarray, int, dict[str, Any]]:
    """Run frontend → tokenize → generate → decode; return wav + metrics."""
    t0 = time.perf_counter()
    device = bundle.device
    spk = int(bundle.default_speaker_id if speaker_id is None else speaker_id)

    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)

    t_fe0 = time.perf_counter()
    stressed = prepare_text(text)
    frontend_ms = (time.perf_counter() - t_fe0) * 1000.0

    t_tok0 = time.perf_counter()
    ids = tokenize(bundle, stressed)
    tokenize_ms = (time.perf_counter() - t_tok0) * 1000.0

    input_ids = ids.unsqueeze(0).to(device)
    speaker_t = torch.tensor([spk], device=device, dtype=torch.long)

    _sync(device)
    t_gen0 = time.perf_counter()
    with torch.no_grad():
        codes = bundle.model.generate(input_ids, speaker_id=speaker_t, **bundle.gen_kw)
    _sync(device)
    generate_ms = (time.perf_counter() - t_gen0) * 1000.0

    _sync(device)
    t_dec0 = time.perf_counter()
    dec = bundle.codec.decode([codes[0].detach().cpu()])
    _sync(device)
    decode_ms = (time.perf_counter() - t_dec0) * 1000.0

    wav = np.asarray(dec.audio[0], dtype=np.float32)
    sr = int(dec.sample_rate)
    audio_s = float(wav.shape[0]) / float(sr) if sr > 0 else 0.0
    total_ms = (time.perf_counter() - t0) * 1000.0
    rtf = (total_ms / 1000.0) / audio_s if audio_s > 0 else None

    meta: dict[str, Any] = {
        **_device_meta(device),
        **_vram_meta(device),
        "ckpt": str(bundle.ckpt_path),
        "tokenizer_id": bundle.tokenizer_id,
        "codec_model_id": bundle.codec.model_id,
        "speaker_id": spk,
        "stressed_text": stressed,
        "n_text_tokens": int(ids.numel()),
        "n_frames": int(codes.shape[1]),
        "n_codebooks": int(codes.shape[2]) if codes.ndim == 3 else None,
        "sample_rate": sr,
        "audio_s": round(audio_s, 4),
        "frontend_ms": round(frontend_ms, 2),
        "tokenize_ms": round(tokenize_ms, 2),
        "generate_ms": round(generate_ms, 2),
        "decode_ms": round(decode_ms, 2),
        "total_ms": round(total_ms, 2),
        "rtf": round(rtf, 4) if rtf is not None else None,
        "infer": dict(bundle.gen_kw),
        "wav_path": None,
    }

    if save_wav:
        dest = Path(out_dir) if out_dir is not None else OUT_DIR
        dest.mkdir(parents=True, exist_ok=True)
        stem = _ascii_slug(stressed)
        wav_path = dest / f"{stem}_{spk}_{device.type}.wav"
        try:
            import soundfile as sf

            sf.write(str(wav_path), wav, sr)
        except Exception:
            import wave

            pcm = np.clip(wav, -1.0, 1.0)
            pcm_i16 = (pcm * 32767.0).astype(np.int16)
            with wave.open(str(wav_path), "wb") as w:
                w.setnchannels(1)
                w.setsampwidth(2)
                w.setframerate(sr)
                w.writeframes(pcm_i16.tobytes())
        meta["wav_path"] = str(wav_path)

    return wav, sr, meta

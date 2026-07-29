"""Autoregressive generation + streaming decode."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import torch

from minispeech.codec.stats import LatentNormalizer
from minispeech.codec.wrapper import VibeVoiceCodec
from minispeech.constants import SR
from minispeech.frontend.normalize import normalize_text
from minispeech.frontend.stress import accentuate
from minispeech.infer.stream import fill_stream_timing
from minispeech.models.minispeech import MiniSpeech
from minispeech.utils import expand


def load_checkpoint(
    path: str | Path,
    device: torch.device,
    *,
    require_talker: bool = True,
) -> tuple[MiniSpeech, dict]:
    ckpt = torch.load(expand(path), map_location=device, weights_only=False)
    cfg = ckpt.get("cfg", {})
    mode = "flow" if "head_flow" in cfg.get("model", {}) or cfg.get("mode") == "flow" else "regression"
    # Infer mode from state dict keys
    sd = ckpt.get("ema") or ckpt["model"]
    if any(k.startswith("head.blocks") or k.startswith("head.time") for k in sd):
        mode = "flow"
    from minispeech.train.trainer import build_model_from_cfg

    model = build_model_from_cfg(cfg if cfg else {"model": {}}, mode=mode)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(
        f"load_checkpoint: loaded={len(sd)} missing={len(missing)} unexpected={len(unexpected)}"
    )
    if missing:
        print(f"  missing (first 10): {missing[:10]}")
    if unexpected:
        print(f"  unexpected (first 10): {unexpected[:10]}")
    if require_talker:
        talker_missing = [k for k in missing if k.startswith("talker.")]
        if talker_missing:
            raise RuntimeError(
                f"Checkpoint missing Talker keys ({len(talker_missing)}); "
                "refusing silent random init. Example: "
                f"{talker_missing[:5]}"
            )
    model.to(device).eval()
    return model, ckpt


@torch.inference_mode()
def cache_text_hidden_online(
    texts: list[str],
    tokenizer,
    text_model,
    layer_indices: tuple[int, ...] = (11, 17, 23),
    device: torch.device | None = None,
    max_length: int = 256,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return text_hidden [B, n_layers, L, H] and text_mask [B, L]."""
    device = device or next(text_model.parameters()).device
    enc = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
    )
    enc = {k: v.to(device) for k, v in enc.items()}
    out = text_model(**enc, output_hidden_states=True)
    # hidden_states[0] = embed; hidden_states[i+1] = after layer i
    hs = out.hidden_states
    layers = []
    for idx in layer_indices:
        layers.append(hs[idx + 1])
    stacked = torch.stack(layers, dim=1)  # [B, n, L, H]
    return stacked, enc["attention_mask"].bool()


def synthesize(
    text: str,
    model: MiniSpeech,
    codec: VibeVoiceCodec,
    normalizer: LatentNormalizer,
    tokenizer,
    text_model,
    *,
    cfg_scale: float = 1.3,
    flow_steps: int = 4,
    use_stress: bool = True,
    stop: dict[str, Any] | None = None,
    stream: bool = False,
    chunk_latents: int = 4,
    min_pcm_ms: float = 40.0,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Synthesize audio. Prefer ``stream=True`` for real TTFB marks."""
    if stream:
        return synthesize_streaming(
            text,
            model,
            codec,
            normalizer,
            tokenizer,
            text_model,
            cfg_scale=cfg_scale,
            flow_steps=flow_steps,
            use_stress=use_stress,
            stop=stop,
            chunk_latents=chunk_latents,
            min_pcm_ms=min_pcm_ms,
        )
    text_n = normalize_text(text)
    text_s = accentuate(text_n) if use_stress else text_n
    # TTFB clock starts after text normalization / stress (protocol).
    t_start = time.perf_counter()
    model_dtype = next(model.parameters()).dtype
    device = next(model.parameters()).device
    th, mask = cache_text_hidden_online([text_s], tokenizer, text_model, device=device)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t_qwen = time.perf_counter()
    th = th.to(dtype=model_dtype)
    stop_cfg = stop or {}
    z_norm, info = model.generate(
        th,
        mask,
        cfg_scale=cfg_scale,
        flow_steps=flow_steps,
        stop_threshold=float(stop_cfg.get("threshold", 0.45)),
        stop_consecutive=int(stop_cfg.get("consecutive", 3)),
        min_frac=float(stop_cfg.get("min_frac", 0.94)),
        max_frac=float(stop_cfg.get("max_frac", 1.6)),
        duration_scale=float(stop_cfg.get("duration_scale", 1.04)),
    )
    z = normalizer.denormalize(z_norm)
    audio = codec.decode(z).audio.squeeze().float().cpu().numpy()
    if device.type == "cuda":
        torch.cuda.synchronize()
    t_end = time.perf_counter()
    info["text_stressed"] = text_s
    info["qwen_ms"] = (t_qwen - t_start) * 1000.0
    # Non-streaming path: first_pcm is end of full decode (not a streaming TTFB).
    fill_stream_timing(info, t_start=t_start, t_text_done=t_qwen, t_first_pcm=t_end)
    info["estimated"] = True
    return audio, info


@torch.inference_mode()
def synthesize_streaming(
    text: str,
    model: MiniSpeech,
    codec: VibeVoiceCodec,
    normalizer: LatentNormalizer,
    tokenizer,
    text_model,
    *,
    cfg_scale: float = 1.3,
    flow_steps: int = 4,
    use_stress: bool = True,
    stop: dict[str, Any] | None = None,
    chunk_latents: int = 4,
    min_pcm_ms: float = 40.0,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Interleaved AR + cached causal decode; stamps real ``first_pcm_ms`` (≥ min_pcm_ms)."""
    text_n = normalize_text(text)
    text_s = accentuate(text_n) if use_stress else text_n
    t_start = time.perf_counter()
    model_dtype = next(model.parameters()).dtype
    device = next(model.parameters()).device
    th, mask = cache_text_hidden_online([text_s], tokenizer, text_model, device=device)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t_qwen = time.perf_counter()
    th = th.to(dtype=model_dtype)
    stop_cfg = stop or {}

    chunk = max(1, int(chunk_latents))
    min_samples = max(1, int(round(min_pcm_ms / 1000.0 * SR)))
    pending: list[torch.Tensor] = []
    pcm_chunks: list[np.ndarray] = []
    pcm_samples = 0
    cache = None
    t_first_latent: float | None = None
    t_first_pcm: float | None = None
    t_codec0: float | None = None
    codec_first_ms = 0.0
    last: dict[str, Any] = {}

    for z_t, step in model.iter_generate(
        th,
        mask,
        cfg_scale=cfg_scale,
        flow_steps=flow_steps,
        stop_threshold=float(stop_cfg.get("threshold", 0.45)),
        stop_consecutive=int(stop_cfg.get("consecutive", 3)),
        min_frac=float(stop_cfg.get("min_frac", 0.94)),
        max_frac=float(stop_cfg.get("max_frac", 1.6)),
        duration_scale=float(stop_cfg.get("duration_scale", 1.04)),
    ):
        last = step
        if t_first_latent is None and step.get("first_latent_ms", 0) > 0:
            t_first_latent = time.perf_counter()
        pending.append(z_t)
        flush = step.get("should_stop", False) or len(pending) >= chunk
        if not flush:
            continue
        piece_norm = torch.cat(pending, dim=1)
        pending.clear()
        piece = normalizer.denormalize(piece_norm)
        if t_codec0 is None:
            t_codec0 = time.perf_counter()
        out = codec.decode(piece, use_cache=True, padding_cache=cache)
        cache = out.padding_cache
        wav = out.audio.squeeze().float().cpu().numpy()
        pcm_chunks.append(wav)
        pcm_samples += int(wav.shape[0])
        if t_first_pcm is None and pcm_samples >= min_samples:
            if device.type == "cuda":
                torch.cuda.synchronize()
            t_first_pcm = time.perf_counter()
            if t_codec0 is not None:
                codec_first_ms = (t_first_pcm - t_codec0) * 1000.0

    if pending:
        piece_norm = torch.cat(pending, dim=1)
        piece = normalizer.denormalize(piece_norm)
        if t_codec0 is None:
            t_codec0 = time.perf_counter()
        out = codec.decode(piece, use_cache=True, padding_cache=cache)
        wav = out.audio.squeeze().float().cpu().numpy()
        pcm_chunks.append(wav)
        pcm_samples += int(wav.shape[0])
        if t_first_pcm is None and pcm_samples >= min_samples:
            if device.type == "cuda":
                torch.cuda.synchronize()
            t_first_pcm = time.perf_counter()
            if t_codec0 is not None:
                codec_first_ms = (t_first_pcm - t_codec0) * 1000.0

    if not pcm_chunks:
        audio = np.zeros(0, dtype=np.float32)
    else:
        audio = np.concatenate(pcm_chunks, axis=0)

    info = {
        "dur_pred": last.get("dur_pred", 0),
        "dur_pred_raw": last.get("dur_pred_raw", 0.0),
        "duration_scale": last.get("duration_scale", 1.0),
        "stop_probs": last.get("stop_probs", []),
        "n_frames": int(last.get("n_frames", 0)),
        "stop_triggered": bool(last.get("should_stop", False)),
        "hit_max_frames": bool(last.get("hit_max_frames", False)),
        "talker_text_ms": float(last.get("talker_text_ms", 0.0)),
        "first_latent_ms_talker": float(last.get("first_latent_ms", 0.0)),
        "sampler_ms": float(last.get("sampler_ms", 0.0)),
        "codec_first_ms": codec_first_ms,
        "chunk_latents": chunk,
        "min_pcm_ms": float(min_pcm_ms),
        "text_stressed": text_s,
        "qwen_ms": (t_qwen - t_start) * 1000.0,
        "estimated": False,
        "stream": True,
    }
    fill_stream_timing(
        info,
        t_start=t_start,
        t_text_done=t_qwen,
        t_first_latent=t_first_latent,
        t_first_pcm=t_first_pcm,
    )
    # Protocol text_encode_ms = Qwen (post-normalization), not Talker projector.
    info["text_encode_ms"] = info["qwen_ms"]
    info["talker_ms"] = float(info.get("first_latent_ms", 0.0)) - float(info["qwen_ms"])
    return audio, info


def stream_decode_latents(
    codec: VibeVoiceCodec,
    latents: torch.Tensor,
    chunk: int = 4,
) -> Iterator[np.ndarray]:
    """Cached causal decode in latent chunks. latents [1,T,64] raw."""
    cache = None
    T = latents.shape[1]
    for i in range(0, T, chunk):
        piece = latents[:, i : i + chunk]
        out = codec.decode(piece, use_cache=True, padding_cache=cache)
        cache = out.padding_cache
        yield out.audio.squeeze().float().cpu().numpy()


def save_wav(path: str | Path, audio: np.ndarray, sr: int = SR) -> None:
    path = expand(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    import soundfile as sf

    arr = np.asarray(audio, dtype=np.float32)
    sf.write(str(path), arr, sr)

"""Autoregressive generation + streaming decode."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterator

import numpy as np
import torch

from minispeech.codec.stats import LatentNormalizer
from minispeech.codec.wrapper import VibeVoiceCodec
from minispeech.constants import SR
from minispeech.frontend.normalize import normalize_text
from minispeech.frontend.stress import accentuate
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
) -> tuple[np.ndarray, dict[str, Any]]:
    text_n = normalize_text(text)
    text_s = accentuate(text_n) if use_stress else text_n
    model_dtype = next(model.parameters()).dtype
    th, mask = cache_text_hidden_online([text_s], tokenizer, text_model, device=next(model.parameters()).device)
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
    info["text_stressed"] = text_s
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

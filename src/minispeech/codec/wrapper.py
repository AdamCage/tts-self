"""Frozen VibeVoice Acoustic Tokenizer wrapper."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
try:
    import torchaudio
except ImportError:  # optional on Windows authoring hosts without torchaudio
    torchaudio = None  # type: ignore

from minispeech.constants import CODEC_MODEL_ID, LATENT_DIM, SAMPLES_PER_FRAME, SR


@dataclass
class EncodeResult:
    latents: torch.Tensor  # [B, T, 64] canonical layout for this project
    n_frames_true: list[int]
    raw: Any = None


@dataclass
class DecodeResult:
    audio: torch.Tensor  # [B, 1, N]
    padding_cache: Any = None
    raw: Any = None


def _to_bt64(latents: torch.Tensor) -> torch.Tensor:
    """Normalize latent layout to [B, T, 64]."""
    if latents.ndim != 3:
        raise ValueError(f"Expected 3D latents, got {tuple(latents.shape)}")
    if latents.shape[-1] == LATENT_DIM:
        return latents
    if latents.shape[1] == LATENT_DIM:
        return latents.transpose(1, 2).contiguous()
    raise ValueError(
        f"Cannot interpret latents shape {tuple(latents.shape)} as [B,T,64] or [B,64,T]"
    )


def _to_model_layout(latents: torch.Tensor, layout: str) -> torch.Tensor:
    latents = _to_bt64(latents)
    if layout == "btd":
        return latents
    if layout == "bdt":
        return latents.transpose(1, 2).contiguous()
    raise ValueError(layout)


class VibeVoiceCodec:
    """Encode/decode with sample=False and optional padding_cache streaming."""

    def __init__(
        self,
        model_id: str = CODEC_MODEL_ID,
        device: str | torch.device | None = None,
        dtype: torch.dtype = torch.bfloat16,
    ):
        import os
        from transformers import AutoFeatureExtractor, VibeVoiceAcousticTokenizerModel

        # Prefer explicit local checkout if present
        local = os.path.expanduser("~/datasets/tts/models/VibeVoice-AcousticTokenizer")
        if model_id == CODEC_MODEL_ID and os.path.isdir(local):
            model_id = local

        self.model_id = model_id
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.dtype = dtype
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(model_id)
        self.model = VibeVoiceAcousticTokenizerModel.from_pretrained(model_id)
        self.model.to(self.device, dtype=dtype)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

        # Probe layout once
        self._latent_layout = self._detect_layout()

    def _detect_layout(self) -> str:
        """Return 'btd' if encode yields [B,T,64], else 'bdt'."""
        n = SAMPLES_PER_FRAME * 4
        wav = torch.zeros(1, 1, n, device=self.device, dtype=self.dtype)
        with torch.no_grad():
            out = self.model.encode(wav, sample=False)
            z = out.latents
        if z.shape[-1] == LATENT_DIM:
            return "btd"
        if z.shape[1] == LATENT_DIM:
            return "bdt"
        raise RuntimeError(f"Unexpected latent shape from codec: {tuple(z.shape)}")

    @torch.inference_mode()
    def preprocess(
        self,
        audio: np.ndarray | torch.Tensor,
        sr: int,
    ) -> torch.Tensor:
        """Resample to 24 kHz in memory; feature extractor handles -25 dBFS."""
        if isinstance(audio, np.ndarray):
            audio_t = torch.from_numpy(np.asarray(audio, dtype=np.float32))
        else:
            audio_t = audio.float().cpu()
        if audio_t.ndim > 1:
            audio_t = audio_t.mean(dim=0) if audio_t.shape[0] <= 8 else audio_t.squeeze()
        audio_t = audio_t.flatten()
        # DC offset removal only (no loudness norm — FE does it)
        audio_t = audio_t - audio_t.mean()
        if sr != SR:
            if torchaudio is None:
                raise RuntimeError("torchaudio required to resample to 24 kHz for VibeVoice")
            audio_t = torchaudio.functional.resample(audio_t.unsqueeze(0), sr, SR).squeeze(0)
        arr = audio_t.numpy()
        inputs = self.feature_extractor(
            arr,
            sampling_rate=SR,
            pad_to_multiple_of=SAMPLES_PER_FRAME,
            return_tensors="pt",
        )
        return inputs.input_values.to(self.device, self.dtype)

    @torch.inference_mode()
    def encode(
        self,
        audio: np.ndarray | torch.Tensor,
        sr: int,
        *,
        sample: bool = False,
        use_cache: bool = False,
        padding_cache=None,
    ) -> EncodeResult:
        if sample:
            raise ValueError("Training/TTS encoding must use sample=False (deterministic mean)")
        n_src = int(np.asarray(audio).size) if not torch.is_tensor(audio) else int(audio.numel())
        # Approximate true frames after resample
        if sr != SR:
            n_24 = int(round(n_src * SR / sr))
        else:
            n_24 = n_src
        n_true = (n_24 + SAMPLES_PER_FRAME - 1) // SAMPLES_PER_FRAME

        input_values = self.preprocess(audio, sr)
        kwargs = {"sample": False}
        if use_cache:
            kwargs["use_cache"] = True
            kwargs["padding_cache"] = padding_cache
        out = self.model.encode(input_values, **kwargs)
        latents = _to_bt64(out.latents)
        assert latents.shape[-1] == LATENT_DIM, latents.shape
        return EncodeResult(latents=latents, n_frames_true=[n_true], raw=out)

    @torch.inference_mode()
    def decode(
        self,
        latents: torch.Tensor,
        *,
        use_cache: bool = False,
        padding_cache=None,
    ) -> DecodeResult:
        z = _to_model_layout(latents.to(self.device, self.dtype), self._latent_layout)
        kwargs = {}
        if use_cache:
            kwargs["use_cache"] = True
            kwargs["padding_cache"] = padding_cache
        out = self.model.decode(z, **kwargs)
        return DecodeResult(audio=out.audio, padding_cache=getattr(out, "padding_cache", None), raw=out)

    @torch.inference_mode()
    def decode_chunked(self, latents: torch.Tensor, *, chunk_frames: int = 4) -> DecodeResult:
        """Streaming decode via padding_cache; audio layout matches full ``decode``."""
        from minispeech.codec.cache_check import decode_chunked as _decode_chunked

        audio = _decode_chunked(self, latents, chunk_frames=chunk_frames)
        return DecodeResult(audio=audio, padding_cache=None, raw=None)

    @torch.inference_mode()
    def encode_tensor(self, input_values: torch.Tensor, *, sample: bool = False) -> torch.Tensor:
        out = self.model.encode(input_values.to(self.device, self.dtype), sample=sample)
        return _to_bt64(out.latents)

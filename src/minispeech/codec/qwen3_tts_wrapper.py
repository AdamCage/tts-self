"""Frozen Qwen3-TTS Tokenizer-12Hz wrapper (encode / decode only).

Never trains or fine-tunes codec weights. Geometry is registered in
``minispeech.constants`` as ``qwen3tts_12hz``.

Dependency note
---------------
``qwen-tts`` currently requires ``transformers~=4.57`` and conflicts with the
VibeVoice path (``transformers>=5`` exposing
``VibeVoiceAcousticTokenizerModel``). Install via optional extra and keep
probes that need both codecs in separate env passes (see Wave 0 D0 probe).
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from minispeech.constants import QWEN3_TTS_TOKENIZER_ID, QWEN3TTS_12HZ, get_target

try:
    import torchaudio
except ImportError:  # optional on Windows authoring hosts without torchaudio
    torchaudio = None  # type: ignore


@dataclass
class DiscreteEncodeResult:
    """Multi-codebook codes for one or more utterances.

    ``codes`` is a list of ``[T, n_codebooks]`` int64 tensors (CPU or device).
    """

    codes: list[torch.Tensor]
    n_frames_true: list[int]
    sample_rate: int
    raw: Any = None


@dataclass
class DiscreteDecodeResult:
    """Decoded waveforms from discrete codes."""

    audio: list[np.ndarray]  # each [N] float32
    sample_rate: int
    raw: Any = None


def _require_qwen_tts():
    try:
        from qwen_tts.inference.qwen3_tts_tokenizer import Qwen3TTSTokenizer
    except Exception as exc:  # noqa: BLE001 — surface install / pin guidance
        raise RuntimeError(
            "Failed to import Qwen3TTSTokenizer from qwen-tts. Install with "
            '`uv pip install "qwen-tts>=0.1.1"` (pulls transformers~=4.57). '
            "This pin conflicts with VibeVoice (transformers>=5). "
            f"Underlying error: {type(exc).__name__}: {exc}"
        ) from exc
    return Qwen3TTSTokenizer


def resolve_qwen3_tts_tokenizer_id(model_id: str | None = None) -> str:
    """Prefer a local checkout under ``~/datasets/tts/models/`` when present."""
    mid = model_id or QWEN3_TTS_TOKENIZER_ID
    if mid == QWEN3_TTS_TOKENIZER_ID:
        local = Path(
            os.path.expanduser("~/datasets/tts/models/Qwen3-TTS-Tokenizer-12Hz")
        )
        if local.is_dir():
            return str(local)
    return mid


class Qwen3TTSCodec:
    """Read-only encode/decode for frozen Qwen3-TTS Tokenizer-12Hz."""

    def __init__(
        self,
        model_id: str | None = None,
        device: str | torch.device | None = None,
        *,
        dtype: torch.dtype | None = None,
        device_map: str | None = None,
    ):
        Qwen3TTSTokenizer = _require_qwen_tts()
        self.spec = get_target(QWEN3TTS_12HZ.name)
        self.model_id = resolve_qwen3_tts_tokenizer_id(model_id)
        if device_map is None:
            if device is None:
                device_map = "cuda:0" if torch.cuda.is_available() else "cpu"
            else:
                device_map = str(device)
        kwargs: dict[str, Any] = {"device_map": device_map}
        if dtype is not None:
            kwargs["torch_dtype"] = dtype
        self._tok = Qwen3TTSTokenizer.from_pretrained(self.model_id, **kwargs)
        self.device = torch.device(getattr(self._tok, "device", device_map))
        # Freeze every parameter we can see.
        model = getattr(self._tok, "model", None)
        if model is not None:
            model.eval()
            for p in model.parameters():
                p.requires_grad_(False)

        self._validate_geometry()

    def _validate_geometry(self) -> None:
        in_sr = int(self._tok.get_input_sample_rate())
        out_sr = int(self._tok.get_output_sample_rate())
        hop = int(self._tok.get_encode_downsample_rate())
        if in_sr != self.spec.sample_rate or out_sr != self.spec.sample_rate:
            raise RuntimeError(
                f"Unexpected sample rate in/out={in_sr}/{out_sr}; "
                f"registered {self.spec.sample_rate}"
            )
        if hop != self.spec.samples_per_frame:
            raise RuntimeError(
                f"Unexpected encode hop {hop}; registered "
                f"{self.spec.samples_per_frame}"
            )
        measured_rate = in_sr / float(hop)
        if abs(measured_rate - self.spec.frame_rate) > 1e-6:
            raise RuntimeError(
                f"Unexpected frame rate {measured_rate}; registered "
                f"{self.spec.frame_rate}"
            )

    @property
    def n_codebooks(self) -> int:
        assert self.spec.n_codebooks is not None
        return int(self.spec.n_codebooks)

    @property
    def codebook_size(self) -> int:
        assert self.spec.codebook_size is not None
        return int(self.spec.codebook_size)

    @property
    def sample_rate(self) -> int:
        return int(self.spec.sample_rate)

    @property
    def frame_rate(self) -> float:
        return float(self.spec.frame_rate)

    def weight_fingerprint(self, *, n_keys: int = 8) -> dict[str, Any]:
        """Stable partial hash of codec weights for manifests (not a full digest)."""
        model = getattr(self._tok, "model", None)
        if model is None:
            return {"status": "unavailable"}
        sd = model.state_dict()
        keys = sorted(sd.keys())[: max(1, n_keys)]
        h = hashlib.sha256()
        for k in keys:
            h.update(k.encode("utf-8"))
            arr = sd[k].detach().float().cpu().contiguous().numpy()
            h.update(arr.tobytes())
        return {
            "status": "ok",
            "model_id": self.model_id,
            "n_state_keys": len(sd),
            "partial_sha256_16": h.hexdigest()[:16],
            "partial_keys": keys,
            "license": "Apache-2.0",
            "target": self.spec.name,
        }

    def _to_mono_float(self, audio: np.ndarray | torch.Tensor) -> np.ndarray:
        if isinstance(audio, torch.Tensor):
            audio_t = audio.detach().float().cpu()
            if audio_t.ndim > 1:
                audio_t = (
                    audio_t.mean(dim=0) if audio_t.shape[0] <= 8 else audio_t.squeeze()
                )
            arr = audio_t.numpy()
        else:
            arr = np.asarray(audio, dtype=np.float32)
            if arr.ndim > 1:
                arr = arr.mean(axis=0) if arr.shape[0] <= 8 else arr.squeeze()
        return np.asarray(arr, dtype=np.float32).reshape(-1)

    def preprocess(
        self,
        audio: np.ndarray | torch.Tensor,
        sr: int,
    ) -> tuple[np.ndarray, int]:
        """Resample to codec sample rate; return mono float32 + sr."""
        wav = self._to_mono_float(audio)
        wav = wav - float(wav.mean())
        if sr != self.sample_rate:
            if torchaudio is None:
                raise RuntimeError(
                    "torchaudio required to resample to 24 kHz for Qwen3-TTS"
                )
            wav_t = torch.from_numpy(wav).unsqueeze(0)
            wav = (
                torchaudio.functional.resample(wav_t, sr, self.sample_rate)
                .squeeze(0)
                .numpy()
                .astype(np.float32)
            )
        return wav, self.sample_rate

    @torch.inference_mode()
    def encode(
        self,
        audio: np.ndarray | torch.Tensor | list[np.ndarray] | list[str] | str,
        sr: int | None = None,
        *,
        sample: bool = False,
    ) -> DiscreteEncodeResult:
        """Encode waveform(s) to ``[T, 16]`` int64 codes.

        ``sample`` must be False — discrete encode is deterministic; the flag
        exists only to fail closed if a caller tries stochastic sampling.
        """
        if sample:
            raise ValueError(
                "Qwen3-TTS TTS targets must use sample=False (deterministic encode)"
            )

        if isinstance(audio, (str, Path)):
            enc = self._tok.encode(str(audio), sr=None, return_dict=True)
            codes = [c.detach().long().cpu() for c in enc.audio_codes]
            n_true = [int(c.shape[0]) for c in codes]
            return DiscreteEncodeResult(
                codes=codes,
                n_frames_true=n_true,
                sample_rate=self.sample_rate,
                raw=enc,
            )

        if isinstance(audio, list) and audio and isinstance(audio[0], str):
            enc = self._tok.encode(audio, sr=None, return_dict=True)
            codes = [c.detach().long().cpu() for c in enc.audio_codes]
            return DiscreteEncodeResult(
                codes=codes,
                n_frames_true=[int(c.shape[0]) for c in codes],
                sample_rate=self.sample_rate,
                raw=enc,
            )

        # numpy / tensor path (single or batch of arrays)
        if isinstance(audio, list):
            if sr is None:
                raise ValueError("sr is required when encoding numpy waveforms")
            wavs = [self.preprocess(a, sr)[0] for a in audio]
        else:
            if sr is None:
                raise ValueError("sr is required when encoding numpy waveforms")
            wavs = [self.preprocess(audio, sr)[0]]

        enc = self._tok.encode(wavs, sr=self.sample_rate, return_dict=True)
        codes = [c.detach().long().cpu() for c in enc.audio_codes]
        for c in codes:
            self._check_codes(c)
        n_true = []
        for wav, c in zip(wavs, codes):
            expected = self.spec.n_frames_from_samples(len(wav))
            # Encoder may drop a trailing incomplete frame; allow off-by-one.
            if abs(int(c.shape[0]) - expected) > 1:
                # Prefer measured length; still record expected for diagnostics.
                pass
            n_true.append(int(c.shape[0]))
        return DiscreteEncodeResult(
            codes=codes,
            n_frames_true=n_true,
            sample_rate=self.sample_rate,
            raw=enc,
        )

    def _check_codes(self, codes: torch.Tensor) -> None:
        if codes.ndim != 2 or codes.shape[-1] != self.n_codebooks:
            raise ValueError(
                f"Expected codes [T,{self.n_codebooks}], got {tuple(codes.shape)}"
            )
        if codes.dtype not in (torch.int64, torch.long, torch.int32):
            raise ValueError(f"Expected integer codes, got {codes.dtype}")
        if int(codes.min()) < 0 or int(codes.max()) >= self.codebook_size:
            raise ValueError(
                f"Codes out of range [0,{self.codebook_size}): "
                f"min={int(codes.min())} max={int(codes.max())}"
            )

    @torch.inference_mode()
    def decode(
        self,
        codes: torch.Tensor | list[torch.Tensor] | Any,
    ) -> DiscreteDecodeResult:
        """Decode ``[T, 16]`` codes (or a list / encoder output) to 24 kHz wav.

        ``qwen_tts`` decode accepts ModelOutput, a dict with ``audio_codes``,
        or a list of such dicts — not a bare list of tensors.
        """
        if hasattr(codes, "audio_codes"):
            payload: Any = codes
        elif isinstance(codes, dict) and "audio_codes" in codes:
            payload = codes
        elif isinstance(codes, torch.Tensor):
            self._check_codes(codes.long())
            payload = {"audio_codes": [codes.long()]}
        elif isinstance(codes, list):
            if codes and isinstance(codes[0], dict):
                payload = codes
            else:
                tensors = []
                for c in codes:
                    t = c.long() if isinstance(c, torch.Tensor) else torch.as_tensor(c).long()
                    self._check_codes(t)
                    tensors.append(t)
                payload = {"audio_codes": tensors}
        else:
            raise TypeError(f"Unsupported codes type: {type(codes)!r}")

        wavs, out_sr = self._tok.decode(payload)
        if int(out_sr) != self.sample_rate:
            raise RuntimeError(
                f"Decode sample rate {out_sr} != registered {self.sample_rate}"
            )
        out = [np.asarray(w, dtype=np.float32).reshape(-1) for w in wavs]
        return DiscreteDecodeResult(audio=out, sample_rate=int(out_sr), raw=None)

    @torch.inference_mode()
    def roundtrip(
        self,
        audio: np.ndarray | torch.Tensor,
        sr: int,
    ) -> np.ndarray:
        """Encode then decode; returns mono float32 at codec sample rate."""
        enc = self.encode(audio, sr, sample=False)
        dec = self.decode(enc.codes)
        return dec.audio[0]

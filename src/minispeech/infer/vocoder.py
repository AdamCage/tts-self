"""Vocoder hook for Track A mel → waveform.

Order of preference:
1. Local checkpoint path from config / ``VOCODER_CKPT`` (HiFi-GAN / UnivNet-style
   ``forward(mel)`` module — loader is a thin stub until weights are present).
2. Griffin–Lim (librosa or torchaudio) — no neural weights.
3. Save mel as ``.pt`` / ``.npy`` so the pipeline stays testable without audio.

Never downloads Hub models unless the caller explicitly opts in.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch

from minispeech.constants import MEL80, get_target
from minispeech.utils import expand

log = logging.getLogger(__name__)

VocoderBackend = Literal["neural", "griffin_lim", "save_mel", "none"]


@dataclass
class VocoderResult:
    wav: np.ndarray | None
    sample_rate: int
    backend: VocoderBackend
    mel_path: str | None = None
    note: str = ""


def find_local_vocoder(path: str | Path | None = None) -> Path | None:
    """Return first existing local vocoder path, or None."""
    candidates: list[Path] = []
    if path:
        candidates.append(expand(path))
    env = os.environ.get("VOCODER_CKPT")
    if env:
        candidates.append(expand(env))
    for rel in (
        "~/models/hifigan/generator.pt",
        "~/models/hifigan/generator.pth",
        "~/models/hifigan/generator.ckpt",
        "~/models/hifigan",
        "~/models/hifigan_universal_v1/generator",
        "~/models/univnet/generator.pt",
        "~/models/vocoder/generator.pt",
    ):
        candidates.append(expand(rel))
    for c in candidates:
        if c.is_file() or (c.is_dir() and any(c.iterdir())):
            return c
    return None


class NeuralVocoderStub:
    """Local HiFi-GAN / UnivNet-style generator loader.

    Accepts:
    - full ``nn.Module`` checkpoint or ``dict[generator|model|vocoder]=Module``
    - flat / nested state_dicts (jik876 or SpeechBrain ``*.conv.*`` layout)
      loaded into :class:`minispeech.infer.hifigan.HifiGanGenerator`
    """

    def __init__(self, ckpt_path: str | Path, *, device: torch.device | None = None):
        self.path = expand(ckpt_path)
        self.device = device or torch.device("cpu")
        self.model: torch.nn.Module | None = None
        self.available = False
        self.note = ""
        self._try_load()

    def _try_load(self) -> None:
        if not self.path.exists():
            self.note = f"missing path {self.path}"
            return
        # Directory: prefer generator.pt / generator.ckpt / generator.pth
        path = self.path
        if path.is_dir():
            for name in ("generator.pt", "generator.pth", "generator.ckpt", "g_latest"):
                cand = path / name
                if cand.is_file():
                    path = cand
                    break
            else:
                files = [p for p in path.iterdir() if p.is_file()]
                if not files:
                    self.note = f"empty vocoder dir {self.path}"
                    return
                path = files[0]
        try:
            obj = torch.load(path, map_location="cpu", weights_only=False)
        except TypeError:
            obj = torch.load(path, map_location="cpu")
        except Exception as exc:  # noqa: BLE001
            self.note = f"load failed: {exc}"
            return
        if isinstance(obj, torch.nn.Module):
            self.model = obj.to(self.device).eval()
            self.available = True
            self.note = f"loaded nn.Module checkpoint ({path.name})"
            return
        if isinstance(obj, dict):
            for key in ("generator", "model", "vocoder"):
                if key in obj and isinstance(obj[key], torch.nn.Module):
                    self.model = obj[key].to(self.device).eval()
                    self.available = True
                    self.note = f"loaded dict[{key!r}] module ({path.name})"
                    return
            try:
                from minispeech.infer.hifigan import (
                    extract_generator_state_dict,
                    load_hifigan_generator,
                )

                state = extract_generator_state_dict(obj)
                if state is None:
                    self.note = (
                        "checkpoint is a dict without generator tensors; "
                        "expected jik876/SpeechBrain HiFi-GAN state_dict. "
                        "Falling back to griffin-lim / save-mel."
                    )
                    return
                self.model = load_hifigan_generator(state, device=self.device)
                self.available = True
                self.note = (
                    f"loaded HiFi-GAN V1 from state_dict ({path.name}; "
                    f"{len(state)} tensors)"
                )
                return
            except Exception as exc:  # noqa: BLE001
                self.note = f"HiFi-GAN state_dict load failed: {exc}"
                return
        self.note = f"unsupported checkpoint type {type(obj)}"

    @torch.no_grad()
    def __call__(self, mel: torch.Tensor) -> torch.Tensor:
        if self.model is None:
            raise RuntimeError(f"neural vocoder unavailable: {self.note}")
        # Accept [T, n_mels] or [B, T, n_mels] or [B, n_mels, T]
        x = mel
        if x.ndim == 2:
            x = x.unsqueeze(0)
        if x.shape[-1] == MEL80.dim or (x.ndim == 3 and x.shape[-1] < x.shape[1]):
            x = x.transpose(1, 2)  # [B, n_mels, T]
        x = x.to(self.device)
        wav = self.model(x)
        if torch.is_tensor(wav) and wav.ndim == 3:
            wav = wav.squeeze(1)
        return wav.squeeze(0).detach().cpu()


def griffin_lim(
    mel: torch.Tensor | np.ndarray,
    *,
    target: str = "mel80",
    n_iter: int = 32,
) -> np.ndarray:
    """Invert log-mel approximately via Griffin–Lim (librosa preferred)."""
    spec = get_target(target)
    if torch.is_tensor(mel):
        logmel = mel.detach().cpu().float().numpy()
    else:
        logmel = np.asarray(mel, dtype=np.float32)
    if logmel.ndim != 2:
        raise ValueError(f"expected [T, n_mels], got {logmel.shape}")
    # [n_mels, T]
    S = np.exp(logmel.T)
    try:
        import librosa

        # Approximate linear spectrogram from mel then Griffin–Lim
        n_fft = int(spec.n_fft or 1024)
        hop = int(spec.hop_length or 256)
        win = int(spec.win_length or n_fft)
        # mel_to_stft needs filters; use mel_to_audio when available
        if hasattr(librosa.feature, "inverse") and hasattr(librosa.feature.inverse, "mel_to_audio"):
            wav = librosa.feature.inverse.mel_to_audio(
                S,
                sr=spec.sample_rate,
                n_fft=n_fft,
                hop_length=hop,
                win_length=win,
                power=1.0,
                n_iter=n_iter,
            )
        else:
            wav = librosa.griffinlim(
                S,
                n_iter=n_iter,
                hop_length=hop,
                win_length=win,
                n_fft=n_fft,
            )
        return np.asarray(wav, dtype=np.float32)
    except Exception as exc:
        log.warning("librosa griffin-lim failed (%s); trying torchaudio", exc)

    import torchaudio.transforms as T

    # torchaudio InverseMelScale + GriffinLim expects power spectrogram
    mel_t = torch.from_numpy(S).float()
    n_stft = (int(spec.n_fft or 1024) // 2) + 1
    inv_mel = T.InverseMelScale(
        n_stft=n_stft,
        n_mels=int(spec.n_mels or 80),
        sample_rate=spec.sample_rate,
        f_min=0.0,
        f_max=spec.sample_rate / 2.0,
    )
    gl = T.GriffinLim(
        n_fft=int(spec.n_fft or 1024),
        hop_length=int(spec.hop_length or 256),
        win_length=int(spec.win_length or 1024),
        n_iter=n_iter,
        power=1.0,
    )
    spec_lin = inv_mel(mel_t)
    wav_t = gl(spec_lin)
    return wav_t.numpy().astype(np.float32)


def save_mel(mel: torch.Tensor | np.ndarray, path: str | Path) -> Path:
    path = expand(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if torch.is_tensor(mel):
        torch.save(mel.detach().cpu().float(), path.with_suffix(".pt"))
        return path.with_suffix(".pt")
    np.save(str(path.with_suffix(".npy")), np.asarray(mel, dtype=np.float32))
    return path.with_suffix(".npy")


def mel_to_wav(
    mel: torch.Tensor | np.ndarray,
    *,
    vocoder_ckpt: str | Path | None = None,
    target: str = "mel80",
    prefer: VocoderBackend | None = None,
    save_mel_path: str | Path | None = None,
    device: torch.device | None = None,
) -> VocoderResult:
    """Convert mel to waveform with graceful fallbacks.

    Documentation placeholder when no neural vocoder is present:
    place a full-module generator at ``~/models/hifigan/generator.pt`` or set
    ``VOCODER_CKPT``, then this function will use it. Until then, Griffin–Lim
    or ``save_mel`` keeps the pipeline testable.
    """
    spec = get_target(target)
    sr = int(spec.sample_rate)

    if prefer == "save_mel":
        if save_mel_path is None:
            raise ValueError("save_mel requires save_mel_path")
        p = save_mel(mel, save_mel_path)
        return VocoderResult(wav=None, sample_rate=sr, backend="save_mel", mel_path=str(p), note="mel only")

    local = find_local_vocoder(vocoder_ckpt)
    if prefer in (None, "neural") and local is not None:
        stub = NeuralVocoderStub(local, device=device)
        if stub.available:
            mel_t = mel if torch.is_tensor(mel) else torch.from_numpy(np.asarray(mel))
            wav_t = stub(mel_t)
            return VocoderResult(
                wav=wav_t.numpy().astype(np.float32),
                sample_rate=sr,
                backend="neural",
                note=f"{stub.note}; path={local}",
            )
        log.info("neural vocoder not usable (%s); falling back", stub.note)

    if prefer in (None, "griffin_lim", "neural"):
        try:
            wav = griffin_lim(mel, target=target)
            return VocoderResult(
                wav=wav,
                sample_rate=sr,
                backend="griffin_lim",
                note="no local neural vocoder; used griffin-lim",
            )
        except Exception as exc:  # noqa: BLE001
            log.warning("griffin-lim failed: %s", exc)

    if save_mel_path is not None:
        p = save_mel(mel, save_mel_path)
        return VocoderResult(
            wav=None,
            sample_rate=sr,
            backend="save_mel",
            mel_path=str(p),
            note="vocoder unavailable; saved mel",
        )

    return VocoderResult(
        wav=None,
        sample_rate=sr,
        backend="none",
        note=(
            "No neural vocoder and griffin-lim failed. "
            "Install librosa, or place weights at ~/models/hifigan/generator.pt, "
            "or pass save_mel_path."
        ),
    )


def vocoder_status(vocoder_ckpt: str | Path | None = None) -> dict[str, Any]:
    """Machine-readable status for docs / smoke tests."""
    local = find_local_vocoder(vocoder_ckpt)
    return {
        "local_vocoder": str(local) if local else None,
        "env_VOCODER_CKPT": os.environ.get("VOCODER_CKPT"),
        "fallback": "griffin_lim then save_mel",
        "placeholder": (
            "Expected local HiFi-GAN under ~/models/hifigan/generator.pt "
            "(full module or jik876/SpeechBrain state_dict) or VOCODER_CKPT. "
            "No Hub download from this helper."
        ),
    }

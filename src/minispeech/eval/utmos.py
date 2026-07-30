"""UTMOS / SpeechMOS naturalness scoring with lazy model load.

Preferred backend: ``tarepan/SpeechMOS`` via ``torch.hub`` (``utmos22_strong``),
which reimplements UTMOS22 strong learner. Returns ``None`` when deps or the
hub model are unavailable — never raises to callers of ``utmos_score``.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import numpy as np

log = logging.getLogger(__name__)

_PREDICTOR: Any | None = None
_LOAD_ATTEMPTED: bool = False
_LOAD_ERROR: str | None = None

DEFAULT_HUB_REPO = "tarepan/SpeechMOS:v1.2.0"
DEFAULT_HUB_MODEL = "utmos22_strong"


def _torch_hub_dir() -> Path:
    """Prefer ~/models/utmos for hub cache; fall back to TORCH_HOME / default."""
    env = os.environ.get("MINISPEECH_UTMOS_DIR") or os.environ.get("TORCH_HOME")
    if env:
        return Path(env).expanduser()
    return Path.home() / "models" / "utmos"


def reset_utmos_cache() -> None:
    """Clear cached predictor (tests)."""
    global _PREDICTOR, _LOAD_ATTEMPTED, _LOAD_ERROR
    _PREDICTOR = None
    _LOAD_ATTEMPTED = False
    _LOAD_ERROR = None


def _load_predictor() -> Any | None:
    global _PREDICTOR, _LOAD_ATTEMPTED, _LOAD_ERROR
    if _LOAD_ATTEMPTED:
        return _PREDICTOR
    _LOAD_ATTEMPTED = True
    try:
        import torch
    except ImportError as exc:
        _LOAD_ERROR = f"torch missing: {exc}"
        log.info("utmos: %s", _LOAD_ERROR)
        return None

    hub_dir = _torch_hub_dir()
    try:
        hub_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        _LOAD_ERROR = f"cannot create hub dir {hub_dir}: {exc}"
        log.warning("utmos: %s", _LOAD_ERROR)
        return None

    # torch.hub uses $TORCH_HOME/hub; point it at our cache root.
    os.environ.setdefault("TORCH_HOME", str(hub_dir))

    try:
        predictor = torch.hub.load(
            DEFAULT_HUB_REPO,
            DEFAULT_HUB_MODEL,
            trust_repo=True,
        )
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        predictor = predictor.to(device)
        predictor.eval()
        _PREDICTOR = predictor
        log.info("utmos: loaded %s/%s on %s (TORCH_HOME=%s)", DEFAULT_HUB_REPO, DEFAULT_HUB_MODEL, device, hub_dir)
        return _PREDICTOR
    except Exception as exc:  # noqa: BLE001
        _LOAD_ERROR = str(exc)
        log.warning("utmos: hub load failed: %s", exc)
        _PREDICTOR = None
        return None


def utmos_load_error() -> str | None:
    """Return last load error string after a failed attempt, else None."""
    _load_predictor()
    return _LOAD_ERROR


def utmos_score(audio_path: str) -> float | None:
    """Predict UTMOS naturalness in roughly [1, 5]; ``None`` if unavailable.

    Lazy-loads and caches the SpeechMOS ``utmos22_strong`` model. Missing
    dependencies, download failures, or unreadable audio return ``None``.
    """
    path = Path(str(audio_path)).expanduser()
    if not path.is_file():
        return None

    predictor = _load_predictor()
    if predictor is None:
        return None

    try:
        import torch

        try:
            import soundfile as sf

            wav, sr = sf.read(str(path), always_2d=False)
            wav = np.asarray(wav, dtype=np.float32)
            if wav.ndim > 1:
                wav = wav.mean(axis=-1)
            sr = int(sr)
        except Exception:
            import librosa

            wav, sr = librosa.load(str(path), sr=None, mono=True)
            wav = np.asarray(wav, dtype=np.float32)
            sr = int(sr)

        if wav.size < int(0.05 * sr):
            return None

        device = next(predictor.parameters()).device
        wave_t = torch.from_numpy(wav).unsqueeze(0).to(device=device, dtype=torch.float32)
        with torch.inference_mode():
            score = predictor(wave_t, sr)
        if hasattr(score, "detach"):
            val = float(score.detach().cpu().reshape(-1)[0].item())
        else:
            val = float(score)
        return val
    except Exception as exc:  # noqa: BLE001
        log.warning("utmos_score failed for %s: %s", path, exc)
        return None


def utmos_score_wav(wav: np.ndarray, sr: int) -> float | None:
    """Score an in-memory mono waveform; ``None`` if backend unavailable."""
    predictor = _load_predictor()
    if predictor is None:
        return None
    try:
        import torch

        x = np.asarray(wav, dtype=np.float32).squeeze()
        if x.ndim != 1 or x.size < int(0.05 * sr):
            return None
        device = next(predictor.parameters()).device
        wave_t = torch.from_numpy(x).unsqueeze(0).to(device=device, dtype=torch.float32)
        with torch.inference_mode():
            score = predictor(wave_t, int(sr))
        return float(score.detach().cpu().reshape(-1)[0].item())
    except Exception as exc:  # noqa: BLE001
        log.warning("utmos_score_wav failed: %s", exc)
        return None

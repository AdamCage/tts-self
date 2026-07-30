"""Log-mel extract / cache helpers for Track A ``mel80`` target.

Uses ``TargetSpec`` from ``constants.py`` (sr=22050, n_fft=1024, hop=256,
n_mels=80). Prefers ``librosa`` then ``torchaudio``; both are optional at
import time so unit tests can inject synthetic mels without audio deps.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch

from minispeech.constants import MEL80, TargetSpec, get_target
from minispeech.data.shards import ShardReader, ShardWriter
from minispeech.utils import expand

log = logging.getLogger(__name__)

MEL_CACHE_META_KEY = "__mel_cache_meta__"


def mel_filter_kwargs(spec: TargetSpec, *, sample_rate: int | None = None) -> dict[str, Any]:
    """Resolve fmin/fmax/mel_scale/mel_norm for a mel ``TargetSpec``."""
    sr = int(sample_rate or spec.sample_rate)
    fmin = float(spec.fmin if spec.fmin is not None else 0.0)
    fmax = float(spec.fmax if spec.fmax is not None else sr / 2.0)
    mel_scale = str(spec.mel_scale or "slaney")
    mel_norm = spec.mel_norm  # may be None
    return {"fmin": fmin, "fmax": fmax, "mel_scale": mel_scale, "mel_norm": mel_norm}


def mel_backend_name() -> str:
    """Return ``librosa``, ``torchaudio``, or ``none``."""
    try:
        import librosa  # noqa: F401

        return "librosa"
    except ImportError:
        pass
    try:
        import torchaudio  # noqa: F401

        return "torchaudio"
    except ImportError:
        return "none"


def waveform_to_logmel(
    audio: np.ndarray | torch.Tensor,
    *,
    target: str | TargetSpec | None = "mel80",
    sample_rate: int | None = None,
) -> torch.Tensor:
    """Convert mono waveform to log-mel ``[T, n_mels]`` (float32).

    Resamples to ``target.sample_rate`` when ``sample_rate`` differs.
    Raises ``RuntimeError`` if neither librosa nor torchaudio is available.
    """
    spec = get_target(target)
    if spec.kind != "mel":
        raise ValueError(f"target {spec.name!r} is not a mel target")
    assert spec.n_fft and spec.hop_length and spec.n_mels and spec.win_length

    if torch.is_tensor(audio):
        wav = audio.detach().cpu().float().numpy()
    else:
        wav = np.asarray(audio, dtype=np.float32)
    wav = np.squeeze(wav)
    if wav.ndim != 1:
        raise ValueError(f"expected mono waveform, got shape {wav.shape}")

    sr_in = int(sample_rate or spec.sample_rate)
    sr_tgt = int(spec.sample_rate)
    backend = mel_backend_name()
    if backend == "none":
        raise RuntimeError(
            "Neither librosa nor torchaudio is available for mel extraction. "
            "Install project deps or pass precomputed mel tensors."
        )

    fk = mel_filter_kwargs(spec, sample_rate=sr_tgt)
    if backend == "librosa":
        import librosa

        if sr_in != sr_tgt:
            wav = librosa.resample(wav, orig_sr=sr_in, target_sr=sr_tgt)
        mel = librosa.feature.melspectrogram(
            y=wav.astype(np.float32),
            sr=sr_tgt,
            n_fft=spec.n_fft,
            hop_length=spec.hop_length,
            win_length=spec.win_length,
            n_mels=spec.n_mels,
            fmin=fk["fmin"],
            fmax=fk["fmax"],
            htk=fk["mel_scale"] == "htk",
            norm=fk["mel_norm"],
            power=1.0,
        )
        logmel = np.log(np.maximum(mel, 1e-5)).T  # [T, n_mels]
        return torch.from_numpy(logmel.astype(np.float32))

    import torchaudio
    import torchaudio.functional as AF

    wav_t = torch.from_numpy(wav.astype(np.float32)).unsqueeze(0)
    if sr_in != sr_tgt:
        wav_t = AF.resample(wav_t, sr_in, sr_tgt)
    transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=sr_tgt,
        n_fft=spec.n_fft,
        win_length=spec.win_length,
        hop_length=spec.hop_length,
        n_mels=spec.n_mels,
        f_min=fk["fmin"],
        f_max=fk["fmax"],
        norm=fk["mel_norm"],
        mel_scale=fk["mel_scale"],
        power=1.0,
    )
    mel = transform(wav_t).squeeze(0)  # [n_mels, T]
    logmel = torch.log(mel.clamp_min(1e-5)).transpose(0, 1).contiguous()
    return logmel.float()


def load_wav_mono(path: str | Path, *, target_sr: int | None = None) -> tuple[np.ndarray, int]:
    """Load mono float32 wav; optional resample to ``target_sr``."""
    path = expand(path)
    try:
        import soundfile as sf

        wav, sr = sf.read(str(path), always_2d=False)
        wav = np.asarray(wav, dtype=np.float32)
        if wav.ndim > 1:
            wav = wav.mean(axis=-1)
    except Exception:
        import librosa

        wav, sr = librosa.load(str(path), sr=None, mono=True)
        wav = np.asarray(wav, dtype=np.float32)
        sr = int(sr)

    sr = int(sr)
    if target_sr is not None and sr != int(target_sr):
        try:
            import librosa

            wav = librosa.resample(wav, orig_sr=sr, target_sr=int(target_sr))
            sr = int(target_sr)
        except ImportError:
            import torchaudio.functional as AF

            t = torch.from_numpy(wav).unsqueeze(0)
            wav = AF.resample(t, sr, int(target_sr)).squeeze(0).numpy()
            sr = int(target_sr)
    return wav, sr


def extract_mel_file(
    wav_path: str | Path,
    *,
    target: str | TargetSpec | None = "mel80",
) -> torch.Tensor:
    """Load wav from disk and return log-mel ``[T, n_mels]``."""
    spec = get_target(target)
    wav, sr = load_wav_mono(wav_path, target_sr=spec.sample_rate)
    return waveform_to_logmel(wav, target=spec, sample_rate=sr)


def mel_n_frames(n_samples: int, *, target: str | TargetSpec | None = "mel80") -> int:
    """Approximate mel frame count for a waveform length (centered STFT)."""
    spec = get_target(target)
    hop = spec.samples_per_frame
    if n_samples <= 0:
        return 0
    # librosa / torchaudio default center=True → roughly 1 + n_samples // hop
    return 1 + int(n_samples) // hop


class MelCacheWriter:
    """Write mel shards via ``ShardWriter`` (prefix ``mels``)."""

    def __init__(
        self,
        out_dir: str | Path,
        shard_size: int = 2000,
        *,
        resume: bool = True,
        target: str = "mel80",
    ):
        self.target = str(target)
        self.backend = mel_backend_name()
        self._w = ShardWriter(out_dir, "mels", shard_size=shard_size, resume=resume)
        # Guard against silently mixing domains in one directory.
        existing = self._w.index.get(MEL_CACHE_META_KEY)
        if isinstance(existing, dict):
            prev_t = existing.get("target")
            prev_b = existing.get("mel_backend_name")
            if prev_t and prev_t != self.target:
                raise RuntimeError(
                    f"Mel cache at {self._w.out_dir} was built for target={prev_t!r}, "
                    f"refusing to append target={self.target!r}. Use a separate directory."
                )
            if prev_b and prev_b != self.backend:
                raise RuntimeError(
                    f"Mel cache at {self._w.out_dir} was built with mel_backend={prev_b!r}, "
                    f"current backend={self.backend!r}. Use a separate directory or matching env."
                )

    def has(self, utt_id: str) -> bool:
        if utt_id == MEL_CACHE_META_KEY:
            return True
        return self._w.has(utt_id)

    def add(self, utt_id: str, mel: torch.Tensor, meta: dict[str, Any] | None = None) -> None:
        if utt_id == MEL_CACHE_META_KEY:
            raise ValueError(f"utt_id {MEL_CACHE_META_KEY!r} is reserved")
        merged = {
            "target": self.target,
            "mel_backend_name": self.backend,
            **(meta or {}),
        }
        self._w.add(utt_id, mel.float().contiguous(), meta=merged)

    def close(self) -> Path:
        self._w.index[MEL_CACHE_META_KEY] = {
            "target": self.target,
            "mel_backend_name": self.backend,
            "kind": "mel_cache_meta",
        }
        return self._w.close()


class MelCacheReader:
    """Read mel shards (``mels_index.json``)."""

    def __init__(
        self,
        mel_dir: str | Path,
        *,
        lru_shards: int = 3,
        lazy: bool = True,
        expect_target: str | None = None,
        expect_backend: str | None = None,
    ):
        self._r = ShardReader(mel_dir, "mels", lru_shards=lru_shards, lazy=lazy)
        self._validate_meta(expect_target=expect_target, expect_backend=expect_backend)

    def _validate_meta(
        self,
        *,
        expect_target: str | None,
        expect_backend: str | None,
    ) -> None:
        meta = self._r.index.get(MEL_CACHE_META_KEY)
        if not isinstance(meta, dict):
            # Legacy caches (pre-guard): warn only.
            if expect_target is not None:
                log.warning(
                    "Mel cache %s has no %s; cannot verify expect_target=%r",
                    self._r.out_dir,
                    MEL_CACHE_META_KEY,
                    expect_target,
                )
            return
        got_t = meta.get("target")
        got_b = meta.get("mel_backend_name")
        if expect_target is not None and got_t is not None and got_t != expect_target:
            raise RuntimeError(
                f"Mel cache {self._r.out_dir} target={got_t!r} != expect_target={expect_target!r}. "
                "Do not mix mel80 / mel80_hifigan directories."
            )
        if expect_backend is not None and got_b is not None and got_b != expect_backend:
            raise RuntimeError(
                f"Mel cache {self._r.out_dir} mel_backend={got_b!r} != expect_backend={expect_backend!r}."
            )

    @property
    def cache_meta(self) -> dict[str, Any] | None:
        meta = self._r.index.get(MEL_CACHE_META_KEY)
        return meta if isinstance(meta, dict) else None

    @property
    def index(self) -> dict[str, Any]:
        return {k: v for k, v in self._r.index.items() if k != MEL_CACHE_META_KEY}

    def get(self, utt_id: str) -> torch.Tensor:
        if utt_id == MEL_CACHE_META_KEY:
            raise KeyError(MEL_CACHE_META_KEY)
        return self._r.get(utt_id)


def default_mel_dir(data_cfg: dict[str, Any]) -> str:
    """Resolve mel cache dir from data YAML (``mel_dir`` or sibling of latents)."""
    if data_cfg.get("mel_dir"):
        return str(data_cfg["mel_dir"])
    root = data_cfg.get("ruslan_dir") or data_cfg.get("root")
    if root:
        return str(Path(str(root)).expanduser() / "mels")
    return "mels"


# Re-export mel80 for callers that only import this module
MEL_TARGET = MEL80

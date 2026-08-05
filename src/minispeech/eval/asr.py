"""Swappable ASR backends with deterministic decode and transcript cache."""

from __future__ import annotations

import hashlib
import importlib.metadata
import inspect
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

from minispeech.frontend.normalize import normalize_for_asr
from minispeech.utils import expand, load_yaml


ASR_SR = 16000


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def asr_provenance(
    config_path: str | Path = "configs/asr.yaml",
    *,
    device: str | torch.device | None = None,
    weight_artifact: str | Path | None = None,
    expected_weight_sha256: str | None = None,
) -> dict[str, Any]:
    """Return canonical ASR/normalizer provenance for formal score reports.

    ``expected_weight_sha256`` supports immutable backend registries when model
    weights are managed outside the repository; callers must not silently omit
    both a local artifact hash and an expected registry hash in formal mode.
    """
    path = expand(config_path)
    if not path.is_file():
        raise FileNotFoundError(f"ASR config missing: {path}")
    cfg = AsrConfig.from_yaml(path)
    parsed = asdict(cfg)
    normalizer_path = Path(inspect.getsourcefile(normalize_for_asr) or "")
    normalizer_sha = _sha256_file(normalizer_path) if normalizer_path.is_file() else None
    artifact_path = expand(weight_artifact) if weight_artifact is not None else None
    artifact_sha = _sha256_file(artifact_path) if artifact_path and artifact_path.is_file() else None
    backend_pkg = "openai-whisper" if cfg.backend == "openai_whisper" else "faster-whisper"
    return {
        "asr_config_path": str(path),
        "asr_config_raw_sha256": _sha256_file(path),
        "asr_config_parsed_sha256": hashlib.sha256(
            json.dumps(parsed, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
        "backend": cfg.backend,
        "model": cfg.model,
        "language": cfg.language,
        "decoding": {
            "temperature": cfg.temperature,
            "beam_size": cfg.beam_size,
            "condition_on_previous_text": cfg.condition_on_previous_text,
            "vad_filter": cfg.vad_filter,
        },
        "device_policy": str(device or ("cuda" if torch.cuda.is_available() else "cpu")),
        "precision_policy": cfg.compute_type,
        "package": {"name": backend_pkg, "version": _package_version(backend_pkg)},
        "weight_artifact_sha256": artifact_sha,
        "expected_weight_sha256": expected_weight_sha256,
        "normalizer": {
            "id": "minispeech.frontend.normalize.normalize_for_asr_v1",
            "module": normalize_for_asr.__module__,
            "source_path": str(normalizer_path),
            "source_sha256": normalizer_sha,
        },
        "scoring_packages": {
            "jiwer": _package_version("jiwer"),
            "torch": _package_version("torch"),
            "torchaudio": _package_version("torchaudio"),
            "soundfile": _package_version("soundfile"),
        },
    }


def _ensure_ctranslate2_cuda_libs() -> None:
    """Prepend pip nvidia-cublas-cu12 (and torch cu13) dirs so ctranslate2 finds libcublas.so.12."""
    import os
    import site
    from pathlib import Path

    dirs: list[str] = []
    for sp in site.getsitepackages():
        root = Path(sp) / "nvidia"
        for rel in ("cublas/lib", "cuda_nvrtc/lib", "cu12/lib", "cu13/lib"):
            p = root / rel
            if p.is_dir():
                dirs.append(str(p))
    if not dirs:
        return
    cur = os.environ.get("LD_LIBRARY_PATH", "")
    parts = [d for d in dirs if d not in cur.split(":")]
    if parts:
        os.environ["LD_LIBRARY_PATH"] = ":".join(parts + ([cur] if cur else []))


@dataclass
class AsrConfig:
    backend: str = "openai_whisper"
    model: str = "large-v3"
    language: str = "ru"
    temperature: float = 0.0
    beam_size: int | None = None
    condition_on_previous_text: bool = False
    compute_type: str = "float16"
    batch_size: int = 8
    vad_filter: bool = False
    cache_dir: str = "~/datasets/tts/asr_cache"

    @classmethod
    def from_yaml(cls, path: str | Path) -> "AsrConfig":
        cfg = load_yaml(path)
        raw = cfg.get("asr", cfg)
        known = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in raw.items() if k in known})

    def params_hash(self) -> str:
        payload = {
            "backend": self.backend,
            "model": self.model,
            "language": self.language,
            "temperature": self.temperature,
            "beam_size": self.beam_size,
            "condition_on_previous_text": self.condition_on_previous_text,
            "vad_filter": self.vad_filter,
        }
        return hashlib.sha1(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:12]


def resample_to_16k(audio: np.ndarray, sr: int) -> np.ndarray:
    x = np.asarray(audio, dtype=np.float32)
    if x.ndim > 1:
        x = x.mean(axis=-1)
    if sr == ASR_SR:
        return x
    import torchaudio

    t = torch.from_numpy(x).float().unsqueeze(0)
    y = torchaudio.functional.resample(t, sr, ASR_SR).squeeze(0).numpy()
    return y.astype(np.float32)


def audio_hash(audio_16k: np.ndarray) -> str:
    # float32 bytes — deterministic for identical PCM
    return hashlib.sha1(np.ascontiguousarray(audio_16k).tobytes()).hexdigest()


class AsrEngine:
    """Unified ASR interface: transcribe / transcribe_many with disk cache."""

    def __init__(self, cfg: AsrConfig, device: str | torch.device | None = None):
        self.cfg = cfg
        self.device = str(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.cache_dir = expand(cfg.cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._model: Any = None
        self._backend = cfg.backend

    def _load(self) -> None:
        if self._model is not None:
            return
        if self._backend == "openai_whisper":
            import whisper

            self._model = whisper.load_model(self.cfg.model, device=self.device)
        elif self._backend == "faster_whisper":
            _ensure_ctranslate2_cuda_libs()
            from faster_whisper import WhisperModel

            device = "cuda" if "cuda" in self.device else "cpu"
            self._model = WhisperModel(
                self.cfg.model,
                device=device,
                compute_type=self.cfg.compute_type if device == "cuda" else "int8",
            )
        else:
            raise ValueError(f"Unknown ASR backend: {self._backend}")

    def _cache_path(self, ahash: str) -> Path:
        return self.cache_dir / f"{ahash}_{self._backend}_{self.cfg.model}_{self.cfg.params_hash()}.json"

    def _read_cache(self, ahash: str) -> str | None:
        p = self._cache_path(ahash)
        if not p.exists():
            return None
        try:
            return json.loads(p.read_text(encoding="utf-8")).get("text", "")
        except Exception:
            return None

    def _write_cache(self, ahash: str, text: str) -> None:
        p = self._cache_path(ahash)
        p.write_text(json.dumps({"text": text, "cfg": asdict(self.cfg)}, ensure_ascii=False), encoding="utf-8")

    def _decode_openai(self, audio_16k: np.ndarray) -> str:
        assert self._model is not None
        kwargs: dict[str, Any] = {
            "language": self.cfg.language,
            "temperature": self.cfg.temperature,
            "condition_on_previous_text": self.cfg.condition_on_previous_text,
            "fp16": "cuda" in self.device,
        }
        if self.cfg.beam_size is not None:
            kwargs["beam_size"] = self.cfg.beam_size
        result = self._model.transcribe(audio_16k, **kwargs)
        return (result.get("text") or "").strip()

    def _decode_faster(self, audio_16k: np.ndarray) -> str:
        assert self._model is not None
        segments, _info = self._model.transcribe(
            audio_16k,
            language=self.cfg.language,
            beam_size=self.cfg.beam_size or 1,
            temperature=self.cfg.temperature,
            condition_on_previous_text=self.cfg.condition_on_previous_text,
            vad_filter=self.cfg.vad_filter,
        )
        return "".join(s.text for s in segments).strip()

    def transcribe(self, audio: np.ndarray, sr: int, *, use_cache: bool = True) -> str:
        audio_16k = resample_to_16k(audio, sr)
        ahash = audio_hash(audio_16k)
        if use_cache:
            cached = self._read_cache(ahash)
            if cached is not None:
                return cached
        self._load()
        if self._backend == "openai_whisper":
            text = self._decode_openai(audio_16k)
        else:
            text = self._decode_faster(audio_16k)
        if use_cache:
            self._write_cache(ahash, text)
        return text

    def transcribe_many(
        self,
        audios: Sequence[tuple[np.ndarray, int]],
        *,
        use_cache: bool = True,
    ) -> list[str]:
        return [self.transcribe(a, sr, use_cache=use_cache) for a, sr in audios]

    def transcribe_normalized(self, audio: np.ndarray, sr: int, *, use_cache: bool = True) -> str:
        return normalize_for_asr(self.transcribe(audio, sr, use_cache=use_cache))


def load_asr_engine(config_path: str | Path = "configs/asr.yaml", device=None) -> AsrEngine:
    return AsrEngine(AsrConfig.from_yaml(config_path), device=device)

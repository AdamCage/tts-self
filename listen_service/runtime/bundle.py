"""Load DiscreteSpeechLM + Qwen tokenizer + frozen codec for listen UI."""

from __future__ import annotations

import json
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import yaml
from transformers import AutoTokenizer

from minispeech.codec.qwen3_tts_wrapper import Qwen3TTSCodec
from minispeech.infer.discrete_config import resolve_discrete_infer_settings
from minispeech.models.discrete_speech_lm import DiscreteSpeechLM, build_discrete_speech_lm

SERVICE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INFER_YAML = SERVICE_ROOT / "configs" / "infer_defaults.yaml"
DEFAULT_SPEAKER_JSON = SERVICE_ROOT / "configs" / "speaker_ruslan.json"


def resolve_device(
    requested: str | torch.device | None = None,
    *,
    warn: bool = True,
) -> tuple[torch.device, str | None]:
    """Map ``cuda``/``cpu`` to an available device; never raise on missing CUDA.

    Returns ``(device, warning_or_None)``. Preference when ``requested`` is None
    or empty: cuda if ``torch.cuda.is_available()`` else cpu.
    """
    if requested is None or (isinstance(requested, str) and not requested.strip()):
        pref = "cuda" if torch.cuda.is_available() else "cpu"
        return torch.device(pref), None

    if isinstance(requested, torch.device):
        name = requested.type
    else:
        name = str(requested).strip().lower()
        if name.startswith("cuda"):
            name = "cuda"
        elif name != "cpu":
            name = "cuda" if torch.cuda.is_available() else "cpu"

    if name == "cuda" and not torch.cuda.is_available():
        msg = (
            "DEVICE=cuda requested but torch.cuda.is_available() is False; "
            "falling back to cpu"
        )
        if warn:
            warnings.warn(msg, UserWarning, stacklevel=2)
        return torch.device("cpu"), msg
    return torch.device(name), None


@dataclass
class ListenBundle:
    """In-memory inference stack for one device."""

    model: DiscreteSpeechLM
    cfg: dict[str, Any]
    tokenizer: Any
    codec: Qwen3TTSCodec
    device: torch.device
    ckpt_path: Path
    tokenizer_id: str
    gen_kw: dict[str, Any]
    pad_token_id: int
    speaker_map: dict[str, int] = field(default_factory=dict)
    default_speaker_id: int = 1088


def _load_yaml(path: Path) -> dict[str, Any]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    return raw if isinstance(raw, dict) else {}


def load_speaker_defaults(path: Path | None = None) -> tuple[dict[str, int], int]:
    p = path or DEFAULT_SPEAKER_JSON
    obj = json.loads(p.read_text(encoding="utf-8"))
    speakers = {str(k): int(v) for k, v in (obj.get("speakers") or {}).items()}
    default_name = str(obj.get("default_speaker") or "ruslan")
    default_id = int(speakers.get(default_name, 1088))
    return speakers, default_id


def _merge_infer_cfg(ckpt_cfg: dict[str, Any]) -> dict[str, Any]:
    """Ensure ``cfg['infer']`` exists (file defaults < ckpt)."""
    cfg = dict(ckpt_cfg) if ckpt_cfg else {}
    file_infer = _load_yaml(DEFAULT_INFER_YAML) if DEFAULT_INFER_YAML.is_file() else {}
    merged_infer = dict(file_infer)
    ckpt_infer = cfg.get("infer") if isinstance(cfg.get("infer"), dict) else {}
    merged_infer.update(ckpt_infer)
    cfg["infer"] = merged_infer
    return cfg


def load_model(ckpt_path: Path, device: torch.device) -> tuple[DiscreteSpeechLM, dict[str, Any]]:
    payload = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = payload.get("cfg") or {}
    if not isinstance(cfg, dict):
        cfg = {}
    model = build_discrete_speech_lm(cfg)
    sd = payload.get("model") or payload
    if isinstance(sd, dict) and any(k.startswith("module.") for k in sd):
        sd = {k.removeprefix("module."): v for k, v in sd.items()}
    model.load_state_dict(sd, strict=True)
    model.to(device).eval()
    return model, cfg


def resolve_tokenizer_id(cfg: dict[str, Any], override: str | None) -> str:
    if override:
        return str(override)
    model = cfg.get("model") or {}
    tid = model.get("text_tokenizer_id") if isinstance(model, dict) else None
    if isinstance(tid, str) and tid.strip():
        return tid.strip()
    return "/home/adam/datasets/tts/models/Qwen3.5-0.8B-Base"


def load_bundle(
    ckpt: str | Path,
    *,
    device: str | torch.device | None = None,
    tokenizer_id: str | None = None,
    codec_model_id: str | None = None,
) -> ListenBundle:
    """Load model, tokenizer, and frozen codec on ``device``.

    If ``device`` is ``cuda`` but CUDA is unavailable, coerces to CPU (no raise).
    """
    ckpt_path = Path(ckpt).expanduser().resolve()
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"checkpoint not found: {ckpt_path}")

    dev, _fallback_msg = resolve_device(device, warn=True)

    model, ckpt_cfg = load_model(ckpt_path, dev)
    cfg = _merge_infer_cfg(ckpt_cfg)
    tok_id = resolve_tokenizer_id(cfg, tokenizer_id)
    tokenizer = AutoTokenizer.from_pretrained(tok_id, trust_remote_code=True)

    codebook_size = int((cfg.get("model") or {}).get("codebook_size", 2048))
    gen_kw = resolve_discrete_infer_settings(cfg, codebook_size=codebook_size)
    pad_id = int((cfg.get("model") or {}).get("pad_token_id", 248044))

    codec = Qwen3TTSCodec(model_id=codec_model_id, device=dev)
    speakers, default_spk = load_speaker_defaults()

    return ListenBundle(
        model=model,
        cfg=cfg,
        tokenizer=tokenizer,
        codec=codec,
        device=dev,
        ckpt_path=ckpt_path,
        tokenizer_id=tok_id,
        gen_kw=gen_kw,
        pad_token_id=pad_id,
        speaker_map=speakers,
        default_speaker_id=default_spk,
    )


def move_bundle(bundle: ListenBundle, device: str | torch.device) -> ListenBundle:
    """Rebuild codec on a new device; move LM weights.

    Missing CUDA coerces to CPU instead of raising.
    """
    new_dev, _ = resolve_device(device, warn=True)
    if new_dev == bundle.device:
        return bundle
    bundle.model.to(new_dev).eval()
    codec = Qwen3TTSCodec(model_id=bundle.codec.model_id, device=new_dev)
    return ListenBundle(
        model=bundle.model,
        cfg=bundle.cfg,
        tokenizer=bundle.tokenizer,
        codec=codec,
        device=new_dev,
        ckpt_path=bundle.ckpt_path,
        tokenizer_id=bundle.tokenizer_id,
        gen_kw=dict(bundle.gen_kw),
        pad_token_id=bundle.pad_token_id,
        speaker_map=dict(bundle.speaker_map),
        default_speaker_id=bundle.default_speaker_id,
    )

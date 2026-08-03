"""Shared discrete Speech LM inference / stop-EOS contract (D1b).

Stop / EOS contract
-------------------
On D1b the **only** trainable termination mechanism is the frame-level
``stop_head`` (binary label on the last valid ``frame_mask`` frame).

- ``SPEECH_EOS_ID=2048`` may remain in the codec-token cache schema for forward
  compatibility, but it is **not** a trainable CB0 EOS id
  (``codebook_size=2048`` → valid CB0 ids are ``0..2047``).
- The model backbone consumes text prefix + acoustic frames; there is **no**
  separate EOS token in the LM sequence.
- ``stop_targets`` are built via
  :meth:`DiscreteSpeechLM.stop_targets_from_mask` (last valid frame only).
- Collate may still emit unused ``eos_targets`` for packing diagnostics; those
  are **not** model-EOS supervision. Prefer ``stop_targets`` / stop_head loss.
  Trainer path should ignore ``eos_targets`` (FIXME for trainer agent if still
  referenced).

Scripts 73 (infer) and 74 (eval) **must** call
:func:`resolve_discrete_infer_settings` so defaults match the frozen WAVE3c
eval contract. Do not fork fallback defaults in those scripts.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping

# Frozen D1 WAVE3c canonical infer knobs (eval_contract / plan §6.4).
WAVE3C_INFER_DEFAULTS: dict[str, Any] = {
    "max_frames": 750,
    "temperature": 0.0,
    "top_k": 0,
    "eos_cb0": None,
    "stop_threshold": 0.5,
    "stop_consecutive": 1,
    "min_frames": 4,
    "max_cb0_repeat": 12,
    "repetition_penalty": 2.0,
    "frames_per_text_token": 2.5,
    "stop_overrun_bias": 0.0,
    "stop_earliest_frac": 0.7,
}

# Keys passed through to ``DiscreteSpeechLM.generate``.
GENERATE_INFER_KEYS: tuple[str, ...] = (
    "max_frames",
    "temperature",
    "top_k",
    "eos_cb0",
    "stop_threshold",
    "stop_consecutive",
    "min_frames",
    "max_cb0_repeat",
    "repetition_penalty",
    "frames_per_text_token",
    "stop_overrun_bias",
    "stop_earliest_frac",
)

# Contract / diagnostic keys allowed in ``infer:`` but not generate kwargs.
INFER_META_KEYS: frozenset[str] = frozenset(
    {
        "sampler",
        "source",
        "shared_resolver_required",
        "shared_resolver_module",
        "shared_resolver_status",
        "calibration_allowed_keys",
    }
)


class DiscreteConfigError(ValueError):
    """Invalid discrete Speech LM config / infer settings."""


def wave3c_infer_defaults() -> dict[str, Any]:
    """Return a copy of the frozen WAVE3c inference defaults."""
    return deepcopy(WAVE3C_INFER_DEFAULTS)


def _coerce_eos_cb0(raw: Any) -> int | None:
    if raw is None:
        return None
    if isinstance(raw, str) and raw.strip().lower() in {"", "null", "none"}:
        return None
    return int(raw)


def resolve_discrete_infer_settings(
    cfg: Mapping[str, Any] | None,
    *,
    overrides: Mapping[str, Any] | None = None,
    max_frames_override: int | None = None,
    codebook_size: int = 2048,
    validate: bool = True,
) -> dict[str, Any]:
    """Resolve generate kwargs for model / script 73 / script 74 identically.

    Precedence: ``WAVE3C_INFER_DEFAULTS`` < ``cfg['infer']`` < ``overrides``
    < ``max_frames_override`` (when not None).
    """
    out = wave3c_infer_defaults()
    if cfg is not None:
        icfg = cfg.get("infer") if isinstance(cfg, Mapping) else None
        if isinstance(icfg, Mapping):
            for key in GENERATE_INFER_KEYS:
                if key in icfg and icfg[key] is not None:
                    out[key] = icfg[key]
            # Allow explicit null eos_cb0 in YAML.
            if "eos_cb0" in icfg:
                out["eos_cb0"] = icfg["eos_cb0"]
    if overrides:
        for key, val in overrides.items():
            if key in GENERATE_INFER_KEYS or key in INFER_META_KEYS:
                out[key] = val
    if max_frames_override is not None:
        out["max_frames"] = int(max_frames_override)

    # Typed coerce for generate().
    resolved = {
        "max_frames": int(out["max_frames"]),
        "temperature": float(out["temperature"]),
        "top_k": int(out.get("top_k") or 0),
        "eos_cb0": _coerce_eos_cb0(out.get("eos_cb0")),
        "stop_threshold": float(out["stop_threshold"]),
        "stop_consecutive": int(out["stop_consecutive"]),
        "min_frames": int(out["min_frames"]),
        "max_cb0_repeat": int(out["max_cb0_repeat"]),
        "repetition_penalty": float(out["repetition_penalty"]),
        "frames_per_text_token": float(out["frames_per_text_token"]),
        "stop_overrun_bias": float(out["stop_overrun_bias"]),
        "stop_earliest_frac": float(out["stop_earliest_frac"]),
    }
    if validate:
        validate_discrete_infer_settings(resolved, codebook_size=codebook_size)
    return resolved


def validate_discrete_infer_settings(
    settings: Mapping[str, Any],
    *,
    codebook_size: int = 2048,
) -> None:
    """Fail closed on incompatible stop / EOS infer settings."""
    eos = settings.get("eos_cb0")
    if eos is not None:
        eos_i = int(eos)
        if eos_i < 0 or eos_i >= int(codebook_size):
            raise DiscreteConfigError(
                f"eos_cb0={eos_i} outside codebook vocab [0, {codebook_size}). "
                "SPEECH_EOS_ID=2048 is not a trainable CB0 EOS; use stop_head "
                "(eos_cb0: null) instead."
            )

    thr = float(settings.get("stop_threshold", 0.5))
    if not (0.0 < thr <= 1.0):
        raise DiscreteConfigError(f"stop_threshold={thr} must be in (0, 1]")

    consec = int(settings.get("stop_consecutive", 1))
    if consec < 1:
        raise DiscreteConfigError(f"stop_consecutive={consec} must be >= 1")

    min_f = int(settings.get("min_frames", 4))
    if min_f < 1:
        raise DiscreteConfigError(f"min_frames={min_f} must be >= 1")

    max_f = int(settings.get("max_frames", 750))
    if max_f < min_f:
        raise DiscreteConfigError(
            f"max_frames={max_f} must be >= min_frames={min_f}"
        )

    frac = float(settings.get("stop_earliest_frac", 0.7))
    if not (0.0 <= frac <= 1.5):
        raise DiscreteConfigError(
            f"stop_earliest_frac={frac} must be in [0, 1.5]"
        )

    temp = float(settings.get("temperature", 0.0))
    if temp < 0.0:
        raise DiscreteConfigError(f"temperature={temp} must be >= 0")


def validate_discrete_lm_config(cfg: Mapping[str, Any]) -> None:
    """Validate production / D1b discrete Speech LM YAML (fail closed)."""
    if not isinstance(cfg, Mapping):
        raise DiscreteConfigError("config must be a mapping")

    model = cfg.get("model") or {}
    if not isinstance(model, Mapping):
        raise DiscreteConfigError("model: must be a mapping")

    codebook_size = int(model.get("codebook_size", 2048))
    vocab = int(model.get("vocab_size", 0))
    pad_id = model.get("pad_token_id")
    if vocab <= 0:
        raise DiscreteConfigError("model.vocab_size must be > 0")
    if pad_id is not None and not (0 <= int(pad_id) < vocab):
        raise DiscreteConfigError(
            f"model.pad_token_id={pad_id} outside vocab [0, {vocab})"
        )

    tic = model.get("text_init") or {}
    if isinstance(tic, Mapping) and bool(tic.get("enabled", False)):
        # Fail-closed reminder: init path must not silently truncate vocab.
        if "allow_unsafe_vocab_truncation" not in tic:
            # enabled without explicit unsafe flag is OK only when vocabs match;
            # apply_qwen_text_init enforces that at runtime.
            pass
        min_cov = tic.get("min_coverage")
        if min_cov is not None and not (0.0 <= float(min_cov) <= 1.0):
            raise DiscreteConfigError(
                f"model.text_init.min_coverage={min_cov} must be in [0, 1]"
            )

    infer = cfg.get("infer") or {}
    if infer:
        if not isinstance(infer, Mapping):
            raise DiscreteConfigError("infer: must be a mapping")
        # Merge with defaults then validate (catches eos_cb0=2048 etc.).
        resolve_discrete_infer_settings(
            {"infer": infer}, codebook_size=codebook_size, validate=True
        )

    train = cfg.get("train") or {}
    if isinstance(train, Mapping):
        step_unit = train.get("step_unit") or train.get("global_step_unit")
        if step_unit is not None and str(step_unit) not in {
            "optimizer",
            "optimizer_step",
            "optimizer_update",
        }:
            raise DiscreteConfigError(
                f"train.step_unit={step_unit!r} must be optimizer-step semantics "
                "(optimizer / optimizer_step / optimizer_update); microbatch "
                "counting is forbidden on D1b"
            )


def infer_settings_match_wave3c(
    settings: Mapping[str, Any],
    *,
    ignore_max_frames: bool = False,
) -> list[str]:
    """Return list of keys that differ from frozen WAVE3c defaults."""
    ref = wave3c_infer_defaults()
    diffs: list[str] = []
    for key in GENERATE_INFER_KEYS:
        if ignore_max_frames and key == "max_frames":
            continue
        if key not in settings:
            diffs.append(key)
            continue
        left = settings[key]
        right = ref[key]
        if key == "eos_cb0":
            if _coerce_eos_cb0(left) != _coerce_eos_cb0(right):
                diffs.append(key)
        elif isinstance(right, float):
            if abs(float(left) - float(right)) > 1e-9:
                diffs.append(key)
        elif isinstance(right, int):
            if int(left) != int(right):
                diffs.append(key)
        elif left != right:
            diffs.append(key)
    return diffs


def contract_infer_from_eval_contract(contract: Mapping[str, Any]) -> dict[str, Any]:
    """Extract generate-relevant infer knobs from ``eval_contract.json``."""
    block = contract.get("infer") or {}
    if not isinstance(block, Mapping):
        raise DiscreteConfigError("eval_contract.infer must be a mapping")
    # Contract may omit max_frames / eos_cb0; fill from WAVE3c defaults.
    merged = wave3c_infer_defaults()
    for key in GENERATE_INFER_KEYS:
        if key in block:
            merged[key] = block[key]
    return resolve_discrete_infer_settings({"infer": merged}, validate=True)


__all__ = [
    "DiscreteConfigError",
    "GENERATE_INFER_KEYS",
    "WAVE3C_INFER_DEFAULTS",
    "contract_infer_from_eval_contract",
    "infer_settings_match_wave3c",
    "resolve_discrete_infer_settings",
    "validate_discrete_infer_settings",
    "validate_discrete_lm_config",
    "wave3c_infer_defaults",
]

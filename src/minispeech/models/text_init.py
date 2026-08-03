"""Optional Qwen text-block initialization for DiscreteSpeechLM.

Policy (D1 / D1b)
-----------------
- Text embeddings + transformer blocks: **may** copy from a Qwen causal LM.
- Speech embeddings, modality adapters, MTP / output heads, stop head: **always**
  scratch.
- Default is **fail-closed** on tokenizer / vocab mismatch and on silent embedding
  row truncation. Truncation requires an explicit unsafe flag.
- Mapping coverage below ``min_coverage`` fails (no silent under-init).
- D1b production runs keep ``text_init.enabled: false``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn as nn


class TextInitError(RuntimeError):
    """Fail-closed text-init policy violation."""


@dataclass
class TextInitReport:
    """Coverage of Qwen → DiscreteSpeechLM weight mapping."""

    source: str
    mapped_keys: list[str] = field(default_factory=list)
    skipped_keys: list[str] = field(default_factory=list)
    missing_targets: list[str] = field(default_factory=list)
    shape_mismatches: list[str] = field(default_factory=list)
    n_mapped_params: int = 0
    n_total_src_params: int = 0

    @property
    def coverage(self) -> float:
        if self.n_total_src_params <= 0:
            return 0.0
        return float(self.n_mapped_params) / float(self.n_total_src_params)

    def as_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "coverage": self.coverage,
            "n_mapped_params": self.n_mapped_params,
            "n_total_src_params": self.n_total_src_params,
            "n_mapped_keys": len(self.mapped_keys),
            "n_skipped_keys": len(self.skipped_keys),
            "n_missing_targets": len(self.missing_targets),
            "n_shape_mismatches": len(self.shape_mismatches),
            "mapped_keys_head": self.mapped_keys[:32],
            "shape_mismatches_head": self.shape_mismatches[:16],
            "skipped_keys_head": self.skipped_keys[:32],
        }


# Keys that must never receive Qwen text weights (speech / heads / adapters).
_SCRATCH_PREFIXES: tuple[str, ...] = (
    "speech_emb",
    "speech_start",
    "speaker_emb",
    "heads",
    "ru_cond",
    "modality",
    "modality_speech",
    "stop_head",
    "acoustic",
)


def _is_scratch_key(name: str) -> bool:
    return any(name == p or name.startswith(p + ".") for p in _SCRATCH_PREFIXES)


def _load_qwen_state(model_id: str) -> tuple[dict[str, torch.Tensor], str]:
    """Load Qwen causal-LM weights without constructing a full generation graph."""
    try:
        from transformers import AutoModelForCausalLM
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "transformers is required for Qwen text-init"
        ) from exc

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
    )
    sd = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    del model
    return sd, model_id


def _candidate_maps(src_key: str) -> list[str]:
    """Map HuggingFace Qwen module names → DiscreteSpeechLM names."""
    aliases: list[str] = []
    k = src_key
    if k.startswith("model."):
        k = k[len("model.") :]
    if k.startswith("transformer."):
        k = k[len("transformer.") :]

    if k in {"embed_tokens.weight", "wte.weight"}:
        aliases.append("text_emb.weight")
    if k.startswith("layers."):
        # layers.N.* → blocks.N.*
        rest = k[len("layers.") :]
        aliases.append("blocks." + rest)
        # HF Llama/Qwen attention naming → our block naming
        aliases.append(
            "blocks."
            + rest.replace("self_attn.", "attn.")
            .replace("input_layernorm.", "norm1.")
            .replace("post_attention_layernorm.", "norm2.")
            .replace("mlp.", "ff.")
        )
    if k in {"norm.weight", "norm.bias", "final_layernorm.weight"}:
        aliases.append("norm.weight" if "bias" not in k else "norm.bias")
    return list(dict.fromkeys(aliases))


def _src_embed_rows(src_state: dict[str, torch.Tensor]) -> int | None:
    for key in (
        "model.embed_tokens.weight",
        "embed_tokens.weight",
        "transformer.wte.weight",
        "wte.weight",
    ):
        t = src_state.get(key)
        if t is not None and t.ndim == 2:
            return int(t.shape[0])
    for key, t in src_state.items():
        if key.endswith("embed_tokens.weight") and t.ndim == 2:
            return int(t.shape[0])
    return None


def _model_vocab_size(model: nn.Module) -> int | None:
    cfg = getattr(model, "cfg", None)
    if cfg is not None and hasattr(cfg, "vocab_size"):
        return int(cfg.vocab_size)
    emb = getattr(model, "text_emb", None)
    if isinstance(emb, nn.Embedding):
        return int(emb.num_embeddings)
    return None


@torch.no_grad()
def apply_qwen_text_init(
    model: nn.Module,
    *,
    model_id: str = "Qwen/Qwen3-0.6B",
    src_state: dict[str, torch.Tensor] | None = None,
    strict_shapes: bool = True,
    expected_vocab_size: int | None = None,
    allow_unsafe_vocab_truncation: bool = False,
    min_coverage: float = 0.05,
    tokenizer_id: str | None = None,
) -> TextInitReport:
    """Copy compatible text embeddings / blocks from Qwen into ``model``.

    Speech / MTP / speaker / stop_head / modality_speech are never overwritten.

    Fail-closed defaults
    --------------------
    - Source embed rows must match ``expected_vocab_size`` / model vocab unless
      ``allow_unsafe_vocab_truncation=True``.
    - Silent partial-row copy is forbidden without that flag.
    - ``report.coverage < min_coverage`` raises :class:`TextInitError`.
    """
    if src_state is None:
        src_state, source = _load_qwen_state(model_id)
    else:
        source = model_id

    target_vocab = expected_vocab_size
    if target_vocab is None:
        target_vocab = _model_vocab_size(model)
    src_rows = _src_embed_rows(src_state)

    if (
        target_vocab is not None
        and src_rows is not None
        and int(src_rows) != int(target_vocab)
        and not allow_unsafe_vocab_truncation
    ):
        raise TextInitError(
            f"text-init vocab mismatch: source embed rows={src_rows} vs "
            f"model vocab_size={target_vocab} (source={source!r}"
            + (f", tokenizer_id={tokenizer_id!r}" if tokenizer_id else "")
            + "). Refusing silent truncation; set "
            "allow_unsafe_vocab_truncation=true only for explicit experiments."
        )

    if (
        tokenizer_id is not None
        and model_id is not None
        and str(tokenizer_id).rstrip("/") != str(model_id).rstrip("/")
        and not allow_unsafe_vocab_truncation
    ):
        # Different ids often imply different vocabs; fail closed unless unsafe.
        # (D1b keeps text_init.enabled=false; Wave-1 text_ids are Qwen3.5.)
        raise TextInitError(
            f"text-init tokenizer/model_id mismatch: tokenizer_id={tokenizer_id!r} "
            f"vs model_id={model_id!r}. Refusing; align ids or set "
            "allow_unsafe_vocab_truncation=true."
        )

    target_sd = model.state_dict()
    report = TextInitReport(source=source)
    report.n_total_src_params = int(sum(v.numel() for v in src_state.values()))

    mapped: dict[str, torch.Tensor] = {}
    used_targets: set[str] = set()

    for src_key, src_tensor in src_state.items():
        candidates = _candidate_maps(src_key)
        if not candidates:
            report.skipped_keys.append(src_key)
            continue
        hit = None
        for cand in candidates:
            if cand in target_sd and not _is_scratch_key(cand):
                hit = cand
                break
        if hit is None:
            report.skipped_keys.append(src_key)
            continue
        tgt = target_sd[hit]
        if tuple(tgt.shape) != tuple(src_tensor.shape):
            # Vocab row mismatch: never silently truncate without explicit flag.
            if (
                hit == "text_emb.weight"
                and src_tensor.ndim == 2
                and tgt.ndim == 2
                and src_tensor.shape[1] == tgt.shape[1]
            ):
                if not allow_unsafe_vocab_truncation:
                    raise TextInitError(
                        f"text-init embedding shape mismatch {src_key}->{hit}: "
                        f"{tuple(src_tensor.shape)} vs {tuple(tgt.shape)}. "
                        "Silent row truncation is forbidden; set "
                        "allow_unsafe_vocab_truncation=true to override."
                    )
                rows = min(src_tensor.shape[0], tgt.shape[0])
                buf = tgt.clone()
                buf[:rows].copy_(src_tensor[:rows])
                mapped[hit] = buf
                used_targets.add(hit)
                report.mapped_keys.append(f"{src_key}->{hit}[:{rows}]")
                report.n_mapped_params += int(rows * src_tensor.shape[1])
                report.shape_mismatches.append(
                    f"{src_key}->{hit}: truncated rows {tuple(src_tensor.shape)} "
                    f"→ {tuple(tgt.shape)} (unsafe)"
                )
                continue
            report.shape_mismatches.append(
                f"{src_key}->{hit}: {tuple(src_tensor.shape)} vs {tuple(tgt.shape)}"
            )
            if strict_shapes:
                continue
        mapped[hit] = src_tensor.to(dtype=tgt.dtype)
        used_targets.add(hit)
        report.mapped_keys.append(f"{src_key}->{hit}")
        report.n_mapped_params += int(src_tensor.numel())

    # Ensure we did not map scratch modules.
    for k in list(mapped):
        if _is_scratch_key(k):
            del mapped[k]

    incompatible = model.load_state_dict(mapped, strict=False)
    report.missing_targets = list(incompatible.missing_keys)

    # Double-check scratch modules unchanged policy: re-assert their keys exist
    for name, _ in model.named_parameters():
        if _is_scratch_key(name) and name in mapped:  # pragma: no cover
            raise RuntimeError(f"text-init illegally mapped scratch param {name}")

    if float(report.coverage) < float(min_coverage):
        raise TextInitError(
            f"text-init coverage {report.coverage:.6f} < min_coverage "
            f"{float(min_coverage):.6f} (source={source!r})"
        )

    return report


def text_init_manifest_extra(report: TextInitReport | None) -> dict[str, Any]:
    """Fragment for ``write_run_manifest(..., extra=...)``."""
    if report is None:
        return {"text_init": {"enabled": False}}
    return {"text_init": {"enabled": True, **report.as_dict()}}


def apply_text_init_from_cfg(
    model: nn.Module, cfg: dict[str, Any]
) -> TextInitReport | None:
    """Apply text-init from a full YAML dict when ``model.text_init.enabled``.

    Prefer this over ad-hoc kwargs so fail-closed flags
    (``allow_unsafe_vocab_truncation``, ``min_coverage``, tokenizer id) are
    honored. Trainer agent can switch ``maybe_apply_text_init`` to this helper.
    """
    model_block = cfg.get("model") if isinstance(cfg, dict) else None
    tic = {}
    if isinstance(model_block, dict):
        raw = model_block.get("text_init") or {}
        if isinstance(raw, dict):
            tic = raw
        tokenizer_id = model_block.get("text_tokenizer_id")
    else:
        tokenizer_id = None
    if not tic and isinstance(cfg, dict):
        raw = cfg.get("text_init") or {}
        if isinstance(raw, dict):
            tic = raw
    if not bool(tic.get("enabled", False)):
        return None
    return apply_qwen_text_init(
        model,
        model_id=str(tic.get("model_id", "Qwen/Qwen3-0.6B")),
        allow_unsafe_vocab_truncation=bool(
            tic.get("allow_unsafe_vocab_truncation", False)
        ),
        min_coverage=float(tic.get("min_coverage", 0.05)),
        tokenizer_id=str(tokenizer_id) if tokenizer_id else None,
        expected_vocab_size=(
            int(model_block["vocab_size"])
            if isinstance(model_block, dict) and "vocab_size" in model_block
            else None
        ),
    )

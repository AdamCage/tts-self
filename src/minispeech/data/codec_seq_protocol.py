"""Stable batch contract for discrete Speech LM (Wave 2).

Wave 1 (`codec_tokens.py` / `collate_codec_seq.py` / `71_encode_*`) may land in
parallel. Trainers import :func:`resolve_collate_codec_seq`, which prefers the
Wave-1 collate when present and otherwise uses the stub below.

Expected per-item keys (dataset / shard row)
-------------------------------------------
- ``codes``: ``[T, 16]`` int64 (CB0..CB15), values in ``[0, codebook_size)``
- ``input_ids``: 1-D long text token ids (stressed natural Russian BPE)
- ``speaker_id``: int (trainable speaker-ID embedding index from SpeakerRegistry;
  strings are rejected — resolve labels before collate)
- optional masks / lengths: ``n_frames``, ``text_len``
- optional RU condition channels (token-aligned, off until text-only base PASS):
  ``phoneme_ids``, ``stress``, ``word_bound``, ``phrase_bound``, ``span_type``
- optional: ``utt_id``, ``sample_weight``, ``dur_s``, ``corpus``
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Protocol, Sequence, runtime_checkable

import torch

from minispeech.constants import QWEN3TTS_12HZ, get_target
from minispeech.data.speaker_registry import assert_integer_speaker_id


@runtime_checkable
class CodecSeqItem(Protocol):
    """Minimal protocol for a single training example."""

    def __getitem__(self, key: str) -> Any: ...


@dataclass
class CodecSeqBatch:
    """Padded batch consumed by :class:`DiscreteSpeechLM` / trainer."""

    input_ids: torch.Tensor  # [B, L]
    text_mask: torch.Tensor  # [B, L] bool
    codes: torch.Tensor  # [B, T, n_codebooks]
    frame_mask: torch.Tensor  # [B, T] bool
    speaker_id: torch.Tensor  # [B] long
    text_lens: torch.Tensor  # [B]
    acoustic_lens: torch.Tensor  # [B]
    sample_weight: torch.Tensor  # [B]
    # Optional RU channels (None when disabled / absent)
    phoneme_ids: torch.Tensor | None = None
    stress: torch.Tensor | None = None
    word_bound: torch.Tensor | None = None
    phrase_bound: torch.Tensor | None = None
    span_type: torch.Tensor | None = None
    utt_ids: list[str] | None = None

    def as_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "input_ids": self.input_ids,
            "text_mask": self.text_mask,
            "codes": self.codes,
            "frame_mask": self.frame_mask,
            "speaker_id": self.speaker_id,
            "text_lens": self.text_lens,
            "acoustic_lens": self.acoustic_lens,
            "sample_weight": self.sample_weight,
        }
        for key in (
            "phoneme_ids",
            "stress",
            "word_bound",
            "phrase_bound",
            "span_type",
        ):
            val = getattr(self, key)
            if val is not None:
                out[key] = val
        if self.utt_ids is not None:
            out["utt_ids"] = self.utt_ids
        return out


def _as_long_1d(x: Any) -> torch.Tensor:
    if not torch.is_tensor(x):
        x = torch.tensor(x, dtype=torch.long)
    return x.long().view(-1)


def _as_codes(x: Any, *, n_codebooks: int) -> torch.Tensor:
    if not torch.is_tensor(x):
        x = torch.tensor(x, dtype=torch.long)
    x = x.long()
    if x.ndim != 2 or x.shape[-1] != n_codebooks:
        raise ValueError(
            f"codes must be [T, {n_codebooks}], got shape {tuple(x.shape)}"
        )
    return x


def collate_codec_seq(
    batch: Sequence[Mapping[str, Any]],
    *,
    pad_token_id: int = 0,
    target: str = QWEN3TTS_12HZ.name,
    include_ru_channels: bool = False,
) -> dict[str, Any]:
    """Pad text ids + multi-codebook codes for discrete LM training.

    This stub matches the Wave-1 contract so trainers can run unit / oracle
    tests before shards land. Prefer
    ``minispeech.data.collate_codec_seq.collate_codec_seq`` when available.
    """
    if not batch:
        raise ValueError("empty batch")
    spec = get_target(target)
    n_cb = int(spec.n_codebooks or 16)
    B = len(batch)

    ids_list = [_as_long_1d(b["input_ids"]) for b in batch]
    codes_list = [_as_codes(b["codes"], n_codebooks=n_cb) for b in batch]
    L = max(int(x.numel()) for x in ids_list)
    T = max(int(c.shape[0]) for c in codes_list)

    input_ids = torch.full((B, L), pad_token_id, dtype=torch.long)
    text_mask = torch.zeros(B, L, dtype=torch.bool)
    codes = torch.zeros(B, T, n_cb, dtype=torch.long)
    frame_mask = torch.zeros(B, T, dtype=torch.bool)
    speaker_id = torch.zeros(B, dtype=torch.long)
    sample_weight = torch.ones(B)
    text_lens = torch.zeros(B, dtype=torch.long)
    acoustic_lens = torch.zeros(B, dtype=torch.long)
    utt_ids: list[str] = []

    ru_keys = ("phoneme_ids", "stress", "word_bound", "phrase_bound", "span_type")
    ru_bufs: dict[str, torch.Tensor] | None = None
    if include_ru_channels or any(k in batch[0] for k in ru_keys):
        ru_bufs = {k: torch.zeros(B, L, dtype=torch.long) for k in ru_keys}

    for i, b in enumerate(batch):
        ids = ids_list[i]
        tl = int(ids.numel())
        input_ids[i, :tl] = ids
        text_mask[i, :tl] = True
        text_lens[i] = tl

        c = codes_list[i]
        al = int(c.shape[0])
        codes[i, :al] = c
        frame_mask[i, :al] = True
        acoustic_lens[i] = al

        if "speaker_id" not in b or b["speaker_id"] is None:
            raise KeyError(
                "batch item missing integer speaker_id; resolve labels with "
                "SpeakerRegistry before collate"
            )
        speaker_id[i] = assert_integer_speaker_id(b["speaker_id"])
        if "sample_weight" in b:
            sample_weight[i] = float(b["sample_weight"])
        utt_ids.append(str(b.get("utt_id", i)))

        if ru_bufs is not None:
            for k in ru_keys:
                if k not in b:
                    continue
                ch = _as_long_1d(b[k])
                n = min(tl, int(ch.numel()))
                ru_bufs[k][i, :n] = ch[:n]

    out = CodecSeqBatch(
        input_ids=input_ids,
        text_mask=text_mask,
        codes=codes,
        frame_mask=frame_mask,
        speaker_id=speaker_id,
        text_lens=text_lens,
        acoustic_lens=acoustic_lens,
        sample_weight=sample_weight,
        phoneme_ids=None if ru_bufs is None else ru_bufs["phoneme_ids"],
        stress=None if ru_bufs is None else ru_bufs["stress"],
        word_bound=None if ru_bufs is None else ru_bufs["word_bound"],
        phrase_bound=None if ru_bufs is None else ru_bufs["phrase_bound"],
        span_type=None if ru_bufs is None else ru_bufs["span_type"],
        utt_ids=utt_ids,
    )
    return out.as_dict()


def _normalize_item_for_collate(b: Mapping[str, Any]) -> dict[str, Any]:
    """Map Wave-1 shard keys → trainer contract (``input_ids``, int ``speaker_id``)."""
    out = dict(b)
    if "input_ids" not in out and "text_ids" in out:
        out["input_ids"] = out["text_ids"]
    if "speaker_id" not in out or out["speaker_id"] is None:
        raise KeyError(
            "item missing integer speaker_id; resolve labels with SpeakerRegistry "
            "before collate (string hash fallback removed)"
        )
    out["speaker_id"] = assert_integer_speaker_id(out["speaker_id"])
    if "phoneme_ids" not in out and "phoneme" in out:
        out["phoneme_ids"] = out["phoneme"]
    return out


def _adapt_landed_batch(batch: dict[str, Any], items: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Expose trainer keys alongside Wave-1 pack fields."""
    out = dict(batch)
    if "codes" not in out and "codes_target" in out:
        out["codes"] = out["codes_target"]
    if "speaker_id" not in out:
        spk = [assert_integer_speaker_id(b["speaker_id"]) for b in items]
        out["speaker_id"] = torch.tensor(spk, dtype=torch.long)
    if "phoneme_ids" not in out and "phoneme" in out:
        out["phoneme_ids"] = out["phoneme"]
    return out


def resolve_collate_codec_seq():
    """Prefer Wave-1 collate when landed; adapt keys to DiscreteLMTrainer contract."""
    try:
        from minispeech.data.collate_codec_seq import (  # type: ignore
            collate_codec_seq as landed,
        )

        def _wrapped(
            batch: Sequence[Mapping[str, Any]],
            *,
            pad_token_id: int = 0,
            **kwargs: Any,
        ) -> dict[str, Any]:
            items = [_normalize_item_for_collate(b) for b in batch]
            raw = landed(items, pad_token_id=pad_token_id, **kwargs)
            return _adapt_landed_batch(raw, items)

        return _wrapped
    except ImportError:
        def _stub_wrapped(
            batch: Sequence[Mapping[str, Any]],
            *,
            pad_token_id: int = 0,
            **kwargs: Any,
        ) -> dict[str, Any]:
            items = [_normalize_item_for_collate(b) for b in batch]
            return collate_codec_seq(items, pad_token_id=pad_token_id, **kwargs)

        return _stub_wrapped

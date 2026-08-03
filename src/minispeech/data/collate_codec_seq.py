"""Collate packed ``[text ∥ codes ∥ EOS]`` sequences for discrete Speech LM.

Mirrors the N2 latent collator contract but for multi-codebook integer codes.
Train/infer packing equality is enforced via
``minispeech.data.codec_tokens.pack_codec_sequence`` (shared helper).
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import torch

from minispeech.constants import QWEN3TTS_12HZ, get_target
from minispeech.data.codec_tokens import (
    SPEECH_EOS_ID,
    RuTextFeatures,
    assert_pack_train_infer_equal,
    pack_codec_sequence,
    pack_for_infer,
)
from minispeech.data.speaker_registry import assert_integer_speaker_id


def collate_codec_seq(
    batch: list[dict[str, Any]],
    *,
    pad_token_id: int = 0,
    speech_eos_id: int = SPEECH_EOS_ID,
    target: str = QWEN3TTS_12HZ.name,
) -> dict[str, Any]:
    """Pad text ids + ``[T, C]`` codes; build modality / AR-shift masks.

    Expected per-item keys
    ----------------------
    - ``text_ids`` / ``input_ids``: 1-D long
    - ``codes``: ``[T, n_codebooks]``
    - optional RU features: ``stress``, ``phoneme``, ``span_type``,
      ``word_bound``, ``phrase_bound``
    - optional: ``utt_id``, ``sample_weight``, ``dur_s``, ``corpus``, ``speaker_id``
    """
    if not batch:
        raise ValueError("empty batch")
    spec = get_target(target)
    n_codebooks = int(spec.n_codebooks or 16)
    B = len(batch)

    packed_list = []
    for b in batch:
        text_ids = b.get("text_ids", b.get("input_ids"))
        codes = b["codes"]
        if not torch.is_tensor(codes):
            codes = torch.as_tensor(codes, dtype=torch.long)
        codes = codes.long()
        if codes.ndim != 2 or codes.shape[-1] != n_codebooks:
            raise ValueError(
                f"codes expected [T,{n_codebooks}], got {tuple(codes.shape)}"
            )
        if all(k in b for k in ("stress", "phoneme", "span_type", "word_bound", "phrase_bound")):
            feats = RuTextFeatures(
                stress=torch.as_tensor(b["stress"]).long().view(-1),
                phoneme=torch.as_tensor(b["phoneme"]).long().view(-1),
                span_type=torch.as_tensor(b["span_type"]).long().view(-1),
                word_bound=torch.as_tensor(b["word_bound"]).long().view(-1),
                phrase_bound=torch.as_tensor(b["phrase_bound"]).long().view(-1),
            )
        else:
            feats = None
        packed_list.append(
            pack_codec_sequence(
                text_ids, codes, feats, speech_eos_id=speech_eos_id
            )
        )

    L = max(p.text_lens for p in packed_list)
    T = max(p.acoustic_lens for p in packed_list)
    S = L + T + 1  # text + codes + EOS

    input_ids = torch.full((B, L), pad_token_id, dtype=torch.long)
    text_mask = torch.zeros(B, L, dtype=torch.bool)
    codes_target = torch.zeros(B, T, n_codebooks, dtype=torch.long)
    codes_input = torch.zeros(B, T, n_codebooks, dtype=torch.long)
    frame_mask = torch.zeros(B, T, dtype=torch.bool)
    seq_mask = torch.zeros(B, S, dtype=torch.bool)
    is_text = torch.zeros(B, S, dtype=torch.bool)
    is_acoustic = torch.zeros(B, S, dtype=torch.bool)
    is_eos = torch.zeros(B, S, dtype=torch.bool)

    stress = torch.zeros(B, L, dtype=torch.long)
    phoneme = torch.zeros(B, L, dtype=torch.long)
    span_type = torch.zeros(B, L, dtype=torch.long)
    word_bound = torch.zeros(B, L, dtype=torch.long)
    phrase_bound = torch.zeros(B, L, dtype=torch.long)

    text_lens = torch.zeros(B, dtype=torch.long)
    acoustic_lens = torch.zeros(B, dtype=torch.long)
    sample_weight = torch.ones(B)
    durations = torch.zeros(B)
    eos_targets = torch.zeros(B, T + 1, dtype=torch.float32)  # stop on final + EOS slot

    for i, (b, p) in enumerate(zip(batch, packed_list)):
        tl, al = p.text_lens, p.acoustic_lens
        text_lens[i] = tl
        acoustic_lens[i] = al
        input_ids[i, :tl] = p.input_ids
        text_mask[i, :tl] = True
        if al > 0:
            codes_target[i, :al] = p.codes_target
            codes_input[i, :al] = p.codes_input
            frame_mask[i, :al] = True
        # Place modalities into padded full sequence [L_max + T_max + 1]
        is_text[i, :tl] = True
        is_acoustic[i, L : L + al] = True
        is_eos[i, L + al] = True
        seq_mask[i, :tl] = True
        seq_mask[i, L : L + al] = True
        seq_mask[i, L + al] = True
        # EOS supervision: 1 on the EOS position relative to acoustic stream
        if al >= 0:
            eos_targets[i, al] = 1.0

        stress[i, :tl] = p.stress
        phoneme[i, :tl] = p.phoneme
        span_type[i, :tl] = p.span_type
        word_bound[i, :tl] = p.word_bound
        phrase_bound[i, :tl] = p.phrase_bound

        fr = float(spec.frame_rate)
        durations[i] = float(b.get("dur_s", al / fr if fr else 0.0))
        sample_weight[i] = float(b.get("sample_weight", 1.0))

    # Integer speaker ids only (resolved via SpeakerRegistry before collate).
    speaker_id = torch.zeros(B, dtype=torch.long)
    for i, b in enumerate(batch):
        if "speaker_id" not in b or b["speaker_id"] is None:
            raise KeyError(
                "batch item missing integer speaker_id; resolve labels with "
                "SpeakerRegistry before collate"
            )
        speaker_id[i] = assert_integer_speaker_id(b["speaker_id"])

    out: dict[str, Any] = {
        "utt_ids": [b.get("utt_id", f"utt_{i}") for i, b in enumerate(batch)],
        "input_ids": input_ids,
        "text_mask": text_mask,
        "codes": codes_target,  # trainer / DiscreteSpeechLM contract
        "codes_target": codes_target,
        "codes_input": codes_input,
        "speaker_id": speaker_id,
        "frame_mask": frame_mask,
        "seq_mask": seq_mask,
        "is_text": is_text,
        "is_acoustic": is_acoustic,
        "is_eos": is_eos,
        "eos_targets": eos_targets,
        "text_lens": text_lens,
        "acoustic_lens": acoustic_lens,
        "durations": durations,
        "sample_weight": sample_weight,
        "stress": stress,
        "phoneme": phoneme,
        "span_type": span_type,
        "word_bound": word_bound,
        "phrase_bound": phrase_bound,
        "pad_token_id": pad_token_id,
        "speech_eos_id": speech_eos_id,
        "target": target,
        "n_codebooks": n_codebooks,
    }
    if "corpus" in batch[0]:
        out["corpora"] = [b.get("corpus") for b in batch]
    if "speaker_id" in batch[0]:
        out["speaker_ids"] = [b.get("speaker_id") for b in batch]
    return out


def masked_ce_placeholder(
    logits: torch.Tensor,
    target: torch.Tensor,
    frame_mask: torch.Tensor,
) -> torch.Tensor:
    """Padding-invariant mean of per-frame CE-like scores (for order tests).

    ``logits`` / ``target`` are arbitrary same-shaped float tensors here; real
    CE lives in the Speech LM trainer. Used only to prove collate order
    invariance in unit tests.
    """
    err = (logits - target.float()).abs().mean(dim=-1)  # [B, T]
    m = frame_mask.to(dtype=err.dtype)
    return (err * m).sum() / m.sum().clamp_min(1.0)


def items_from_codec_example(item: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize a dataset item / CodecTokenExample-dict for ``collate_codec_seq``."""
    text_ids = item.get("text_ids", item.get("input_ids"))
    if text_ids is None:
        raise KeyError("item missing text_ids/input_ids")
    if "codes" not in item:
        raise KeyError("item missing codes")
    out = {
        "utt_id": item.get("utt_id"),
        "text_ids": text_ids,
        "codes": item["codes"],
        "sample_weight": item.get("sample_weight", 1.0),
        "dur_s": item.get("dur_s"),
        "corpus": item.get("corpus"),
        "speaker_id": item.get("speaker_id"),
    }
    for k in ("stress", "phoneme", "span_type", "word_bound", "phrase_bound"):
        if k in item:
            out[k] = item[k]
    return out


def assert_batch_pack_matches_items(
    batch_items: Sequence[Mapping[str, Any]],
    *,
    speech_eos_id: int = SPEECH_EOS_ID,
) -> None:
    """Per-item train/infer pack equality (no batching artifacts)."""
    from minispeech.data.codec_tokens import CodecTokenExample

    for raw in batch_items:
        item = items_from_codec_example(raw)
        feats = None
        if all(
            k in item
            for k in ("stress", "phoneme", "span_type", "word_bound", "phrase_bound")
        ):
            feats = RuTextFeatures(
                stress=torch.as_tensor(item["stress"]).long().view(-1),
                phoneme=torch.as_tensor(item["phoneme"]).long().view(-1),
                span_type=torch.as_tensor(item["span_type"]).long().view(-1),
                word_bound=torch.as_tensor(item["word_bound"]).long().view(-1),
                phrase_bound=torch.as_tensor(item["phrase_bound"]).long().view(-1),
            )
        text_ids = torch.as_tensor(item["text_ids"]).long().view(-1)
        if feats is None:
            z = torch.zeros(int(text_ids.numel()), dtype=torch.long)
            feats = RuTextFeatures(
                stress=z,
                phoneme=z.clone(),
                span_type=z.clone(),
                word_bound=z.clone(),
                phrase_bound=z.clone(),
            )
        ex = CodecTokenExample(
            utt_id=str(item.get("utt_id") or "x"),
            codes=torch.as_tensor(item["codes"]).long(),
            text_ids=text_ids,
            features=feats,
            corpus=str(item.get("corpus") or "x"),
            speaker_id=str(item.get("speaker_id") or "x"),
        )
        assert_pack_train_infer_equal(ex, speech_eos_id=speech_eos_id)
        # Infer empty-codes pack keeps text features identical.
        inf = pack_for_infer(
            ex.text_ids, ex.features, n_codebooks=ex.codes.shape[-1], speech_eos_id=speech_eos_id
        )
        assert torch.equal(inf.input_ids, ex.text_ids.long().view(-1))
        assert torch.equal(inf.stress, ex.features.stress)

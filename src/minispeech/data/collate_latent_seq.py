"""Collate mixed ``[text tokens ∥ latents]`` sequences for N2 (A3).

Curriculum max duration and batch packing knobs live in
``configs/n2_qwen_latent_v01.yaml`` (``train.curriculum`` / ``train.batch``),
not hard-coded here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import torch

from minispeech.constants import get_target
from minispeech.models.latent_io import build_modality_masks, shift_latents_for_ar


@dataclass(slots=True, frozen=True)
class CurriculumStage:
    max_dur_s: float
    steps: int


def parse_curriculum(cfg: Mapping[str, Any] | None) -> list[CurriculumStage]:
    """Read ``train.curriculum.stages`` from the N2 config."""
    if not cfg:
        return []
    stages = cfg.get("stages") if "stages" in cfg else cfg
    if not isinstance(stages, Sequence):
        raise TypeError("curriculum.stages must be a list")
    out: list[CurriculumStage] = []
    for s in stages:
        out.append(
            CurriculumStage(max_dur_s=float(s["max_dur_s"]), steps=int(s["steps"]))
        )
    return out


def curriculum_max_dur_s(stages: Sequence[CurriculumStage], global_step: int) -> float:
    """Resolve active ``max_dur_s`` for ``global_step`` (exclusive end of stages)."""
    if not stages:
        return float("inf")
    acc = 0
    for st in stages:
        acc += st.steps
        if global_step < acc:
            return st.max_dur_s
    return stages[-1].max_dur_s


def filter_indices_by_duration(
    durations_s: Sequence[float],
    max_dur_s: float,
) -> list[int]:
    """Indices with ``dur_s <= max_dur_s`` (inclusive)."""
    return [i for i, d in enumerate(durations_s) if float(d) <= float(max_dur_s) + 1e-9]


def frames_from_duration_s(dur_s: float, *, target: str = "vibevoice64") -> int:
    """Convert seconds → latent frames via ``TARGETS`` frame_rate."""
    fr = get_target(target).frame_rate
    return max(1, int(round(float(dur_s) * fr)))


def collate_latent_seq(
    batch: list[dict[str, Any]],
    *,
    pad_token_id: int = 0,
    bos_latent: torch.Tensor | None = None,
    target: str = "vibevoice64",
) -> dict[str, Any]:
    """Pad text token ids + latents; build modality / AR-shift masks.

    Expected per-item keys
    ----------------------
    - ``input_ids``: 1-D long tensor / list of token ids
    - ``z_target``: ``[T, latent_dim]``
    - optional: ``utt_id``, ``sample_weight``, ``dur_s``, ``corpus``, ``speaker_id``
    """
    if not batch:
        raise ValueError("empty batch")
    latent_dim = get_target(target).dim
    B = len(batch)

    ids_list: list[torch.Tensor] = []
    for b in batch:
        ids = b["input_ids"]
        if not torch.is_tensor(ids):
            ids = torch.tensor(ids, dtype=torch.long)
        else:
            ids = ids.long().view(-1)
        ids_list.append(ids)

    L = max(int(x.numel()) for x in ids_list)
    T = max(int(b["z_target"].shape[0]) for b in batch)

    input_ids = torch.full((B, L), pad_token_id, dtype=torch.long)
    text_mask = torch.zeros(B, L, dtype=torch.bool)
    z_target = torch.zeros(B, T, latent_dim)
    frame_mask = torch.zeros(B, T, dtype=torch.bool)
    sample_weight = torch.ones(B)
    durations = torch.zeros(B)
    text_lens = torch.zeros(B, dtype=torch.long)
    acoustic_lens = torch.zeros(B, dtype=torch.long)

    for i, b in enumerate(batch):
        ids = ids_list[i]
        tl = int(ids.numel())
        input_ids[i, :tl] = ids
        text_mask[i, :tl] = True
        text_lens[i] = tl

        z = b["z_target"]
        if not torch.is_tensor(z):
            z = torch.tensor(z, dtype=torch.float32)
        z = z.float()
        if z.shape[-1] != latent_dim:
            raise ValueError(
                f"z_target dim {z.shape[-1]} != TARGETS[{target}].dim={latent_dim}"
            )
        t = int(z.shape[0])
        z_target[i, :t] = z
        frame_mask[i, :t] = True
        acoustic_lens[i] = t
        durations[i] = float(b.get("dur_s", t / get_target(target).frame_rate))
        sample_weight[i] = float(b.get("sample_weight", 1.0))

    z_input = shift_latents_for_ar(z_target, frame_mask, bos=bos_latent)
    masks = build_modality_masks(text_lens, acoustic_lens)

    out: dict[str, Any] = {
        "utt_ids": [b.get("utt_id", f"utt_{i}") for i, b in enumerate(batch)],
        "input_ids": input_ids,
        "text_mask": text_mask,
        "z_target": z_target,
        "z_input": z_input,
        "frame_mask": frame_mask,
        "seq_mask": masks.seq_mask,
        "is_acoustic": masks.is_acoustic,
        "text_lens": text_lens,
        "acoustic_lens": acoustic_lens,
        "durations": durations,
        "sample_weight": sample_weight,
        "pad_token_id": pad_token_id,
        "target": target,
    }
    if "corpus" in batch[0]:
        out["corpora"] = [b.get("corpus") for b in batch]
    if "speaker_id" in batch[0]:
        out["speaker_ids"] = [b.get("speaker_id") for b in batch]
    return out


def masked_l1_latent_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    frame_mask: torch.Tensor,
    *,
    sample_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    """Padding-invariant L1 on latents (for collator order tests / training)."""
    err = (pred - target).abs().mean(dim=-1)  # [B, T]
    m = frame_mask.to(dtype=err.dtype)
    if sample_weight is not None:
        w = sample_weight.to(dtype=err.dtype).unsqueeze(-1)
        m = m * w
    return (err * m).sum() / m.sum().clamp_min(1.0)


def items_compatible_with_multicorpus(
    mc_item: Mapping[str, Any],
    *,
    input_ids_key: str = "input_ids",
) -> dict[str, Any]:
    """Adapt a ``MultiCorpusDataset`` item when ``input_ids`` is present.

    C4 items expose ``z_target`` / ``text_hidden``. N2 collator needs token ids;
    when the dataset (or A4 wrapper) attaches ``input_ids``, this helper maps
    fields without modifying ``dataset.py``.
    """
    if input_ids_key not in mc_item:
        raise KeyError(
            f"MultiCorpus item missing {input_ids_key!r}; attach token ids before "
            "collate_latent_seq (A4 wiring)"
        )
    out = {
        "utt_id": mc_item.get("utt_id"),
        "input_ids": mc_item[input_ids_key],
        "z_target": mc_item["z_target"],
        "sample_weight": mc_item.get("sample_weight", 1.0),
        "dur_s": mc_item.get("dur_s"),
        "corpus": mc_item.get("corpus"),
        "speaker_id": mc_item.get("speaker_id"),
    }
    if out["dur_s"] is None and "n_frames" in mc_item:
        out["dur_s"] = float(mc_item["n_frames"]) / get_target().frame_rate
    return out

"""Alignment priors: A word-level, B projected BPE, C monotonic Gaussian."""

from __future__ import annotations

import math
from enum import Enum

import torch


class AlignLevel(str, Enum):
    WORD = "A"
    TOKEN = "B"
    GAUSSIAN = "C"


def monotonic_gaussian_prior(T: int, L: int, sigma: float = 0.2) -> torch.Tensor:
    """[T, L] soft diagonal prior."""
    if T == 0 or L == 0:
        return torch.zeros(T, L)
    t = torch.arange(T).float().unsqueeze(1) / max(T - 1, 1)
    l = torch.arange(L).float().unsqueeze(0) / max(L - 1, 1)
    dist2 = (t - l) ** 2
    prior = torch.exp(-dist2 / (2 * sigma**2))
    prior = prior / prior.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    return prior


def word_level_prior(
    word_spans_frames: list[tuple[int, int]],
    token_word_ids: list[int],
    T: int,
    L: int,
) -> torch.Tensor:
    """Alignment A: frames belonging to a word attend to all tokens of that word."""
    prior = torch.zeros(T, L)
    for t in range(T):
        # find word for frame t
        w_id = None
        for wi, (s, e) in enumerate(word_spans_frames):
            if s <= t < e:
                w_id = wi
                break
        if w_id is None:
            continue
        toks = [j for j, wid in enumerate(token_word_ids) if wid == w_id and j < L]
        if not toks:
            continue
        prior[t, toks] = 1.0 / len(toks)
    # fallback empty rows
    for t in range(T):
        if prior[t].sum() == 0:
            prior[t] = monotonic_gaussian_prior(T, L)[t]
    return prior


def project_word_to_token_prior(
    word_spans_frames: list[tuple[int, int]],
    token_char_lens: list[int],
    token_word_ids: list[int],
    T: int,
    L: int,
) -> tuple[torch.Tensor, float]:
    """Alignment B: split word span across tokens by char length. Returns prior, defect_rate."""
    prior = torch.zeros(T, L)
    defects = 0
    words = sorted(set(token_word_ids))
    for w in words:
        toks = [j for j, wid in enumerate(token_word_ids) if wid == w]
        if not toks or w >= len(word_spans_frames):
            defects += 1
            continue
        s, e = word_spans_frames[w]
        if e <= s:
            defects += 1
            continue
        lengths = [max(1, token_char_lens[j]) for j in toks]
        total = sum(lengths)
        cursor = s
        for j, ln in zip(toks, lengths):
            span = max(1, int(round((e - s) * ln / total)))
            te = min(e, cursor + span)
            if te <= cursor:
                defects += 1
                continue
            prior[cursor:te, j] = 1.0
            cursor = te
        if cursor < e and toks:
            prior[cursor:e, toks[-1]] = 1.0
    # normalize rows
    for t in range(T):
        s = prior[t].sum()
        if s <= 0:
            defects += 1
            prior[t] = monotonic_gaussian_prior(T, L)[t]
        else:
            prior[t] /= s
    defect_rate = defects / max(len(words), 1)
    return prior, defect_rate


def validate_token_projection(defect_rate: float, threshold: float = 0.10) -> bool:
    return defect_rate <= threshold


def frames_from_seconds(t0: float, t1: float, frame_rate: float = 7.5) -> tuple[int, int]:
    return int(round(t0 * frame_rate)), int(round(t1 * frame_rate))

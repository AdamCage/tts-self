"""Phoneme/token → acoustic-frame projection (Track A / A3).

Consumes ``forced.json`` second-level segments (or phone dicts) and builds
hard or soft overlap matrices for an arbitrary ``frame_rate`` (VibeVoice 7.5 Hz
or mel ≈86.13 Hz).

Projection convention
---------------------
Frame ``i`` occupies the half-open time interval ``[i / fr, (i + 1) / fr)``.
A segment ``[t0, t1]`` contributes overlap length
``max(0, min(t1, frame_end) - max(t0, frame_start))``.

- **hard**: each frame is assigned to the segment with maximum overlap
  (ties → lowest segment index); empty frames stay all-zero.
- **soft**: ``W[p, t] = overlap(p, t) / frame_duration``; rows are *not*
  renormalized (a short phone may sum to <1 across frames; a frame may sum
  across phones to ≤1). Use ``normalize_columns`` / ``normalize_rows`` if needed.

CTC gap filling
---------------
Char-CTC ``merge_tokens`` spans often leave blank gaps (~60% of the utterance).
``fill_segment_gaps`` repairs the timeline over ``[0, utt_end]`` so
``sum(durs)`` matches the full mel length (load-path fix; no CTC re-run):

- ``expand_neighbor`` (default for mel train): absorb gaps into adjacent
  phone/char boundaries (split between neighbors). No ``sil`` token —
  free infer is content labels + mean/pred durs only.
- ``sil_fill`` (ablation / S1d): insert explicit ``sil`` over blank gaps.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Sequence

import numpy as np

from minispeech.constants import FRAME_RATE, get_target
from minispeech.data.forced_align import load_forced_json

# Explicit silence / CTC-blank gap label for mel packing (in PhoneVocab).
SIL_LABEL = "sil"

GapMode = Literal["expand_neighbor", "sil_fill", "none", "raw"]
DEFAULT_GAP_MODE: GapMode = "expand_neighbor"


@dataclass(frozen=True)
class TimeSegment:
    """One aligned unit in seconds."""

    start: float
    end: float
    label: str = ""

    def __post_init__(self) -> None:
        if self.end < self.start:
            raise ValueError(f"empty/inverted segment: {self.start} > {self.end}")

    @property
    def duration(self) -> float:
        return float(self.end - self.start)


def segments_from_pairs(
    spans_s: Sequence[Sequence[float] | tuple[float, float]],
    labels: Sequence[str] | None = None,
) -> list[TimeSegment]:
    """Build segments from ``[[t0, t1], ...]`` (+ optional labels)."""
    out: list[TimeSegment] = []
    for i, span in enumerate(spans_s):
        t0, t1 = float(span[0]), float(span[1])
        lab = str(labels[i]) if labels is not None and i < len(labels) else f"seg{i}"
        out.append(TimeSegment(t0, t1, lab))
    return out


def segments_from_forced_entry(entry: dict[str, Any]) -> list[TimeSegment]:
    """Extract second-level segments from one ``forced.json`` value.

    Preference order:
    1. ``phones`` list of dicts with ``start``/``end`` (and optional ``phone``/``label``)
    2. ``phone_spans_seconds`` + ``phones`` (string list) or ``phone_labels``
    3. ``word_spans_seconds`` (+ optional ``words``)

    Does **not** fill CTC blank gaps — use ``fill_segment_gaps`` /
    ``segments_covering_utterance`` for training timelines.
    """
    phones = entry.get("phones")
    if isinstance(phones, list) and phones and isinstance(phones[0], dict):
        segs: list[TimeSegment] = []
        for p in phones:
            t0 = float(p["start"] if "start" in p else p["t0"])
            t1 = float(p["end"] if "end" in p else p["t1"])
            lab = str(p.get("phone") or p.get("label") or p.get("token") or "")
            segs.append(TimeSegment(t0, t1, lab))
        return segs

    phone_spans = entry.get("phone_spans_seconds")
    if phone_spans:
        labels = entry.get("phone_labels")
        if labels is None and isinstance(phones, list) and phones and isinstance(phones[0], str):
            labels = phones
        return segments_from_pairs(phone_spans, labels)

    word_spans = entry.get("word_spans_seconds")
    if word_spans:
        return segments_from_pairs(word_spans, entry.get("words"))

    raise ValueError(
        "forced entry has no phone/word second spans "
        "(need phones[{start,end}], phone_spans_seconds, or word_spans_seconds)"
    )


def utterance_end_seconds(
    entry: dict[str, Any],
    segments: Sequence[TimeSegment] | None = None,
    *,
    fallback_dur_s: float | None = None,
) -> float:
    """Best-effort utterance end time for gap filling."""
    segs = list(segments) if segments is not None else []
    if not segs:
        try:
            segs = segments_from_forced_entry(entry)
        except ValueError:
            segs = []
    candidates = [
        float(entry["dur_s"]) if entry.get("dur_s") is not None else None,
        float(fallback_dur_s) if fallback_dur_s is not None else None,
        max((s.end for s in segs), default=None),
    ]
    # n_frames @ vibevoice rate is a weak fallback; prefer seconds fields.
    n_fr = entry.get("n_frames")
    if n_fr is not None and float(n_fr) > 0:
        candidates.append(float(n_fr) / FRAME_RATE)
    vals = [c for c in candidates if c is not None and c > 0]
    return float(max(vals)) if vals else 0.0


def fill_segment_gaps(
    segments: Sequence[TimeSegment],
    *,
    utt_end: float | None = None,
    utt_start: float = 0.0,
    gap_label: str = SIL_LABEL,
    min_gap_s: float = 1e-4,
    mode: GapMode = DEFAULT_GAP_MODE,
) -> list[TimeSegment]:
    """Cover ``[utt_start, utt_end]`` by filling CTC blank gaps.

    ``sil_fill``
        Insert ``gap_label`` spans for leading / inter / trailing blanks.
        Phone spans themselves are unchanged.
    ``expand_neighbor``
        Absorb gaps into neighboring segment ends (split mid-gap between
        neighbors; lead→first start, trail→last end). No new labels.
        Empty input returns ``[]`` (no synthetic ``sil``).
    """
    start = float(utt_start)
    if mode in ("none", "raw"):
        return list(segments)
    if mode == "expand_neighbor":
        return _expand_neighbor_gaps(
            segments, utt_end=utt_end, utt_start=start, min_gap_s=min_gap_s
        )

    if not segments:
        end = float(utt_end) if utt_end is not None else start
        if end - start >= min_gap_s:
            return [TimeSegment(start, end, gap_label)]
        return []

    segs = sorted(segments, key=lambda s: (s.start, s.end))
    end = float(utt_end) if utt_end is not None else float(segs[-1].end)
    end = max(end, float(segs[-1].end), start)

    out: list[TimeSegment] = []
    cursor = start
    for seg in segs:
        if seg.start - cursor >= min_gap_s:
            out.append(TimeSegment(cursor, float(seg.start), gap_label))
        out.append(seg)
        cursor = max(cursor, float(seg.end))
    if end - cursor >= min_gap_s:
        out.append(TimeSegment(cursor, end, gap_label))
    return out


def _expand_neighbor_gaps(
    segments: Sequence[TimeSegment],
    *,
    utt_end: float | None,
    utt_start: float,
    min_gap_s: float,
) -> list[TimeSegment]:
    """Absorb CTC blanks into adjacent phone/char boundaries (no ``sil``)."""
    start = float(utt_start)
    if not segments:
        return []

    segs = sorted(segments, key=lambda s: (s.start, s.end))
    end = float(utt_end) if utt_end is not None else float(segs[-1].end)
    end = max(end, float(segs[-1].end), start)
    n = len(segs)

    # Cut points: utt_start | mid(gap)… | utt_end → contiguous cover.
    cuts = [start]
    for i in range(n - 1):
        a_end = float(segs[i].end)
        b_start = float(segs[i + 1].start)
        mid = 0.5 * (a_end + b_start)
        cuts.append(mid)
    cuts.append(end)

    # Enforce non-decreasing cuts (degenerate / heavy overlap).
    for i in range(1, len(cuts)):
        if cuts[i] < cuts[i - 1]:
            cuts[i] = cuts[i - 1]

    out: list[TimeSegment] = []
    for i, seg in enumerate(segs):
        t0, t1 = float(cuts[i]), float(cuts[i + 1])
        if t1 < t0:
            t1 = t0
        # Keep zero-length units (frame snap later uses min 1); skip only
        # when the whole timeline collapsed and min_gap would discard noise.
        if t1 - t0 < min_gap_s and n > 1 and t1 == t0:
            # Still emit so phone count matches content labels.
            out.append(TimeSegment(t0, t0, seg.label))
        else:
            out.append(TimeSegment(t0, t1, seg.label))
    return out


def segments_covering_utterance(
    entry: dict[str, Any],
    *,
    fallback_dur_s: float | None = None,
    gap_label: str = SIL_LABEL,
    fill_gaps: bool = True,
    gap_mode: GapMode = DEFAULT_GAP_MODE,
) -> list[TimeSegment]:
    """``segments_from_forced_entry`` then optional CTC gap repair."""
    segs = segments_from_forced_entry(entry)
    # ``none`` / ``raw``: keep CTC blank gaps (paper2 ablation arm).
    if not fill_gaps or gap_mode in ("none", "raw"):
        return segs
    utt_end = utterance_end_seconds(entry, segs, fallback_dur_s=fallback_dur_s)
    return fill_segment_gaps(
        segs,
        utt_end=utt_end,
        utt_start=0.0,
        gap_label=gap_label,
        mode=gap_mode,  # type: ignore[arg-type]
    )


def segment_cover_ratio(
    segments: Sequence[TimeSegment],
    *,
    utt_end: float | None = None,
) -> float:
    """``sum(phone durations) / utt_end`` (raw CTC spans often ≈0.38)."""
    if not segments:
        return 0.0
    end = float(utt_end) if utt_end is not None else float(max(s.end for s in segments))
    if end <= 1e-12:
        return 0.0
    return float(sum(s.duration for s in segments) / end)


def snap_frame_durations(durs: Sequence[int], n_frames: int) -> list[int]:
    """Adjust integer frame counts so ``sum(durs) == n_frames`` (min 1 each)."""
    if n_frames <= 0:
        return [max(1, int(d)) for d in durs]
    if not durs:
        return [int(n_frames)]
    out = [max(1, int(d)) for d in durs]
    # If more phones than frames, collapse to 1 each on a prefix (degenerate).
    if len(out) > n_frames:
        out = [1] * n_frames
        return out
    # Guarantee min-1 fit
    while sum(out) > n_frames:
        # shrink largest above 1
        i = max(range(len(out)), key=lambda j: out[j])
        if out[i] <= 1:
            break
        out[i] -= 1
    rem = int(n_frames) - sum(out)
    if rem > 0:
        order = sorted(range(len(out)), key=lambda j: -out[j])
        for k in range(rem):
            out[order[k % len(order)]] += 1
    return out


def phones_and_frame_durs_from_forced(
    entry: dict[str, Any],
    *,
    frame_rate: float | str | None = None,
    fallback_dur_s: float | None = None,
    n_frames: int | None = None,
    fill_gaps: bool = True,
    gap_label: str = SIL_LABEL,
    gap_mode: GapMode = DEFAULT_GAP_MODE,
) -> tuple[list[str], list[int], dict[str, Any]]:
    """Phones + integer frame durations covering the full utterance timeline.

    Returns ``(phones, durs, meta)`` where meta includes raw/filled cover ratios.
    When ``n_frames`` is set (e.g. mel length), durations are snapped to that T.
    Default ``gap_mode='expand_neighbor'`` (no ``sil``); use ``sil_fill`` for S1d.
    """
    raw = segments_from_forced_entry(entry)
    utt_end = utterance_end_seconds(entry, raw, fallback_dur_s=fallback_dur_s)
    cover_raw = segment_cover_ratio(raw, utt_end=utt_end)
    segs = (
        fill_segment_gaps(
            raw,
            utt_end=utt_end,
            utt_start=0.0,
            gap_label=gap_label,
            mode=gap_mode,
        )
        if fill_gaps
        else list(raw)
    )
    cover_filled = segment_cover_ratio(segs, utt_end=utt_end)
    phones = [s.label or "?" for s in segs]
    durs_s = [s.duration for s in segs]
    durs = expand_durations_to_frames(durs_s, frame_rate=frame_rate)
    if n_frames is not None:
        durs = snap_frame_durations(durs, int(n_frames))
    gap_s = 0.0
    for a, b in zip(raw, raw[1:]):
        gap_s += max(0.0, b.start - a.end)
    if raw:
        gap_s += max(0.0, raw[0].start - 0.0)
        gap_s += max(0.0, utt_end - raw[-1].end)
    meta = {
        "n_phones": len(phones),
        "n_phones_raw": len(raw),
        "n_sil": sum(1 for p in phones if p == gap_label),
        "sum_dur_s": float(sum(durs_s)),
        "gap_s": float(gap_s),
        "utt_end_s": float(utt_end),
        "n_dur_frames": int(sum(durs)),
        "cover_ratio_raw": float(cover_raw),
        "cover_ratio_vs_end": float(cover_filled),
        "fill_gaps": bool(fill_gaps),
        "gap_label": gap_label if gap_mode == "sil_fill" else None,
        "gap_mode": gap_mode if fill_gaps else "none",
    }
    return phones, durs, meta


def load_forced_segments(
    path: str | Path,
    utt_id: str,
) -> list[TimeSegment]:
    """Load ``forced.json`` and return segments for ``utt_id``."""
    forced = load_forced_json(path)
    if utt_id not in forced:
        raise KeyError(f"{utt_id!r} not in {path}")
    return segments_from_forced_entry(forced[utt_id])


def n_frames_for_duration(dur_s: float, frame_rate: float) -> int:
    """Ceil frame count covering ``[0, dur_s]``."""
    if dur_s <= 0:
        return 0
    return max(1, int(np.ceil(dur_s * frame_rate - 1e-12)))


def resolve_frame_rate(frame_rate: float | str | None = None) -> float:
    """``None`` → VibeVoice 7.5; target name → registry; else float Hz."""
    if frame_rate is None:
        return FRAME_RATE
    if isinstance(frame_rate, str):
        return float(get_target(frame_rate).frame_rate)
    return float(frame_rate)


def overlap_matrix(
    segments: Sequence[TimeSegment],
    *,
    frame_rate: float | str | None = None,
    n_frames: int | None = None,
    mode: Literal["soft", "hard"] = "soft",
) -> np.ndarray:
    """Phoneme×frame overlap weights.

    Returns
    -------
    W : np.ndarray, shape ``[P, T]``, dtype float64
        Soft fractional overlaps in ``[0, 1]`` per cell, or hard 0/1 assignment.
    """
    fr = resolve_frame_rate(frame_rate)
    if not segments:
        T = 0 if n_frames is None else int(n_frames)
        return np.zeros((0, T), dtype=np.float64)

    dur_s = max(s.end for s in segments)
    T = int(n_frames) if n_frames is not None else n_frames_for_duration(dur_s, fr)
    P = len(segments)
    frame_dur = 1.0 / fr
    soft = np.zeros((P, T), dtype=np.float64)

    for p, seg in enumerate(segments):
        # Frames that can possibly overlap [start, end]
        t0 = max(0, int(np.floor(seg.start * fr + 1e-12)))
        t1 = min(T, int(np.ceil(seg.end * fr - 1e-12)))
        for t in range(t0, t1):
            f0, f1 = t * frame_dur, (t + 1) * frame_dur
            ov = min(seg.end, f1) - max(seg.start, f0)
            if ov > 0:
                soft[p, t] = ov / frame_dur

    if mode == "soft":
        return soft

    hard = np.zeros_like(soft)
    for t in range(T):
        col = soft[:, t]
        if col.sum() <= 0:
            continue
        # max overlap; lowest index on ties
        p_star = int(np.argmax(col))
        hard[p_star, t] = 1.0
    return hard


def soft_overlap_projection(
    segments: Sequence[TimeSegment],
    *,
    frame_rate: float | str | None = None,
    n_frames: int | None = None,
) -> np.ndarray:
    """Alias for ``overlap_matrix(..., mode=\"soft\")``."""
    return overlap_matrix(segments, frame_rate=frame_rate, n_frames=n_frames, mode="soft")


def hard_overlap_projection(
    segments: Sequence[TimeSegment],
    *,
    frame_rate: float | str | None = None,
    n_frames: int | None = None,
) -> np.ndarray:
    """Alias for ``overlap_matrix(..., mode=\"hard\")``."""
    return overlap_matrix(segments, frame_rate=frame_rate, n_frames=n_frames, mode="hard")


def normalize_columns(W: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Column-stochastic (each frame distributes mass over phones)."""
    out = W.astype(np.float64, copy=True)
    col = out.sum(axis=0, keepdims=True)
    out /= np.maximum(col, eps)
    out[:, col.ravel() <= eps] = 0.0
    return out


def normalize_rows(W: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Row-stochastic (each phone distributes mass over frames)."""
    out = W.astype(np.float64, copy=True)
    row = out.sum(axis=1, keepdims=True)
    out /= np.maximum(row, eps)
    out[row.ravel() <= eps, :] = 0.0
    return out


def frame_coverage(W: np.ndarray) -> np.ndarray:
    """Per-frame total soft mass (shape ``[T]``)."""
    return W.sum(axis=0)


def phone_durations_seconds(segments: Sequence[TimeSegment]) -> list[float]:
    return [s.duration for s in segments]


def expand_durations_to_frames(
    durations_s: Sequence[float],
    *,
    frame_rate: float | str | None = None,
) -> list[int]:
    """Round phone durations (seconds) to integer frame counts (min 1 each)."""
    fr = resolve_frame_rate(frame_rate)
    counts = [max(1, int(round(d * fr))) for d in durations_s]
    return counts

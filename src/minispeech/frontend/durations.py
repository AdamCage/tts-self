"""Phoneme duration stats / predictor stub (Track A / A3).

Rule-based mean durations from ``forced.json`` for now; interface is ready for
a trained duration predictor later.

Free-infer packing (``pack_infer_phones_and_durs``):

- Default ``gap_mode='expand_neighbor'``: content labels (char/``|``) + mean
  / uniform / oracle durs — no ``sil`` insertion (matches S1e+ train).
- Optional ``gap_mode='sil_fill'``: insert ``sil`` for S1d ablation vocabs.
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Protocol, Sequence

from minispeech.data.frame_projection import (
    DEFAULT_GAP_MODE,
    SIL_LABEL,
    GapMode,
    TimeSegment,
    expand_durations_to_frames,
    phones_and_frame_durs_from_forced,
    resolve_frame_rate,
    segments_covering_utterance,
    snap_frame_durations,
)
from minispeech.data.forced_align import load_forced_json
from minispeech.frontend.phone_vocab import SIL


class DurationPredictor(Protocol):
    """Predict per-unit durations in seconds (phones or tokens)."""

    def predict_seconds(self, units: Sequence[str]) -> list[float]:
        ...


@dataclass
class MeanDurationPredictor:
    """Lookup mean duration per label; unknown labels → ``default_s``.

    Built from forced-align segments (word or phone). Suitable as a wiring
    stub until a learned duration model exists.
    """

    means: dict[str, float] = field(default_factory=dict)
    default_s: float = 0.08
    counts: dict[str, int] = field(default_factory=dict)

    def predict_seconds(self, units: Sequence[str]) -> list[float]:
        return [float(self.means.get(u, self.default_s)) for u in units]

    def predict_frames(
        self,
        units: Sequence[str],
        *,
        frame_rate: float | str | None = None,
    ) -> list[int]:
        return expand_durations_to_frames(
            self.predict_seconds(units),
            frame_rate=frame_rate,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "means": dict(self.means),
            "default_s": self.default_s,
            "counts": dict(self.counts),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> MeanDurationPredictor:
        return cls(
            means={str(k): float(v) for k, v in (d.get("means") or {}).items()},
            default_s=float(d.get("default_s", 0.08)),
            counts={str(k): int(v) for k, v in (d.get("counts") or {}).items()},
        )

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: str | Path) -> MeanDurationPredictor:
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    @classmethod
    def from_segments(
        cls,
        segments: Sequence[TimeSegment],
        *,
        default_s: float = 0.08,
        min_count: int = 1,
    ) -> MeanDurationPredictor:
        sums: dict[str, float] = defaultdict(float)
        counts: dict[str, int] = defaultdict(int)
        for seg in segments:
            lab = seg.label or ""
            if not lab:
                continue
            d = seg.duration
            if d <= 0:
                continue
            sums[lab] += d
            counts[lab] += 1
        means = {
            lab: sums[lab] / counts[lab]
            for lab in sums
            if counts[lab] >= min_count
        }
        global_mean = (
            sum(sums.values()) / sum(counts.values()) if counts else default_s
        )
        return cls(means=means, default_s=float(global_mean), counts=dict(counts))

    @classmethod
    def from_forced_json(
        cls,
        path: str | Path,
        *,
        default_s: float = 0.08,
        min_count: int = 1,
        max_utts: int | None = None,
        gap_mode: GapMode = DEFAULT_GAP_MODE,
    ) -> MeanDurationPredictor:
        forced = load_forced_json(path)
        segs: list[TimeSegment] = []
        for i, (_uid, entry) in enumerate(forced.items()):
            if max_utts is not None and i >= max_utts:
                break
            if not isinstance(entry, dict):
                continue
            try:
                segs.extend(
                    segments_covering_utterance(
                        entry, fill_gaps=True, gap_mode=gap_mode
                    )
                )
            except ValueError:
                continue
        return cls.from_segments(segs, default_s=default_s, min_count=min_count)


def duration_stats_from_forced(
    path: str | Path,
    *,
    max_utts: int | None = None,
    gap_mode: GapMode = DEFAULT_GAP_MODE,
) -> dict[str, Any]:
    """Summary stats (mean/std/count per label) for notebooks / docs."""
    pred = MeanDurationPredictor.from_forced_json(
        path, max_utts=max_utts, gap_mode=gap_mode
    )
    # recompute std
    forced = load_forced_json(path)
    vals: dict[str, list[float]] = defaultdict(list)
    for i, (_uid, entry) in enumerate(forced.items()):
        if max_utts is not None and i >= max_utts:
            break
        if not isinstance(entry, dict):
            continue
        try:
            for seg in segments_covering_utterance(
                entry, fill_gaps=True, gap_mode=gap_mode
            ):
                if seg.label and seg.duration > 0:
                    vals[seg.label].append(seg.duration)
        except ValueError:
            continue
    per_label: dict[str, Any] = {}
    for lab, xs in sorted(vals.items()):
        arr_mean = sum(xs) / len(xs)
        var = sum((x - arr_mean) ** 2 for x in xs) / max(len(xs) - 1, 1)
        per_label[lab] = {
            "mean_s": arr_mean,
            "std_s": var**0.5,
            "count": len(xs),
        }
    return {
        "n_labels": len(per_label),
        "default_s": pred.default_s,
        "per_label": per_label,
    }


SilPolicy = Literal["bigram", "interleave", "word", "none"]
DurationMode = Literal["mean_stats", "oracle_forced", "uniform"]


@dataclass
class SilGapModel:
    """Bigram-conditioned silence insertion from gap-filled ``forced.json``.

    Ablation on S1d: oracle sil *mask* + mean durs ≈ oracle CER; naive
    always-interleave sil destroys intelligibility. This model inserts ``sil``
    between content labels when ``P(sil|a→b) >= threshold``, with unigram
    backoff for rare bigrams.
    """

    p_sil: dict[str, float] = field(default_factory=dict)  # key "a\tb"
    mean_sil_frames: dict[str, float] = field(default_factory=dict)
    counts: dict[str, int] = field(default_factory=dict)
    p_after: dict[str, float] = field(default_factory=dict)  # P(sil after label a)
    p_before: dict[str, float] = field(default_factory=dict)  # P(sil before label b)
    count_after: dict[str, int] = field(default_factory=dict)
    count_before: dict[str, int] = field(default_factory=dict)
    default_p: float = 0.8
    default_sil_frames: float = 3.0
    lead_sil_frames: float = 3.0
    trail_sil_frames: float = 3.0
    frames_per_content: float = 7.0
    threshold: float = 0.5
    min_count: int = 2

    @staticmethod
    def _key(a: str, b: str) -> str:
        return f"{a}\t{b}"

    def prob(self, a: str, b: str) -> float:
        k = self._key(a, b)
        if self.counts.get(k, 0) >= self.min_count:
            return float(self.p_sil.get(k, self.default_p))
        # Unigram backoff (average of after-a and before-b)
        parts: list[float] = []
        if self.count_after.get(a, 0) >= self.min_count:
            parts.append(float(self.p_after[a]))
        if self.count_before.get(b, 0) >= self.min_count:
            parts.append(float(self.p_before[b]))
        if parts:
            return float(sum(parts) / len(parts))
        return float(self.default_p)

    def sil_frames_between(self, a: str, b: str) -> int:
        k = self._key(a, b)
        if self.counts.get(k, 0) >= self.min_count and k in self.mean_sil_frames:
            return max(1, int(round(self.mean_sil_frames[k])))
        return max(1, int(round(self.default_sil_frames)))

    def should_insert(self, a: str, b: str) -> bool:
        return self.prob(a, b) >= self.threshold

    def to_dict(self) -> dict[str, Any]:
        return {
            "p_sil": dict(self.p_sil),
            "mean_sil_frames": dict(self.mean_sil_frames),
            "counts": dict(self.counts),
            "p_after": dict(self.p_after),
            "p_before": dict(self.p_before),
            "count_after": dict(self.count_after),
            "count_before": dict(self.count_before),
            "default_p": self.default_p,
            "default_sil_frames": self.default_sil_frames,
            "lead_sil_frames": self.lead_sil_frames,
            "trail_sil_frames": self.trail_sil_frames,
            "frames_per_content": self.frames_per_content,
            "threshold": self.threshold,
            "min_count": self.min_count,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> SilGapModel:
        return cls(
            p_sil={str(k): float(v) for k, v in (d.get("p_sil") or {}).items()},
            mean_sil_frames={str(k): float(v) for k, v in (d.get("mean_sil_frames") or {}).items()},
            counts={str(k): int(v) for k, v in (d.get("counts") or {}).items()},
            p_after={str(k): float(v) for k, v in (d.get("p_after") or {}).items()},
            p_before={str(k): float(v) for k, v in (d.get("p_before") or {}).items()},
            count_after={str(k): int(v) for k, v in (d.get("count_after") or {}).items()},
            count_before={str(k): int(v) for k, v in (d.get("count_before") or {}).items()},
            default_p=float(d.get("default_p", 0.8)),
            default_sil_frames=float(d.get("default_sil_frames", 3.0)),
            lead_sil_frames=float(d.get("lead_sil_frames", 3.0)),
            trail_sil_frames=float(d.get("trail_sil_frames", 3.0)),
            frames_per_content=float(d.get("frames_per_content", 7.0)),
            threshold=float(d.get("threshold", 0.5)),
            min_count=int(d.get("min_count", 2)),
        )

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: str | Path) -> SilGapModel:
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    @classmethod
    def from_forced_json(
        cls,
        path: str | Path,
        *,
        frame_rate: float | str | None = "mel80",
        max_utts: int | None = None,
        threshold: float = 0.5,
        min_count: int = 2,
    ) -> SilGapModel:
        fr = resolve_frame_rate(frame_rate)
        forced = load_forced_json(path)
        with_sil: dict[str, int] = defaultdict(int)
        total: dict[str, int] = defaultdict(int)
        sil_fr_sum: dict[str, float] = defaultdict(float)
        sil_fr_n: dict[str, int] = defaultdict(int)
        after_sil: dict[str, int] = defaultdict(int)
        after_tot: dict[str, int] = defaultdict(int)
        before_sil: dict[str, int] = defaultdict(int)
        before_tot: dict[str, int] = defaultdict(int)
        lead_fr: list[int] = []
        trail_fr: list[int] = []
        utt_fpc: list[float] = []

        for i, (_uid, entry) in enumerate(forced.items()):
            if max_utts is not None and i >= max_utts:
                break
            if not isinstance(entry, dict):
                continue
            try:
                phones, durs, _meta = phones_and_frame_durs_from_forced(
                    entry,
                    frame_rate=fr,
                    fill_gaps=True,
                    gap_mode="sil_fill",
                    gap_label=SIL,
                )
            except ValueError:
                continue
            n_content = sum(1 for p in phones if p != SIL)
            if n_content > 0:
                utt_fpc.append(sum(durs) / n_content)
            if phones and phones[0] == SIL:
                lead_fr.append(int(durs[0]))
            if phones and phones[-1] == SIL:
                trail_fr.append(int(durs[-1]))

            j = 0
            while j < len(phones):
                if phones[j] == SIL:
                    j += 1
                    continue
                a = phones[j]
                k = j + 1
                gap = 0
                while k < len(phones) and phones[k] == SIL:
                    gap += int(durs[k])
                    k += 1
                if k >= len(phones):
                    break
                b = phones[k]
                key = cls._key(a, b)
                total[key] += 1
                after_tot[a] += 1
                before_tot[b] += 1
                if gap > 0:
                    with_sil[key] += 1
                    after_sil[a] += 1
                    before_sil[b] += 1
                    sil_fr_sum[key] += gap
                    sil_fr_n[key] += 1
                j = k

        p_sil = {k: with_sil[k] / total[k] for k in total}
        mean_sil = {
            k: sil_fr_sum[k] / sil_fr_n[k] for k in sil_fr_n if sil_fr_n[k] > 0
        }
        p_after = {k: after_sil[k] / after_tot[k] for k in after_tot}
        p_before = {k: before_sil[k] / before_tot[k] for k in before_tot}
        n_trans = sum(total.values())
        n_with = sum(with_sil.values())
        default_p = (n_with / n_trans) if n_trans else 0.8
        default_sil = (
            sum(sil_fr_sum.values()) / sum(sil_fr_n.values()) if sil_fr_n else 3.0
        )
        # Median frames/content is less biased by long sil tails than the mean.
        fpc = float(sorted(utt_fpc)[len(utt_fpc) // 2]) if utt_fpc else 7.0
        return cls(
            p_sil=p_sil,
            mean_sil_frames=mean_sil,
            counts=dict(total),
            p_after=p_after,
            p_before=p_before,
            count_after=dict(after_tot),
            count_before=dict(before_tot),
            default_p=float(default_p),
            default_sil_frames=float(default_sil),
            lead_sil_frames=float(sum(lead_fr) / len(lead_fr)) if lead_fr else 3.0,
            trail_sil_frames=float(sum(trail_fr) / len(trail_fr)) if trail_fr else 3.0,
            frames_per_content=fpc,
            threshold=threshold,
            min_count=min_count,
        )


def interleave_sil_labels(
    labels: Sequence[str],
    *,
    sil: str = SIL,
    policy: SilPolicy = "bigram",
    leading: bool = True,
    trailing: bool = True,
    gap_model: SilGapModel | None = None,
) -> list[str]:
    """Insert ``sil`` so free-infer timelines match gap-filled train packing.

    - ``bigram``: insert sil between a→b when ``SilGapModel`` says so (default).
    - ``interleave``: sil before/between/after every non-sil label.
    - ``word``: sil at edges and around ``|`` only.
    - ``none``: keep labels as-is (optional lead/trail sil).
    """
    base = [str(x) for x in labels if str(x) and str(x) != sil]

    if policy == "none":
        # Content-only (expand_neighbor train). Callers wanting edge sil must
        # pass leading/trailing=True explicitly.
        out = list(base)
        if leading:
            out = [sil] + out
        if trailing:
            out = out + [sil]
        return out if out else ([sil] if (leading or trailing) else [])

    if policy == "word":
        out = []
        if leading:
            out.append(sil)
        for lab in base:
            if lab == "|":
                if not out or out[-1] != sil:
                    out.append(sil)
                out.append(lab)
                out.append(sil)
            else:
                out.append(lab)
        if trailing and (not out or out[-1] != sil):
            out.append(sil)
        return out if out else [sil]

    if policy == "bigram":
        if not base:
            return [sil] if (leading or trailing) else []
        gm = gap_model or SilGapModel()
        out = []
        if leading:
            out.append(sil)
        for i, lab in enumerate(base):
            out.append(lab)
            if i < len(base) - 1 and gm.should_insert(lab, base[i + 1]):
                out.append(sil)
        if trailing and (not out or out[-1] != sil):
            out.append(sil)
        return out

    # interleave: sil, lab, sil, lab, ..., sil
    if not base:
        return [sil] if (leading or trailing) else []
    out = []
    if leading:
        out.append(sil)
    for i, lab in enumerate(base):
        out.append(lab)
        if i < len(base) - 1 or trailing:
            out.append(sil)
    return out


def sil_aware_snap_durs(
    phones: Sequence[str],
    durs: Sequence[int],
    n_frames: int,
    *,
    sil: str = SIL,
) -> list[int]:
    """Snap ``sum(durs)`` to ``n_frames``, preferring to resize ``sil`` tokens."""
    out = [max(1, int(d)) for d in durs]
    if n_frames <= 0 or not out:
        return out
    sil_idx = [i for i, p in enumerate(phones) if p == sil]
    content_idx = [i for i, p in enumerate(phones) if p != sil]
    while sum(out) > n_frames:
        cands = [i for i in sil_idx if out[i] > 1] or [i for i in content_idx if out[i] > 1]
        if not cands:
            break
        i = max(cands, key=lambda j: out[j])
        out[i] -= 1
    rem = int(n_frames) - sum(out)
    if rem > 0:
        order = sil_idx + content_idx
        for k in range(rem):
            out[order[k % len(order)]] += 1
    return out


def estimate_target_frames_from_text(
    n_content_labels: int,
    predictor: MeanDurationPredictor,
    *,
    frame_rate: float | str | None = "mel80",
    sil_policy: SilPolicy = "bigram",
    gap_model: SilGapModel | None = None,
) -> int:
    """Heuristic T from content length (+ optional ``frames_per_content``)."""
    n = max(0, int(n_content_labels))
    if gap_model is not None and gap_model.frames_per_content > 0:
        return max(1, int(round(n * gap_model.frames_per_content)))
    fr = resolve_frame_rate(frame_rate)
    char_s = float(predictor.default_s)
    non_sil = [v for k, v in predictor.means.items() if k not in (SIL, SIL_LABEL, "|")]
    if non_sil:
        char_s = sum(non_sil) / len(non_sil)
    sil_s = float(predictor.means.get(SIL, predictor.means.get(SIL_LABEL, predictor.default_s)))
    if sil_policy == "interleave":
        n_sil = n + 1 if n else 1
    elif sil_policy == "word":
        n_sil = max(2, n // 4 + 2)
    elif sil_policy == "bigram":
        n_sil = max(2, int(round(0.8 * max(n - 1, 0))) + 2)
    else:
        n_sil = 0  # content-only (expand_neighbor)
    total_s = n * char_s + n_sil * sil_s
    return max(1, int(round(total_s * fr)))


def _assign_durs_with_gap_model(
    phones: Sequence[str],
    predictor: MeanDurationPredictor,
    gap_model: SilGapModel | None,
    *,
    frame_rate: float,
    sil: str = SIL,
) -> list[int]:
    """Per-label mean frames; sil tokens use bigram-specific means when possible."""
    fr = float(frame_rate)
    base_means = predictor.predict_frames(phones, frame_rate=fr)
    if gap_model is None:
        return base_means
    out = list(base_means)
    # Leading / trailing sil
    if phones and phones[0] == sil:
        out[0] = max(1, int(round(gap_model.lead_sil_frames)))
    if phones and phones[-1] == sil and len(phones) > 1:
        out[-1] = max(1, int(round(gap_model.trail_sil_frames)))
    # Interior sil: look at neighboring content labels
    for i, p in enumerate(phones):
        if p != sil or i == 0 or i == len(phones) - 1:
            continue
        left = phones[i - 1]
        right = phones[i + 1]
        if left != sil and right != sil:
            out[i] = gap_model.sil_frames_between(left, right)
        else:
            out[i] = max(1, int(round(gap_model.default_sil_frames)))
    return out


def lookup_forced_entry_by_content(
    forced: dict[str, Any],
    content_labels: Sequence[str],
    *,
    frame_rate: float | str | None = "mel80",
    sil: str = SIL,
    gap_mode: GapMode = DEFAULT_GAP_MODE,
) -> tuple[str, dict[str, Any]] | None:
    """Return ``(utt_id, entry)`` whose gap-filled content labels match exactly."""
    want = [str(x) for x in content_labels if str(x) and str(x) != sil]
    if not want:
        return None
    fr = resolve_frame_rate(frame_rate)
    for uid, entry in forced.items():
        if not isinstance(entry, dict):
            continue
        try:
            phones, _durs, _meta = phones_and_frame_durs_from_forced(
                entry,
                frame_rate=fr,
                fill_gaps=True,
                gap_label=sil,
                gap_mode=gap_mode,
            )
        except ValueError:
            continue
        got = [p for p in phones if p != sil]
        if got == want:
            return str(uid), entry
    return None


def pack_infer_phones_and_durs(
    labels: Sequence[str],
    predictor: MeanDurationPredictor | None = None,
    *,
    mode: DurationMode = "mean_stats",
    frame_rate: float | str | None = "mel80",
    sil_policy: SilPolicy = "none",
    sil: str = SIL,
    target_frames: int | None = None,
    target_seconds: float | None = None,
    uniform_frames_per_unit: int = 4,
    forced_entry: dict[str, Any] | None = None,
    forced_index: dict[str, Any] | None = None,
    gap_model: SilGapModel | None = None,
    gap_mode: GapMode = DEFAULT_GAP_MODE,
    snap: bool = True,
    auto_target: bool = True,
    allow_forced_lookup: bool = False,
) -> tuple[list[str], list[int], dict[str, Any]]:
    """Build train-consistent ``(phones, frame_durs, meta)`` for free / oracle infer.

    Modes
    -----
    mean_stats
        Content labels (+ optional ``sil`` if ``sil_policy`` ≠ ``none``), mean
        frame durs; snap when ``auto_target``. Default ``gap_mode`` /
        ``sil_policy`` match expand-neighbor train (no sil).
        If ``forced_index`` + ``allow_forced_lookup``, reuse that entry's
        gap-filled timeline when content labels match.
    uniform
        Same label packing; equal integer frames per unit (then optional snap).
    oracle_forced
        Ignore ``labels``; use ``forced_entry`` via ``phones_and_frame_durs_from_forced``.
    """
    fr = resolve_frame_rate(frame_rate)
    # sil_fill train → default sil packing if caller left sil_policy at none.
    effective_sil: SilPolicy = sil_policy
    if gap_mode == "sil_fill" and sil_policy == "none" and gap_model is not None:
        effective_sil = "bigram"
    meta: dict[str, Any] = {
        "duration_mode": mode,
        "sil_policy": effective_sil,
        "gap_mode": gap_mode,
        "frame_rate": float(fr),
    }

    if mode == "oracle_forced":
        if forced_entry is None:
            raise ValueError("oracle_forced requires forced_entry")
        phones, durs, ometa = phones_and_frame_durs_from_forced(
            forced_entry,
            frame_rate=fr,
            n_frames=target_frames,
            fill_gaps=True,
            gap_label=sil,
            gap_mode=gap_mode,
        )
        meta.update(ometa)
        meta["n_sil"] = sum(1 for p in phones if p == sil)
        meta["sum_dur_frames"] = int(sum(durs))
        return phones, durs, meta

    content = [str(x) for x in labels if str(x) and str(x) != sil]

    # Exact content match in forced.json → oracle gap-filled timeline.
    if (
        allow_forced_lookup
        and forced_index is not None
        and mode == "mean_stats"
    ):
        hit = lookup_forced_entry_by_content(
            forced_index, content, frame_rate=fr, sil=sil, gap_mode=gap_mode
        )
        if hit is not None:
            uid, entry = hit
            phones, durs, ometa = phones_and_frame_durs_from_forced(
                entry,
                frame_rate=fr,
                n_frames=target_frames,
                fill_gaps=True,
                gap_label=sil,
                gap_mode=gap_mode,
            )
            meta.update(ometa)
            meta["n_sil"] = sum(1 for p in phones if p == sil)
            meta["sum_dur_frames"] = int(sum(durs))
            meta["sum_dur_s"] = float(sum(durs) / fr) if fr > 0 else 0.0
            meta["cover_ratio_vs_end"] = 1.0
            meta["forced_lookup_utt_id"] = uid
            meta["sil_policy"] = "forced_lookup"
            return phones, durs, meta

    lead_trail = effective_sil != "none"
    phones = interleave_sil_labels(
        content,
        sil=sil,
        policy=effective_sil,
        gap_model=gap_model,
        leading=lead_trail,
        trailing=lead_trail,
    )
    meta["n_labels_raw"] = len(content)
    meta["n_phones"] = len(phones)
    meta["n_sil"] = sum(1 for p in phones if p == sil)
    meta["phones"] = list(phones)

    if not phones:
        # expand_neighbor: no sil fallback; use a single unk-like placeholder.
        phones = ["|"] if effective_sil == "none" else [sil]
        meta["n_phones"] = 1
        meta["n_sil"] = 0 if effective_sil == "none" else 1

    if mode == "uniform":
        durs = [max(1, int(uniform_frames_per_unit)) for _ in phones]
    else:
        if predictor is None:
            raise ValueError("mean_stats requires MeanDurationPredictor")
        use_gap = gap_model if effective_sil != "none" else None
        durs = _assign_durs_with_gap_model(
            phones, predictor, use_gap, frame_rate=fr, sil=sil
        )

    n_tgt = target_frames
    if n_tgt is None and target_seconds is not None:
        n_tgt = max(1, int(round(float(target_seconds) * fr)))
    if (
        n_tgt is None
        and auto_target
        and mode == "mean_stats"
        and predictor is not None
        and snap
    ):
        # For expand_neighbor, prefer sum of per-label means (already absorb gaps).
        if effective_sil == "none":
            n_tgt = max(1, int(sum(durs)))
        else:
            n_tgt = estimate_target_frames_from_text(
                len(content),
                predictor,
                frame_rate=fr,
                sil_policy=effective_sil,
                gap_model=gap_model,
            )
    if n_tgt is not None and snap:
        durs = sil_aware_snap_durs(phones, durs, int(n_tgt), sil=sil)

    meta["sum_dur_frames"] = int(sum(durs))
    meta["sum_dur_s"] = float(sum(durs) / fr) if fr > 0 else 0.0
    meta["target_frames"] = int(n_tgt) if n_tgt is not None else None
    meta["cover_ratio_vs_end"] = 1.0
    if gap_model is not None and effective_sil != "none":
        meta["frames_per_content"] = gap_model.frames_per_content
        meta["sil_threshold"] = gap_model.threshold
    return phones, durs, meta

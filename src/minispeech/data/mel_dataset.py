"""Dataset / collate for non-AR mel baseline (Track A / A4).

Loads phone ids + frame durations from ``forced.json`` (or uniform G2P
durations) and mel targets from a mel cache or synthetic tensors.

Unit tests can construct items without audio / Hub models.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from minispeech.constants import MEL80, get_target
from minispeech.data.forced_align import load_forced_json
from minispeech.data.frame_projection import (
    DEFAULT_GAP_MODE,
    SIL_LABEL,
    GapMode,
    n_frames_for_duration,
    phones_and_frame_durs_from_forced,
    segments_covering_utterance,
    snap_frame_durations,
    soft_overlap_projection,
)
from minispeech.data.manifest import Utterance, filter_split, read_manifest
from minispeech.data.mel import MelCacheReader
from minispeech.frontend.durations import MeanDurationPredictor
from minispeech.frontend.g2p import text_to_phonemes
from minispeech.frontend.phone_vocab import PhoneVocab
from minispeech.utils import expand


class MelSpeechDataset(Dataset):
    """Phoneme → mel training rows for S1-scale mel baseline."""

    def __init__(
        self,
        manifest_path: str,
        *,
        split: str = "train",
        mel_dir: str | None = None,
        forced_path: str | None = None,
        vocab: PhoneVocab | None = None,
        max_utterances: int | None = None,
        max_hours: float | None = None,
        exclude_tags: Sequence[str] = ("exclude",),
        review_weight: float | None = 0.5,
        target: str = "mel80",
        upsample: str = "hard",
        preload_mel: bool = False,
        duration_predictor: MeanDurationPredictor | None = None,
        allow_missing_mel: bool = False,
        gap_mode: GapMode = DEFAULT_GAP_MODE,
    ):
        self.target_spec = get_target(target)
        self.upsample = upsample
        self.frame_rate = float(self.target_spec.frame_rate)
        self.gap_mode: GapMode = gap_mode
        rows = filter_split(read_manifest(expand(manifest_path)), split)
        if exclude_tags:
            excl = set(exclude_tags)
            rows = [r for r in rows if (r.qc_tag or "clean") not in excl]
        if max_utterances is not None:
            rows = rows[:max_utterances]
        if max_hours is not None:
            budget = max_hours * 3600.0
            acc, kept = 0.0, []
            for r in rows:
                if acc + r.dur_s > budget:
                    break
                kept.append(r)
                acc += r.dur_s
            rows = kept
        self.review_weight = review_weight
        self.allow_missing_mel = allow_missing_mel
        self.forced = load_forced_json(expand(forced_path)) if forced_path else {}
        self.dur_pred = duration_predictor
        self.vocab = vocab or PhoneVocab()
        self.mels: MelCacheReader | None = None
        if mel_dir is not None:
            mel_p = expand(mel_dir)
            if (mel_p / "mels_index.json").is_file():
                self.mels = MelCacheReader(mel_p, lazy=True)
        # Prefer forced+mel coverage only: char-CTC vocab mismatches G2P fallback,
        # and missing mel raises in __getitem__. Dry-run (allow_missing_mel) keeps all.
        if not allow_missing_mel:
            if self.forced:
                rows = [r for r in rows if r.utt_id in self.forced]
            if self.mels is not None:
                rows = [r for r in rows if r.utt_id in self.mels.index]
        self.rows = rows
        self._mel_cache: dict[str, torch.Tensor] | None = {} if preload_mel else None
        if preload_mel and self.mels is not None:
            for u in self.rows:
                if u.utt_id in self.mels.index:
                    self._mel_cache[u.utt_id] = self.mels.get(u.utt_id)

    def __len__(self) -> int:
        return len(self.rows)

    def _phones_and_durs(
        self,
        u: Utterance,
        *,
        n_mel_frames: int | None = None,
    ) -> tuple[list[str], list[int]]:
        entry = self.forced.get(u.utt_id)
        if isinstance(entry, dict):
            try:
                phones, durs, _meta = phones_and_frame_durs_from_forced(
                    entry,
                    frame_rate=self.frame_rate,
                    fallback_dur_s=float(u.dur_s or 0.0) or None,
                    n_frames=n_mel_frames,
                    fill_gaps=True,
                    gap_label=SIL_LABEL,
                    gap_mode=self.gap_mode,
                )
                return phones, durs
            except ValueError:
                pass
        text = u.text_stressed or u.text_raw
        phones = text_to_phonemes(text)
        if not phones:
            phones = ["?"]
        if self.dur_pred is not None:
            durs = self.dur_pred.predict_frames(phones, frame_rate=self.frame_rate)
        else:
            # Uniform over utterance duration
            n_frames = n_mel_frames or max(
                1, n_frames_for_duration(float(u.dur_s or 1.0), self.frame_rate)
            )
            base = max(1, n_frames // len(phones))
            durs = [base] * len(phones)
            rem = n_frames - base * len(phones)
            for i in range(max(0, rem)):
                durs[i % len(phones)] += 1
        if n_mel_frames is not None:
            durs = snap_frame_durations(durs, int(n_mel_frames))
        return phones, durs

    def _get_mel(self, u: Utterance, n_frames: int) -> torch.Tensor | None:
        if self._mel_cache is not None and u.utt_id in self._mel_cache:
            return self._mel_cache[u.utt_id]
        if self.mels is not None and u.utt_id in self.mels.index:
            mel = self.mels.get(u.utt_id).float()
            if self._mel_cache is not None:
                self._mel_cache[u.utt_id] = mel
            return mel
        if self.allow_missing_mel:
            return torch.zeros(n_frames, self.target_spec.dim)
        return None

    def __getitem__(self, idx: int) -> dict[str, Any]:
        u = self.rows[idx]
        # Prefer mel length as the timeline so sum(durs) == T (no prefix crop).
        mel_probe = self._get_mel(u, n_frames_for_duration(float(u.dur_s or 1.0), self.frame_rate))
        n_mel = int(mel_probe.shape[0]) if mel_probe is not None else None
        phones, durs = self._phones_and_durs(u, n_mel_frames=n_mel)
        self.vocab.add_phones(phones)
        ids = self.vocab.encode(phones)
        n_frames = int(sum(durs))
        mel = mel_probe if mel_probe is not None else self._get_mel(u, n_frames)
        if mel is None:
            raise KeyError(
                f"missing mel for {u.utt_id}; build cache with scripts/29_cache_mels.py "
                f"or pass allow_missing_mel for dry-run"
            )
        # After gap-fill + snap, lengths should match; keep pad/crop as safety.
        if mel.shape[0] < n_frames:
            pad = torch.zeros(n_frames - mel.shape[0], mel.shape[-1])
            mel = torch.cat([mel, pad], dim=0)
        elif mel.shape[0] > n_frames:
            mel = mel[:n_frames]

        weight = 1.0
        if u.sample_weight is not None:
            weight = float(u.sample_weight)
        elif self.review_weight is not None and u.qc_tag == "review":
            weight = float(self.review_weight)

        item: dict[str, Any] = {
            "utt_id": u.utt_id,
            "phone_ids": torch.tensor(ids, dtype=torch.long),
            "durations": torch.tensor(durs, dtype=torch.float32),
            "mel_target": mel.float(),
            "n_phones": len(ids),
            "n_frames": n_frames,
            "sample_weight": weight,
        }
        if self.upsample == "soft":
            from minispeech.data.frame_projection import TimeSegment

            entry = self.forced.get(u.utt_id)
            segs = None
            if isinstance(entry, dict):
                try:
                    segs = segments_covering_utterance(
                        entry,
                        fallback_dur_s=float(u.dur_s or 0.0) or None,
                        gap_mode=self.gap_mode,
                    )
                except ValueError:
                    segs = None
            if segs is None:
                t0 = 0.0
                segs = []
                for i, d in enumerate(durs):
                    t1 = t0 + float(d) / self.frame_rate
                    segs.append(TimeSegment(t0, t1, phones[i] if i < len(phones) else "?"))
                    t0 = t1
            W = soft_overlap_projection(segs, frame_rate=self.frame_rate, n_frames=n_frames)
            item["proj_W"] = torch.from_numpy(W.astype(np.float32))
        return item


def collate_mel(batch: list[dict[str, Any]], *, pad_id: int = 0) -> dict[str, Any]:
    B = len(batch)
    P = max(b["n_phones"] for b in batch)
    T = max(b["n_frames"] for b in batch)
    n_mels = batch[0]["mel_target"].shape[-1]

    phone_ids = torch.full((B, P), pad_id, dtype=torch.long)
    durations = torch.zeros(B, P)
    mel_target = torch.zeros(B, T, n_mels)
    phone_mask = torch.zeros(B, P, dtype=torch.bool)
    frame_mask = torch.zeros(B, T, dtype=torch.bool)
    sample_weight = torch.ones(B)
    has_W = all("proj_W" in b for b in batch)
    proj_W = torch.zeros(B, P, T) if has_W else None

    for i, b in enumerate(batch):
        p, t = b["n_phones"], b["n_frames"]
        phone_ids[i, :p] = b["phone_ids"]
        durations[i, :p] = b["durations"]
        mel_target[i, :t] = b["mel_target"][:t]
        phone_mask[i, :p] = True
        frame_mask[i, :t] = True
        sample_weight[i] = float(b.get("sample_weight", 1.0))
        if proj_W is not None and "proj_W" in b:
            W = b["proj_W"]
            pp, tt = min(p, W.shape[0]), min(t, W.shape[1])
            proj_W[i, :pp, :tt] = W[:pp, :tt]

    out: dict[str, Any] = {
        "utt_ids": [b["utt_id"] for b in batch],
        "phone_ids": phone_ids,
        "durations": durations,
        "mel_target": mel_target,
        "phone_mask": phone_mask,
        "frame_mask": frame_mask,
        "sample_weight": sample_weight,
    }
    if proj_W is not None:
        out["proj_W"] = proj_W
    return out


def build_vocab_from_forced(
    forced_path: str | Path,
    *,
    max_utts: int | None = None,
    fill_gaps: bool = True,
    gap_mode: GapMode = DEFAULT_GAP_MODE,
) -> PhoneVocab:
    forced = load_forced_json(expand(forced_path))
    v = PhoneVocab()
    unit = "phone"
    for i, (_uid, entry) in enumerate(forced.items()):
        if max_utts is not None and i >= max_utts:
            break
        if not isinstance(entry, dict):
            continue
        gran = str(entry.get("granularity") or entry.get("align_unit") or "")
        if gran == "char" or entry.get("align_unit") == "char" or entry.get("method") == "ctc":
            unit = "char"
        try:
            segs = segments_covering_utterance(
                entry, fill_gaps=fill_gaps, gap_mode=gap_mode
            )
            v.add_phones(s.label for s in segs if s.label)
        except ValueError:
            continue
    # ``sil`` only required for sil_fill packing; expand_neighbor never emits it.
    if fill_gaps and gap_mode == "sil_fill":
        v.add_phones([SIL_LABEL])
    v.unit = unit
    return v

"""JSONL manifest schema, multi-corpus fields, dedup, and streaming IO."""

from __future__ import annotations

import hashlib
import json
import os
import re
import unicodedata
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator, Literal, Sequence

from minispeech.data.corpus_registry import (
    CHANNELS,
    get_corpus,
    infer_corpus_id,
    validate_channel,
)
from minispeech.utils import expand

_WS = re.compile(r"\s+")
_PUNCT = re.compile(r"[^\w\s]", re.UNICODE)

@dataclass
class Utterance:
    utt_id: str
    wav: str
    text_raw: str
    text_stressed: str
    speaker: str = "ruslan"
    # Track N multi-corpus fields (filled by apply_defaults when absent).
    corpus: str | None = None
    speaker_id: str | None = None
    channel: str | None = None  # studio | crowd | telephony | farfield
    sample_rate: int | None = None
    license: str | None = None
    source_url: str | None = None
    source_id: str | None = None
    container_id: str | None = None
    audio_sha256: str | None = None
    text_sha256: str | None = None
    n_samples_24k: int = 0
    n_frames: int = 0
    dur_s: float = 0.0
    split: str = "train"
    asr_cer: float | None = None
    asr_wer: float | None = None
    qc_tag: str | None = None
    stress_agreement: float | None = None
    input_ids: list[int] | None = None
    challenge: bool = False
    sample_weight: float | None = None
    extra: dict[str, Any] = field(default_factory=dict)
    # Fields absent in source JSON and filled by apply_utterance_defaults;
    # omitted again on to_json so Track A read→write does not migrate manifests.
    _defaulted_fields: set[str] = field(default_factory=set, repr=False, compare=False)

    def to_json(self) -> dict[str, Any]:
        d = asdict(self)
        extra = d.pop("extra", {})
        d.pop("_defaulted_fields", None)
        for k in self._defaulted_fields:
            d.pop(k, None)
        d = {k: v for k, v in d.items() if v is not None}
        d.update(extra)
        return d

    @classmethod
    def from_json(cls, obj: dict[str, Any], *, apply_defaults: bool = True) -> "Utterance":
        known = {
            f.name
            for f in cls.__dataclass_fields__.values()  # type: ignore
            if f.name != "_defaulted_fields"
        }
        kwargs = {k: v for k, v in obj.items() if k in known}
        extra = {k: v for k, v in obj.items() if k not in known}
        nested_extra = kwargs.pop("extra", {})
        if isinstance(nested_extra, dict):
            extra = {**nested_extra, **extra}
        if kwargs.get("container_id") is None and extra.get("container_id") is not None:
            kwargs["container_id"] = str(extra.pop("container_id"))
        u = cls(**kwargs)
        u.extra = extra
        if apply_defaults:
            apply_utterance_defaults(u)
        return u

    def effective_speaker_id(self) -> str:
        return self.speaker_id or self.speaker or "unknown"

    def materialize_defaults(self) -> "Utterance":
        """Keep registry defaults in ``to_json`` (for new multi-corpus writes)."""
        self._defaulted_fields.clear()
        return self


def apply_utterance_defaults(u: Utterance) -> Utterance:
    """Fill multi-corpus fields from the registry for in-memory use.

    Legacy RUSLAN rows stay readable. Fields filled here are tracked in
    ``_defaulted_fields`` and omitted by ``to_json`` so existing on-disk
    manifests are not rewritten when Track A scripts do read→write.
    """
    defaulted: set[str] = set()
    corpus_id = infer_corpus_id(corpus=u.corpus, speaker=u.speaker, utt_id=u.utt_id)
    info = get_corpus(corpus_id)
    if u.corpus is None:
        u.corpus = corpus_id
        defaulted.add("corpus")
    if u.speaker_id is None:
        if u.speaker:
            u.speaker_id = u.speaker
        elif info is not None and info.default_speaker_id:
            u.speaker_id = info.default_speaker_id
        else:
            u.speaker_id = "unknown"
        defaulted.add("speaker_id")
    if u.channel is None:
        u.channel = info.default_channel if info else "studio"
        defaulted.add("channel")
    else:
        validate_channel(u.channel)
    if u.sample_rate is None and info is not None and info.default_sample_rate is not None:
        u.sample_rate = info.default_sample_rate
        defaulted.add("sample_rate")
    if u.license is None and info is not None:
        u.license = info.license
        defaulted.add("license")
    if u.source_url is None and info is not None:
        u.source_url = info.source_url
        defaulted.add("source_url")
    if u.source_id is None:
        u.source_id = u.utt_id
        defaulted.add("source_id")
    if u.text_sha256 is None:
        key = text_dedup_key(u.text_raw or u.text_stressed or "")
        if key:
            u.text_sha256 = hashlib.sha256(key.encode("utf-8")).hexdigest()
            defaulted.add("text_sha256")
    if u.channel is not None and u.channel not in CHANNELS:
        raise ValueError(
            f"utt {u.utt_id}: invalid channel {u.channel!r}; "
            f"expected one of {sorted(CHANNELS)}"
        )
    u._defaulted_fields |= defaulted
    return u


def write_manifest(path: str | Path, rows: list[Utterance]) -> None:
    """Atomic JSONL write (tmp + os.replace)."""
    path = expand(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r.to_json(), ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def write_manifest_iter(path: str | Path, rows: Iterable[Utterance]) -> int:
    """Stream utterances to JSONL without materializing the full list. Returns count."""
    path = expand(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    n = 0
    with open(tmp, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r.to_json(), ensure_ascii=False) + "\n")
            n += 1
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    return n


def read_manifest(path: str | Path) -> list[Utterance]:
    """Load entire manifest into memory (Track A / small corpora)."""
    return list(iter_manifest(path))


def iter_manifest(path: str | Path) -> Iterator[Utterance]:
    """Stream utterances from JSONL one line at a time (large corpora)."""
    path = expand(path)
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield Utterance.from_json(json.loads(line))


def iter_manifests(paths: Sequence[str | Path]) -> Iterator[Utterance]:
    """Stream utterances from multiple JSONL manifests in order."""
    for p in paths:
        yield from iter_manifest(p)


def filter_split(rows: list[Utterance], split: str) -> list[Utterance]:
    return [r for r in rows if r.split == split]


def filter_split_iter(path: str | Path, split: str) -> Iterator[Utterance]:
    """Stream only rows with the given split (no full-list materialization)."""
    for r in iter_manifest(path):
        if r.split == split:
            yield r


def contiguous_split(
    rows: list[Utterance],
    train: float = 0.9,
    val: float = 0.05,
    test: float = 0.05,
) -> list[Utterance]:
    """Assign split by contiguous index ranges (sorted by utt_id).

    Book-/container-range rule for single-speaker corpora: no random utterance
    shuffle across the timeline.
    """
    assert abs(train + val + test - 1.0) < 1e-6
    rows = sorted(rows, key=lambda r: r.utt_id)
    n = len(rows)
    n_train = int(n * train)
    n_val = int(n * val)
    for i, r in enumerate(rows):
        if i < n_train:
            r.split = "train"
        elif i < n_train + n_val:
            r.split = "val"
        else:
            r.split = "test"
    return rows


def speaker_split(
    rows: list[Utterance],
    train: float = 0.9,
    val: float = 0.05,
    test: float = 0.05,
) -> list[Utterance]:
    """Assign each speaker wholly to one split (no speaker leakage).

    Speakers are ordered by (duration, speaker_id); smallest speakers fill
    test then val quotas first so large speakers remain in train. If fewer
    than two distinct speakers are present, falls back to ``contiguous_split``
    (container ranges).
    """
    assert abs(train + val + test - 1.0) < 1e-6
    by_spk: dict[str, list[Utterance]] = defaultdict(list)
    for r in rows:
        by_spk[r.effective_speaker_id()].append(r)
    if len(by_spk) < 2:
        return contiguous_split(rows, train=train, val=val, test=test)

    spk_durs = {s: sum(u.dur_s for u in utts) for s, utts in by_spk.items()}
    total = sum(spk_durs.values()) or 1.0
    target_test = total * test
    target_val = total * val
    # Smallest speakers → held-out first; tie-break by id for determinism.
    order = sorted(by_spk.keys(), key=lambda s: (spk_durs[s], s))
    assignment: dict[str, str] = {}
    acc_test = 0.0
    acc_val = 0.0
    for s in order:
        if acc_test < target_test:
            assignment[s] = "test"
            acc_test += spk_durs[s]
        elif acc_val < target_val:
            assignment[s] = "val"
            acc_val += spk_durs[s]
        else:
            assignment[s] = "train"

    # Guarantee a non-empty train when possible.
    if all(assignment[s] != "train" for s in assignment) and order:
        assignment[order[-1]] = "train"

    for s, utts in by_spk.items():
        sp = assignment[s]
        for u in utts:
            u.split = sp
    return rows


def container_speaker_split(
    rows: list[Utterance],
    train: float = 0.9,
    val: float = 0.05,
    test: float = 0.05,
) -> list[Utterance]:
    """Train/val/test by speakers when multi-speaker, else contiguous containers.

    Never assigns the same speaker_id to more than one split.
    """
    return speaker_split(rows, train=train, val=val, test=test)


def speakers_by_split(rows: Iterable[Utterance]) -> dict[str, set[str]]:
    out: dict[str, set[str]] = defaultdict(set)
    for r in rows:
        out[r.split].add(r.effective_speaker_id())
    return dict(out)


def assert_no_speaker_leak(rows: Iterable[Utterance]) -> None:
    """Raise if any speaker_id appears in more than one of train/val/test."""
    by = speakers_by_split(rows)
    splits = ("train", "val", "test")
    for i, a in enumerate(splits):
        for b in splits[i + 1 :]:
            leak = by.get(a, set()) & by.get(b, set())
            if leak:
                raise AssertionError(
                    f"speaker leak between {a!r} and {b!r}: {sorted(leak)}"
                )


def text_dedup_key(text: str) -> str:
    """Normalized text key for cross-corpus duplicate detection.

    Lowercase, strip stress/punct, ё→е — aligned with ASR-style normalization
    but kept local so manifest IO does not require the full frontend stack.
    """
    if not text:
        return ""
    t = unicodedata.normalize("NFC", text.strip())
    t = t.replace("\u0301", "").replace("\u0300", "")
    t = t.lower().replace("ё", "е")
    t = _PUNCT.sub(" ", t)
    t = _WS.sub(" ", t)
    return t.strip()


def text_fingerprint(text: str) -> str:
    """SHA-256 of the normalized cross-corpus text dedup key."""
    key = text_dedup_key(text)
    return hashlib.sha256(key.encode("utf-8")).hexdigest() if key else ""


def utterance_text_fingerprint(
    row: Utterance,
    *,
    mode: Literal["canonical", "literal_lower"] = "canonical",
) -> str:
    """Return the configured text key for within- or cross-corpus dedup.

    ``literal_lower`` preserves the historical RUSLAN builder behavior so a
    manifest rebuild does not silently change its membership. Cross-corpus
    ingestion should use the default canonical key.
    """
    if mode == "canonical":
        return row.text_sha256 or text_fingerprint(
            row.text_raw or row.text_stressed or ""
        )
    if mode == "literal_lower":
        key = (row.text_raw or row.text_stressed or "").lower()
        # Hash the empty key too: the historical builder retained the first
        # empty normalized transcript and dropped subsequent empty duplicates.
        return hashlib.sha256(key.encode("utf-8")).hexdigest()
    raise ValueError(f"unknown text dedup mode {mode!r}")


def audio_fingerprint(
    wav_path: str | Path,
    *,
    chunk_size: int = 1 << 20,
) -> str:
    """SHA-256 fingerprint of an audio file, streamed in chunks.

    Does not load the whole file into memory. Raises FileNotFoundError if
    the path is missing.
    """
    path = Path(wav_path)
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def audio_fingerprint_bytes(data: bytes) -> str:
    """Fingerprint from in-memory bytes (tests / preloaded buffers)."""
    return hashlib.sha256(data).hexdigest()


def populate_integrity_fields(
    row: Utterance,
    *,
    compute_audio: bool = True,
) -> Utterance:
    """Populate stable source/text/audio identity for a newly built row.

    Audio hashing streams the source file and therefore stays opt-in for callers
    that only have remote audio. Newly populated values are materialized in
    ``to_json``; this helper is never called implicitly while reading legacy
    manifests.
    """
    row.source_id = row.source_id or row.utt_id
    row.text_sha256 = text_fingerprint(row.text_raw or row.text_stressed or "")
    if compute_audio and row.audio_sha256 is None and row.wav:
        row.audio_sha256 = audio_fingerprint(row.wav)
    row._defaulted_fields.difference_update(
        {"source_id", "audio_sha256", "text_sha256"}
    )
    return row


@dataclass
class DuplicateHit:
    key_type: str  # "text" | "audio"
    key: str
    utt_ids: list[str]


@dataclass
class DedupStats:
    n_in: int = 0
    n_out: int = 0
    n_dropped_text: int = 0
    n_dropped_audio: int = 0
    text_duplicate_groups: int = 0
    audio_duplicate_groups: int = 0


def detect_duplicates(
    rows: Iterable[Utterance],
    *,
    by_text: bool = True,
    by_audio: bool = True,
    text_key_mode: Literal["canonical", "literal_lower"] = "canonical",
    audio_fps: dict[str, str] | None = None,
    compute_missing_audio: bool = False,
) -> list[DuplicateHit]:
    """Find duplicate groups by normalized text and/or audio fingerprint.

    ``audio_fps`` maps ``utt_id`` → fingerprint. When ``compute_missing_audio``
    is True, fingerprints are computed from ``wav`` paths (streamed).
    """
    text_groups: dict[str, list[str]] = defaultdict(list)
    audio_groups: dict[str, list[str]] = defaultdict(list)
    fps = dict(audio_fps or {})

    for r in rows:
        if by_text:
            key = utterance_text_fingerprint(
                r,
                mode=text_key_mode,
            )
            if key:
                text_groups[key].append(r.utt_id)
        if by_audio:
            fp = fps.get(r.utt_id) or r.audio_sha256
            if fp is None and compute_missing_audio and r.wav:
                try:
                    fp = audio_fingerprint(r.wav)
                    fps[r.utt_id] = fp
                except OSError:
                    fp = None
            if fp:
                audio_groups[fp].append(r.utt_id)

    hits: list[DuplicateHit] = []
    if by_text:
        for key, ids in text_groups.items():
            if len(ids) > 1:
                hits.append(DuplicateHit("text", key, ids))
    if by_audio:
        for key, ids in audio_groups.items():
            if len(ids) > 1:
                hits.append(DuplicateHit("audio", key, ids))
    return hits


def deduplicate_utterances(
    rows: Iterable[Utterance],
    *,
    by_text: bool = True,
    by_audio: bool = True,
    text_key_mode: Literal["canonical", "literal_lower"] = "canonical",
    audio_fps: dict[str, str] | None = None,
    compute_missing_audio: bool = False,
) -> tuple[list[Utterance], DedupStats]:
    """Keep first occurrence; drop later text/audio duplicates.

    Streaming-friendly over the input iterable: only fingerprint/text keys are
    retained in memory for the seen-set (plus the kept list).
    """
    stats = DedupStats()
    seen_text: set[str] = set()
    seen_audio: set[str] = set()
    fps = dict(audio_fps or {})
    kept: list[Utterance] = []

    for r in rows:
        stats.n_in += 1
        drop = False
        if by_text:
            tkey = utterance_text_fingerprint(
                r,
                mode=text_key_mode,
            )
            if tkey:
                if tkey in seen_text:
                    stats.n_dropped_text += 1
                    drop = True
                else:
                    seen_text.add(tkey)
        if not drop and by_audio:
            fp = fps.get(r.utt_id) or r.audio_sha256
            if fp is None and compute_missing_audio and r.wav:
                try:
                    fp = audio_fingerprint(r.wav)
                    fps[r.utt_id] = fp
                except OSError:
                    fp = None
            if fp:
                if fp in seen_audio:
                    stats.n_dropped_audio += 1
                    drop = True
                else:
                    seen_audio.add(fp)
        if not drop:
            kept.append(r)
            stats.n_out += 1
    return kept, stats


def iter_deduped_manifests(
    paths: Sequence[str | Path],
    *,
    by_text: bool = True,
    by_audio: bool = False,
    compute_missing_audio: bool = False,
) -> Iterator[Utterance]:
    """Stream one or more manifests, yielding first occurrence of each key.

    Only dedup keys (and optional fingerprints) are held in memory — not the
    full corpus. Audio fingerprinting is opt-in (``by_audio``).
    """
    seen_text: set[str] = set()
    seen_audio: set[str] = set()
    for r in iter_manifests(paths):
        if by_text:
            tkey = r.text_sha256 or text_fingerprint(
                r.text_raw or r.text_stressed or ""
            )
            if tkey and tkey in seen_text:
                continue
            if tkey:
                seen_text.add(tkey)
        if by_audio:
            fp: str | None = r.audio_sha256
            if fp is None and compute_missing_audio and r.wav:
                try:
                    fp = audio_fingerprint(r.wav)
                except OSError:
                    fp = None
            if fp:
                if fp in seen_audio:
                    continue
                seen_audio.add(fp)
        yield r

"""Corpus metadata registry for multi-corpus manifests (Track N / N11).

Provides default ``corpus`` / ``channel`` / ``license`` / ``source_url`` /
``sample_rate`` values when reading legacy single-corpus JSONL rows
(e.g. RUSLAN) that omit the new fields.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final, Iterable, Literal

Channel = Literal["studio", "crowd", "telephony", "farfield"]

CHANNELS: Final[frozenset[str]] = frozenset({"studio", "crowd", "telephony", "farfield"})


@dataclass(frozen=True)
class CorpusInfo:
    """Static provenance defaults for a known corpus id."""

    id: str
    license: str
    source_url: str
    default_channel: Channel
    default_sample_rate: int | None = None
    default_speaker_id: str | None = None
    notes: str = ""


# Defaults grounded in docs/registry_corpora.md and out/probe_corpus_inventory.json.
CORPORA: Final[dict[str, CorpusInfo]] = {
    "ruslan": CorpusInfo(
        id="ruslan",
        license="CC BY-NC-SA 4.0",
        source_url="https://ruslan-corpus.github.io/",
        default_channel="studio",
        # Project config sample_rate_src; local WAVs may be 44100 (see registry).
        default_sample_rate=22050,
        default_speaker_id="ruslan",
        notes="Track A production corpus; single speaker.",
    ),
}


def get_corpus(corpus_id: str | None) -> CorpusInfo | None:
    if not corpus_id:
        return None
    return CORPORA.get(corpus_id.lower())


def known_corpus_ids() -> tuple[str, ...]:
    return tuple(sorted(CORPORA))


def validate_channel(channel: str | None) -> str | None:
    """Return channel if valid; raise ValueError if non-empty and unknown."""
    if channel is None or channel == "":
        return None
    if channel not in CHANNELS:
        raise ValueError(
            f"invalid channel {channel!r}; expected one of {sorted(CHANNELS)}"
        )
    return channel


def infer_corpus_id(
    *,
    corpus: str | None = None,
    speaker: str | None = None,
    utt_id: str | None = None,
) -> str:
    """Best-effort corpus id for legacy rows missing ``corpus``."""
    if corpus:
        return corpus.lower()
    if speaker and speaker.lower() in CORPORA:
        return speaker.lower()
    if utt_id:
        stem = utt_id.split("_", 1)[0].lower()
        if stem in CORPORA:
            return stem
    return "ruslan"


def register_corpus(info: CorpusInfo, *, overwrite: bool = False) -> None:
    """Register or replace a corpus entry (tests / local extensions)."""
    key = info.id.lower()
    if key in CORPORA and not overwrite:
        raise ValueError(f"corpus {key!r} already registered")
    CORPORA[key] = info


def iter_registered() -> Iterable[CorpusInfo]:
    for cid in known_corpus_ids():
        yield CORPORA[cid]

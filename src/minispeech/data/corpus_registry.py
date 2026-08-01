"""Audited corpus registry and defaults for multi-corpus manifests."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final, Iterable, Literal

Channel = Literal["studio", "crowd", "telephony", "farfield"]

CHANNELS: Final[frozenset[str]] = frozenset({"studio", "crowd", "telephony", "farfield"})


@dataclass(frozen=True)
class CorpusInfo:
    """Audited source metadata plus manifest defaults."""

    id: str
    name: str
    source: str
    license: str
    source_url: str
    training_allowed: bool | None
    default_channel: Channel
    default_sample_rate: int | None
    n_speakers: int | None
    hours: float | None
    checked_at: str
    gated: bool
    default_speaker_id: str | None = None
    notes: str = ""


# Facts checked in P1 D03. ``None`` means the source did not provide the fact
# or legal review is still open; it is never an estimate.
CORPORA: Final[dict[str, CorpusInfo]] = {
    "ruslan": CorpusInfo(
        id="ruslan",
        name="RUSLAN",
        source="Official RUSLAN corpus site",
        license="CC BY-NC-SA 4.0",
        source_url="https://ruslan-corpus.github.io/",
        training_allowed=True,
        default_channel="studio",
        default_sample_rate=22050,
        n_speakers=1,
        hours=31.5486,
        checked_at="2026-08-01",
        gated=False,
        default_speaker_id="ruslan",
        notes="Research/non-commercial training only; ShareAlike applies.",
    ),
    "silero_tts_v4_ru": CorpusInfo(
        id="silero_tts_v4_ru",
        name="Silero TTS v4_ru",
        source="snakers4/silero-models official repository",
        license="CC BY-NC 4.0",
        source_url="https://github.com/snakers4/silero-models",
        training_allowed=None,
        default_channel="studio",
        default_sample_rate=24000,
        n_speakers=6,
        hours=None,
        checked_at="2026-08-01",
        gated=False,
        notes=(
            "External evaluation baseline only. The official model table lists "
            "aidar, baya, kseniya, xenia, eugene, and random; model outputs were "
            "not audited as a training corpus."
        ),
    ),
    "common_voice_ru_17_hf_mirror": CorpusInfo(
        id="common_voice_ru_17_hf_mirror",
        name="Common Voice 17.0 Russian (fixie-ai mirror)",
        source="Mozilla Common Voice via pinned public Hugging Face mirror",
        license="CC0-1.0",
        source_url="https://huggingface.co/datasets/fixie-ai/common_voice_17_0",
        training_allowed=True,
        default_channel="crowd",
        default_sample_rate=48000,
        n_speakers=2671,
        hours=69.10033731481552,
        checked_at="2026-08-01",
        gated=False,
        notes=(
            "Selected P1 shards contain 46,783 clips and 2,671 observed "
            "client_id values; the source index reports 2,858 speakers. "
            "Clean hours are measured separately by D10 QC."
        ),
    ),
    "common_voice_ru_25_mdc": CorpusInfo(
        id="common_voice_ru_25_mdc",
        name="Common Voice Scripted Speech 25.0 Russian",
        source="Mozilla Data Collective",
        license="CC0-1.0",
        source_url="https://mozilladatacollective.com/datasets/cmn2h1dg201gro107lpynbbd6",
        training_allowed=True,
        default_channel="crowd",
        default_sample_rate=48000,
        n_speakers=3695,
        hours=290.51,
        checked_at="2026-08-01",
        gated=True,
        notes="Download blocked until dataset terms are accepted and MDC_API_KEY is set.",
    ),
    "golos": CorpusInfo(
        id="golos",
        name="Golos",
        source="SberDevices source plus Sh1man/golos_opus mirror",
        license="CUSTOM-GOLOS",
        source_url="https://github.com/salute-developers/golos",
        training_allowed=None,
        default_channel="crowd",
        default_sample_rate=16000,
        n_speakers=None,
        hours=1240.0,
        checked_at="2026-08-01",
        gated=False,
        notes="Original custom terms conflict with the mirror CC-BY-NC-4.0 tag.",
    ),
    "openstt": CorpusInfo(
        id="openstt",
        name="Russian Open STT",
        source="snakers4/open_stt",
        license="CC BY-NC 4.0",
        source_url="https://github.com/snakers4/open_stt",
        training_allowed=True,
        default_channel="crowd",
        default_sample_rate=16000,
        n_speakers=None,
        hours=21852.0,
        checked_at="2026-08-01",
        gated=False,
        notes="Research/non-commercial only; preserve component provenance.",
    ),
    "sova_rudevices": CorpusInfo(
        id="sova_rudevices",
        name="SOVA RuDevices",
        source="SOVA.ai via bond005/sova_rudevices",
        license="CC BY 4.0",
        source_url="https://huggingface.co/datasets/bond005/sova_rudevices",
        training_allowed=True,
        default_channel="crowd",
        default_sample_rate=16000,
        n_speakers=None,
        hours=101.0,
        checked_at="2026-08-01",
        gated=False,
        notes="Published schema has no speaker IDs.",
    ),
    "m_ailabs_ru": CorpusInfo(
        id="m_ailabs_ru",
        name="M-AILABS Russian",
        source="OpenSLR 94",
        license="M-AILABS",
        source_url="https://www.openslr.org/94/",
        training_allowed=None,
        default_channel="studio",
        default_sample_rate=16000,
        n_speakers=3,
        hours=46.78,
        checked_at="2026-08-01",
        gated=False,
        notes="Original terms conflict with the reviewed filtered mirror tag.",
    ),
    "yodas_ru": CorpusInfo(
        id="yodas_ru",
        name="YODAS Russian",
        source="espnet/yodas",
        license="CC BY 3.0",
        source_url="https://huggingface.co/datasets/espnet/yodas",
        training_allowed=None,
        default_channel="farfield",
        default_sample_rate=16000,
        n_speakers=None,
        hours=39622.289,
        checked_at="2026-08-01",
        gated=False,
        notes="YouTube source rights and deletion obligations require legal review.",
    ),
    "ruls": CorpusInfo(
        id="ruls",
        name="Russian LibriSpeech (RuLS)",
        source="OpenSLR 96",
        license="Public Domain in the USA",
        source_url="https://www.openslr.org/96/",
        training_allowed=True,
        default_channel="studio",
        default_sample_rate=16000,
        n_speakers=None,
        hours=98.0,
        checked_at="2026-08-01",
        gated=False,
        notes="Speaker count is not reported on the source page.",
    ),
    "balalaika": CorpusInfo(
        id="balalaika",
        name="Balalaika",
        source="lab260/Balalaika metadata and external source audio",
        license="CC BY-NC-ND 4.0",
        source_url="https://huggingface.co/datasets/lab260/Balalaika",
        training_allowed=None,
        default_channel="crowd",
        default_sample_rate=None,
        n_speakers=None,
        hours=5078.0,
        checked_at="2026-08-01",
        gated=False,
        notes="Audio keeps per-source licenses; derivative status requires review.",
    ),
    "espeech_podcasts": CorpusInfo(
        id="espeech_podcasts",
        name="ESpeech Podcasts",
        source="ESpeech/ESpeech-podcasts",
        license="CC BY-NC 4.0",
        source_url="https://huggingface.co/datasets/ESpeech/ESpeech-podcasts",
        training_allowed=True,
        default_channel="farfield",
        default_sample_rate=None,
        n_speakers=None,
        hours=None,
        checked_at="2026-08-01",
        gated=True,
        notes="Manual Hub approval is not confirmed in this environment.",
    ),
    "espeech_webinars2": CorpusInfo(
        id="espeech_webinars2",
        name="ESpeech Webinars 2",
        source="ESpeech/ESpeech-webinars2",
        license="CONFLICT: Apache-2.0 / MIT",
        source_url="https://huggingface.co/datasets/ESpeech/ESpeech-webinars2",
        training_allowed=None,
        default_channel="farfield",
        default_sample_rate=44100,
        n_speakers=None,
        hours=850.0,
        checked_at="2026-08-01",
        gated=False,
        notes="Hub front matter and card body declare different licenses.",
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

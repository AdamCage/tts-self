"""Discrete Qwen3-TTS 12Hz codec-token caches + train/infer packing.

Wave 1 (D1): frozen encode → ``[T, 16]`` int codes with provenance and
offset-aligned RU conditioning features. Does **not** reuse VibeVoice latents
or text-hidden caches.

Shard layout (per corpus root)
------------------------------
- ``codec_tokens_XXXXX.safetensors`` — packed tensors keyed ``{utt_id}__{field}``
- ``codec_tokens_index.json`` — utt → shard + provenance meta
- ``encode_meta.json`` — codec fingerprint, tokenizer ids, coverage

Packed sequence contract (train == infer)
-----------------------------------------
``[text_ids || acoustic_codes || EOS]`` where EOS is a single reserved speech
marker after the last codebook frame. Parallel RU features align to the text
prefix only (length ``L``); acoustic positions carry pad feature ids.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence  # Any: tokenizer duck-typing

import torch
from safetensors.torch import load_file, save_file
from torch.utils.data import Dataset

from minispeech.constants import QWEN3TTS_12HZ, STRESS_CHAR, get_target
from minispeech.data.manifest import Utterance, iter_manifest
from minispeech.eval.critical_spans import CriticalSpan, parse_critical_spans
from minispeech.frontend.g2p import text_to_phonemes
from minispeech.frontend.phone_vocab import PhoneVocab
from minispeech.frontend.stress import to_canonical_stress
from minispeech.utils import expand

# Reserved speech EOS id (outside codebook vocab 0..2047).
SPEECH_EOS_ID: int = 2048

# Feature channel pad / none
FEAT_PAD: int = 0
STRESS_NONE: int = 0
STRESS_MARKED: int = 1

# Critical-span type → id (0 = none). Stable for ablations.
SPAN_TYPE_TO_ID: dict[str, int] = {
    "": 0,
    "none": 0,
    "number": 1,
    "digit": 1,
    "code": 2,
    "phone": 3,
    "telephone": 3,
    "date": 4,
    "time": 5,
    "latin": 6,
    "entity": 7,
    "money": 8,
    "percent": 9,
    "homograph": 10,
    "ordinal": 11,
}
SPAN_ID_TO_TYPE: dict[int, str] = {
    0: "none",
    1: "number",
    2: "code",
    3: "phone",
    4: "date",
    5: "time",
    6: "latin",
    7: "entity",
    8: "money",
    9: "percent",
    10: "homograph",
    11: "ordinal",
}

CODEC_TOKEN_FIELDS: tuple[str, ...] = (
    "codes",
    "text_ids",
    "stress",
    "phoneme",
    "span_type",
    "word_bound",
    "phrase_bound",
)

_WORD_RE = re.compile(r"[0-9A-Za-zА-Яа-яЁё]+", re.UNICODE)
_PHRASE_BREAK = re.compile(r"[.!?…;:—–\-]")


@dataclass(slots=True)
class RuTextFeatures:
    """Offset-aligned RU channels for a text prefix of length ``L``."""

    stress: torch.Tensor  # [L] int64
    phoneme: torch.Tensor  # [L] int64
    span_type: torch.Tensor  # [L] int64
    word_bound: torch.Tensor  # [L] int64 (1 at first token of each word)
    phrase_bound: torch.Tensor  # [L] int64 (1 at first token after phrase break)


@dataclass(slots=True)
class CodecTokenExample:
    """One utterance worth of discrete codes + text conditioning."""

    utt_id: str
    codes: torch.Tensor  # [T, n_codebooks] int64
    text_ids: torch.Tensor  # [L] int64
    features: RuTextFeatures
    corpus: str
    speaker_id: str
    license: str | None = None
    codec_id: str = QWEN3TTS_12HZ.name
    codec_hash: str | None = None
    n_frames: int = 0
    text_len: int = 0
    dur_s: float = 0.0
    text_stressed: str = ""
    sample_weight: float = 1.0
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.n_frames <= 0:
            self.n_frames = int(self.codes.shape[0])
        if self.text_len <= 0:
            self.text_len = int(self.text_ids.numel())


@dataclass(slots=True, frozen=True)
class PackedCodecSeq:
    """Train/infer-identical packed view of one utterance."""

    input_ids: torch.Tensor  # [S] text token ids (prefix only; acoustic uses 0)
    codes_target: torch.Tensor  # [T, C] acoustic codes (no EOS row)
    codes_input: torch.Tensor  # [T, C] AR-shifted codes (BOS zeros → c_{t-1})
    text_lens: int
    acoustic_lens: int  # T (codes only; EOS is separate)
    seq_len: int  # L + T + 1 (EOS)
    is_text: torch.Tensor  # [S_full] bool
    is_acoustic: torch.Tensor  # [S_full] bool
    is_eos: torch.Tensor  # [S_full] bool
    seq_mask: torch.Tensor  # [S_full] bool
    stress: torch.Tensor  # [L]
    phoneme: torch.Tensor  # [L]
    span_type: torch.Tensor  # [L]
    word_bound: torch.Tensor  # [L]
    phrase_bound: torch.Tensor  # [L]
    speech_eos_id: int = SPEECH_EOS_ID


def span_type_id(name: str | None) -> int:
    if not name:
        return 0
    key = str(name).strip().lower()
    return int(SPAN_TYPE_TO_ID.get(key, SPAN_TYPE_TO_ID.get("entity", 7) if key else 0))


def _char_stress_mask(text: str) -> list[bool]:
    """Per-character stress: True on a vowel that carries combining acute."""
    t = to_canonical_stress(text)
    out: list[bool] = [False] * len(t)
    i = 0
    while i < len(t):
        if t[i] == STRESS_CHAR:
            if i > 0:
                out[i - 1] = True
            i += 1
            continue
        i += 1
    return out


def _char_span_ids(text: str, spans: Sequence[CriticalSpan]) -> list[int]:
    ids = [0] * len(text)
    for sp in spans:
        sid = span_type_id(sp.type)
        s = max(0, min(len(text), int(sp.start)))
        e = max(s, min(len(text), int(sp.end)))
        for i in range(s, e):
            ids[i] = sid
    return ids


def _char_word_bounds(text: str) -> list[bool]:
    bounds = [False] * len(text)
    for m in _WORD_RE.finditer(text):
        bounds[m.start()] = True
    return bounds


def _char_phrase_bounds(text: str) -> list[bool]:
    """Mark first character of a word that follows a phrase-break punctuation."""
    bounds = [False] * len(text)
    break_at = {m.end() for m in _PHRASE_BREAK.finditer(text)}
    for m in _WORD_RE.finditer(text):
        # Look back for break punctuation / whitespace before this word.
        j = m.start() - 1
        while j >= 0 and text[j].isspace():
            j -= 1
        if j < 0:
            bounds[m.start()] = True  # utterance onset
        elif (j + 1) in break_at or text[j] in ".!?…;:":
            bounds[m.start()] = True
    if text and not any(bounds):
        m0 = _WORD_RE.search(text)
        if m0:
            bounds[m0.start()] = True
    return bounds


def _offsets_to_token_feats(
    offsets: Sequence[tuple[int, int]],
    char_vals: Sequence[int] | Sequence[bool],
    *,
    reduce: str = "max",
) -> list[int]:
    """Project char-level ints/bools onto BPE tokens via offset mapping."""
    out: list[int] = []
    for start, end in offsets:
        if end <= start:
            out.append(0)
            continue
        chunk = [int(char_vals[i]) for i in range(start, min(end, len(char_vals)))]
        if not chunk:
            out.append(0)
            continue
        if reduce == "any":
            out.append(1 if any(chunk) else 0)
        elif reduce == "first":
            out.append(int(chunk[0]))
        else:
            out.append(int(max(chunk)))
    return out


def _phoneme_ids_for_offsets(
    text: str,
    offsets: Sequence[tuple[int, int]],
    phone_vocab: PhoneVocab,
) -> list[int]:
    """Map G2P phones onto BPE tokens by uniform char-span projection.

    Phones are laid out across the unstressed character skeleton proportionally;
    each BPE token takes the phone covering its midpoint character.
    """
    plain = to_canonical_stress(text).replace(STRESS_CHAR, "")
    phones = text_to_phonemes(text)
    if not phones or not plain:
        return [phone_vocab.pad_id] * len(offsets)
    phone_vocab.add_phones(phones)
    # Uniform assignment of phones over characters.
    n = len(plain)
    char_phone = [phone_vocab.pad_id] * n
    for i, ph in enumerate(phones):
        a = int(i * n / len(phones))
        b = int((i + 1) * n / len(phones))
        if b <= a:
            b = min(n, a + 1)
        pid = phone_vocab.stoi.get(ph, phone_vocab.unk_id)
        for j in range(a, b):
            char_phone[j] = pid
    # Map NFC stressed offsets → plain indices by stripping stress in a cursor.
    stressed = to_canonical_stress(text)
    # Build map stressed_char_idx → plain_char_idx
    plain_idx_of: list[int] = []
    pi = 0
    for ch in stressed:
        if ch == STRESS_CHAR:
            plain_idx_of.append(max(0, pi - 1) if pi > 0 else 0)
        else:
            plain_idx_of.append(min(pi, n - 1) if n else 0)
            pi += 1
    out: list[int] = []
    for start, end in offsets:
        if end <= start or not plain_idx_of:
            out.append(phone_vocab.pad_id)
            continue
        mid = (start + end - 1) // 2
        mid = max(0, min(len(plain_idx_of) - 1, mid))
        out.append(int(char_phone[plain_idx_of[mid]]))
    return out


def build_ru_text_features(
    text_stressed: str,
    offsets: Sequence[tuple[int, int]],
    *,
    phone_vocab: PhoneVocab,
    critical_spans: Sequence[CriticalSpan] | Sequence[Mapping[str, Any]] | None = None,
) -> RuTextFeatures:
    """Build L-aligned RU feature channels from stressed text + BPE offsets."""
    text = to_canonical_stress(text_stressed or "")
    L = len(offsets)
    if isinstance(critical_spans, Sequence) and critical_spans and not isinstance(
        critical_spans[0], CriticalSpan
    ):
        spans = parse_critical_spans(list(critical_spans))
    else:
        spans = list(critical_spans or [])  # type: ignore[arg-type]

    stress = _offsets_to_token_feats(
        offsets, _char_stress_mask(text), reduce="any"
    )
    # Upgrade bool stress to STRESS_MARKED / NONE
    stress_ids = [STRESS_MARKED if s else STRESS_NONE for s in stress]
    span_ids = _offsets_to_token_feats(
        offsets, _char_span_ids(text, spans), reduce="max"
    )
    word_b = _offsets_to_token_feats(
        offsets, _char_word_bounds(text), reduce="any"
    )
    phrase_b = _offsets_to_token_feats(
        offsets, _char_phrase_bounds(text), reduce="any"
    )
    phoneme = _phoneme_ids_for_offsets(text, offsets, phone_vocab)

    def _t(xs: Sequence[int]) -> torch.Tensor:
        if len(xs) != L:
            raise ValueError(f"feature len {len(xs)} != L={L}")
        return torch.tensor(list(xs), dtype=torch.long)

    return RuTextFeatures(
        stress=_t(stress_ids),
        phoneme=_t(phoneme),
        span_type=_t(span_ids),
        word_bound=_t(word_b),
        phrase_bound=_t(phrase_b),
    )


def tokenize_text_with_offsets(
    tokenizer: Any,
    text: str,
    *,
    max_length: int = 256,
    add_special_tokens: bool = True,
) -> tuple[torch.Tensor, list[tuple[int, int]]]:
    """Return ``input_ids`` ``[L]`` and char offsets for feature alignment."""
    text = to_canonical_stress(text or "")
    try:
        enc = tokenizer(
            text,
            return_tensors=None,
            return_offsets_mapping=True,
            add_special_tokens=add_special_tokens,
            truncation=True,
            max_length=max_length,
        )
        ids = enc["input_ids"]
        offs = enc["offset_mapping"]
        if ids and isinstance(ids[0], (list, tuple)):
            ids = ids[0]
            offs = offs[0]
        offsets = [(int(a), int(b)) for a, b in offs]
        return torch.tensor(ids, dtype=torch.long), offsets
    except Exception:
        # Slow tokenizer / no offset mapping: approximate with uniform char spans.
        enc = tokenizer(
            text,
            return_tensors=None,
            add_special_tokens=add_special_tokens,
            truncation=True,
            max_length=max_length,
        )
        ids = enc["input_ids"]
        if ids and isinstance(ids[0], (list, tuple)):
            ids = ids[0]
        L = len(ids)
        n = max(1, len(text))
        offsets = []
        for i in range(L):
            a = int(i * n / L)
            b = int((i + 1) * n / L)
            if b <= a:
                b = min(n, a + 1)
            offsets.append((a, b))
        return torch.tensor(ids, dtype=torch.long), offsets


def pack_codec_sequence(
    text_ids: torch.Tensor | Sequence[int],
    codes: torch.Tensor,
    features: RuTextFeatures | None = None,
    *,
    speech_eos_id: int = SPEECH_EOS_ID,
) -> PackedCodecSeq:
    """Pack text prefix + codes + EOS. Identical for train and infer paths."""
    if not torch.is_tensor(text_ids):
        text_ids_t = torch.tensor(list(text_ids), dtype=torch.long)
    else:
        text_ids_t = text_ids.long().view(-1)
    codes_t = codes.long()
    if codes_t.ndim != 2:
        raise ValueError(f"codes must be [T, C], got {tuple(codes_t.shape)}")
    L = int(text_ids_t.numel())
    T = int(codes_t.shape[0])
    C = int(codes_t.shape[1])
    S = L + T + 1

    is_text = torch.zeros(S, dtype=torch.bool)
    is_acoustic = torch.zeros(S, dtype=torch.bool)
    is_eos = torch.zeros(S, dtype=torch.bool)
    seq_mask = torch.ones(S, dtype=torch.bool)
    is_text[:L] = True
    is_acoustic[L : L + T] = True
    is_eos[L + T] = True

    # AR shift for CB streams: input[t] = codes[t-1], input[0] = 0.
    codes_input = torch.zeros_like(codes_t)
    if T > 0:
        codes_input[1:] = codes_t[:-1]

    if features is None:
        z = torch.zeros(L, dtype=torch.long)
        features = RuTextFeatures(
            stress=z.clone(),
            phoneme=z.clone(),
            span_type=z.clone(),
            word_bound=z.clone(),
            phrase_bound=z.clone(),
        )
    for name in ("stress", "phoneme", "span_type", "word_bound", "phrase_bound"):
        feat = getattr(features, name)
        if int(feat.numel()) != L:
            raise ValueError(f"{name} len {feat.numel()} != text_len {L}")

    return PackedCodecSeq(
        input_ids=text_ids_t,
        codes_target=codes_t,
        codes_input=codes_input,
        text_lens=L,
        acoustic_lens=T,
        seq_len=S,
        is_text=is_text,
        is_acoustic=is_acoustic,
        is_eos=is_eos,
        seq_mask=seq_mask,
        stress=features.stress.long(),
        phoneme=features.phoneme.long(),
        span_type=features.span_type.long(),
        word_bound=features.word_bound.long(),
        phrase_bound=features.phrase_bound.long(),
        speech_eos_id=int(speech_eos_id),
    )


def pack_for_train(example: CodecTokenExample, *, speech_eos_id: int = SPEECH_EOS_ID) -> PackedCodecSeq:
    return pack_codec_sequence(
        example.text_ids,
        example.codes,
        example.features,
        speech_eos_id=speech_eos_id,
    )


def pack_for_infer(
    text_ids: torch.Tensor | Sequence[int],
    features: RuTextFeatures | None = None,
    *,
    n_codebooks: int | None = None,
    speech_eos_id: int = SPEECH_EOS_ID,
) -> PackedCodecSeq:
    """Infer-time pack with empty acoustic codes (generation starts after text)."""
    C = int(n_codebooks or get_target(QWEN3TTS_12HZ.name).n_codebooks or 16)
    empty = torch.zeros(0, C, dtype=torch.long)
    return pack_codec_sequence(
        text_ids, empty, features, speech_eos_id=speech_eos_id
    )


def assert_pack_train_infer_equal(
    example: CodecTokenExample,
    *,
    speech_eos_id: int = SPEECH_EOS_ID,
) -> PackedCodecSeq:
    """Verify train packing matches reconstructing from text + codes + features."""
    train = pack_for_train(example, speech_eos_id=speech_eos_id)
    # Infer path with known codes must match train for the text prefix / features /
    # and when codes are supplied via pack_codec_sequence (shared helper).
    recon = pack_codec_sequence(
        example.text_ids,
        example.codes,
        example.features,
        speech_eos_id=speech_eos_id,
    )
    for key in (
        "input_ids",
        "codes_target",
        "codes_input",
        "is_text",
        "is_acoustic",
        "is_eos",
        "seq_mask",
        "stress",
        "phoneme",
        "span_type",
        "word_bound",
        "phrase_bound",
    ):
        a = getattr(train, key)
        b = getattr(recon, key)
        if not torch.equal(a, b):
            raise AssertionError(f"train/infer pack mismatch on {key}")
    if train.seq_len != recon.seq_len or train.text_lens != recon.text_lens:
        raise AssertionError("train/infer length mismatch")
    if train.speech_eos_id != speech_eos_id:
        raise AssertionError("EOS id mismatch")
    return train


# ---------------------------------------------------------------------------
# Shard I/O (multi-tensor per utterance)
# ---------------------------------------------------------------------------


def _field_key(utt_id: str, field: str) -> str:
    return f"{utt_id}__{field}"


class CodecTokenShardWriter:
    """Write multi-field codec-token shards with resume via index."""

    def __init__(
        self,
        out_dir: str | Path,
        prefix: str = "codec_tokens",
        shard_size: int = 2000,
        *,
        resume: bool = True,
    ):
        self.out_dir = expand(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.prefix = prefix
        self.shard_size = int(shard_size)
        self.index: dict[str, dict[str, Any]] = {}
        self._buf: dict[str, torch.Tensor] = {}
        self._buf_utts: list[str] = []
        self._shard_id = 0
        self._idx_path = self.out_dir / f"{self.prefix}_index.json"
        if resume and self._idx_path.exists():
            self.index = json.loads(self._idx_path.read_text(encoding="utf-8"))
            max_id = -1
            for info in self.index.values():
                name = info.get("shard", "")
                try:
                    stem = Path(name).stem
                    sid = int(stem.rsplit("_", 1)[-1])
                    max_id = max(max_id, sid)
                except Exception:
                    continue
            self._shard_id = max_id + 1 if max_id >= 0 else 0

    def has(self, utt_id: str) -> bool:
        return utt_id in self.index

    def _shard_name(self) -> str:
        return f"{self.prefix}_{self._shard_id:05d}.safetensors"

    def _write_index(self) -> None:
        self._idx_path.write_text(
            json.dumps(self.index, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    def add(self, example: CodecTokenExample, meta: Mapping[str, Any] | None = None) -> None:
        if example.utt_id in self.index:
            return
        uid = example.utt_id
        tensors = {
            "codes": example.codes.contiguous().cpu().long(),
            "text_ids": example.text_ids.contiguous().cpu().long(),
            "stress": example.features.stress.contiguous().cpu().long(),
            "phoneme": example.features.phoneme.contiguous().cpu().long(),
            "span_type": example.features.span_type.contiguous().cpu().long(),
            "word_bound": example.features.word_bound.contiguous().cpu().long(),
            "phrase_bound": example.features.phrase_bound.contiguous().cpu().long(),
        }
        for field_name, tensor in tensors.items():
            self._buf[_field_key(uid, field_name)] = tensor
        self._buf_utts.append(uid)
        entry = {
            "shard": self._shard_name(),
            "n_frames": int(example.n_frames),
            "text_len": int(example.text_len),
            "corpus": example.corpus,
            "speaker_id": example.speaker_id,
            "license": example.license,
            "codec_id": example.codec_id,
            "codec_hash": example.codec_hash,
            "dur_s": float(example.dur_s),
            "fields": list(CODEC_TOKEN_FIELDS),
        }
        if meta:
            entry.update(dict(meta))
        self.index[uid] = entry
        if len(self._buf_utts) >= self.shard_size:
            self.flush()

    def flush(self) -> None:
        if not self._buf:
            return
        path = self.out_dir / self._shard_name()
        save_file(self._buf, str(path))
        self._buf.clear()
        self._buf_utts.clear()
        self._shard_id += 1
        self._write_index()

    def close(self) -> Path:
        self.flush()
        self._write_index()
        return self._idx_path


class CodecTokenShardReader:
    """Read multi-field codec-token shards."""

    def __init__(
        self,
        out_dir: str | Path,
        prefix: str = "codec_tokens",
        *,
        lazy: bool = True,
        lru_shards: int = 3,
    ):
        self.out_dir = expand(out_dir)
        self.prefix = prefix
        idx_path = self.out_dir / f"{prefix}_index.json"
        self.index = json.loads(idx_path.read_text(encoding="utf-8"))
        self.lazy = lazy
        self.lru_shards = lru_shards
        self._cache: dict[str, dict[str, torch.Tensor]] = {}
        self._lru: list[str] = []

    def __contains__(self, utt_id: str) -> bool:
        return utt_id in self.index

    def _touch(self, shard: str) -> None:
        if shard in self._lru:
            self._lru.remove(shard)
        self._lru.append(shard)
        while len(self._lru) > self.lru_shards:
            old = self._lru.pop(0)
            self._cache.pop(old, None)

    def _shard_dict(self, shard: str) -> dict[str, torch.Tensor]:
        if shard not in self._cache:
            self._cache[shard] = load_file(str(self.out_dir / shard))
        self._touch(shard)
        return self._cache[shard]

    def get_field(self, utt_id: str, field: str) -> torch.Tensor:
        info = self.index[utt_id]
        shard = info["shard"]
        key = _field_key(utt_id, field)
        if self.lazy:
            try:
                from safetensors import safe_open

                with safe_open(str(self.out_dir / shard), framework="pt") as f:
                    return f.get_tensor(key).clone()
            except ImportError:
                self.lazy = False
        return self._shard_dict(shard)[key].clone()

    def get(self, utt_id: str) -> CodecTokenExample:
        info = self.index[utt_id]
        codes = self.get_field(utt_id, "codes")
        text_ids = self.get_field(utt_id, "text_ids")
        feats = RuTextFeatures(
            stress=self.get_field(utt_id, "stress"),
            phoneme=self.get_field(utt_id, "phoneme"),
            span_type=self.get_field(utt_id, "span_type"),
            word_bound=self.get_field(utt_id, "word_bound"),
            phrase_bound=self.get_field(utt_id, "phrase_bound"),
        )
        return CodecTokenExample(
            utt_id=utt_id,
            codes=codes.long(),
            text_ids=text_ids.long(),
            features=feats,
            corpus=str(info.get("corpus") or "unknown"),
            speaker_id=str(info.get("speaker_id") or "unknown"),
            license=info.get("license"),
            codec_id=str(info.get("codec_id") or QWEN3TTS_12HZ.name),
            codec_hash=info.get("codec_hash"),
            n_frames=int(info.get("n_frames") or codes.shape[0]),
            text_len=int(info.get("text_len") or text_ids.numel()),
            dur_s=float(info.get("dur_s") or 0.0),
        )


class MultiRootCodecTokenReader:
    """Route codec-token reads across per-corpus roots."""

    def __init__(
        self,
        corpus_dirs: Mapping[str, str | Path],
        utt_to_corpus: Mapping[str, str],
        *,
        prefix: str = "codec_tokens",
        lazy: bool = True,
    ):
        self.utt_to_corpus = dict(utt_to_corpus)
        self._readers: dict[str, CodecTokenShardReader] = {}
        for corpus, directory in corpus_dirs.items():
            path = expand(directory)
            if (path / f"{prefix}_index.json").exists():
                self._readers[corpus] = CodecTokenShardReader(
                    path, prefix=prefix, lazy=lazy
                )

    def __contains__(self, utt_id: str) -> bool:
        corpus = self.utt_to_corpus.get(utt_id)
        if corpus is None:
            return False
        r = self._readers.get(corpus)
        return r is not None and utt_id in r

    def get(self, utt_id: str) -> CodecTokenExample:
        corpus = self.utt_to_corpus.get(utt_id)
        if corpus is None:
            raise KeyError(f"utt_id {utt_id!r} missing from utt_to_corpus")
        r = self._readers.get(corpus)
        if r is None:
            raise KeyError(f"no codec_tokens root for corpus {corpus!r}")
        return r.get(utt_id)


class CodecTokenDataset(Dataset):
    """Track N discrete-token dataset over multicorpus codec shards."""

    def __init__(
        self,
        manifest_path: str | Path,
        corpus_caches: Mapping[str, Mapping[str, str]],
        *,
        split: str = "train",
        trackn_sets: Mapping[str, str] | None = None,
        utt_ids: Sequence[str] | None = None,
        max_utterances: int | None = None,
        max_hours: float | None = None,
        exclude_tags: Sequence[str] = ("exclude", "duplicate"),
        review_weight: float | None = 0.5,
        require_caches: bool = True,
        prefix: str = "codec_tokens",
    ):
        from minispeech.data.dataset import (
            DEFAULT_TRACKN_SET_PATHS,
            MultiCorpusRow,
            RUSLAN_CORPUS_ID,
            load_trackn_utt_ids,
        )

        if utt_ids is not None:
            wanted_list = [str(u) for u in utt_ids]
        else:
            paths = dict(DEFAULT_TRACKN_SET_PATHS if trackn_sets is None else trackn_sets)
            wanted_list = load_trackn_utt_ids(paths, splits=(split,))[split]
        wanted = set(wanted_list)
        excl = set(exclude_tags) if exclude_tags else set()
        found: dict[str, MultiCorpusRow] = {}
        for r in iter_manifest(manifest_path):
            if r.utt_id not in wanted:
                continue
            if excl and (r.qc_tag or "clean") in excl:
                continue
            corpus = str(r.corpus or RUSLAN_CORPUS_ID)
            found[r.utt_id] = MultiCorpusRow(
                utt_id=r.utt_id,
                corpus=corpus,
                speaker_id=r.effective_speaker_id(),
                qc_tag=r.qc_tag,
                sample_weight=r.sample_weight,
                dur_s=float(r.dur_s or 0.0),
            )
        rows = [found[uid] for uid in wanted_list if uid in found]
        if max_utterances is not None:
            rows = rows[: int(max_utterances)]
        if max_hours is not None:
            budget = float(max_hours) * 3600.0
            acc, kept = 0.0, []
            for r in rows:
                if acc + r.dur_s > budget:
                    break
                kept.append(r)
                acc += r.dur_s
            rows = kept
        self.rows = rows
        self.split = split
        self.review_weight = review_weight
        self.utt_to_corpus = {r.utt_id: r.corpus for r in rows}
        dirs = {
            c: cfg["codec_tokens_dir"]
            for c, cfg in corpus_caches.items()
            if cfg.get("codec_tokens_dir")
        }
        self.reader = MultiRootCodecTokenReader(
            dirs, self.utt_to_corpus, prefix=prefix, lazy=True
        )
        if require_caches:
            missing = [r.utt_id for r in self.rows if r.utt_id not in self.reader]
            if missing:
                raise KeyError(
                    f"missing codec_tokens for {len(missing)} utts "
                    f"(e.g. {missing[:3]})"
                )

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.rows[idx]
        ex = self.reader.get(row.utt_id)
        weight = 1.0
        if row.sample_weight is not None:
            weight = float(row.sample_weight)
        elif self.review_weight is not None and row.qc_tag == "review":
            weight = float(self.review_weight)
        packed = pack_for_train(ex)
        return {
            "utt_id": row.utt_id,
            "corpus": row.corpus,
            "speaker_id": row.speaker_id,
            "codes": ex.codes,
            "text_ids": ex.text_ids,
            "stress": ex.features.stress,
            "phoneme": ex.features.phoneme,
            "span_type": ex.features.span_type,
            "word_bound": ex.features.word_bound,
            "phrase_bound": ex.features.phrase_bound,
            "n_frames": ex.n_frames,
            "text_len": ex.text_len,
            "dur_s": float(row.dur_s or ex.dur_s),
            "sample_weight": weight,
            "license": ex.license,
            "codec_id": ex.codec_id,
            "codec_hash": ex.codec_hash,
            "packed": packed,
        }


def verify_trackn_no_leak(
    manifest_path: str | Path,
    trackn_sets: Mapping[str, str],
) -> dict[str, Any]:
    """Confirm frozen Track N lists are speaker-disjoint and have no test leak."""
    from minispeech.data.dataset import load_utt_ids_from_eval_set

    lists = {k: set(load_utt_ids_from_eval_set(v)) for k, v in trackn_sets.items()}
    overlaps: dict[str, int] = {}
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        if a in lists and b in lists:
            overlaps[f"{a}∩{b}"] = len(lists[a] & lists[b])
    by_id: dict[str, Utterance] = {}
    for u in iter_manifest(manifest_path):
        by_id[u.utt_id] = u

    def speakers(split: str) -> set[str]:
        out: set[str] = set()
        for uid in lists.get(split, ()):
            u = by_id.get(uid)
            if u is not None:
                out.add(u.effective_speaker_id())
        return out

    spk = {s: speakers(s) for s in ("train", "val", "test") if s in lists}
    spk_leaks: dict[str, list[str]] = {}
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        if a in spk and b in spk:
            leak = sorted(spk[a] & spk[b])
            if leak:
                spk_leaks[f"{a}∩{b}"] = leak[:20]

    hours = {}
    for split, ids in lists.items():
        hours[split] = sum(float(by_id[i].dur_s or 0.0) for i in ids if i in by_id) / 3600.0

    ok = all(v == 0 for v in overlaps.values()) and not spk_leaks
    return {
        "ok": ok,
        "utt_overlaps": overlaps,
        "speaker_leaks_sample": spk_leaks,
        "n": {k: len(v) for k, v in lists.items()},
        "hours": hours,
    }


def write_encode_meta(path: str | Path, meta: Mapping[str, Any]) -> Path:
    p = expand(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(dict(meta), indent=2, ensure_ascii=False), encoding="utf-8")
    return p

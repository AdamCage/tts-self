"""Source-specific ingestion into the versioned multi-corpus manifest schema."""

from __future__ import annotations

import csv
import glob
import hashlib
import json
import re
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import soundfile as sf

from minispeech.constants import SR, n_frames_from_samples
from minispeech.data.corpus_registry import CorpusInfo, get_corpus, validate_channel
from minispeech.data.manifest import (
    Utterance,
    apply_utterance_defaults,
    contiguous_split,
    populate_integrity_fields,
    speaker_split,
    write_manifest,
    write_manifest_iter,
)
from minispeech.frontend.normalize import normalize_text
from minispeech.frontend.stress import to_canonical_stress
from minispeech.utils import expand, load_yaml

_ID_PART = re.compile(r"[^a-z0-9_]+")
_SOURCE_SPLITS = {"train": "train", "validation": "val", "val": "val", "test": "test"}


def stable_utt_id(corpus: str, source_id: str) -> str:
    """Return a deterministic, corpus-namespaced 96-bit source identifier."""
    corpus_slug = _ID_PART.sub("_", corpus.strip().lower()).strip("_")
    if not corpus_slug:
        raise ValueError("corpus id must contain an ASCII letter or digit")
    source_id = source_id.strip()
    if not source_id:
        raise ValueError("source_id must be non-empty")
    digest = hashlib.sha256(source_id.encode("utf-8")).hexdigest()[:24]
    return f"{corpus_slug}__{digest}"


def _value(record: Mapping[str, Any], path: str | None, default: Any = None) -> Any:
    if not path:
        return default
    value: Any = record
    for key in path.split("."):
        if not isinstance(value, Mapping) or key not in value:
            return default
        value = value[key]
    return value


def _expand_inputs(value: str | list[str], *, base_dir: Path) -> list[Path]:
    patterns = [value] if isinstance(value, str) else value
    paths: list[Path] = []
    for pattern in patterns:
        candidate = Path(pattern).expanduser()
        if not candidate.is_absolute():
            candidate = base_dir / candidate
        matched = sorted(Path(p).resolve() for p in glob.glob(str(candidate)))
        if matched:
            paths.extend(matched)
        elif candidate.is_file():
            paths.append(candidate.resolve())
    unique = list(dict.fromkeys(paths))
    if not unique:
        raise FileNotFoundError(f"no inputs matched: {patterns}")
    return unique


def _iter_jsonl(paths: Iterable[Path]) -> Iterator[tuple[dict[str, Any], str | None]]:
    for path in paths:
        with path.open(encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError(f"{path}:{line_no}: JSONL row must be an object")
                yield value, None


def _iter_delimited(
    paths: Iterable[Path],
    *,
    delimiter: str,
) -> Iterator[tuple[dict[str, Any], str | None]]:
    for path in paths:
        with path.open(encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle, delimiter=delimiter):
                yield dict(row), None


def _iter_parquet(
    inputs: Mapping[str, str | list[str]] | str | list[str],
    *,
    base_dir: Path,
    audio_field: str,
) -> Iterator[tuple[dict[str, Any], str | None]]:
    try:
        from datasets import Audio, load_dataset
    except ImportError as exc:
        raise RuntimeError("parquet ingestion requires the 'datasets' dependency") from exc

    if isinstance(inputs, Mapping):
        data_files = {
            split: [str(path) for path in _expand_inputs(patterns, base_dir=base_dir)]
            for split, patterns in inputs.items()
        }
    else:
        data_files = {
            "train": [str(path) for path in _expand_inputs(inputs, base_dir=base_dir)]
        }
    dataset = load_dataset("parquet", data_files=data_files)
    for source_split, split_rows in dataset.items():
        if audio_field in split_rows.column_names:
            split_rows = split_rows.cast_column(audio_field, Audio(decode=False))
        for record in split_rows:
            yield dict(record), source_split


def iter_source_records(
    cfg: Mapping[str, Any],
    *,
    base_dir: Path,
) -> Iterator[tuple[dict[str, Any], str | None]]:
    """Yield source records plus an optional source split."""
    fmt = str(cfg.get("format") or "").lower()
    inputs = cfg.get("inputs")
    if not inputs:
        raise ValueError("ingest.inputs is required")
    if fmt == "parquet":
        field_map = cfg.get("field_map") or {}
        yield from _iter_parquet(
            inputs,
            base_dir=base_dir,
            audio_field=str(field_map.get("audio") or "audio"),
        )
        return

    if isinstance(inputs, Mapping):
        raise ValueError(f"mapping inputs are only supported for parquet, not {fmt!r}")
    paths = _expand_inputs(inputs, base_dir=base_dir)
    if fmt == "jsonl":
        yield from _iter_jsonl(paths)
    elif fmt == "csv":
        yield from _iter_delimited(paths, delimiter=",")
    elif fmt == "tsv":
        yield from _iter_delimited(paths, delimiter="\t")
    else:
        raise ValueError(f"unsupported ingest format {fmt!r}")


def _audio_path(
    value: Any,
    *,
    source_id: str,
    utt_id: str,
    audio_root: Path | None,
    audio_cache: Path,
) -> Path:
    source_path: str | None = None
    payload: bytes | None = None
    if isinstance(value, Mapping):
        path_value = value.get("path")
        source_path = str(path_value) if path_value else None
        bytes_value = value.get("bytes")
        if isinstance(bytes_value, bytes):
            payload = bytes_value
        elif isinstance(bytes_value, bytearray):
            payload = bytes(bytes_value)
    elif isinstance(value, str):
        source_path = value
    else:
        raise ValueError(f"{source_id}: unsupported audio value {type(value).__name__}")

    if payload is not None:
        suffix = Path(source_path or source_id).suffix.lower()
        if not suffix or len(suffix) > 8:
            suffix = ".audio"
        path = audio_cache / f"{utt_id}{suffix}"
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists() or path.stat().st_size != len(payload):
            path.write_bytes(payload)
        return path.resolve()

    if not source_path:
        raise ValueError(f"{source_id}: audio has neither bytes nor path")
    path = Path(source_path).expanduser()
    if not path.is_absolute():
        path = (audio_root or Path.cwd()) / path
    if not path.is_file():
        raise FileNotFoundError(f"{source_id}: audio file not found: {path}")
    return path.resolve()


def _source_split(record: Mapping[str, Any], source_split: str | None, field: str | None) -> str:
    raw = _value(record, field, source_split)
    return _SOURCE_SPLITS.get(str(raw).lower(), "train")


def ingest_records(
    records: Iterable[tuple[dict[str, Any], str | None]],
    cfg: Mapping[str, Any],
    *,
    base_dir: Path,
) -> tuple[list[Utterance], dict[str, Any]]:
    """Convert source records to materialized multi-corpus utterances."""
    corpus = str(cfg.get("corpus") or "").strip().lower()
    if not corpus:
        raise ValueError("ingest.corpus is required")
    registry_info = get_corpus(corpus)
    if registry_info is None:
        raise ValueError(f"corpus {corpus!r} is absent from the audited registry")
    channel = registry_info.default_channel
    validate_channel(channel)
    license_id = registry_info.license
    source_url = registry_info.source_url
    if cfg.get("license") and str(cfg["license"]) != license_id:
        raise ValueError(
            f"config license {cfg['license']!r} != registry {license_id!r}"
        )
    if cfg.get("source_url") and str(cfg["source_url"]).strip() != source_url:
        raise ValueError("config source_url does not match audited registry")

    field_map = dict(cfg.get("field_map") or {})
    required_map = ("audio", "text", "source_id")
    missing_map = [name for name in required_map if not field_map.get(name)]
    if missing_map:
        raise ValueError(f"ingest.field_map missing: {', '.join(missing_map)}")

    audio_root_value = cfg.get("audio_root")
    audio_root = (
        (base_dir / str(audio_root_value)).resolve()
        if audio_root_value and not Path(str(audio_root_value)).expanduser().is_absolute()
        else Path(str(audio_root_value)).expanduser().resolve()
        if audio_root_value
        else None
    )
    cache_value = cfg.get("audio_cache") or f"data/{corpus}/audio"
    audio_cache = Path(str(cache_value)).expanduser()
    if not audio_cache.is_absolute():
        audio_cache = (base_dir / audio_cache).resolve()

    min_dur = float(cfg.get("min_dur_s", 0.5))
    max_dur = float(cfg.get("max_dur_s", 30.0))
    extra_fields = [str(name) for name in cfg.get("extra_fields", [])]
    speaker_field = field_map.get("speaker_id")
    container_field = field_map.get("container_id")
    stressed_field = field_map.get("text_stressed")
    split_field = field_map.get("split")
    limit = int(cfg["limit"]) if cfg.get("limit") is not None else None

    counters = {
        "records_seen": 0,
        "kept": 0,
        "dropped_empty_text": 0,
        "dropped_duration": 0,
        "dropped_unreadable_audio": 0,
        "dropped_duplicate_source_id": 0,
    }
    rows: list[Utterance] = []
    seen_source_ids: set[str] = set()
    seen_utt_ids: set[str] = set()
    for record, source_split in records:
        counters["records_seen"] += 1
        if limit is not None and len(rows) >= limit:
            break
        source_id = str(_value(record, str(field_map["source_id"]), "") or "").strip()
        if not source_id:
            raise ValueError(f"record {counters['records_seen']}: empty source_id")
        if source_id in seen_source_ids:
            counters["dropped_duplicate_source_id"] += 1
            continue
        seen_source_ids.add(source_id)
        utt_id = stable_utt_id(corpus, source_id)
        if utt_id in seen_utt_ids:
            raise ValueError(f"utt_id collision for source_id {source_id!r}: {utt_id}")
        seen_utt_ids.add(utt_id)

        text_source = str(_value(record, str(field_map["text"]), "") or "").strip()
        if not text_source:
            counters["dropped_empty_text"] += 1
            continue
        text_raw = normalize_text(text_source)
        stressed_source = str(_value(record, stressed_field, text_source) or text_source)
        text_stressed = to_canonical_stress(stressed_source)

        try:
            wav = _audio_path(
                _value(record, str(field_map["audio"])),
                source_id=source_id,
                utt_id=utt_id,
                audio_root=audio_root,
                audio_cache=audio_cache,
            )
            info = sf.info(str(wav))
        except (OSError, RuntimeError, ValueError, FileNotFoundError):
            counters["dropped_unreadable_audio"] += 1
            continue
        n_24 = int(round(info.frames * SR / info.samplerate))
        dur_s = n_24 / SR
        if dur_s < min_dur or dur_s > max_dur:
            counters["dropped_duration"] += 1
            continue

        source_speaker = str(_value(record, speaker_field, "unknown") or "unknown").strip()
        speaker_id = f"{corpus}:{source_speaker}"
        source_container = str(
            _value(record, container_field, source_speaker or source_id)
            or source_speaker
            or source_id
        ).strip()
        container_id = f"{corpus}:{source_container}"
        extra = {name: _value(record, name) for name in extra_fields}
        extra["source_split"] = source_split
        row = Utterance(
            utt_id=utt_id,
            wav=str(wav),
            text_raw=text_raw,
            text_stressed=text_stressed,
            speaker=speaker_id,
            corpus=corpus,
            speaker_id=speaker_id,
            channel=channel,
            sample_rate=int(info.samplerate),
            license=license_id,
            source_url=source_url,
            source_id=source_id,
            container_id=container_id,
            n_samples_24k=n_24,
            n_frames=n_frames_from_samples(n_24),
            dur_s=dur_s,
            split=_source_split(record, source_split, split_field),
            extra=extra,
        )
        apply_utterance_defaults(row)
        populate_integrity_fields(row)
        rows.append(row.materialize_defaults())
        counters["kept"] += 1

    split_cfg = dict(cfg.get("split") or {})
    strategy = str(split_cfg.get("strategy") or "source").lower()
    ratios = {
        "train": float(split_cfg.get("train", 0.9)),
        "val": float(split_cfg.get("val", 0.05)),
        "test": float(split_cfg.get("test", 0.05)),
    }
    if strategy == "speaker":
        rows = speaker_split(rows, **ratios)
    elif strategy == "contiguous":
        rows = contiguous_split(rows, **ratios)
    elif strategy != "source":
        raise ValueError("ingest.split.strategy must be source, speaker, or contiguous")

    meta = {
        "schema_version": 1,
        "corpus": corpus,
        "format": cfg.get("format"),
        "n": len(rows),
        "hours": sum(row.dur_s for row in rows) / 3600.0,
        "speakers": len({row.effective_speaker_id() for row in rows}),
        "by_split": {
            split: {
                "n": sum(row.split == split for row in rows),
                "hours": sum(row.dur_s for row in rows if row.split == split) / 3600.0,
            }
            for split in ("train", "val", "test")
        },
        "filter_counters": counters,
        "utt_id_scheme": f"{corpus}__sha256(source_id)[:24]",
    }
    return rows, meta


@dataclass
class _DirectoryScan:
    n: int
    hours: float
    speaker_durations: dict[str, float]
    counters: dict[str, int]


def _iter_directory_audio(root: Path, pattern: str) -> Iterator[Path]:
    absolute_pattern = str(root / pattern)
    for value in glob.iglob(absolute_pattern, recursive=True):
        path = Path(value)
        if path.is_file():
            yield path.resolve()


def _layout_value(relative: Path, mode: str, *, fallback: str) -> str:
    if mode == "parent":
        return relative.parent.name or fallback
    if mode == "topdir":
        return relative.parts[0] if len(relative.parts) > 1 else fallback
    if mode == "stem":
        return relative.stem
    if mode == "relative_parent":
        value = relative.parent.as_posix()
        return value if value != "." else fallback
    raise ValueError(
        f"unsupported layout selector {mode!r}; expected parent, topdir, stem, "
        "or relative_parent"
    )


def _transcript_path(root: Path, relative: Path, template: str) -> Path:
    values = {
        "relative": relative.as_posix(),
        "relative_no_suffix": relative.with_suffix("").as_posix(),
        "stem": relative.stem,
        "name": relative.name,
        "parent": relative.parent.as_posix(),
    }
    rendered = Path(template.format(**values)).expanduser()
    return rendered if rendered.is_absolute() else (root / rendered)


def _directory_row(
    path: Path,
    *,
    root: Path,
    info: CorpusInfo,
    cfg: Mapping[str, Any],
    compute_hashes: bool,
) -> Utterance | None:
    relative = path.relative_to(root)
    transcript = _transcript_path(
        root,
        relative,
        str(cfg.get("transcript_template") or "{relative_no_suffix}.txt"),
    )
    if not transcript.is_file():
        return None
    text_source = transcript.read_text(encoding="utf-8").strip()
    if not text_source:
        return None
    try:
        audio_info = sf.info(str(path))
    except (OSError, RuntimeError):
        return None
    n_24 = int(round(audio_info.frames * SR / audio_info.samplerate))
    dur_s = n_24 / SR
    if dur_s < float(cfg.get("min_dur_s", 0.5)) or dur_s > float(
        cfg.get("max_dur_s", 30.0)
    ):
        return None

    source_id = relative.as_posix()
    speaker_source = _layout_value(
        relative,
        str(cfg.get("speaker_from") or "parent"),
        fallback="unknown",
    )
    container_source = _layout_value(
        relative,
        str(cfg.get("container_from") or "relative_parent"),
        fallback=source_id,
    )
    speaker_id = f"{info.id}:{speaker_source}"
    row = Utterance(
        utt_id=stable_utt_id(info.id, source_id),
        wav=str(path),
        text_raw=normalize_text(text_source),
        text_stressed=to_canonical_stress(text_source),
        speaker=speaker_id,
        corpus=info.id,
        speaker_id=speaker_id,
        sample_rate=int(audio_info.samplerate),
        source_id=source_id,
        container_id=f"{info.id}:{container_source}",
        n_samples_24k=n_24,
        n_frames=n_frames_from_samples(n_24),
        dur_s=dur_s,
    )
    apply_utterance_defaults(row)
    if compute_hashes:
        populate_integrity_fields(row)
    return row.materialize_defaults()


def _scan_directory(
    *,
    root: Path,
    info: CorpusInfo,
    cfg: Mapping[str, Any],
) -> _DirectoryScan:
    counters = {
        "audio_seen": 0,
        "kept": 0,
        "dropped_missing_text_or_invalid_audio": 0,
    }
    durations: dict[str, float] = {}
    total = 0.0
    for path in _iter_directory_audio(root, str(cfg.get("audio_glob") or "**/*.wav")):
        counters["audio_seen"] += 1
        row = _directory_row(
            path,
            root=root,
            info=info,
            cfg=cfg,
            compute_hashes=False,
        )
        if row is None:
            counters["dropped_missing_text_or_invalid_audio"] += 1
            continue
        counters["kept"] += 1
        total += row.dur_s
        speaker = row.effective_speaker_id()
        durations[speaker] = durations.get(speaker, 0.0) + row.dur_s
    return _DirectoryScan(
        n=counters["kept"],
        hours=total / 3600.0,
        speaker_durations=durations,
        counters=counters,
    )


def _speaker_assignment(
    durations: Mapping[str, float],
    *,
    train: float,
    val: float,
    test: float,
) -> dict[str, str]:
    placeholders = [
        Utterance(
            utt_id=f"speaker__{hashlib.sha256(speaker.encode()).hexdigest()[:16]}",
            wav="",
            text_raw="placeholder",
            text_stressed="placeholder",
            speaker=speaker,
            speaker_id=speaker,
            dur_s=duration,
        )
        for speaker, duration in durations.items()
    ]
    speaker_split(placeholders, train=train, val=val, test=test)
    return {row.effective_speaker_id(): row.split for row in placeholders}


def build_directory_manifest(config_path: str | Path) -> dict[str, Any]:
    """Two-pass, constant-per-file-memory directory ingestion."""
    path = expand(config_path)
    raw = load_yaml(path)
    cfg = raw.get("ingest")
    if not isinstance(cfg, dict) or str(cfg.get("format") or "") != "directory":
        raise ValueError(f"{path}: ingest.format must be 'directory'")
    corpus = str(cfg.get("corpus") or "").strip().lower()
    info = get_corpus(corpus)
    if info is None:
        raise ValueError(f"corpus {corpus!r} is absent from the audited registry")
    if info.training_allowed is not True and not bool(cfg.get("allow_unapproved")):
        raise ValueError(
            f"corpus {corpus!r} is not approved for training; legal review is required"
        )
    root_value = str(cfg.get("root") or "").strip()
    if not root_value:
        raise ValueError("ingest.root is required")
    root = Path(root_value).expanduser()
    if not root.is_absolute():
        root = (path.parent / root).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"corpus root not found: {root}")

    output_value = str(cfg.get("manifest") or "").strip()
    if not output_value:
        raise ValueError("ingest.manifest is required")
    output = Path(output_value).expanduser()
    if not output.is_absolute():
        output = (path.parent / output).resolve()

    scan = _scan_directory(root=root, info=info, cfg=cfg)
    split_cfg = dict(cfg.get("split") or {})
    strategy = str(split_cfg.get("strategy") or "speaker_holdout")
    train = float(split_cfg.get("train", 0.9))
    val = float(split_cfg.get("val", 0.05))
    test = float(split_cfg.get("test", 0.05))
    if abs(train + val + test - 1.0) >= 1e-6:
        raise ValueError("split ratios must sum to 1")
    if strategy == "speaker_holdout":
        assignment = _speaker_assignment(
            scan.speaker_durations,
            train=train,
            val=val,
            test=test,
        )
    elif strategy == "contiguous_ranges":
        assignment = {}
    else:
        raise ValueError(
            "ingest.split.strategy must be 'speaker_holdout' or 'contiguous_ranges'"
        )

    split_counts = {"train": 0, "val": 0, "test": 0}
    split_hours = {"train": 0.0, "val": 0.0, "test": 0.0}
    index = 0

    def rows() -> Iterator[Utterance]:
        nonlocal index
        n_train = int(scan.n * train)
        n_val = int(scan.n * val)
        for audio_path in _iter_directory_audio(
            root,
            str(cfg.get("audio_glob") or "**/*.wav"),
        ):
            row = _directory_row(
                audio_path,
                root=root,
                info=info,
                cfg=cfg,
                compute_hashes=True,
            )
            if row is None:
                continue
            if strategy == "speaker_holdout":
                row.split = assignment[row.effective_speaker_id()]
            elif index < n_train:
                row.split = "train"
            elif index < n_train + n_val:
                row.split = "val"
            else:
                row.split = "test"
            index += 1
            split_counts[row.split] += 1
            split_hours[row.split] += row.dur_s / 3600.0
            yield row

    n_written = write_manifest_iter(output, rows())
    meta = {
        "schema_version": 1,
        "corpus": corpus,
        "manifest": str(output),
        "n": n_written,
        "hours": scan.hours,
        "speakers": len(scan.speaker_durations),
        "split_strategy": strategy,
        "by_split": {
            split: {"n": split_counts[split], "hours": split_hours[split]}
            for split in ("train", "val", "test")
        },
        "filter_counters": scan.counters,
        "streaming": True,
        "utt_id_scheme": f"{corpus}__sha256(source_id)[:24]",
    }
    meta_path = output.with_suffix(".meta.json")
    meta_path.write_text(
        json.dumps(meta, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return meta


def build_manifest_from_config(config_path: str | Path) -> dict[str, Any]:
    path = expand(config_path)
    raw = load_yaml(path)
    cfg = raw.get("ingest")
    if not isinstance(cfg, dict):
        raise ValueError(f"{path}: missing top-level 'ingest' mapping")
    if str(cfg.get("format") or "") == "directory":
        return build_directory_manifest(path)
    records = iter_source_records(cfg, base_dir=path.parent)
    rows, meta = ingest_records(records, cfg, base_dir=path.parent)
    output_value = str(cfg.get("manifest") or "").strip()
    if not output_value:
        raise ValueError("ingest.manifest is required")
    output = Path(output_value).expanduser()
    if not output.is_absolute():
        output = (path.parent / output).resolve()
    write_manifest(output, rows)
    meta_path = output.with_suffix(".meta.json")
    meta_path.write_text(
        json.dumps(meta, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return {"manifest": str(output), "meta": str(meta_path), **meta}

"""JSONL manifest schema and validation."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterator

from minispeech.utils import expand


@dataclass
class Utterance:
    utt_id: str
    wav: str
    text_raw: str
    text_stressed: str
    speaker: str = "ruslan"
    n_samples_24k: int = 0
    n_frames: int = 0
    dur_s: float = 0.0
    split: str = "train"
    asr_cer: float | None = None
    stress_agreement: float | None = None
    input_ids: list[int] | None = None
    challenge: bool = False
    extra: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        d = asdict(self)
        extra = d.pop("extra", {})
        d = {k: v for k, v in d.items() if v is not None}
        d.update(extra)
        return d

    @classmethod
    def from_json(cls, obj: dict[str, Any]) -> "Utterance":
        known = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore
        kwargs = {k: v for k, v in obj.items() if k in known}
        extra = {k: v for k, v in obj.items() if k not in known}
        u = cls(**kwargs)
        u.extra = extra
        return u


def write_manifest(path: str | Path, rows: list[Utterance]) -> None:
    path = expand(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r.to_json(), ensure_ascii=False) + "\n")


def read_manifest(path: str | Path) -> list[Utterance]:
    path = expand(path)
    rows: list[Utterance] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(Utterance.from_json(json.loads(line)))
    return rows


def iter_manifest(path: str | Path) -> Iterator[Utterance]:
    path = expand(path)
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield Utterance.from_json(json.loads(line))


def filter_split(rows: list[Utterance], split: str) -> list[Utterance]:
    return [r for r in rows if r.split == split]


def contiguous_split(
    rows: list[Utterance],
    train: float = 0.9,
    val: float = 0.05,
    test: float = 0.05,
) -> list[Utterance]:
    """Assign split by contiguous index ranges (sorted by utt_id)."""
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

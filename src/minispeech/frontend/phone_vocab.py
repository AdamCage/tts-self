"""Phoneme / grapheme vocabulary for Track A mel baseline."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence


PAD = "<pad>"
UNK = "<unk>"
BOS = "<bos>"
EOS = "<eos>"
# Optional silence token for sil_fill gap packing (not used by expand_neighbor)
SIL = "sil"


@dataclass
class PhoneVocab:
    """Bidirectional phone ↔ id map. ``pad_id=0``, ``unk_id=1`` by default.

    ``unit`` is ``\"phone\"`` (espeak/G2P) or ``\"char\"`` (CTC char alphabet).
    Infer must use the same unit as training.
    """

    stoi: dict[str, int] = field(default_factory=dict)
    itos: list[str] = field(default_factory=list)
    unit: str = "phone"

    def __post_init__(self) -> None:
        if not self.itos:
            for tok in (PAD, UNK, BOS, EOS):
                self._add(tok)
        elif not self.stoi:
            self.stoi = {t: i for i, t in enumerate(self.itos)}

    def _add(self, tok: str) -> int:
        if tok in self.stoi:
            return self.stoi[tok]
        idx = len(self.itos)
        self.itos.append(tok)
        self.stoi[tok] = idx
        return idx

    @property
    def pad_id(self) -> int:
        return self.stoi[PAD]

    @property
    def unk_id(self) -> int:
        return self.stoi[UNK]

    def __len__(self) -> int:
        return len(self.itos)

    def encode(self, phones: Sequence[str], *, add_bos_eos: bool = False) -> list[int]:
        ids: list[int] = []
        if add_bos_eos:
            ids.append(self.stoi[BOS])
        for p in phones:
            ids.append(self.stoi.get(p, self.unk_id))
        if add_bos_eos:
            ids.append(self.stoi[EOS])
        return ids

    def decode(self, ids: Sequence[int]) -> list[str]:
        out: list[str] = []
        for i in ids:
            if 0 <= int(i) < len(self.itos):
                tok = self.itos[int(i)]
                if tok not in (PAD, BOS, EOS):
                    out.append(tok)
        return out

    def add_phones(self, phones: Iterable[str]) -> None:
        for p in phones:
            if p and p not in self.stoi:
                self._add(p)

    def to_dict(self) -> dict:
        return {"itos": list(self.itos), "unit": self.unit}

    @classmethod
    def from_dict(cls, d: dict) -> PhoneVocab:
        itos = list(d["itos"])
        unit = str(d.get("unit") or "phone")
        # Heuristic for older vocabs saved without unit
        if "unit" not in d:
            content = [t for t in itos if t not in (PAD, UNK, BOS, EOS)]
            if content and ("|" in content or all(len(t) == 1 for t in content[:16])):
                unit = "char"
        return cls(itos=itos, stoi={t: i for i, t in enumerate(itos)}, unit=unit)

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: str | Path) -> PhoneVocab:
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    @classmethod
    def build(cls, phone_lists: Iterable[Sequence[str]]) -> PhoneVocab:
        v = cls()
        for phones in phone_lists:
            v.add_phones(phones)
        return v

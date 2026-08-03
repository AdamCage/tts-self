"""Deterministic global speaker-ID registry for discrete Speech LM (D1b).

Contract
--------
- ``0`` is reserved for explicit unknown (infer-only unless ``allow_unknown``).
- Known speakers are sorted by stable string key and mapped to ``1..N``.
- Train / eval are fail-closed on unknown speakers.
- Mapping does not depend on Python ``hash()``, DataLoader order, or DDP rank.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Literal, Mapping, Sequence

from minispeech.utils import expand

UNKNOWN_SPEAKER_ID = 0
REGISTRY_VERSION = "2.0.0"
SpeakerMode = Literal["train", "eval", "infer"]


def _canonical_json(obj: Any) -> bytes:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _items_sha256(items: Sequence[str]) -> str:
    payload = _canonical_json(list(items))
    return _sha256_bytes(payload)


def normalize_speaker_key(speaker: str | int | None) -> str:
    """Stable string key for a speaker label (not the dense embedding id)."""
    if speaker is None:
        return "unknown"
    if isinstance(speaker, int):
        raise TypeError(
            "normalize_speaker_key expects a string label, not an embedding id; "
            f"got int {speaker}"
        )
    key = str(speaker).strip()
    return key if key else "unknown"


def build_dense_mapping(speakers: Iterable[str]) -> dict[str, int]:
    """Sort unique speaker strings and assign ``1..N`` (``0`` reserved)."""
    unique: set[str] = set()
    for s in speakers:
        key = normalize_speaker_key(s)
        if key == "unknown":
            continue
        unique.add(key)
    ordered = sorted(unique)
    return {name: i + 1 for i, name in enumerate(ordered)}


@dataclass(frozen=True)
class SpeakerRegistry:
    """Dense speaker → embedding-id map with content hashes."""

    version: str
    source_manifest_sha256: str
    train_set_sha256: str
    n_speakers: int
    mapping: dict[str, int]
    sha256: str
    source_manifest: str | None = None
    train_set: str | None = None
    notes: str | None = None

    @property
    def max_speakers(self) -> int:
        """Embedding table size: ``n_speakers + 1`` (includes reserved unknown)."""
        return int(self.n_speakers) + 1

    def to_payload(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "source_manifest": self.source_manifest,
            "source_manifest_sha256": self.source_manifest_sha256,
            "train_set": self.train_set,
            "train_set_sha256": self.train_set_sha256,
            "n_speakers": self.n_speakers,
            "max_speakers": self.max_speakers,
            "unknown_id": UNKNOWN_SPEAKER_ID,
            "mapping": dict(sorted(self.mapping.items(), key=lambda kv: kv[1])),
            "sha256": self.sha256,
            "notes": self.notes,
        }

    def save(self, path: str | Path) -> Path:
        path = expand(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.to_payload(), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        return path

    @classmethod
    def from_mapping(
        cls,
        mapping: Mapping[str, int],
        *,
        source_manifest_sha256: str = "",
        train_set_sha256: str = "",
        source_manifest: str | None = None,
        train_set: str | None = None,
        version: str = REGISTRY_VERSION,
        notes: str | None = None,
    ) -> SpeakerRegistry:
        cleaned: dict[str, int] = {}
        for name, sid in mapping.items():
            key = normalize_speaker_key(name)
            if key == "unknown":
                raise ValueError("mapping must not include reserved key 'unknown'")
            sid_i = int(sid)
            if sid_i < 1:
                raise ValueError(f"speaker id for {key!r} must be >= 1, got {sid_i}")
            if key in cleaned and cleaned[key] != sid_i:
                raise ValueError(f"conflicting ids for speaker {key!r}")
            cleaned[key] = sid_i

        ids = list(cleaned.values())
        if len(ids) != len(set(ids)):
            raise ValueError("speaker mapping has colliding integer ids")
        expected = set(range(1, len(cleaned) + 1))
        if set(ids) != expected:
            raise ValueError(
                "speaker mapping must be a dense permutation of 1..N; "
                f"got ids={sorted(ids)[:16]}{'...' if len(ids) > 16 else ''}"
            )

        n_speakers = len(cleaned)
        content = {
            "version": version,
            "source_manifest_sha256": source_manifest_sha256,
            "train_set_sha256": train_set_sha256,
            "n_speakers": n_speakers,
            "mapping": dict(sorted(cleaned.items(), key=lambda kv: kv[1])),
            "unknown_id": UNKNOWN_SPEAKER_ID,
        }
        digest = _sha256_bytes(_canonical_json(content))
        return cls(
            version=version,
            source_manifest_sha256=source_manifest_sha256,
            train_set_sha256=train_set_sha256,
            n_speakers=n_speakers,
            mapping=cleaned,
            sha256=digest,
            source_manifest=source_manifest,
            train_set=train_set,
            notes=notes,
        )

    @classmethod
    def from_speakers(
        cls,
        speakers: Iterable[str],
        *,
        source_manifest_sha256: str = "",
        train_set_sha256: str = "",
        source_manifest: str | None = None,
        train_set: str | None = None,
        version: str = REGISTRY_VERSION,
        notes: str | None = None,
    ) -> SpeakerRegistry:
        return cls.from_mapping(
            build_dense_mapping(speakers),
            source_manifest_sha256=source_manifest_sha256,
            train_set_sha256=train_set_sha256,
            source_manifest=source_manifest,
            train_set=train_set,
            version=version,
            notes=notes,
        )

    @classmethod
    def load(cls, path: str | Path) -> SpeakerRegistry:
        path = expand(path)
        obj = json.loads(path.read_text(encoding="utf-8"))
        mapping = obj.get("mapping") or {}
        reg = cls.from_mapping(
            mapping,
            source_manifest_sha256=str(obj.get("source_manifest_sha256") or ""),
            train_set_sha256=str(obj.get("train_set_sha256") or ""),
            source_manifest=obj.get("source_manifest"),
            train_set=obj.get("train_set"),
            version=str(obj.get("version") or REGISTRY_VERSION),
            notes=obj.get("notes"),
        )
        # Prefer stored digest when present and content matches rebuild.
        stored = obj.get("sha256")
        if stored and str(stored) != reg.sha256:
            raise ValueError(
                f"speaker map sha256 mismatch at {path}: "
                f"stored={stored} recomputed={reg.sha256}"
            )
        return reg

    def to_id(
        self,
        speaker: str | int | None,
        *,
        mode: SpeakerMode = "train",
        allow_unknown: bool | None = None,
    ) -> int:
        """Map a speaker label (or already-int id) to an embedding index.

        ``allow_unknown`` defaults to ``True`` only for ``mode='infer'``.
        """
        if allow_unknown is None:
            allow_unknown = mode == "infer"

        if isinstance(speaker, int):
            sid = int(speaker)
            if sid < 0 or sid > self.n_speakers:
                raise ValueError(
                    f"speaker id {sid} out of range for registry "
                    f"(valid 0..{self.n_speakers}; max_speakers={self.max_speakers})"
                )
            if sid == UNKNOWN_SPEAKER_ID and not allow_unknown:
                raise ValueError(
                    f"unknown speaker id 0 is not allowed in mode={mode!r}"
                )
            return sid

        key = normalize_speaker_key(speaker)
        if key == "unknown":
            if allow_unknown:
                return UNKNOWN_SPEAKER_ID
            raise KeyError(f"unknown speaker not allowed in mode={mode!r}")
        if key in self.mapping:
            return int(self.mapping[key])
        if allow_unknown:
            return UNKNOWN_SPEAKER_ID
        raise KeyError(
            f"speaker {key!r} not in registry (mode={mode!r}, "
            f"n_speakers={self.n_speakers})"
        )

    def require_int_id(self, speaker_id: Any, *, mode: SpeakerMode = "train") -> int:
        """Fail-closed helper for collate / protocol: ints only, in-range."""
        if isinstance(speaker_id, str):
            raise TypeError(
                "speaker_id must be an int from SpeakerRegistry; "
                f"got string {speaker_id!r}. Resolve labels before collate."
            )
        if not isinstance(speaker_id, int) and not (
            hasattr(speaker_id, "item") and not isinstance(speaker_id, (bytes, str))
        ):
            # torch scalar / numpy integer
            try:
                speaker_id = int(speaker_id)
            except Exception as exc:  # pragma: no cover
                raise TypeError(
                    f"speaker_id must be an int, got {type(speaker_id)!r}"
                ) from exc
        else:
            speaker_id = int(speaker_id)
        return self.to_id(speaker_id, mode=mode)


def assert_integer_speaker_id(speaker_id: Any, *, context: str = "speaker_id") -> int:
    """Reject string / hash fallbacks; return a plain Python int."""
    if isinstance(speaker_id, str):
        raise TypeError(
            f"{context} must be an int from SpeakerRegistry; got string {speaker_id!r}"
        )
    if isinstance(speaker_id, bool):
        raise TypeError(f"{context} must be an int, not bool")
    try:
        return int(speaker_id)
    except Exception as exc:
        raise TypeError(
            f"{context} must be an int from SpeakerRegistry; got {type(speaker_id)!r}"
        ) from exc


def collect_speakers_from_trackn(
    *,
    train_set_path: str | Path,
    manifest_path: str | Path,
) -> tuple[list[str], str, str, list[str]]:
    """Return (sorted unique speakers, manifest_sha256, train_set_sha256, utt_ids)."""
    from minispeech.data.manifest import iter_manifest

    train_set_path = expand(train_set_path)
    manifest_path = expand(manifest_path)
    if not train_set_path.is_file():
        raise FileNotFoundError(train_set_path)
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)

    train_obj = json.loads(train_set_path.read_text(encoding="utf-8"))
    utt_ids = list(train_obj["utt_ids"])
    want = set(utt_ids)
    train_set_sha = str(train_obj.get("items_sha256") or _items_sha256(utt_ids))
    manifest_sha = _sha256_file(manifest_path)

    speakers: set[str] = set()
    seen = 0
    for row in iter_manifest(manifest_path):
        if row.utt_id not in want:
            continue
        seen += 1
        speakers.add(normalize_speaker_key(row.effective_speaker_id()))
    if seen != len(want):
        raise RuntimeError(
            f"trackn speaker collect: expected {len(want)} utts, found {seen} in manifest"
        )
    return sorted(s for s in speakers if s != "unknown"), manifest_sha, train_set_sha, utt_ids


def build_trackn_v2_registry(
    *,
    train_set_path: str | Path = "configs/eval_sets/trackn_train.json",
    manifest_path: str | Path | None = None,
    out_path: str | Path = "configs/speaker_maps/trackn_v2.json",
) -> SpeakerRegistry:
    """Build canonical Track N v2 speaker map from frozen train inventory."""
    train_set_path = expand(train_set_path)
    train_obj = json.loads(train_set_path.read_text(encoding="utf-8"))
    if manifest_path is None:
        manifest_path = train_obj.get("source", {}).get("manifest")
    if not manifest_path:
        raise ValueError("manifest_path required (not found in train set source)")
    speakers, man_sha, train_sha, _ = collect_speakers_from_trackn(
        train_set_path=train_set_path, manifest_path=manifest_path
    )
    expected_n = int(train_obj.get("source", {}).get("speakers") or 0)
    if expected_n and len(speakers) != expected_n:
        raise RuntimeError(
            f"speaker count {len(speakers)} != trackn_train source.speakers={expected_n}"
        )
    reg = SpeakerRegistry.from_speakers(
        speakers,
        source_manifest_sha256=man_sha,
        train_set_sha256=train_sha,
        source_manifest=str(manifest_path),
        train_set=str(train_set_path),
        notes=(
            "Canonical Track N v2 map: 0=unknown reserved; known speakers 1..N "
            "from frozen trackn_train, sorted by stable speaker string."
        ),
    )
    reg.save(out_path)
    return reg


def write_scaffold_map(
    out_path: str | Path = "configs/speaker_maps/trackn_v2.json",
    *,
    expected_n_speakers: int = 1088,
) -> Path:
    """Write a placeholder map when the multicorpus manifest is unavailable."""
    out_path = expand(out_path)
    payload = {
        "version": REGISTRY_VERSION,
        "source_manifest": None,
        "source_manifest_sha256": "",
        "train_set": "configs/eval_sets/trackn_train.json",
        "train_set_sha256": "",
        "n_speakers": 0,
        "max_speakers": 1,
        "unknown_id": UNKNOWN_SPEAKER_ID,
        "mapping": {},
        "sha256": "",
        "scaffold": True,
        "expected_n_speakers": expected_n_speakers,
        "notes": (
            "SCAFFOLD ONLY — rebuild in WSL with: "
            "python -m minispeech.data.speaker_registry build-trackn-v2 "
            "--train-set configs/eval_sets/trackn_train.json "
            "--out configs/speaker_maps/trackn_v2.json"
        ),
    }
    # Fill sha256 of scaffold content (empty mapping).
    content = {
        "version": REGISTRY_VERSION,
        "source_manifest_sha256": "",
        "train_set_sha256": "",
        "n_speakers": 0,
        "mapping": {},
        "unknown_id": UNKNOWN_SPEAKER_ID,
    }
    payload["sha256"] = _sha256_bytes(_canonical_json(content))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return out_path


def _cli(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build-trackn-v2", help="Build configs/speaker_maps/trackn_v2.json")
    b.add_argument("--train-set", default="configs/eval_sets/trackn_train.json")
    b.add_argument("--manifest", default=None)
    b.add_argument("--out", default="configs/speaker_maps/trackn_v2.json")

    s = sub.add_parser("scaffold", help="Write empty scaffold map")
    s.add_argument("--out", default="configs/speaker_maps/trackn_v2.json")
    s.add_argument("--expected-n-speakers", type=int, default=1088)

    v = sub.add_parser("validate", help="Load and print registry summary")
    v.add_argument("path", default="configs/speaker_maps/trackn_v2.json", nargs="?")

    args = ap.parse_args(argv)
    if args.cmd == "build-trackn-v2":
        reg = build_trackn_v2_registry(
            train_set_path=args.train_set,
            manifest_path=args.manifest,
            out_path=args.out,
        )
        print(
            json.dumps(
                {
                    "out": str(expand(args.out)),
                    "n_speakers": reg.n_speakers,
                    "max_speakers": reg.max_speakers,
                    "sha256": reg.sha256,
                    "source_manifest_sha256": reg.source_manifest_sha256,
                    "train_set_sha256": reg.train_set_sha256,
                },
                indent=2,
            )
        )
        return 0
    if args.cmd == "scaffold":
        path = write_scaffold_map(args.out, expected_n_speakers=args.expected_n_speakers)
        print(f"wrote scaffold {path}")
        return 0
    if args.cmd == "validate":
        reg = SpeakerRegistry.load(args.path)
        print(
            json.dumps(
                {
                    "path": str(expand(args.path)),
                    "n_speakers": reg.n_speakers,
                    "max_speakers": reg.max_speakers,
                    "sha256": reg.sha256,
                },
                indent=2,
            )
        )
        return 0
    raise SystemExit(f"unknown cmd {args.cmd}")


if __name__ == "__main__":
    raise SystemExit(_cli())

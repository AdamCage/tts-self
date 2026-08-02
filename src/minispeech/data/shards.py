"""Sharded safetensors caches for latents and text hidden states."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file, save_file

from minispeech.utils import expand


class ShardWriter:
    """Append-friendly shard writer with resume from existing index."""

    def __init__(
        self,
        out_dir: str | Path,
        prefix: str,
        shard_size: int = 2000,
        *,
        resume: bool = True,
    ):
        self.out_dir = expand(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.prefix = prefix
        self.shard_size = shard_size
        self.index: dict[str, dict[str, Any]] = {}
        self._buf: dict[str, torch.Tensor] = {}
        self._shard_id = 0
        self._idx_path = self.out_dir / f"{self.prefix}_index.json"
        if resume and self._idx_path.exists():
            self.index = json.loads(self._idx_path.read_text(encoding="utf-8"))
            max_id = -1
            for info in self.index.values():
                name = info.get("shard", "")
                # latents_00012.safetensors
                try:
                    stem = Path(name).stem
                    sid = int(stem.rsplit("_", 1)[-1])
                    max_id = max(max_id, sid)
                except Exception:
                    continue
            self._shard_id = max_id + 1 if max_id >= 0 else 0

    def has(self, utt_id: str) -> bool:
        return utt_id in self.index

    def add(self, utt_id: str, tensor: torch.Tensor, meta: dict[str, Any] | None = None) -> None:
        if utt_id in self.index:
            return  # skip existing
        key = f"{utt_id}"
        self._buf[key] = tensor.contiguous().cpu()
        self.index[utt_id] = {
            "shard": self._shard_name(),
            "key": key,
            "shape": list(tensor.shape),
            **(meta or {}),
        }
        if len(self._buf) >= self.shard_size:
            self.flush()

    def _shard_name(self) -> str:
        return f"{self.prefix}_{self._shard_id:05d}.safetensors"

    def _write_index(self) -> None:
        self._idx_path.write_text(json.dumps(self.index, indent=2), encoding="utf-8")

    def flush(self) -> None:
        if not self._buf:
            return
        path = self.out_dir / self._shard_name()
        save_file(self._buf, str(path))
        self._buf.clear()
        self._shard_id += 1
        self._write_index()

    def close(self) -> Path:
        self.flush()
        self._write_index()
        return self._idx_path


class ShardReader:
    """Read sharded safetensors caches.

    Parameters
    ----------
    lazy:
        If True (default), fetch a single tensor via ``safe_open`` without
        materializing the whole shard into a Python dict (lower peak RAM).
        An LRU of recently opened shard handles is still kept. Set False to
        load entire shards with ``load_file`` (previous behaviour).
    """

    def __init__(
        self,
        out_dir: str | Path,
        prefix: str,
        lru_shards: int = 3,
        *,
        lazy: bool = True,
    ):
        self.out_dir = expand(out_dir)
        self.prefix = prefix
        idx_path = self.out_dir / f"{prefix}_index.json"
        self.index = json.loads(idx_path.read_text(encoding="utf-8"))
        self._cache: dict[str, dict[str, torch.Tensor]] = {}
        self._lru: list[str] = []
        self.lru_shards = lru_shards
        self.lazy = lazy

    def _touch(self, shard: str) -> None:
        if shard in self._lru:
            self._lru.remove(shard)
        self._lru.append(shard)
        while len(self._lru) > self.lru_shards:
            old = self._lru.pop(0)
            self._cache.pop(old, None)

    def get(self, utt_id: str) -> torch.Tensor:
        info = self.index[utt_id]
        shard = info["shard"]
        key = info["key"]
        path = str(self.out_dir / shard)
        if self.lazy:
            try:
                from safetensors import safe_open

                with safe_open(path, framework="pt") as f:
                    return f.get_tensor(key).clone()
            except ImportError:
                self.lazy = False

        if shard not in self._cache:
            self._cache[shard] = load_file(path)
        self._touch(shard)
        return self._cache[shard][key]

    def preload_all(self) -> dict[str, torch.Tensor]:
        return {utt_id: self.get(utt_id) for utt_id in self.index}

    def __contains__(self, utt_id: str) -> bool:
        return utt_id in self.index


class MultiRootShardReader:
    """Route ``get(utt_id)`` across per-corpus shard roots.

    Built from a ``utt_id → corpus`` map (from the multicorpus manifest) and a
    ``corpus → directory`` map. Indexes are loaded once; tensors stay lazy via
    underlying :class:`ShardReader` instances (``lazy=True`` by default).
    """

    def __init__(
        self,
        corpus_dirs: dict[str, str | Path],
        prefix: str,
        utt_to_corpus: dict[str, str],
        *,
        lru_shards: int = 3,
        lazy: bool = True,
    ):
        self.prefix = prefix
        self.utt_to_corpus = dict(utt_to_corpus)
        self._readers: dict[str, ShardReader] = {}
        for corpus, directory in corpus_dirs.items():
            path = expand(directory)
            idx = path / f"{prefix}_index.json"
            if not idx.exists():
                # Allow constructing the dataset before a corpus cache exists;
                # access to missing corpora raises on get().
                continue
            self._readers[corpus] = ShardReader(
                path, prefix, lru_shards=lru_shards, lazy=lazy
            )

    def __contains__(self, utt_id: str) -> bool:
        corpus = self.utt_to_corpus.get(utt_id)
        if corpus is None:
            return False
        reader = self._readers.get(corpus)
        return reader is not None and utt_id in reader.index

    def get(self, utt_id: str) -> torch.Tensor:
        corpus = self.utt_to_corpus.get(utt_id)
        if corpus is None:
            raise KeyError(f"utt_id {utt_id!r} missing from utt_to_corpus map")
        reader = self._readers.get(corpus)
        if reader is None:
            raise KeyError(
                f"no {self.prefix!r} shard root for corpus {corpus!r} (utt_id={utt_id!r})"
            )
        return reader.get(utt_id)

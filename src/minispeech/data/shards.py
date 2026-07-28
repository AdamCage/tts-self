"""Sharded safetensors caches for latents and text hidden states."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file, save_file

from minispeech.utils import expand


class ShardWriter:
    def __init__(self, out_dir: str | Path, prefix: str, shard_size: int = 2000):
        self.out_dir = expand(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.prefix = prefix
        self.shard_size = shard_size
        self.index: dict[str, dict[str, Any]] = {}
        self._buf: dict[str, torch.Tensor] = {}
        self._shard_id = 0

    def add(self, utt_id: str, tensor: torch.Tensor, meta: dict[str, Any] | None = None) -> None:
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

    def flush(self) -> None:
        if not self._buf:
            return
        path = self.out_dir / self._shard_name()
        save_file(self._buf, str(path))
        self._buf.clear()
        self._shard_id += 1

    def close(self) -> Path:
        self.flush()
        idx_path = self.out_dir / f"{self.prefix}_index.json"
        idx_path.write_text(json.dumps(self.index, indent=2), encoding="utf-8")
        return idx_path


class ShardReader:
    def __init__(self, out_dir: str | Path, prefix: str):
        self.out_dir = expand(out_dir)
        self.prefix = prefix
        idx_path = self.out_dir / f"{prefix}_index.json"
        self.index = json.loads(idx_path.read_text(encoding="utf-8"))
        self._cache: dict[str, dict[str, torch.Tensor]] = {}

    def get(self, utt_id: str) -> torch.Tensor:
        info = self.index[utt_id]
        shard = info["shard"]
        if shard not in self._cache:
            self._cache[shard] = load_file(str(self.out_dir / shard))
        return self._cache[shard][info["key"]]

    def preload_all(self) -> dict[str, torch.Tensor]:
        return {utt_id: self.get(utt_id) for utt_id in self.index}

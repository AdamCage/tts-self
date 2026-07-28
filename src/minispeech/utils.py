"""Shared utilities: paths, config, run manifests, seeding."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import random
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import yaml


def expand(path: str | Path) -> Path:
    return Path(os.path.expanduser(str(path))).resolve()


def load_yaml(path: str | Path) -> dict[str, Any]:
    with open(expand(path), encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def deep_merge(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    out = dict(a)
    for k, v in b.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def git_sha(cwd: Path | None = None) -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(cwd or Path.cwd()),
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return out.strip()
    except Exception:
        return "unknown"


def config_hash(cfg: dict[str, Any]) -> str:
    blob = json.dumps(cfg, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:16]


def write_run_manifest(
    run_dir: str | Path,
    cfg: dict[str, Any],
    *,
    extra: dict[str, Any] | None = None,
) -> Path:
    run_dir = expand(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    info: dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "git_sha": git_sha(),
        "config_hash": config_hash(cfg),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "hostname": platform.node(),
    }
    try:
        import torch

        info["torch"] = torch.__version__
        info["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
            info["vram_gb"] = round(torch.cuda.get_device_properties(0).total_memory / 1e9, 2)
    except ImportError:
        info["torch"] = None
    try:
        import transformers

        info["transformers"] = transformers.__version__
    except ImportError:
        info["transformers"] = None
    if extra:
        info.update(extra)
    path = run_dir / "run_manifest.json"
    path.write_text(json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8")
    with open(run_dir / "config_snapshot.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)
    return path


def auto_batch_size(
    probe_fn,
    candidates: list[int],
    target_vram_gb: float = 14.5,
) -> int:
    """Pick largest batch that fits in target VRAM. `probe_fn(bs)` runs a few steps."""
    import torch

    if not torch.cuda.is_available():
        return candidates[0]
    chosen = candidates[0]
    for bs in sorted(candidates):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        try:
            probe_fn(bs)
            peak = torch.cuda.max_memory_allocated() / 1e9
            if peak <= target_vram_gb:
                chosen = bs
            else:
                break
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                torch.cuda.empty_cache()
                break
            raise
    return chosen

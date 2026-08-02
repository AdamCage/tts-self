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


def _find_git_root(start: Path | None = None) -> Path | None:
    """Walk parents for a ``.git`` directory (file or dir)."""
    cur = (start or Path.cwd()).resolve()
    for p in [cur, *cur.parents]:
        if (p / ".git").exists():
            return p
    return None


def git_sha(cwd: Path | None = None) -> str:
    """Return HEAD SHA, searching upward for the repo root.

    Previously used only ``Path.cwd()``, which yielded ``\"unknown\"`` when
    training was launched from a non-repo working directory.
    """
    root = _find_git_root(cwd)
    if root is None:
        return "unknown"
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(root),
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return out.strip()
    except Exception:
        return "unknown"


def git_is_dirty(cwd: Path | None = None) -> bool:
    """Return True if the git worktree has tracked or untracked changes.

    Ignored paths (``runs/``, ``out/``, datasets, caches) do not count.
    Missing git / non-repo → treated as dirty (unknown provenance).
    """
    root = _find_git_root(cwd)
    if root is None:
        return True
    try:
        out = subprocess.check_output(
            ["git", "status", "--porcelain", "--untracked-files=normal"],
            cwd=str(root),
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return bool(out.strip())
    except Exception:
        return True


def allow_dirty_tree() -> bool:
    """Escape hatch: ``MINISPEECH_ALLOW_DIRTY=1`` permits a dirty-tree run."""
    return os.environ.get("MINISPEECH_ALLOW_DIRTY", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def enforce_clean_git_tree(*, cwd: Path | None = None) -> bool:
    """Refuse dirty-tree training unless ``MINISPEECH_ALLOW_DIRTY`` is set.

    Returns ``git_is_dirty()``. Raises ``RuntimeError`` when dirty and the
    escape hatch is unset. See ``docs/reproducibility.md``.
    """
    dirty = git_is_dirty(cwd)
    if not dirty:
        return False
    if allow_dirty_tree():
        print(
            "WARNING: git worktree is dirty; continuing because "
            "MINISPEECH_ALLOW_DIRTY is set. run_manifest will have "
            "git_dirty=true.",
            flush=True,
        )
        return True
    raise RuntimeError(
        "Refusing to start: git worktree is dirty (git status --porcelain "
        "non-empty). Commit or stash, sync WSL via git pull (never "
        "copy-instead-of-pull), then retry. Emergency override: "
        "MINISPEECH_ALLOW_DIRTY=1 (manifest will record git_dirty=true). "
        "See docs/reproducibility.md."
    )


def config_hash(cfg: dict[str, Any]) -> str:
    blob = json.dumps(cfg, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:16]


def write_run_manifest(
    run_dir: str | Path,
    cfg: dict[str, Any],
    *,
    extra: dict[str, Any] | None = None,
    enforce_clean: bool = True,
) -> Path:
    """Write ``run_manifest.json`` (+ config snapshot).

    Always records ``git_sha`` and ``git_dirty``. By default refuses to write
    when the tree is dirty unless ``MINISPEECH_ALLOW_DIRTY=1``. Pass
    ``enforce_clean=False`` only for unit tests / non-training callers.
    """
    dirty = (
        enforce_clean_git_tree()
        if enforce_clean
        else git_is_dirty()
    )
    run_dir = expand(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    info: dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "git_sha": git_sha(),
        "git_dirty": bool(dirty),
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

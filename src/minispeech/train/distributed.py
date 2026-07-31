"""Distributed training helpers for Track A / Track N trainers.

Single-process remains the default. DDP activates when the process is launched
via ``torchrun`` / ``torch.distributed`` (``RANK`` + ``WORLD_SIZE`` in the
environment), including ``WORLD_SIZE=1`` for a smoke init on one GPU.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP


@dataclass(frozen=True)
class DistInfo:
    rank: int = 0
    local_rank: int = 0
    world_size: int = 1
    distributed: bool = False
    backend: str | None = None


_INFO: DistInfo | None = None


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    return int(raw)


def init_distributed(*, backend: str | None = None) -> DistInfo:
    """Initialize the process group when launched under torchrun.

    No-op (returns single-process :class:`DistInfo`) when ``RANK`` /
    ``WORLD_SIZE`` are absent — the default interactive / script path.
    """
    global _INFO
    if _INFO is not None:
        return _INFO

    launched = "RANK" in os.environ and "WORLD_SIZE" in os.environ
    world_size = _env_int("WORLD_SIZE", 1)
    if not launched:
        _INFO = DistInfo()
        return _INFO

    rank = _env_int("RANK", 0)
    local_rank = _env_int("LOCAL_RANK", 0)
    if backend is None:
        backend = "nccl" if torch.cuda.is_available() else "gloo"

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)

    if not dist.is_initialized():
        dist.init_process_group(backend=backend, rank=rank, world_size=world_size)

    _INFO = DistInfo(
        rank=rank,
        local_rank=local_rank,
        world_size=world_size,
        distributed=True,
        backend=backend,
    )
    return _INFO


def get_dist_info() -> DistInfo:
    return _INFO if _INFO is not None else DistInfo()


def is_distributed() -> bool:
    return get_dist_info().distributed


def get_rank() -> int:
    return get_dist_info().rank


def get_world_size() -> int:
    return get_dist_info().world_size


def get_local_rank() -> int:
    return get_dist_info().local_rank


def is_main_process() -> bool:
    return get_rank() == 0


def barrier() -> None:
    if is_distributed() and dist.is_initialized():
        dist.barrier()


def cleanup_distributed() -> None:
    """Destroy the process group (safe to call in single-process mode)."""
    global _INFO
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()
    _INFO = None


def unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    return model.module if isinstance(model, DDP) else model


def wrap_ddp(
    model: torch.nn.Module,
    *,
    local_rank: int | None = None,
    find_unused_parameters: bool = False,
) -> torch.nn.Module:
    """Wrap ``model`` in DDP when distributed; otherwise return it unchanged."""
    info = get_dist_info()
    if not info.distributed:
        return model
    lr = info.local_rank if local_rank is None else local_rank
    device_ids = [lr] if torch.cuda.is_available() else None
    output_device = lr if torch.cuda.is_available() else None
    return DDP(
        model,
        device_ids=device_ids,
        output_device=output_device,
        find_unused_parameters=find_unused_parameters,
    )


def device_for_rank(fallback: torch.device | None = None) -> torch.device:
    """CUDA device for the local rank, or ``fallback`` / CPU."""
    info = get_dist_info()
    if info.distributed and torch.cuda.is_available():
        return torch.device(f"cuda:{info.local_rank}")
    if fallback is not None:
        return fallback
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def broadcast_object(obj: Any, src: int = 0) -> Any:
    """Broadcast a picklable object from ``src`` (no-op if not distributed)."""
    if not is_distributed():
        return obj
    container: list[Any] = [obj]
    dist.broadcast_object_list(container, src=src)
    return container[0]

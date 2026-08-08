#!/usr/bin/env python3
"""Real torchrun WORLD_SIZE=1 DDP init smoke for D1c Wave 3.

Launch only via::

    PYTHONPATH=src torchrun --standalone --nproc_per_node=1 \\
      .cursor/results/d1c_semantic_eos_repair_b81ecc3e/logs/wave3_torchrun_ddp_smoke.py
"""

from __future__ import annotations

import os
import sys

from minispeech.train.distributed import cleanup_distributed, get_dist_info, init_distributed


def main() -> int:
    launched = "RANK" in os.environ and "WORLD_SIZE" in os.environ
    if not launched:
        print("TORCHRUN_SMOKE_FAIL: not launched under torchrun (RANK/WORLD_SIZE missing)")
        return 2
    info = init_distributed()
    try:
        if not info.distributed:
            print(f"TORCHRUN_SMOKE_FAIL: distributed=False info={info}")
            return 3
        if int(info.world_size) != 1:
            print(f"TORCHRUN_SMOKE_FAIL: world_size={info.world_size} expected 1")
            return 4
        if int(info.rank) != 0:
            print(f"TORCHRUN_SMOKE_FAIL: rank={info.rank} expected 0")
            return 5
        print(
            "TORCHRUN_SMOKE_OK "
            f"backend={info.backend} world_size={info.world_size} rank={info.rank} "
            f"local_rank={info.local_rank}"
        )
        return 0
    finally:
        cleanup_distributed()
        _ = get_dist_info()


if __name__ == "__main__":
    sys.exit(main())

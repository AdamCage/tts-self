#!/usr/bin/env python3
"""Wave-0: verify historical D1b hashes unchanged."""
from __future__ import annotations

import json
import pathlib
import sys

EXPECTED = {
    "contract_sha256": "d04295d3ac9864afe5002a8470baead8f5f57a3c8f8d38831fe134aa7feada92",
    "dev_items": "c886ba31f3305546d9b4938e1cd0b7a2fa4fce738d7da342e9da2a30a28b172c",
    "c1_items": "232df2786859820298271a4845006bbcd4f9b78cfd9fb37872da90d0eee7265c",
    "c2_items": "63d7e351602e62d26196d944cd1f7a9e928a241e70d968f6eb58c0d1b7a746b9",
}


def main() -> int:
    root = pathlib.Path(".")
    contract = json.loads(
        (root / "configs/eval_contracts/d1b_n2_reliability.json").read_text(
            encoding="utf-8"
        )
    )
    dev = json.loads(
        (root / "configs/train_subsets/d1b_dev32.json").read_text(encoding="utf-8")
    )
    c1 = json.loads(
        (root / "configs/train_subsets/d1b_confirm32.json").read_text(encoding="utf-8")
    )
    c2 = json.loads(
        (root / "configs/train_subsets/d1b_confirm32_v2.json").read_text(encoding="utf-8")
    )
    got = {
        "contract_sha256": contract.get("sha256"),
        "dev_items": dev.get("items_sha256"),
        "c1_items": c1.get("items_sha256"),
        "c2_items": c2.get("items_sha256"),
    }
    ok = True
    for k, exp in EXPECTED.items():
        match = got[k] == exp
        ok = ok and match
        print(f"{k}: {'OK' if match else 'FAIL'} got={got[k]}")
    print("ALL_OK" if ok else "HASH_MISMATCH")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

"""Blind pairwise preference harness (Phase M / M08)."""

from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class PrefPair:
    pair_id: str
    system_a: str
    system_b: str
    path_left: str
    path_right: str
    left_is_a: bool
    trap: bool = False


def binomial_ci95(k: int, n: int) -> tuple[float, float]:
    """Wilson score interval for proportion k/n."""
    if n <= 0:
        return 0.0, 1.0
    z = 1.96
    p = k / n
    denom = 1.0 + z * z / n
    centre = p + z * z / (2 * n)
    margin = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    lo = (centre - margin) / denom
    hi = (centre + margin) / denom
    return max(0.0, lo), min(1.0, hi)


def build_pairs(
    system_a_files: list[Path],
    system_b_files: list[Path],
    *,
    seed: int = 42,
    n_traps: int = 3,
    system_a_name: str = "A",
    system_b_name: str = "B",
) -> tuple[list[PrefPair], dict[str, Any]]:
    """Match files by name; randomize left/right; add self-comparison traps."""
    by_b = {p.name: p for p in system_b_files}
    matched = []
    for pa in system_a_files:
        if pa.name in by_b:
            matched.append((pa, by_b[pa.name]))
    rng = random.Random(seed)
    rng.shuffle(matched)
    pairs: list[PrefPair] = []
    key: dict[str, Any] = {"seed": seed, "system_a": system_a_name, "system_b": system_b_name, "pairs": {}}
    for i, (pa, pb) in enumerate(matched):
        left_is_a = rng.random() < 0.5
        pid = f"pair_{i:03d}"
        left = pa if left_is_a else pb
        right = pb if left_is_a else pa
        pairs.append(
            PrefPair(
                pair_id=pid,
                system_a=system_a_name,
                system_b=system_b_name,
                path_left=str(left),
                path_right=str(right),
                left_is_a=left_is_a,
                trap=False,
            )
        )
        key["pairs"][pid] = {
            "file": pa.name,
            "left_is_a": left_is_a,
            "trap": False,
        }
    # Traps: same file twice
    for j in range(min(n_traps, len(matched))):
        pa, _ = matched[j]
        pid = f"trap_{j:03d}"
        pairs.append(
            PrefPair(
                pair_id=pid,
                system_a=system_a_name,
                system_b=system_b_name,
                path_left=str(pa),
                path_right=str(pa),
                left_is_a=True,
                trap=True,
            )
        )
        key["pairs"][pid] = {"file": pa.name, "left_is_a": True, "trap": True}
    rng.shuffle(pairs)
    return pairs, key


def ballot_from_pairs(pairs: list[PrefPair]) -> list[dict[str, Any]]:
    """Public ballot without revealing which system is which."""
    return [
        {
            "pair_id": p.pair_id,
            "left": p.path_left,
            "right": p.path_right,
            "instruction": "Choose which is more natural: left | right | same",
        }
        for p in pairs
    ]


def score_responses(
    responses: list[dict[str, Any]],
    key: dict[str, Any],
) -> dict[str, Any]:
    """Score choices. response: {pair_id, choice: left|right|same}."""
    prefer_a = 0
    prefer_b = 0
    same = 0
    n = 0
    trap_same = 0
    trap_n = 0
    for r in responses:
        pid = str(r.get("pair_id"))
        choice = str(r.get("choice") or "").lower().strip()
        meta = (key.get("pairs") or {}).get(pid)
        if not meta:
            continue
        if meta.get("trap"):
            trap_n += 1
            if choice == "same":
                trap_same += 1
            continue
        n += 1
        left_is_a = bool(meta.get("left_is_a"))
        if choice == "same":
            same += 1
        elif choice == "left":
            if left_is_a:
                prefer_a += 1
            else:
                prefer_b += 1
        elif choice == "right":
            if left_is_a:
                prefer_b += 1
            else:
                prefer_a += 1
    decisive = prefer_a + prefer_b
    p_a = prefer_a / decisive if decisive else 0.5
    lo, hi = binomial_ci95(prefer_a, decisive) if decisive else (0.0, 1.0)
    return {
        "n_scored": n,
        "prefer_a": prefer_a,
        "prefer_b": prefer_b,
        "same": same,
        "p_prefer_a": p_a,
        "ci95_low": lo,
        "ci95_high": hi,
        "significant_vs_half": bool(decisive and (hi < 0.5 or lo > 0.5)),
        "trap_n": trap_n,
        "trap_same_rate": trap_same / trap_n if trap_n else None,
        "system_a": key.get("system_a"),
        "system_b": key.get("system_b"),
    }

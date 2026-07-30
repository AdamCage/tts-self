#!/usr/bin/env python3
"""Paper2 T10 — three-way aligner comparison (raw CTC / expand_neighbor / MFA).

Expects:
  - out/paper2_seeds_summary.json from scripts/35_paper2_seeds.py (T08)
  - ~/datasets/tts/ruslan/align/forced_mfa.json from MFA (T09)

Writes out/paper2_threeway.json and docs/figures/paper2_cover_vs_cer.png
(when matplotlib is available).
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from minispeech.data.forced_align import load_forced_json
from minispeech.data.frame_projection import phones_and_frame_durs_from_forced
from minispeech.utils import expand


def _cover_stats(forced_path: Path, *, gap_mode: str, frame_rate: float = 86.1328125) -> dict:
    forced = load_forced_json(forced_path)
    covers = []
    for _uid, entry in forced.items():
        if not isinstance(entry, dict):
            continue
        try:
            fill = gap_mode not in ("raw", "none")
            _p, _d, meta = phones_and_frame_durs_from_forced(
                entry,
                frame_rate=frame_rate,
                fill_gaps=fill,
                gap_mode=gap_mode,  # type: ignore[arg-type]
            )
            # Prefer filled cover (cover_ratio_vs_end); fall back to raw.
            c = meta.get("cover_ratio_vs_end") if fill else meta.get("cover_ratio_raw")
            if c is None:
                c = meta.get("cover_ratio_raw")
            if c is not None:
                covers.append(float(c))
        except ValueError:
            continue
    if not covers:
        return {"n": 0, "mean_cover": None}
    return {
        "n": len(covers),
        "mean_cover": sum(covers) / len(covers),
        "min_cover": min(covers),
        "max_cover": max(covers),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", default="out/paper2_seeds_summary.json")
    ap.add_argument("--forced-ctc", default="~/datasets/tts/ruslan/align/forced.json")
    ap.add_argument("--forced-mfa", default="~/datasets/tts/ruslan/align/forced_mfa.json")
    ap.add_argument("--out", default="out/paper2_threeway.json")
    ap.add_argument("--figure", default="docs/figures/paper2_cover_vs_cer.png")
    args = ap.parse_args()

    seeds_path = expand(args.seeds)
    seeds = json.loads(seeds_path.read_text(encoding="utf-8")) if seeds_path.is_file() else {}
    by_mode = seeds.get("by_mode") or seeds

    def _seed_arm(name: str) -> dict:
        block = by_mode.get(name) or {}
        if not isinstance(block, dict):
            return {}
        # Normalize to keys expected by the figure helper.
        cer = block.get("oracle_cer") or {}
        return {
            "mean_cer": cer.get("mean"),
            "cer_ci95": cer.get("ci95"),
            "std_cer": cer.get("std"),
            "cover": (block.get("cover_ratio") or {}).get("mean"),
        }

    ctc = expand(args.forced_ctc)
    mfa = expand(args.forced_mfa)
    arms = {
        "raw_ctc": {
            "gap_mode": "raw",
            "cover": _cover_stats(ctc, gap_mode="raw") if ctc.is_file() else {},
            "seeds": _seed_arm("raw"),
        },
        "sil_fill": {
            "gap_mode": "sil_fill",
            "cover": _cover_stats(ctc, gap_mode="sil_fill") if ctc.is_file() else {},
            "seeds": _seed_arm("sil_fill"),
        },
        "expand_neighbor": {
            "gap_mode": "expand_neighbor",
            "cover": _cover_stats(ctc, gap_mode="expand_neighbor") if ctc.is_file() else {},
            "seeds": _seed_arm("expand_neighbor"),
        },
        "mfa": {
            "gap_mode": "none",
            "cover": _cover_stats(mfa, gap_mode="none") if mfa.is_file() else {},
            "seeds": _seed_arm("mfa"),
            "n_mfa": len(load_forced_json(mfa)) if mfa.is_file() else 0,
        },
    }

    out = {
        "arms": arms,
        "note": (
            "Plot CER (free or oracle) vs mean cover_ratio. "
            "Fill seeds via T08; MFA arm needs T09 forced_mfa.json + optional MFA train."
        ),
    }
    out_path = expand(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"wrote": str(out_path), "arms": list(arms)}, indent=2))

    # Optional figure
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib missing — skip figure")
        return 0

    xs, ys, labels, xerr, yerr = [], [], [], [], []
    for name, arm in arms.items():
        cover = (arm.get("cover") or {}).get("mean_cover")
        seed = arm.get("seeds") or {}
        cer = seed.get("mean_cer_free") or seed.get("mean_cer") or seed.get("cer")
        if cover is None or cer is None:
            continue
        xs.append(float(cover))
        ys.append(float(cer))
        labels.append(name)
        xerr.append(0.0)
        # Prefer bootstrap half-width if present
        ci = seed.get("cer_ci95") or seed.get("ci95")
        if isinstance(ci, (list, tuple)) and len(ci) == 2:
            yerr.append(0.5 * abs(float(ci[1]) - float(ci[0])))
        else:
            sd = seed.get("std_cer") or seed.get("cer_std") or 0.0
            yerr.append(float(sd) if sd is not None else 0.0)

    if not xs:
        print("no complete arms for figure yet")
        return 0

    fig_path = expand(args.figure)
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5.5, 4.0))
    ax.errorbar(xs, ys, xerr=xerr, yerr=yerr, fmt="o", capsize=3)
    for x, y, lab in zip(xs, ys, labels):
        ax.annotate(lab, (x, y), textcoords="offset points", xytext=(6, 4))
    ax.set_xlabel("timeline cover_ratio")
    ax.set_ylabel("CER")
    ax.set_title("Paper2: cover vs CER")
    ax.set_xlim(0, 1.05)
    ax.set_ylim(0, max(0.2, math.ceil(max(ys) * 10) / 10))
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(fig_path, dpi=150)
    print(f"figure={fig_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

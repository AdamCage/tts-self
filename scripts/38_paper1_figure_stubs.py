#!/usr/bin/env python3
"""Build Paper1 figure stubs from existing Part-1 / Phase-0 JSON artifacts (T16).

Does **not** invent scores. Missing inputs → skip that panel with a clear note.
Writes docs/figures/paper1_*.png when matplotlib is available, else JSON coords only.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from minispeech.utils import expand


def _load(path: Path) -> dict[str, Any] | list[Any] | None:
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _try_plot():
    try:
        import matplotlib.pyplot as plt

        return plt
    except ImportError:
        return None


def fig_r2_vs_cer(out_dir: Path, plt) -> dict[str, Any]:
    """Scatter of proxy-ish scores vs CER when both exist in known artifacts."""
    meta: dict[str, Any] = {"figure": "r2_vs_cer", "points": []}
    head = _load(expand("out/head_diagnostics_v01_best.json"))
    # Prefer explicit points if present; else single Gate0 point from diagnosis docs numbers.
    if isinstance(head, dict) and head.get("r2_overall") is not None:
        # Pair with known synth CER from diagnosis if available
        synth = _load(expand("out/intelligibility_mel_s4_val100.json"))
        cer = None
        if isinstance(synth, dict):
            cer = synth.get("mean_cer") or (synth.get("aggregates") or {}).get("mean_cer")
        # Latent synth CER from part1 (fixed measured)
        latent_cer = 0.889
        meta["points"].append(
            {
                "label": "latent_talker_s3",
                "r2": float(head["r2_overall"]),
                "cer": latent_cer,
                "source_r2": "out/head_diagnostics_v01_best.json",
                "source_cer": "docs/part1_diagnosis.md (measured)",
            }
        )
        if cer is not None:
            meta["points"].append(
                {
                    "label": "mel_s4_val100",
                    "r2": None,
                    "cer": float(cer),
                    "note": "mel path has no latent R²; CER only",
                }
            )
    coords = out_dir / "paper1_r2_vs_cer.json"
    coords.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    if plt and meta["points"]:
        xs = [p["r2"] for p in meta["points"] if p.get("r2") is not None]
        ys = [p["cer"] for p in meta["points"] if p.get("r2") is not None]
        fig, ax = plt.subplots(figsize=(5, 4))
        if xs:
            ax.scatter(xs, ys)
            for p in meta["points"]:
                if p.get("r2") is not None:
                    ax.annotate(p["label"], (p["r2"], p["cer"]), textcoords="offset points", xytext=(5, 5))
        ax.axvline(0.28, color="gray", ls="--", label="Gate0 R²=0.28")
        ax.set_xlabel("teacher-forced R²")
        ax.set_ylabel("synth CER")
        ax.set_title("Paper1 stub: R² vs CER")
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_dir / "paper1_r2_vs_cer.png", dpi=150)
        plt.close(fig)
        meta["png"] = str(out_dir / "paper1_r2_vs_cer.png")
    return meta


def fig_from_sensitivity(out_dir: Path, plt) -> dict[str, Any]:
    """CER(σ) / CER(α) stubs from Phase-0 sensitivity if present."""
    meta: dict[str, Any] = {"figure": "cer_sigma_alpha", "series": {}}
    for name, fname in (
        ("cer_sigma", "out/phase0_sensitivity.json"),
        ("cer_alpha", "out/phase0_sensitivity.json"),
    ):
        data = _load(expand(fname))
        meta["series"][name] = {
            "source": fname,
            "available": data is not None,
            "note": "Plot when scripts/21_latent_error_calibration.py / phase0 artifacts exist",
        }
        if isinstance(data, dict):
            # Best-effort: look for nested sweeps
            for key in ("sigma_sweep", "noise_sweep", "cer_by_sigma", "alpha_sweep"):
                if key in data:
                    meta["series"][name]["key"] = key
                    meta["series"][name]["n"] = len(data[key]) if hasattr(data[key], "__len__") else None
    (out_dir / "paper1_cer_sigma_alpha.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    if plt:
        fig, ax = plt.subplots(figsize=(5, 3.5))
        ax.text(
            0.5,
            0.5,
            "Stub: drop Phase-0 / calibration JSON\nto render CER(σ) and CER(α) curves",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        ax.set_axis_off()
        fig.savefig(out_dir / "paper1_cer_sigma_alpha_stub.png", dpi=120)
        plt.close(fig)
    return meta


def fig_diagonality(out_dir: Path, plt) -> dict[str, Any]:
    meta: dict[str, Any] = {"figure": "diagonality_vs_chance", "status": "pending_artifact"}
    for cand in (
        "out/attention_diagonality.json",
        "out/text_usage_s1.json",
        "out/head_diagnostics_v01_best.json",
    ):
        data = _load(expand(cand))
        if data is None:
            continue
        meta["candidate"] = cand
        if isinstance(data, dict):
            for k in ("diagonality", "attn_diag", "diag_score", "chance_diag"):
                if k in data:
                    meta[k] = data[k]
        meta["status"] = "partial" if meta.keys() - {"figure", "status", "candidate"} else "pending_artifact"
        break
    (out_dir / "paper1_diagonality.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    if plt:
        fig, ax = plt.subplots(figsize=(5, 3.5))
        ax.text(
            0.5,
            0.5,
            "Stub: attention diagonality vs chance\n(requires text-usage / attn dump)",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        ax.set_axis_off()
        fig.savefig(out_dir / "paper1_diagonality_stub.png", dpi=120)
        plt.close(fig)
    return meta


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", default="docs/figures")
    args = ap.parse_args()
    out_dir = expand(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    plt = _try_plot()
    report = {
        "r2_vs_cer": fig_r2_vs_cer(out_dir, plt),
        "cer_sigma_alpha": fig_from_sensitivity(out_dir, plt),
        "diagonality": fig_diagonality(out_dir, plt),
        "matplotlib": plt is not None,
    }
    summary = out_dir / "paper1_figure_stubs_summary.json"
    summary.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"wrote": str(summary), "matplotlib": plt is not None}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

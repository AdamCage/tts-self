#!/usr/bin/env python3
"""Build Paper 1 figures from existing measured JSON artifacts (S4).

Does not invent scores. Each figure records its source artifact path(s).
Writes PNG (if matplotlib available) plus JSON coords under out/paper1_figures/.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from minispeech.utils import expand


def _load(path: str | Path) -> dict[str, Any] | list[Any] | None:
    p = expand(path)
    if not p.is_file():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def _try_plot():
    try:
        import matplotlib.pyplot as plt

        return plt
    except ImportError:
        return None


def _write_meta(out_dir: Path, stem: str, meta: dict[str, Any]) -> Path:
    path = out_dir / f"{stem}.json"
    path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def fig_cer_sigma_alpha(out_dir: Path, plt) -> dict[str, Any]:
    src = "out/latent_error_calibration.json"
    data = _load(src)
    meta: dict[str, Any] = {
        "figure": "cer_sigma_alpha",
        "source": src,
        "status": "missing_source" if data is None else "ok",
        "series": {},
    }
    if not isinstance(data, dict):
        _write_meta(out_dir, "fig01_cer_sigma_alpha", meta)
        return meta

    curves = data.get("curves") or {}
    for key, xlabel in (
        ("white", "σ (white noise)"),
        ("variance_collapse", "α (variance collapse)"),
        ("temporal_smooth", "kernel width (frames)"),
    ):
        rows = curves.get(key) or []
        series = [
            {"x": float(r["value"]), "mean_cer": float(r["mean_cer"]), "n_scored": int(r["n_scored"])}
            for r in rows
            if isinstance(r, dict) and "value" in r and "mean_cer" in r
        ]
        meta["series"][key] = {"xlabel": xlabel, "points": series}

    meta["gt_latent_topline_cer"] = float((data.get("gt_latent_topline") or {}).get("mean_cer") or 0.0)
    meta["n"] = int(data.get("n") or 0)
    _write_meta(out_dir, "fig01_cer_sigma_alpha", meta)

    if plt and any(meta["series"][k]["points"] for k in meta["series"]):
        fig, axes = plt.subplots(1, 3, figsize=(11, 3.6), sharey=True)
        titles = {
            "white": "CER(σ) white",
            "variance_collapse": "CER(α) collapse",
            "temporal_smooth": "CER(smooth)",
        }
        for ax, key in zip(axes, ("white", "variance_collapse", "temporal_smooth")):
            pts = meta["series"][key]["points"]
            xs = [p["x"] for p in pts]
            ys = [p["mean_cer"] for p in pts]
            ax.plot(xs, ys, marker="o")
            ax.axhline(meta["gt_latent_topline_cer"], color="gray", ls="--", lw=1, label="GT topline")
            ax.set_xlabel(meta["series"][key]["xlabel"])
            ax.set_title(titles[key])
            ax.grid(True, alpha=0.3)
        axes[0].set_ylabel("mean CER")
        axes[0].legend(fontsize=8)
        fig.suptitle(f"Latent sensitivity (n={meta['n']}) — source: {src}", fontsize=10)
        fig.tight_layout()
        png = out_dir / "fig01_cer_sigma_alpha.png"
        fig.savefig(png, dpi=150)
        plt.close(fig)
        meta["png"] = str(png)
        _write_meta(out_dir, "fig01_cer_sigma_alpha", meta)
    return meta


def fig_r2_three_systems(out_dir: Path, plt) -> dict[str, Any]:
    src = "out/external_baseline_silero_v3.json"
    data = _load(src)
    meta: dict[str, Any] = {
        "figure": "r2_three_systems_cer_der",
        "source": src,
        "status": "missing_source" if data is None else "ok",
        "bars": [],
    }
    if not isinstance(data, dict):
        _write_meta(out_dir, "fig02_r2_three_systems", meta)
        return meta

    summaries = data.get("summaries") or {}
    sets_of_interest = ("demo5", "val100", "bank_entity", "homograph", "latin")
    conditions = ("silero_raw", "silero_frontend", "track_a")
    for set_name in sets_of_interest:
        row: dict[str, Any] = {"set": set_name, "mean_cer": {}, "der": {}}
        for cond in conditions:
            s = ((summaries.get(cond) or {}).get("sets") or {}).get(set_name) or {}
            if "mean_cer" in s and s["mean_cer"] is not None:
                row["mean_cer"][cond] = float(s["mean_cer"])
            if s.get("der") is not None:
                row["der"][cond] = float(s["der"])
        # bank DER also in top-level bank_entity_der
        meta["bars"].append(row)

    bank = data.get("bank_entity_der") or {}
    meta["bank_entity_der"] = bank
    meta["system_compare_condition"] = data.get("system_compare_condition")
    meta["frontend_contribution_bank_der"] = bank.get("delta_frontend_minus_raw")
    _write_meta(out_dir, "fig02_r2_three_systems", meta)

    if plt and meta["bars"]:
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
        # CER grouped bars
        ax = axes[0]
        x_labels = [b["set"] for b in meta["bars"]]
        import numpy as np

        x = np.arange(len(x_labels))
        width = 0.25
        for i, cond in enumerate(conditions):
            ys = [b["mean_cer"].get(cond) for b in meta["bars"]]
            ys_plot = [y if y is not None else float("nan") for y in ys]
            ax.bar(x + (i - 1) * width, ys_plot, width, label=cond)
        ax.set_xticks(x)
        ax.set_xticklabels(x_labels, rotation=20, ha="right")
        ax.set_ylabel("mean CER")
        ax.set_title("R2 three conditions — CER")
        ax.legend(fontsize=8)
        ax.grid(True, axis="y", alpha=0.3)

        ax = axes[1]
        labels = ["silero_raw", "silero_frontend", "track_a"]
        ders = [float(bank[k]) for k in labels if k in bank]
        ax.bar(labels, ders, color=["#888", "#2a6", "#36a"])
        ax.set_ylabel("bank_entity DER")
        ax.set_title("bank_entity DER (v3)")
        if meta.get("frontend_contribution_bank_der") is not None:
            ax.set_xlabel(f"Δ frontend−raw = {meta['frontend_contribution_bank_der']:.4f}")
        ax.grid(True, axis="y", alpha=0.3)
        fig.suptitle(f"Fair three-system compare — source: {src}", fontsize=10)
        fig.tight_layout()
        png = out_dir / "fig02_r2_three_systems.png"
        fig.savefig(png, dpi=150)
        plt.close(fig)
        meta["png"] = str(png)
        _write_meta(out_dir, "fig02_r2_three_systems", meta)
    return meta


def fig_bank_delta_2x2(out_dir: Path, plt) -> dict[str, Any]:
    src = "out/bank_delta_2x2.json"
    data = _load(src)
    meta: dict[str, Any] = {
        "figure": "bank_delta_2x2",
        "source": src,
        "status": "missing_source" if data is None else "ok",
    }
    if not isinstance(data, dict):
        _write_meta(out_dir, "fig03_bank_delta_2x2", meta)
        return meta

    cells = data.get("cells") or {}
    meta["cells"] = {
        k: {
            "der": v.get("der"),
            "eer": v.get("eer"),
            "mean_cer": v.get("mean_cer"),
            "n": v.get("n"),
            "label_version": v.get("label_version"),
        }
        for k, v in cells.items()
        if isinstance(v, dict)
    }
    meta["decomposition_der"] = data.get("decomposition_der")
    meta["decomposition_eer"] = data.get("decomposition_eer")
    _write_meta(out_dir, "fig03_bank_delta_2x2", meta)

    if plt and meta["cells"]:
        # 2x2 heat-ish bar for DER
        order = [
            "frontend_before_labels_v2",
            "frontend_before_labels_v3",
            "frontend_after_labels_v2",
            "frontend_after_labels_v3",
        ]
        present = [k for k in order if k in meta["cells"]]
        ders = [float(meta["cells"][k]["der"]) for k in present]
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        axes[0].barh(present, ders, color="#4a7")
        axes[0].set_xlabel("DER")
        axes[0].set_title("2×2 cells (DER)")
        axes[0].grid(True, axis="x", alpha=0.3)

        decomp = meta.get("decomposition_der") or {}
        keys = ["label_avg", "frontend_avg", "interaction", "unexplained_remainder_vs_path1"]
        keys = [k for k in keys if k in decomp]
        axes[1].bar(keys, [float(decomp[k]) for k in keys], color="#a63")
        axes[1].axhline(0, color="black", lw=0.8)
        axes[1].set_ylabel("Δ DER")
        axes[1].set_title("Decomposition")
        axes[1].tick_params(axis="x", rotation=20)
        axes[1].grid(True, axis="y", alpha=0.3)
        fig.suptitle(f"bank_entity v2→v3 2×2 — source: {src}", fontsize=10)
        fig.tight_layout()
        png = out_dir / "fig03_bank_delta_2x2.png"
        fig.savefig(png, dpi=150)
        plt.close(fig)
        meta["png"] = str(png)
        _write_meta(out_dir, "fig03_bank_delta_2x2", meta)
    return meta


def fig_gate0_panel(out_dir: Path, plt) -> dict[str, Any]:
    src_head = "out/head_diagnostics_v01_best.json"
    src_triv = "out/trivial_baselines.json"
    head = _load(src_head)
    triv = _load(src_triv)
    meta: dict[str, Any] = {
        "figure": "gate0_panel",
        "sources": [src_head, src_triv],
        "status": "ok" if isinstance(head, dict) else "missing_source",
        "head": {},
        "trivial_legacy_pass": [],
    }
    if isinstance(head, dict):
        for k in (
            "r2_overall",
            "tf_l1_mean",
            "tf_l1_over_constant_mae",
            "var_ratio",
            "hf_power_ratio",
            "gate0_r2_ge_0.28",
            "n_utts",
        ):
            if k in head:
                meta["head"][k] = head[k]
    if isinstance(triv, dict):
        # structure may nest predictors
        predictors = triv.get("predictors") or triv.get("baselines") or triv
        if isinstance(predictors, dict):
            for name, row in predictors.items():
                if isinstance(row, dict) and "passes_legacy_fr_gates" in row:
                    meta["trivial_legacy_pass"].append(
                        {"name": name, "passes_legacy_fr_gates": bool(row["passes_legacy_fr_gates"])}
                    )
    _write_meta(out_dir, "fig04_gate0_panel", meta)

    if plt and meta["head"]:
        fig, axes = plt.subplots(1, 2, figsize=(9, 3.8))
        metrics = ["r2_overall", "var_ratio", "hf_power_ratio", "tf_l1_over_constant_mae"]
        vals = [float(meta["head"][m]) for m in metrics if m in meta["head"]]
        labels = [m for m in metrics if m in meta["head"]]
        axes[0].barh(labels, vals, color="#c45")
        axes[0].axvline(0.28, color="gray", ls="--", label="Gate0 R²=0.28")
        axes[0].set_title("Gate 0 diagnostics (FAIL)")
        axes[0].legend(fontsize=8)
        axes[0].grid(True, axis="x", alpha=0.3)

        names = [r["name"] for r in meta["trivial_legacy_pass"]] or ["(see JSON)"]
        passes = [1.0 if r.get("passes_legacy_fr_gates") else 0.0 for r in meta["trivial_legacy_pass"]] or [0.0]
        axes[1].bar(names, passes, color="#888")
        axes[1].set_ylim(0, 1.2)
        axes[1].set_ylabel("passes_legacy_fr_gates")
        axes[1].set_title("E0 trivial baselines")
        axes[1].tick_params(axis="x", rotation=25)
        fig.suptitle(f"Proxy trap — sources: {src_head}, {src_triv}", fontsize=10)
        fig.tight_layout()
        png = out_dir / "fig04_gate0_panel.png"
        fig.savefig(png, dpi=150)
        plt.close(fig)
        meta["png"] = str(png)
        _write_meta(out_dir, "fig04_gate0_panel", meta)
    return meta


def fig_e2_vs_e4(out_dir: Path, plt) -> dict[str, Any]:
    src_cal = "out/latent_error_calibration.json"
    src_intel = "out/intelligibility_S3.json"
    cal = _load(src_cal)
    intel = _load(src_intel)
    meta: dict[str, Any] = {
        "figure": "e2_vs_e4_gap",
        "sources": [src_cal, src_intel],
        "status": "ok",
        "bars": [],
    }
    if isinstance(cal, dict):
        gt = (cal.get("gt_latent_topline") or {}).get("mean_cer")
        if gt is not None:
            meta["bars"].append(
                {"label": "E2_gt_latent_topline", "mean_cer": float(gt), "source": src_cal}
            )
    if isinstance(intel, dict):
        results = intel.get("results") or {}
        for name in ("S3_calibrated", "S4_fr_best"):
            row = results.get(name) or {}
            if isinstance(row, dict) and row.get("mean_cer") is not None:
                meta["bars"].append(
                    {"label": f"E4_{name}", "mean_cer": float(row["mean_cer"]), "source": src_intel}
                )
    if len(meta["bars"]) < 2:
        meta["status"] = "partial"
    _write_meta(out_dir, "fig05_e2_vs_e4_gap", meta)

    if plt and meta["bars"]:
        fig, ax = plt.subplots(figsize=(6, 3.8))
        labels = [b["label"] for b in meta["bars"]]
        ys = [b["mean_cer"] for b in meta["bars"]]
        ax.bar(labels, ys, color=["#2a6", "#c45", "#a63"][: len(ys)])
        ax.set_ylabel("mean CER")
        ax.set_title("Copy-synth topline vs model synth")
        ax.tick_params(axis="x", rotation=15)
        ax.grid(True, axis="y", alpha=0.3)
        if len(ys) >= 2:
            ax.annotate(
                f"Δ ≈ {ys[1] - ys[0]:+.3f}",
                xy=(1, ys[1]),
                xytext=(1.2, max(ys) * 0.7),
                arrowprops=dict(arrowstyle="->", color="black"),
            )
        fig.tight_layout()
        png = out_dir / "fig05_e2_vs_e4_gap.png"
        fig.savefig(png, dpi=150)
        plt.close(fig)
        meta["png"] = str(png)
        _write_meta(out_dir, "fig05_e2_vs_e4_gap", meta)
    return meta


def fig_diagonality(out_dir: Path, plt) -> dict[str, Any]:
    src = "out/align_audit.json"
    data = _load(src)
    meta: dict[str, Any] = {
        "figure": "diagonality_vs_chance",
        "source": src,
        "status": "missing_source" if data is None else "ok",
    }
    if isinstance(data, dict):
        for k in ("n", "mean_diagonality", "mean_chance", "mean_vs_chance"):
            if k in data:
                meta[k] = data[k]
    _write_meta(out_dir, "fig06_diagonality", meta)

    if plt and "mean_diagonality" in meta:
        fig, ax = plt.subplots(figsize=(5, 3.6))
        ax.bar(
            ["diagonality", "chance"],
            [float(meta["mean_diagonality"]), float(meta["mean_chance"])],
            color=["#36a", "#888"],
        )
        ax.set_ylabel("mean score")
        ax.set_title(f"Cross-attn diagonality (n={meta.get('n')})")
        ax.annotate(
            f"Δ={float(meta.get('mean_vs_chance', 0)):.4f}",
            xy=(0.5, max(float(meta["mean_diagonality"]), float(meta["mean_chance"]))),
        )
        ax.grid(True, axis="y", alpha=0.3)
        fig.tight_layout()
        png = out_dir / "fig06_diagonality.png"
        fig.savefig(png, dpi=150)
        plt.close(fig)
        meta["png"] = str(png)
        _write_meta(out_dir, "fig06_diagonality", meta)
    return meta


def fig_telephony(out_dir: Path, plt) -> dict[str, Any]:
    src = "out/p1_telephony_channels.json"
    data = _load(src)
    meta: dict[str, Any] = {
        "figure": "telephony_channels",
        "source": src,
        "status": "missing_source" if data is None else "ok",
        "conditions": {},
        "paired_significance": {},
    }
    if isinstance(data, dict):
        conds = data.get("conditions") or {}
        if isinstance(conds, dict):
            for name, row in conds.items():
                if isinstance(row, dict):
                    meta["conditions"][name] = {
                        "n": row.get("n"),
                        "mean_cer": row.get("mean_cer"),
                        "median_cer": row.get("median_cer"),
                        "cer_ci95_low": row.get("cer_ci95_low"),
                        "cer_ci95_high": row.get("cer_ci95_high"),
                    }
        ps = data.get("paired_significance") or {}
        meta["paired_significance"] = {
            "n": ps.get("n"),
            "difference_detected_any": ps.get("difference_detected_any"),
            "conclusion_ru": ps.get("conclusion_ru"),
            "channels": {
                ch: {
                    "mean_delta_cer": row.get("mean_delta_cer"),
                    "bootstrap_mean_delta_cer_ci95_low": row.get("bootstrap_mean_delta_cer_ci95_low"),
                    "bootstrap_mean_delta_cer_ci95_high": row.get("bootstrap_mean_delta_cer_ci95_high"),
                    "difference_detected": row.get("difference_detected"),
                    "ci_includes_zero": row.get("ci_includes_zero"),
                }
                for ch, row in (ps.get("channels") or {}).items()
                if isinstance(row, dict)
            },
        }
    _write_meta(out_dir, "fig07_telephony_channels", meta)

    if plt and meta["conditions"]:
        fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
        names = list(meta["conditions"].keys())
        means = [float(meta["conditions"][n]["mean_cer"]) for n in names]
        lo = [float(meta["conditions"][n]["cer_ci95_low"]) for n in names]
        hi = [float(meta["conditions"][n]["cer_ci95_high"]) for n in names]
        yerr = [[m - l for m, l in zip(means, lo)], [h - m for m, h in zip(means, hi)]]
        axes[0].bar(names, means, yerr=yerr, capsize=4, color="#36a")
        axes[0].set_ylabel("mean CER")
        axes[0].set_title("Absolute CER by channel")
        axes[0].tick_params(axis="x", rotation=15)
        axes[0].grid(True, axis="y", alpha=0.3)

        chans = meta["paired_significance"].get("channels") or {}
        cnames = list(chans.keys())
        deltas = [float(chans[c]["mean_delta_cer"]) for c in cnames]
        dlo = [float(chans[c]["bootstrap_mean_delta_cer_ci95_low"]) for c in cnames]
        dhi = [float(chans[c]["bootstrap_mean_delta_cer_ci95_high"]) for c in cnames]
        dyerr = [[d - l for d, l in zip(deltas, dlo)], [h - d for d, h in zip(deltas, dhi)]]
        axes[1].bar(cnames, deltas, yerr=dyerr, capsize=4, color="#a63")
        axes[1].axhline(0, color="black", lw=0.8)
        axes[1].set_ylabel("Δ CER vs wideband")
        axes[1].set_title(
            f"Paired Δ (detected={meta['paired_significance'].get('difference_detected_any')})"
        )
        axes[1].tick_params(axis="x", rotation=15)
        axes[1].grid(True, axis="y", alpha=0.3)
        fig.suptitle(f"Telephony channels — source: {src}", fontsize=10)
        fig.tight_layout()
        png = out_dir / "fig07_telephony_channels.png"
        fig.savefig(png, dpi=150)
        plt.close(fig)
        meta["png"] = str(png)
        _write_meta(out_dir, "fig07_telephony_channels", meta)
    return meta


def fig_track_a_ladder(out_dir: Path, plt) -> dict[str, Any]:
    """Counterpoint figure: mel Track A quality ladder from frozen suite summaries."""
    sources = {
        "baseline": "out/eval_mel_suite_s4_baseline_full.json",
        "stress": "out/eval_mel_suite_stress_s4.json",
        "variance": "out/eval_mel_suite_variance_s4.json",
        "gate": "out/gate_v4_v5_summary.json",
    }
    meta: dict[str, Any] = {
        "figure": "track_a_ladder",
        "sources": list(sources.values()),
        "status": "ok",
        "points": [],
    }

    def _val100_cer(path: str) -> float | None:
        data = _load(path)
        if not isinstance(data, dict):
            return None
        sets = data.get("sets") or {}
        val = sets.get("val100") or {}
        if isinstance(val, dict) and val.get("mean_cer") is not None:
            return float(val["mean_cer"])
        return None

    for label, path in (("mel80_s4_baseline", sources["baseline"]), ("stress_s4", sources["stress"]), ("variance_s4", sources["variance"])):
        cer = _val100_cer(path)
        if cer is not None:
            meta["points"].append({"label": label, "val100_mean_cer": cer, "source": path})
    gate = _load(sources["gate"])
    if isinstance(gate, dict):
        meta["gate_v4_v5"] = {
            k: gate[k]
            for k in gate
            if isinstance(gate[k], (str, bool, int, float, type(None)))
        }
    if not meta["points"]:
        meta["status"] = "missing_source"
    _write_meta(out_dir, "fig08_track_a_ladder", meta)

    if plt and meta["points"]:
        fig, ax = plt.subplots(figsize=(6, 3.6))
        labels = [p["label"] for p in meta["points"]]
        ys = [p["val100_mean_cer"] for p in meta["points"]]
        ax.bar(labels, ys, color=["#888", "#2a6", "#c45"][: len(ys)])
        ax.set_ylabel("val100 mean CER")
        ax.set_title("Track A ladder (counterpoint)")
        ax.grid(True, axis="y", alpha=0.3)
        fig.tight_layout()
        png = out_dir / "fig08_track_a_ladder.png"
        fig.savefig(png, dpi=150)
        plt.close(fig)
        meta["png"] = str(png)
        _write_meta(out_dir, "fig08_track_a_ladder", meta)
    return meta


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", default="out/paper1_figures")
    args = ap.parse_args(argv)
    out_dir = expand(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    plt = _try_plot()

    report = {
        "fig01_cer_sigma_alpha": fig_cer_sigma_alpha(out_dir, plt),
        "fig02_r2_three_systems": fig_r2_three_systems(out_dir, plt),
        "fig03_bank_delta_2x2": fig_bank_delta_2x2(out_dir, plt),
        "fig04_gate0_panel": fig_gate0_panel(out_dir, plt),
        "fig05_e2_vs_e4_gap": fig_e2_vs_e4(out_dir, plt),
        "fig06_diagonality": fig_diagonality(out_dir, plt),
        "fig07_telephony_channels": fig_telephony(out_dir, plt),
        "fig08_track_a_ladder": fig_track_a_ladder(out_dir, plt),
        "matplotlib": plt is not None,
        "mos_status": "unmeasured",
        "note": "Figures built only from on-disk artifacts; no synthetic curves.",
    }
    summary = out_dir / "manifest.json"
    summary.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    n_ok = sum(1 for k, v in report.items() if isinstance(v, dict) and v.get("status") == "ok")
    print(
        json.dumps(
            {
                "wrote": str(summary),
                "matplotlib": plt is not None,
                "figures_ok": n_ok,
                "out_dir": str(out_dir),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

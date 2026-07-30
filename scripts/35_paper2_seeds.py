#!/usr/bin/env python3
"""Paper2 seed ladder: raw CTC / sil_fill / expand_neighbor × 3 seeds (T08).

Trains MelBaseline S1 (32 utts) for each (gap_mode, seed), then scores
cover_ratio + oracle CER + free CER. Writes ``out/paper2_seeds_summary.json``.

Run on frozen tag / mel80 domain (WSL)::

    python scripts/35_paper2_seeds.py --seeds 11,22,33 --max-utts 32
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any

import numpy as np

from minispeech.utils import expand


def _run(cmd: list[str]) -> int:
    print("+", " ".join(cmd), flush=True)
    return subprocess.call(cmd)


def _mean_ci(vals: list[float], *, n_boot: int = 1000, seed: int = 0) -> dict[str, Any]:
    arr = np.asarray(vals, dtype=np.float64)
    if arr.size == 0:
        return {"mean": None, "std": None, "ci95": [None, None], "n": 0}
    rng = np.random.default_rng(seed)
    boots = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        idx = rng.integers(0, arr.size, size=arr.size)
        boots[i] = float(arr[idx].mean())
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=0)),
        "ci95": [float(lo), float(hi)],
        "n": int(arr.size),
        "values": [float(x) for x in arr],
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", default="11,22,33")
    ap.add_argument("--config", default="configs/mel_baseline_v01.yaml")
    ap.add_argument("--data", default="configs/data_ruslan_local.yaml")
    ap.add_argument("--max-utts", type=int, default=32)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--out", default="out/paper2_seeds_summary.json")
    ap.add_argument(
        "--modes",
        default="raw,sil_fill,expand_neighbor",
        help="raw=no gap fill (cover≪1); sil_fill; expand_neighbor",
    )
    ap.add_argument("--skip-train", action="store_true")
    args = ap.parse_args()

    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]
    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    runs: list[dict[str, Any]] = []

    for mode in modes:
        # Train/eval gap_mode: raw/none leave CTC blank gaps; others fill.
        gap = "raw" if mode == "raw" else mode
        fill_gaps = mode not in ("raw", "none")
        for seed in seeds:
            run_dir = expand(f"runs/paper2_{mode}_s{seed}")
            ckpt = run_dir / "best.pt"
            if not args.skip_train:
                cmd = [
                    "python",
                    "scripts/27_train_mel.py",
                    "--config",
                    args.config,
                    "--data",
                    args.data,
                    "--stage",
                    "S1",
                    "--gap-mode",
                    gap,
                    "--run-dir",
                    str(run_dir),
                ]
                # seed via env + config patch is trainer-dependent; pass through YAML override if supported
                env_seed = ["--seed", str(seed)]
                # 27_train_mel may not expose --seed; write a tiny overlay
                overlay = run_dir / "seed_overlay.yaml"
                overlay.parent.mkdir(parents=True, exist_ok=True)
                overlay.write_text(
                    Path(args.config).read_text(encoding="utf-8").replace(
                        "seed: 42", f"seed: {seed}"
                    ),
                    encoding="utf-8",
                )
                cmd[cmd.index(args.config)] = str(overlay)
                rc = _run(cmd)
                if rc != 0:
                    runs.append({"mode": mode, "seed": seed, "error": f"train_rc={rc}"})
                    continue

            # Cover ratio probe (reuse frame_projection over forced)
            cover_out = run_dir / "cover.json"
            _run(
                [
                    "python",
                    "-c",
                    (
                        "import json; from pathlib import Path; "
                        "from minispeech.data.forced_align import load_forced_json; "
                        "from minispeech.data.frame_projection import phones_and_frame_durs_from_forced; "
                        "from minispeech.utils import load_yaml, expand; "
                        f"d=load_yaml('{args.data}')['data']; "
                        "forced=load_forced_json(expand(d['forced_path'])); "
                        f"fill={fill_gaps}; mode='{gap}'; "
                        "ids=list(forced)[:32]; ratios=[]; "
                        "\nfor uid in ids:\n"
                        " e=forced[uid];\n"
                        " _,_,m=phones_and_frame_durs_from_forced(e,frame_rate=86.1328125,"
                        "fill_gaps=fill,gap_mode=mode);\n"
                        " ratios.append(m['cover_ratio_vs_end']);\n"
                        f"Path(r'{cover_out}').write_text(json.dumps({{'mean': sum(ratios)/len(ratios) if ratios else None, 'n': len(ratios), 'mode': '{mode}'}}), encoding='utf-8')"
                    ),
                ]
            )
            cover = None
            if cover_out.is_file():
                cover = json.loads(cover_out.read_text(encoding="utf-8")).get("mean")

            oracle_json = run_dir / "oracle.json"
            free_json = run_dir / "free_demo.json"
            if ckpt.is_file():
                _run(
                    [
                        "python",
                        "scripts/30_mel_copy_synth_eval.py",
                        "--skip-gt",
                        "--ckpt",
                        str(ckpt),
                        "--gap-mode",
                        gap,
                        "--n-oracle",
                        "5",
                        "--out-oracle",
                        str(oracle_json),
                    ]
                )
                _run(
                    [
                        "python",
                        "scripts/28_infer_mel.py",
                        "--ckpt",
                        str(ckpt),
                        "--demo",
                        "--duration-mode",
                        "mean_stats",
                        "--gap-mode",
                        gap,
                        "--sil-policy",
                        "none",
                        "--no-forced-lookup",
                        "--data",
                        args.data,
                        "--out",
                        str(run_dir / "demo_wavs"),
                    ]
                )
            oracle_cer = None
            if oracle_json.is_file():
                oracle_cer = json.loads(oracle_json.read_text(encoding="utf-8")).get("mean_cer")
            runs.append(
                {
                    "mode": mode,
                    "seed": seed,
                    "run_dir": str(run_dir),
                    "cover_ratio": cover,
                    "oracle_cer": oracle_cer,
                    "fill_gaps": fill_gaps,
                    "gap_mode": gap,
                }
            )

    # Aggregate per mode (cover even when oracle CER missing)
    summary: dict[str, Any] = {"runs": runs, "by_mode": {}}
    for mode in modes:
        subset = [r for r in runs if r.get("mode") == mode]
        with_cer = [r for r in subset if r.get("oracle_cer") is not None]
        with_cover = [r for r in subset if r.get("cover_ratio") is not None]
        summary["by_mode"][mode] = {
            "oracle_cer": _mean_ci([float(r["oracle_cer"]) for r in with_cer]),
            "cover_ratio": _mean_ci([float(r["cover_ratio"]) for r in with_cover]),
            "n_runs": len(subset),
            "n_oracle": len(with_cer),
        }
    out = expand(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary["by_mode"], indent=2), flush=True)
    print(f"WROTE {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

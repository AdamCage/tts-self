#!/usr/bin/env python3
"""E0: trivial latent predictors through free-running stop/duration/DTW gates.

Shows that constant-mean / copy-prev / random predictors can pass the same
gates used by ``scripts/09b_eval_free_running.py`` when given oracle length.
No model checkpoint required — only cached latents + stats.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict

import torch

from minispeech.codec.stats import LatentNormalizer
from minispeech.constants import FRAME_RATE
from minispeech.data.dataset import MiniSpeechDataset
from minispeech.eval.free_running import (
    aggregate_stability,
    duration_bucket,
    evaluate_free_running_latents,
    fr_selection_key,
)
from minispeech.eval.trivial_baselines import BaselineName, predict_baseline
from minispeech.utils import expand, load_yaml, set_seed

BASELINES: tuple[BaselineName, ...] = (
    "constant_mean",
    "copy_prev",
    "random_corpus",
    "random_other_utt",
)


def _gates(stab: dict[str, float]) -> dict[str, bool]:
    return {
        "stop_success_ge_0.95": stab.get("stop_success_rate", 0.0) >= 0.95,
        "runaway_le_0.02": stab.get("runaway_rate", 1.0) <= 0.02,
        "nan_rate_eq_0": stab.get("nan_rate", 1.0) == 0.0,
        "duration_in_band_ge_0.90": stab.get("duration_in_band_rate", 0.0) >= 0.90,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default="configs/data_ruslan_local.yaml")
    ap.add_argument("--split", default="val", choices=("train", "val", "test"))
    ap.add_argument("--n", type=int, default=32)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="out/trivial_baselines.json")
    args = ap.parse_args()
    set_seed(args.seed)

    dcfg = load_yaml(args.data)["data"]
    try:
        norm = LatentNormalizer.from_file(dcfg["stats_path"])
        # Normalized space: corpus mean ≈ 0
        mean = torch.zeros(norm.mean.shape[-1])
    except Exception:
        norm = None
        mean = torch.zeros(64)

    ds = MiniSpeechDataset(
        manifest_path=dcfg["manifest"],
        latent_dir=dcfg["latent_dir"],
        text_hidden_dir=dcfg["text_hidden_dir"],
        split=args.split,
        max_utterances=args.n,
        exclude_tags=tuple(dcfg.get("exclude_tags", ("exclude",))),
        preload=True,
    )
    if len(ds) == 0:
        print("No utterances found; check manifest/split.")
        return 2

    # Collect normalized GT latents for pool / other-utt
    gt_list: list[torch.Tensor] = []
    for i in range(len(ds)):
        z = ds[i]["z_target"].float()
        if norm is not None:
            z = norm.normalize(z.unsqueeze(0))[0]
        gt_list.append(z)
    pool = torch.cat(gt_list, dim=0)
    gen = torch.Generator().manual_seed(args.seed)

    reports: dict[str, dict] = {}
    for name in BASELINES:
        rows = []
        bucket_dtw: dict[str, list[float]] = defaultdict(list)
        for z_gt in gt_list:
            z_pred = predict_baseline(
                name,
                z_gt,
                mean=mean,
                pool=pool,
                candidates=gt_list,
                generator=gen,
            )
            # Oracle length ⇒ stop success; content-blind predictors still pass timing gates
            m = evaluate_free_running_latents(
                z_pred,
                z_gt,
                stop_triggered=True,
                hit_max_frames=False,
            )
            rows.append(m)
            bucket_dtw[duration_bucket(z_gt.shape[0] / FRAME_RATE)].append(m.dtw_l1)

        stab = aggregate_stability(rows)
        gates = _gates(stab)
        reports[name] = {
            "n": len(rows),
            "stability": stab,
            "bucket_mean_dtw": {k: sum(v) / len(v) for k, v in sorted(bucket_dtw.items())},
            "gates": gates,
            "selection_key": list(fr_selection_key(stab)),
            "passes_legacy_fr_gates": all(gates.values()),
            "note": (
                "Oracle length (stop_triggered=True, n_pred=n_gt). "
                "Content-blind predictors can still pass stop/duration/DTW timing gates."
            ),
        }

    # Constant-mean MAE diagnostic (should be ≈ √(2/π) ≈ 0.798 in unit-variance space)
    mae_const = float(
        torch.stack([(z - mean.to(z.device)).abs().mean() for z in gt_list]).mean().item()
    )
    payload = {
        "split": args.split,
        "n": len(gt_list),
        "normalized": norm is not None,
        "constant_mean_mae": mae_const,
        "constant_mean_mae_expected_approx": 0.7979,
        "baselines": reports,
        "takeaway": (
            "If any content-blind baseline passes legacy FR gates, those gates do not "
            "prove text→acoustic learning. Prefer Gate 0 (R²) and ASR CER (E4)."
        ),
    }
    print(json.dumps(payload, indent=2))
    out = expand(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote {out}")

    # Exit 0 always for audit; exit 3 if constant_mean failed timing (unexpected)
    if not reports["constant_mean"]["passes_legacy_fr_gates"]:
        print("WARNING: constant_mean did not pass timing gates (unexpected with oracle length)")
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

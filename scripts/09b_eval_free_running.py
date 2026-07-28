#!/usr/bin/env python3
"""Free-running stop/duration/DTW eval on a fixed val (or train) subset."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict

import torch
from torch.utils.data import DataLoader

from minispeech.codec.stats import LatentNormalizer
from minispeech.data.dataset import MiniSpeechDataset, collate_minispeech
from minispeech.eval.free_running import (
    aggregate_stability,
    duration_bucket,
    evaluate_free_running_latents,
)
from minispeech.infer.generate import load_checkpoint
from minispeech.train.losses import masked_l1
from minispeech.utils import expand, load_yaml


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data", default="configs/data_ruslan_local.yaml")
    ap.add_argument("--infer", default="configs/infer.yaml")
    ap.add_argument("--split", default="val", choices=("train", "val", "test"))
    ap.add_argument("--n", type=int, default=32)
    ap.add_argument("--out", default=None, help="JSON path (default: <ckpt_dir>/eval_free_running.json)")
    args = ap.parse_args()

    dcfg = load_yaml(args.data)["data"]
    stop_cfg = load_yaml(args.infer).get("infer", {}).get("stop", {})
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _ = load_checkpoint(args.ckpt, device)
    model.eval()
    try:
        norm = LatentNormalizer.from_file(dcfg["stats_path"])
    except Exception:
        norm = None

    ds = MiniSpeechDataset(
        manifest_path=dcfg["manifest"],
        latent_dir=dcfg["latent_dir"],
        text_hidden_dir=dcfg["text_hidden_dir"],
        split=args.split,
        max_utterances=args.n,
        exclude_tags=tuple(dcfg.get("exclude_tags", ("exclude",))),
    )
    loader = DataLoader(ds, batch_size=1, shuffle=False, collate_fn=collate_minispeech)

    rows = []
    ratios = []
    tf_l1s = []
    bucket_dtw: dict[str, list[float]] = defaultdict(list)

    for batch in loader:
        th = batch["text_hidden"].to(device)
        tm = batch["text_mask"].to(device)
        z_gt = batch["z_target"]
        if norm is not None:
            z_gt_n = norm.normalize(z_gt)
        else:
            z_gt_n = z_gt
        z_gt_n = z_gt_n.to(device)
        n_gt = int(batch["frame_mask"].sum().item())

        # Teacher-forced L1 diagnostic
        z_in = torch.zeros_like(z_gt_n)
        z_in[:, 1:] = z_gt_n[:, :-1]
        with torch.no_grad():
            out = model(z_in, th, frame_mask=batch["frame_mask"].to(device), text_mask=tm, z_target=z_gt_n)
            tf_l1s.append(
                float(
                    masked_l1(out.z_pred, z_gt_n, batch["frame_mask"].to(device)).item()
                )
            )

        z_pred, info = model.generate(
            th,
            tm,
            stop_threshold=float(stop_cfg.get("threshold", 0.45)),
            stop_consecutive=int(stop_cfg.get("consecutive", 3)),
            min_frac=float(stop_cfg.get("min_frac", 0.65)),
            max_frac=float(stop_cfg.get("max_frac", 1.6)),
        )
        m = evaluate_free_running_latents(
            z_pred,
            z_gt_n[:, :n_gt],
            stop_triggered=bool(info.get("stop_triggered")),
            hit_max_frames=bool(info.get("hit_max_frames")),
        )
        rows.append(m)
        ratios.append(m.duration_ratio)
        dur_s = n_gt / 7.5
        bucket_dtw[duration_bucket(dur_s)].append(m.dtw_l1)

    stab = aggregate_stability(rows)
    tf_mean = sum(tf_l1s) / max(len(tf_l1s), 1)
    report = {
        "n": len(rows),
        "split": args.split,
        "stop_cfg": stop_cfg,
        "tf_l1_mean": tf_mean,
        "dtw_over_tf": (stab.get("mean_dtw_l1", 0.0) / max(tf_mean, 1e-8)),
        "stability": stab,
        "bucket_mean_dtw": {k: sum(v) / len(v) for k, v in sorted(bucket_dtw.items())},
        "duration_ratios": ratios,
        "gates": {
            "stop_success_ge_0.95": stab.get("stop_success_rate", 0.0) >= 0.95,
            "runaway_le_0.02": stab.get("runaway_rate", 1.0) <= 0.02,
            "nan_rate_eq_0": stab.get("nan_rate", 1.0) == 0.0,
            "duration_in_band_ge_0.90": stab.get("duration_in_band_rate", 0.0) >= 0.90,
        },
    }
    print(json.dumps(report, indent=2))
    out_path = expand(args.out) if args.out else expand(args.ckpt).parent / "eval_free_running.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote {out_path}")

    ok = report["gates"]["stop_success_ge_0.95"] and report["gates"]["runaway_le_0.02"]
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())

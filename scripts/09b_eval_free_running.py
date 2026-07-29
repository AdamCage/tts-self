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
    fr_selection_key,
)
from minispeech.infer.generate import load_checkpoint
from minispeech.train.losses import masked_l1
from minispeech.utils import expand, load_yaml


def _parse_scales(raw: str | None, default: float) -> list[float]:
    if not raw:
        return [float(default)]
    return [float(x.strip()) for x in raw.split(",") if x.strip()]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data", default="configs/data_ruslan_local.yaml")
    ap.add_argument("--infer", default="configs/infer.yaml")
    ap.add_argument("--split", default="val", choices=("train", "val", "test"))
    ap.add_argument("--n", type=int, default=32)
    ap.add_argument("--duration-scale", type=float, default=None, help="Override infer.stop.duration_scale")
    ap.add_argument(
        "--sweep-duration-scale",
        default=None,
        help="Comma-separated scales to sweep (e.g. 0.95,1.0,1.05,1.1)",
    )
    ap.add_argument("--out", default=None, help="JSON path (default: <ckpt_dir>/eval_free_running.json)")
    args = ap.parse_args()

    dcfg = load_yaml(args.data)["data"]
    stop_cfg = dict(load_yaml(args.infer).get("infer", {}).get("stop", {}))
    if args.duration_scale is not None:
        stop_cfg["duration_scale"] = float(args.duration_scale)
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
    raw_loader = DataLoader(ds, batch_size=1, shuffle=False, collate_fn=collate_minispeech)

    class NormLoader:
        def __iter__(self):
            for batch in raw_loader:
                batch = dict(batch)
                if norm is not None:
                    batch["z_target"] = norm.normalize(batch["z_target"])
                yield batch

    infer_root = load_yaml(args.infer).get("infer", {})
    cfg_scale = float(infer_root.get("cfg_scale", 1.3))
    flow_steps = int(infer_root.get("flow_steps", 4))
    sampler = str(infer_root.get("sampler", "heun"))

    # TF L1 diagnostic (regression only; flow forward has no z_pred)
    tf_l1s: list[float] = []
    if getattr(model, "mode", "regression") == "regression":
        for batch in NormLoader():
            th = batch["text_hidden"].to(device)
            tm = batch["text_mask"].to(device)
            z_gt_n = batch["z_target"].to(device)
            z_in = torch.zeros_like(z_gt_n)
            z_in[:, 1:] = z_gt_n[:, :-1]
            with torch.no_grad():
                out = model(z_in, th, frame_mask=batch["frame_mask"].to(device), text_mask=tm, z_target=z_gt_n)
                if out.z_pred is None:
                    continue
                tf_l1s.append(float(masked_l1(out.z_pred, z_gt_n, batch["frame_mask"].to(device)).item()))
    tf_mean = sum(tf_l1s) / max(len(tf_l1s), 1) if tf_l1s else float("nan")

    scales = _parse_scales(args.sweep_duration_scale, float(stop_cfg.get("duration_scale", 1.0)))
    sweep_rows = []
    best_report = None
    best_key = None

    for scale in scales:
        # Per-utt generate for bucket DTW + consistency with prior reports
        rows = []
        ratios = []
        bucket_dtw: dict[str, list[float]] = defaultdict(list)
        dur_preds = []
        n_gts = []
        for batch in NormLoader():
            th = batch["text_hidden"].to(device)
            tm = batch["text_mask"].to(device)
            z_gt_n = batch["z_target"].to(device)
            n_gt = int(batch["frame_mask"].sum().item())
            gen_kwargs: dict = {
                "stop_threshold": float(stop_cfg.get("threshold", 0.45)),
                "stop_consecutive": int(stop_cfg.get("consecutive", 3)),
                "min_frac": float(stop_cfg.get("min_frac", 0.94)),
                "max_frac": float(stop_cfg.get("max_frac", 1.6)),
                "duration_scale": float(scale),
            }
            if getattr(model, "mode", "regression") == "flow":
                gen_kwargs.update(cfg_scale=cfg_scale, flow_steps=flow_steps, sampler=sampler)
            z_pred, info = model.generate(th, tm, **gen_kwargs)
            m = evaluate_free_running_latents(
                z_pred,
                z_gt_n[:, :n_gt],
                stop_triggered=bool(info.get("stop_triggered")),
                hit_max_frames=bool(info.get("hit_max_frames")),
            )
            rows.append(m)
            ratios.append(m.duration_ratio)
            bucket_dtw[duration_bucket(n_gt / 7.5)].append(m.dtw_l1)
            dur_preds.append(float(info.get("dur_pred", 0)))
            n_gts.append(n_gt)

        stab = aggregate_stability(rows)
        dtw_over_tf = None
        if tf_l1s and tf_mean == tf_mean:  # not NaN
            dtw_over_tf = stab.get("mean_dtw_l1", 0.0) / max(tf_mean, 1e-8)
        report = {
            "n": len(rows),
            "split": args.split,
            "mode": getattr(model, "mode", "regression"),
            "cfg_scale": cfg_scale if getattr(model, "mode", "") == "flow" else None,
            "flow_steps": flow_steps if getattr(model, "mode", "") == "flow" else None,
            "sampler": sampler if getattr(model, "mode", "") == "flow" else None,
            "stop_cfg": {**stop_cfg, "duration_scale": float(scale)},
            "tf_l1_mean": tf_mean,
            "dtw_over_tf": dtw_over_tf,
            "stability": stab,
            "bucket_mean_dtw": {k: sum(v) / len(v) for k, v in sorted(bucket_dtw.items())},
            "duration_ratios": ratios,
            "dur_preds": dur_preds,
            "n_gts": n_gts,
            "selection_key": list(fr_selection_key(stab)),
            "gates": {
                "stop_success_ge_0.95": stab.get("stop_success_rate", 0.0) >= 0.95,
                "runaway_le_0.02": stab.get("runaway_rate", 1.0) <= 0.02,
                "nan_rate_eq_0": stab.get("nan_rate", 1.0) == 0.0,
                "duration_in_band_ge_0.90": stab.get("duration_in_band_rate", 0.0) >= 0.90,
            },
        }
        sweep_rows.append(
            {
                "duration_scale": float(scale),
                "duration_in_band_rate": stab.get("duration_in_band_rate"),
                "stop_success_rate": stab.get("stop_success_rate"),
                "runaway_rate": stab.get("runaway_rate"),
                "mean_duration_ratio": stab.get("mean_duration_ratio"),
                "mean_dtw_l1": stab.get("mean_dtw_l1"),
                "gates": report["gates"],
            }
        )
        key = fr_selection_key(stab)
        if best_key is None or key > best_key:
            best_key = key
            best_report = report

    assert best_report is not None
    if len(scales) > 1:
        payload = {
            "sweep": sweep_rows,
            "best": best_report,
            "best_duration_scale": best_report["stop_cfg"]["duration_scale"],
        }
    else:
        payload = best_report

    print(json.dumps(payload, indent=2))
    out_path = expand(args.out) if args.out else expand(args.ckpt).parent / "eval_free_running.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote {out_path}")

    stab = best_report["stability"]
    ok = stab.get("stop_success_rate", 0.0) >= 0.95 and stab.get("runaway_rate", 1.0) <= 0.02
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())

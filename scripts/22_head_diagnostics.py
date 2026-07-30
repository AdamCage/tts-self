#!/usr/bin/env python3
"""E3: teacher-forced head diagnostics (R², variance ratio, lag corr, HF spectrum).

Uses NormLoader like ``scripts/09b_eval_free_running.py``. Regression ckpts only
for z_pred; flow ckpts report a clear skip message.
"""

from __future__ import annotations

import argparse
import json

import torch
from torch.utils.data import DataLoader

from minispeech.codec.stats import LatentNormalizer
from minispeech.data.dataset import MiniSpeechDataset, collate_minispeech
from minispeech.infer.generate import load_checkpoint
from minispeech.train.losses import masked_l1
from minispeech.utils import expand, load_yaml


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data", default="configs/data_ruslan_local.yaml")
    ap.add_argument("--split", default="val", choices=("train", "val", "test"))
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    dcfg = load_yaml(args.data)["data"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _ = load_checkpoint(args.ckpt, device)
    model.eval()
    if getattr(model, "mode", "regression") != "regression":
        print(json.dumps({"error": "head diagnostics require regression mode (z_pred)", "mode": model.mode}))
        return 2

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
    raw_loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_minispeech)

    class NormLoader:
        def __iter__(self):
            for batch in raw_loader:
                batch = dict(batch)
                if norm is not None:
                    batch["z_target"] = norm.normalize(batch["z_target"])
                    z_in = torch.zeros_like(batch["z_target"])
                    z_in[:, 1:] = batch["z_target"][:, :-1]
                    batch["z_input"] = z_in
                yield batch

    preds, gts, masks = [], [], []
    tf_l1s: list[float] = []
    with torch.inference_mode():
        for batch in NormLoader():
            th = batch["text_hidden"].to(device)
            tm = batch["text_mask"].to(device)
            z_gt = batch["z_target"].to(device)
            fm = batch["frame_mask"].to(device)
            z_in = batch["z_input"].to(device)
            out = model(z_in, th, frame_mask=fm, text_mask=tm, z_target=z_gt)
            if out.z_pred is None:
                continue
            preds.append(out.z_pred.cpu())
            gts.append(z_gt.cpu())
            masks.append(fm.cpu())
            tf_l1s.append(float(masked_l1(out.z_pred, z_gt, fm).item()))

    if not preds:
        print(json.dumps({"error": "no predictions collected"}))
        return 2

    from minispeech.eval.head_diagnostics import (
        lag_correlations,
        per_dim_r2,
        r2_score,
        temporal_highfreq_attenuation,
        variance_ratio,
    )

    # Re-run on concatenated padded tensors with masks (correct)
    # Pad to max T
    max_t = max(p.shape[1] for p in preds)
    d = preds[0].shape[-1]
    b_total = sum(p.shape[0] for p in preds)
    P = torch.zeros(b_total, max_t, d)
    G = torch.zeros(b_total, max_t, d)
    M = torch.zeros(b_total, max_t, dtype=torch.bool)
    off = 0
    for p, g, m in zip(preds, gts, masks):
        b, t, _ = p.shape
        P[off : off + b, :t] = p
        G[off : off + b, :t] = g
        M[off : off + b, :t] = m
        off += b

    tf_mean = sum(tf_l1s) / max(len(tf_l1s), 1)
    const_mae = 0.7979  # E|N(0,1)| for unit-variance normalized latents
    report = {
        "ckpt": str(args.ckpt),
        "split": args.split,
        "n_batches": len(preds),
        "n_utts": int(M.any(dim=1).sum().item()) if M.ndim == 2 else int(b_total),
        "normalized": norm is not None,
        "tf_l1_mean": tf_mean,
        "tf_l1_over_constant_mae": tf_mean / const_mae,
        "gate0_r2_ge_0.28": r2_score(P, G, M) >= 0.28,
        "r2_overall": r2_score(P, G, M),
        "r2_per_dim_mean": float(per_dim_r2(P, G, M).nanmean().item()),
        "r2_per_dim": [float(x) for x in per_dim_r2(P, G, M).tolist()],
        "var_ratio": variance_ratio(P, G, M),
        **lag_correlations(P, G, M),
        **temporal_highfreq_attenuation(P, G, M),
        "interpretation": {
            "r2_near_zero": "head explains little beyond corpus mean",
            "var_ratio_lt_0.3": "strong variance collapse / oversmoothing",
            "corr_gt_tm1_gt_corr_gt_t": "head behaves like copy-prev predictor",
            "hf_power_ratio_lt_0.5": "high-frequency temporal dynamics attenuated",
        },
    }
    # Drop huge per-dim from console; keep in file
    console = {k: v for k, v in report.items() if k != "r2_per_dim"}
    print(json.dumps(console, indent=2))

    out = expand(args.out) if args.out else expand(args.ckpt).parent / "head_diagnostics.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote {out}")
    return 0 if report["r2_overall"] == report["r2_overall"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

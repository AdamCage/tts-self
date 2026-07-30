#!/usr/bin/env python3
"""E5: dump cross-attn diagonality vs chance for ~16 utterances.

Saves JSON; optional matplotlib heatmaps if matplotlib is importable.
"""

from __future__ import annotations

import argparse
import json

import torch
from torch.utils.data import DataLoader

from minispeech.codec.stats import LatentNormalizer
from minispeech.data.dataset import MiniSpeechDataset, collate_minispeech
from minispeech.eval.alignment_metrics import diagonality_vs_chance
from minispeech.infer.generate import load_checkpoint
from minispeech.utils import expand, load_yaml


def _maybe_plot(attn_2d: torch.Tensor, path, title: str) -> bool:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return False
    a = attn_2d.detach().float().cpu().numpy()
    fig, ax = plt.subplots(figsize=(6, 4))
    im = ax.imshow(a.T, aspect="auto", origin="lower", interpolation="nearest")
    ax.set_xlabel("frame t")
    ax.set_ylabel("text token l")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return True


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data", default="configs/data_ruslan_local.yaml")
    ap.add_argument("--split", default="val", choices=("train", "val", "test"))
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--out", default="out/align_audit.json")
    ap.add_argument("--plot-dir", default=None, help="If set, try matplotlib heatmaps")
    args = ap.parse_args()

    dcfg = load_yaml(args.data)["data"]
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

    utt_rows = []
    plot_dir = expand(args.plot_dir) if args.plot_dir else None
    plotted = 0
    with torch.inference_mode():
        for batch in loader:
            batch = dict(batch)
            if norm is not None:
                batch["z_target"] = norm.normalize(batch["z_target"])
                z_in = torch.zeros_like(batch["z_target"])
                z_in[:, 1:] = batch["z_target"][:, :-1]
                batch["z_input"] = z_in
            th = batch["text_hidden"].to(device)
            tm = batch["text_mask"].to(device)
            z_gt = batch["z_target"].to(device)
            fm = batch["frame_mask"].to(device)
            z_in = batch["z_input"].to(device)
            out = model(z_in, th, frame_mask=fm, text_mask=tm, z_target=z_gt)
            if out.cross_attn is None:
                continue
            stats = diagonality_vs_chance(out.cross_attn, fm, tm)
            uids = batch.get("utt_ids") or batch.get("utt_id")
            uid = uids[0] if isinstance(uids, (list, tuple)) else str(uids)
            row = {"utt_id": uid, **stats["rows"][0]}
            utt_rows.append(row)
            if plot_dir is not None:
                attn = out.cross_attn
                if attn.ndim == 4:
                    attn = attn.mean(dim=1)
                t_eff = int(fm[0].sum().item())
                l_eff = int(tm[0].sum().item())
                ok = _maybe_plot(
                    attn[0, :t_eff, :l_eff],
                    plot_dir / f"{uid}_cross_attn.png",
                    f"{uid} diag={row['diagonality']:.3f} chance={row['chance']:.3f}",
                )
                if ok:
                    plotted += 1

    if not utt_rows:
        print(json.dumps({"error": "no cross_attn collected"}))
        return 2

    report = {
        "ckpt": str(args.ckpt),
        "split": args.split,
        "n": len(utt_rows),
        "mean_diagonality": sum(r["diagonality"] for r in utt_rows) / len(utt_rows),
        "mean_chance": sum(r["chance"] for r in utt_rows) / len(utt_rows),
        "mean_vs_chance": sum(r["vs_chance"] for r in utt_rows) / len(utt_rows),
        "plots_written": plotted,
        "utterances": utt_rows,
        "note": (
            "If mean_diagonality ≈ mean_chance (~0.15–0.20 for typical T,L), "
            "cross-attn is not learning monotonic alignment."
        ),
    }
    print(json.dumps({k: v for k, v in report.items() if k != "utterances"}, indent=2))
    out = expand(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

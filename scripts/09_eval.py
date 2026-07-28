#!/usr/bin/env python3
"""Eval: text-usage gates, optional WER, alignment diagonality."""

from __future__ import annotations

import argparse
import json

import torch
from torch.utils.data import DataLoader

from minispeech.data.dataset import MiniSpeechDataset, collate_minispeech
from minispeech.eval.text_usage import (
    text_mask_test,
    text_prefix_counterfactual_test,
    text_shuffle_test,
)
from minispeech.infer.generate import load_checkpoint
from minispeech.train.guided_attn import attention_diagonality
from minispeech.utils import expand, load_yaml


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data", default="configs/data_ruslan.yaml")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument(
        "--split",
        default="train",
        choices=("train", "val", "test"),
        help="S1 overfit gate must use train (32 utts); use val after S2+",
    )
    ap.add_argument("--mask-min-rise", type=float, default=0.20, help="0.35 after S2")
    ap.add_argument("--prefix-min-ratio", type=float, default=3.0)
    args = ap.parse_args()
    dcfg = load_yaml(args.data)["data"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _ = load_checkpoint(args.ckpt, device)
    ds = MiniSpeechDataset(
        manifest_path=dcfg["manifest"],
        latent_dir=dcfg["latent_dir"],
        text_hidden_dir=dcfg["text_hidden_dir"],
        split=args.split,
        max_utterances=32,
        exclude_tags=tuple(dcfg.get("exclude_tags", ("exclude",))),
    )
    if len(ds) == 0 and args.split != "train":
        ds = MiniSpeechDataset(
            manifest_path=dcfg["manifest"],
            latent_dir=dcfg["latent_dir"],
            text_hidden_dir=dcfg["text_hidden_dir"],
            split="train",
            max_utterances=32,
            exclude_tags=tuple(dcfg.get("exclude_tags", ("exclude",))),
        )
    loader = DataLoader(ds, batch_size=args.batch_size, collate_fn=collate_minispeech)
    batch = next(iter(loader))
    # normalize if stats present
    try:
        from minispeech.codec.stats import LatentNormalizer

        norm = LatentNormalizer.from_file(dcfg["stats_path"])
        batch["z_target"] = norm.normalize(batch["z_target"])
        z_in = torch.zeros_like(batch["z_target"])
        z_in[:, 1:] = batch["z_target"][:, :-1]
        batch["z_input"] = z_in
    except Exception:
        pass

    shuffle = text_shuffle_test(model, batch, device)
    mask = text_mask_test(model, batch, device, min_rise=args.mask_min_rise)
    prefix = text_prefix_counterfactual_test(
        model, batch, device, min_ratio=args.prefix_min_ratio
    )
    batch_d = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
    out = model(
        batch_d["z_input"],
        batch_d["text_hidden"],
        frame_mask=batch_d["frame_mask"],
        text_mask=batch_d["text_mask"],
        z_target=batch_d["z_target"],
    )
    diag = attention_diagonality(out.cross_attn, batch_d["frame_mask"]) if out.cross_attn is not None else None
    report = {
        "split": args.split if len(ds) else "train",
        "text_shuffle": shuffle,
        "text_mask": mask,
        "text_prefix_counterfactual": prefix,
        "attention_diagonality": diag,
        # Plan C.4: all three counterfactuals required.
        "s1_text_usage_pass": bool(
            shuffle.get("pass") and mask.get("pass") and prefix.get("pass")
        ),
        "s2_text_usage_pass": bool(
            shuffle.get("pass")
            and mask.get("pass")
            and prefix.get("pass")
            and float(mask.get("relative_rise", 0.0)) >= 0.35
        ),
    }
    print(json.dumps(report, indent=2))
    out_path = expand(args.ckpt).parent / "eval_text_usage.json"
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return 0 if report["s1_text_usage_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

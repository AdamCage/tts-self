#!/usr/bin/env python3
"""N2 text-usage gate (A5). JSON keys match scripts/09_eval.py.

Exit code 2 when ``s1_text_usage_pass`` is false.
"""

from __future__ import annotations

import argparse
import json
from typing import Any

import torch
from transformers import AutoTokenizer

from minispeech.codec.stats import LatentNormalizer
from minispeech.eval.text_usage import (
    n2_attention_diagonality,
    n2_text_mask_test,
    n2_text_prefix_counterfactual_test,
    n2_text_shuffle_test,
)
from minispeech.train.n2_trainer import (
    load_n2_checkpoint,
    make_n2_loaders,
    resolve_data_cfg,
)
from minispeech.utils import expand, load_yaml


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--config", default="configs/n2_qwen_latent_v01.yaml")
    ap.add_argument("--data", default=None)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--split", default="train", choices=("train", "val", "test"))
    ap.add_argument("--max-utts", type=int, default=32)
    ap.add_argument("--mask-min-rise", type=float, default=0.20)
    ap.add_argument("--prefix-min-ratio", type=float, default=3.0)
    ap.add_argument("--out", default="out/n2_text_usage.json")
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    data_cfg = resolve_data_cfg(cfg, args.data)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _ = load_n2_checkpoint(args.ckpt, device)
    qcfg = cfg.get("qwen", {})
    tok = AutoTokenizer.from_pretrained(
        str(qcfg.get("model_id")), trust_remote_code=True
    )
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    try:
        normalizer = LatentNormalizer.from_file(data_cfg["stats_path"])
    except Exception:
        normalizer = None

    stage = {"name": "S1", "max_utterances": args.max_utts}
    train_loader, val_loader, _ = make_n2_loaders(
        cfg,
        data_cfg,
        stage,
        args.batch_size,
        tokenizer=tok,
        bos_latent=model.backbone.bos_latent.detach().cpu(),
        normalizer=normalizer,
    )
    loader = train_loader if args.split == "train" else (val_loader or train_loader)
    batch = next(iter(loader))

    shuffle = n2_text_shuffle_test(model, batch, device)
    mask = n2_text_mask_test(model, batch, device, min_rise=args.mask_min_rise)
    prefix = n2_text_prefix_counterfactual_test(
        model, batch, device, min_ratio=args.prefix_min_ratio
    )
    diag = n2_attention_diagonality(model, batch, device)

    report: dict[str, Any] = {
        "split": args.split,
        "ckpt": str(args.ckpt),
        "text_shuffle": shuffle,
        "text_mask": mask,
        "text_prefix_counterfactual": prefix,
        "attention_diagonality": diag,
        "attention_diagonality_note": (
            "N2: last transformer layer, mean over heads, acoustic-query → "
            "text-key submatrix; then Part-1 attention_diagonality. "
            "Null if backbone cannot emit attentions."
        ),
        # Plan exit gate (n2_first_execution §7 / A5): shuffle + mask must break
        # synthesis. Prefix counterfactual is optional diagnostics ("при необходимости").
        "s1_text_usage_pass": bool(shuffle.get("pass") and mask.get("pass")),
        "s2_text_usage_pass": bool(
            shuffle.get("pass")
            and mask.get("pass")
            and float(mask.get("relative_rise", 0.0)) >= 0.35
        ),
        "prefix_optional_pass": bool(prefix.get("pass")),
    }
    print(json.dumps(report, indent=2))
    out_path = expand(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    ckpt_side = expand(args.ckpt).parent / "eval_text_usage.json"
    ckpt_side.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return 0 if report["s1_text_usage_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

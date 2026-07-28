#!/usr/bin/env python3
"""Inference CLI."""

from __future__ import annotations

import argparse

import torch
from transformers import AutoTokenizer

from minispeech.codec.stats import LatentNormalizer
from minispeech.codec.wrapper import VibeVoiceCodec
from minispeech.constants import QWEN_MODEL_ID
from minispeech.infer.generate import load_checkpoint, save_wav, synthesize
from minispeech.utils import expand, load_yaml


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--text", required=True)
    ap.add_argument("--ckpt", default="runs/v01_regression/best.pt")
    ap.add_argument("--config", default="configs/infer.yaml")
    ap.add_argument("--out", default="out/synth.wav")
    args = ap.parse_args()
    cfg = load_yaml(args.config)["infer"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _ = load_checkpoint(args.ckpt, device)
    codec = VibeVoiceCodec(model_id=cfg.get("codec", "microsoft/VibeVoice-AcousticTokenizer"))
    normalizer = LatentNormalizer.from_file(cfg["stats"])
    tok = AutoTokenizer.from_pretrained(cfg.get("qwen", QWEN_MODEL_ID), trust_remote_code=True)
    try:
        from transformers import Qwen3_5TextModel

        text_model = Qwen3_5TextModel.from_pretrained(
            cfg.get("qwen", QWEN_MODEL_ID), torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
        )
    except Exception:
        from transformers import AutoModel

        text_model = AutoModel.from_pretrained(
            cfg.get("qwen", QWEN_MODEL_ID), torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
        )
    audio, info = synthesize(
        args.text,
        model,
        codec,
        normalizer,
        tok,
        text_model,
        cfg_scale=cfg.get("cfg_scale", 1.3),
        flow_steps=cfg.get("flow_steps", 4),
        use_stress=cfg.get("stress", {}).get("use_accentuator", True),
        stop=cfg.get("stop"),
    )
    save_wav(args.out, audio)
    print(info)
    print(f"Wrote {expand(args.out)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

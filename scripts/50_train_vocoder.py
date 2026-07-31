#!/usr/bin/env python3
"""Fine-tune the local SpeechBrain LibriTTS HiFi-GAN on RUSLAN mel-to-wav pairs."""

from __future__ import annotations

import argparse

import torch

from minispeech.models.hifigan_disc import HiFiGanDiscriminators
from minispeech.train.vocoder_trainer import (
    VocoderTrainer,
    load_generator_checkpoint,
    make_vocoder_loaders,
)
from minispeech.utils import expand, load_yaml, set_seed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/vocoder_ft_v01.yaml")
    parser.add_argument("--data", default="configs/data_ruslan_local.yaml")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--resume", default=None, help="Resume a vocoder last.pt checkpoint")
    parser.add_argument("--run-dir", default=None, help="Override train.run_dir")
    parser.add_argument(
        "--generator-init",
        default=None,
        help="Override train.generator_init (default: ~/models/hifigan/generator.pt)",
    )
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    data_cfg = load_yaml(args.data)["data"]
    train_cfg = cfg.setdefault("train", {})
    if args.run_dir:
        train_cfg["run_dir"] = args.run_dir
    if args.generator_init:
        train_cfg["generator_init"] = args.generator_init
    if data_cfg.get("split", {}).get("strategy") != "contiguous_ranges":
        raise SystemExit(
            "vocoder fine-tuning requires data.split.strategy=contiguous_ranges "
            "to retain the project's book-range train/val convention"
        )
    init_path = train_cfg.get("generator_init")
    if not init_path:
        raise SystemExit("train.generator_init is required; do not train the vocoder from scratch")
    if not expand(init_path).is_file():
        raise SystemExit(f"missing generator checkpoint: {expand(init_path)}")

    set_seed(int(train_cfg.get("seed", 42)))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        print("warning: CUDA is unavailable; HiFi-GAN fine-tuning will be very slow")
    generator = load_generator_checkpoint(
        init_path, inference_padding=int(train_cfg.get("generator_inference_padding", 0))
    )
    batch_size = args.batch_size or int(train_cfg.get("batch_size", 8))
    train_loader, val_loader = make_vocoder_loaders(data_cfg, cfg, batch_size=batch_size)
    print(
        f"target={cfg.get('data', {}).get('target', 'mel80_hifigan')} "
        f"train_pairs={len(train_loader.dataset)} val_pairs={len(val_loader.dataset)} "
        f"batch_size={batch_size}"
    )
    trainer = VocoderTrainer(
        generator,
        HiFiGanDiscriminators(),
        train_loader,
        val_loader,
        cfg,
        device,
        target=str(cfg.get("data", {}).get("target", "mel80_hifigan")),
        resume=args.resume,
    )
    steps = args.steps or int(train_cfg.get("steps", 100_000))
    print(trainer.train_steps(steps))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

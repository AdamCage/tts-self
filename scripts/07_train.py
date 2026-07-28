#!/usr/bin/env python3
"""Train MiniSpeech (regression v0.1 or flow v0.2)."""

from __future__ import annotations

import argparse

import torch

from minispeech.codec.stats import LatentNormalizer
from minispeech.train.trainer import Trainer, build_model_from_cfg, make_loaders
from minispeech.utils import auto_batch_size, deep_merge, load_yaml, set_seed


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/talker_v01.yaml")
    ap.add_argument("--data", default="configs/data_ruslan.yaml")
    ap.add_argument("--mode", choices=["regression", "flow"], default="regression")
    ap.add_argument("--stage", default=None, help="S1/S2/S3/S4 or all")
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument(
        "--init-from",
        default=None,
        help="Load weights from prior stage ckpt (e.g. runs/.../best.pt) before training",
    )
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    data_cfg = load_yaml(args.data)["data"]
    set_seed(cfg.get("train", {}).get("seed", 42))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = build_model_from_cfg(cfg, mode=args.mode)
    if args.init_from:
        from minispeech.infer.generate import load_checkpoint
        from minispeech.utils import expand

        model, _ = load_checkpoint(expand(args.init_from), device)
        print(f"init_from={args.init_from}")
    # Normalize latents are assumed pre-normalized in shards OR we normalize in collate.
    # For simplicity expect encode script stored RAW and Dataset returns raw;
    # Trainer expects normalized — apply via wrapping if stats exist.
    try:
        normalizer = LatentNormalizer.from_file(data_cfg["stats_path"])
    except Exception:
        normalizer = None

    stages = cfg.get("train", {}).get("stages", [{"name": "S1", "max_utterances": 32, "steps": 2000}])
    if args.stage and args.stage != "all":
        stages = [s for s in stages if s["name"] == args.stage]

    bcfg = cfg.get("train", {}).get("batch", {})
    if args.batch_size:
        bs = args.batch_size
    elif bcfg.get("mode") == "auto":

        def probe(b):
            tl, _ = make_loaders(
                data_cfg,
                stages[0],
                b,
                model_cfg=cfg.get("model", {}),
                train_section=cfg.get("train", {}),
            )
            batch = next(iter(tl))
            batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
            if normalizer:
                batch["z_target"] = normalizer.normalize(batch["z_target"])
                batch["z_input"] = normalizer.normalize(batch["z_input"])
                batch["z_input"][:, 0] = 0
            model.to(device)
            out = model(
                batch["z_input"],
                batch["text_hidden"],
                frame_mask=batch["frame_mask"],
                text_mask=batch["text_mask"],
                z_target=batch["z_target"],
            )
            loss = out.stop_logits.float().mean()
            if out.z_pred is not None:
                loss = loss + out.z_pred.float().mean()
            loss.backward()
            model.zero_grad(set_to_none=True)

        bs = auto_batch_size(probe, bcfg.get("candidates", [8, 16, 32, 64]), bcfg.get("target_vram_gb", 14.5))
        print(f"auto batch_size={bs}")
    else:
        bs = 8

    for stage in stages:
        train_loader, val_loader = make_loaders(
            data_cfg,
            stage,
            bs,
            model_cfg=cfg.get("model", {}),
            train_section=cfg.get("train", {}),
        )
        # Wrap loader to normalize
        class NormLoader:
            def __init__(self, loader, norm):
                self.loader = loader
                self.norm = norm
                self.dataset = getattr(loader, "dataset", None)

            def __iter__(self):
                for batch in self.loader:
                    if self.norm is not None:
                        batch["z_target"] = self.norm.normalize(batch["z_target"])
                        z_in = batch["z_target"].clone()
                        # rebuild shift in normalized space
                        z_shift = torch.zeros_like(z_in)
                        z_shift[:, 1:] = z_in[:, :-1]
                        batch["z_input"] = z_shift
                    yield batch

            def __len__(self):
                return len(self.loader)

        tl = NormLoader(train_loader, normalizer)
        vl = NormLoader(val_loader, normalizer) if val_loader else None
        trainer = Trainer(model, tl, vl, cfg, device, normalizer=normalizer)
        steps = args.steps or stage.get("steps", 1000)
        metrics = trainer.train_steps(steps, stage=stage["name"])
        print(stage["name"], metrics)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

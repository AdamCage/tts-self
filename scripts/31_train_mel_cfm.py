#!/usr/bin/env python3
"""Train Track A CFM / rectified-flow mel decoder (plan a5; no VibeVoice)."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from minispeech.data.mel_dataset import build_vocab_from_forced
from minispeech.frontend.g2p import text_to_phonemes
from minispeech.frontend.phone_vocab import PhoneVocab
from minispeech.models.mel_cfm import MelCFM
from minispeech.train.mel_cfm_trainer import MelCFMTrainer, make_mel_loaders
from minispeech.utils import expand, load_yaml, set_seed


def _build_vocab(
    data_cfg: dict,
    max_utts: int | None,
    *,
    gap_mode: str = "expand_neighbor",
) -> PhoneVocab:
    forced = data_cfg.get("forced_path")
    if forced and expand(forced).is_file():
        return build_vocab_from_forced(
            expand(forced), max_utts=max_utts, gap_mode=gap_mode  # type: ignore[arg-type]
        )
    from minispeech.data.manifest import filter_split, read_manifest

    rows = filter_split(read_manifest(expand(data_cfg["manifest"])), "train")
    if max_utts is not None:
        rows = rows[:max_utts]
    v = PhoneVocab()
    for u in rows:
        v.add_phones(text_to_phonemes(u.text_stressed or u.text_raw))
    return v


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/mel_cfm_v01.yaml")
    ap.add_argument("--data", default="configs/data_ruslan_local.yaml")
    ap.add_argument("--stage", default="S1", help="S1/S2/S3 or all")
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument(
        "--allow-missing-mel",
        action="store_true",
        help="Dry-run with zero mels (smoke only; not for real training)",
    )
    ap.add_argument("--vocab-out", default=None, help="Save PhoneVocab JSON")
    ap.add_argument(
        "--vocab",
        default=None,
        help="Load existing phone_vocab.json (e.g. from regression S3 for warm-start)",
    )
    ap.add_argument("--run-dir", default=None, help="Override train.run_dir")
    ap.add_argument(
        "--gap-mode",
        default=None,
        choices=["expand_neighbor", "sil_fill"],
        help="CTC gap packing (default: config train.data.gap_mode)",
    )
    ap.add_argument(
        "--init-ckpt",
        default=None,
        help="Warm-start: MelCFM ckpt, or MelBaseline regression best.pt (encoder/dur)",
    )
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    cfg["mode"] = "cfm"
    data_cfg = load_yaml(args.data)["data"]
    set_seed(cfg.get("train", {}).get("seed", 42))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.run_dir:
        cfg.setdefault("train", {})["run_dir"] = args.run_dir

    stages = cfg.get("train", {}).get("stages", [{"name": "S1", "max_utterances": 32, "steps": 2000}])
    if args.stage and args.stage != "all":
        stages = [s for s in stages if s["name"] == args.stage]
    if not stages:
        raise SystemExit(f"no stages match {args.stage!r}")

    gap_mode = args.gap_mode or (
        cfg.get("train", {}).get("data", {}).get("gap_mode")
        or cfg.get("model", {}).get("mel", {}).get("gap_mode")
        or "expand_neighbor"
    )
    cfg.setdefault("train", {}).setdefault("data", {})["gap_mode"] = gap_mode

    vocab_path = expand(args.vocab_out or Path(cfg["train"]["run_dir"]) / "phone_vocab.json")
    vocab_path.parent.mkdir(parents=True, exist_ok=True)
    if args.vocab:
        vocab = PhoneVocab.load(expand(args.vocab))
        print(f"loaded vocab {args.vocab} size={len(vocab)}")
    else:
        # Prefer full forced coverage when warm-starting a larger regression vocab.
        vocab_cap = stages[0].get("max_utterances")
        if args.init_ckpt or cfg.get("train", {}).get("init_ckpt"):
            vocab_cap = None
        vocab = _build_vocab(data_cfg, vocab_cap, gap_mode=gap_mode)
    vocab.save(vocab_path)
    print(f"vocab_size={len(vocab)} gap_mode={gap_mode} sil_in_vocab={'sil' in vocab.stoi} saved={vocab_path}")

    mcfg = cfg.get("model", {}).get("mel", {})
    model = MelCFM(
        vocab_size=max(len(vocab), int(mcfg.get("vocab_size", 512))),
        d_model=int(mcfg.get("d_model", 256)),
        n_mels=int(mcfg.get("n_mels", 80)),
        enc_layers=int(mcfg.get("enc_layers", 3)),
        n_heads=int(mcfg.get("n_heads", 4)),
        ffn_dim=int(mcfg.get("ffn_dim", 1024)),
        dropout=float(mcfg.get("dropout", 0.1)),
        upsample=str(mcfg.get("upsample", "hard")),  # type: ignore[arg-type]
        predict_duration=bool(mcfg.get("predict_duration", True)),
        pad_id=vocab.pad_id,
        flow_hidden=int(mcfg.get("flow_hidden", 512)),
        flow_layers=int(mcfg.get("flow_layers", 4)),
        time_embed_dim=int(mcfg.get("time_embed_dim", 256)),
        cfg_dropout=float(mcfg.get("cfg_dropout", 0.1)),
        time_sampling=str(mcfg.get("time_sampling", "logit_normal")),
    )

    init_ckpt = args.init_ckpt or cfg.get("train", {}).get("init_ckpt")
    if init_ckpt:
        ckpt = torch.load(expand(init_ckpt), map_location=device, weights_only=False)
        sd = ckpt.get("model") or ckpt.get("ema") or ckpt
        if isinstance(sd, dict) and "state_dict" in sd and "model" not in ckpt:
            sd = sd["state_dict"]
        mode = ckpt.get("mode") or (ckpt.get("cfg") or {}).get("mode")
        model_sd = model.state_dict()
        if mode == "cfm" or any(k.startswith("flow.") or k.startswith("backbone.") for k in sd):
            filtered = {
                k: v
                for k, v in sd.items()
                if k in model_sd and tuple(model_sd[k].shape) == tuple(v.shape)
            }
            missing, unexpected = model.load_state_dict(filtered, strict=False)
            print(
                f"init_ckpt={init_ckpt} (cfm) loaded={len(filtered)}/{len(model_sd)} "
                f"missing={len(missing)} unexpected={len(unexpected)}"
            )
        else:
            # Regression MelBaseline keys → backbone.*
            missing, unexpected = model.load_backbone_from_regression(sd, strict=False)
            n_bb = len(model.backbone.state_dict())
            print(
                f"init_ckpt={init_ckpt} (regression→backbone) "
                f"missing={len(missing)} unexpected={len(unexpected)} backbone_keys={n_bb}"
            )

    if args.allow_missing_mel:
        cfg.setdefault("train", {}).setdefault("data", {})["allow_missing_mel"] = True

    bcfg = cfg.get("train", {}).get("batch", {})
    bs = args.batch_size or int(bcfg.get("size", 8))

    for stage in stages:
        train_loader, val_loader = make_mel_loaders(
            data_cfg,
            stage,
            bs,
            model_cfg=cfg.get("model", {}),
            train_section=cfg.get("train", {}),
            vocab=vocab,
        )
        if len(vocab) > model.emb.num_embeddings:
            raise SystemExit(
                f"vocab grew to {len(vocab)} > model emb {model.emb.num_embeddings}; "
                "rebuild with larger vocab_size"
            )
        trainer = MelCFMTrainer(model, train_loader, val_loader, cfg, device)
        steps = args.steps or stage.get("steps", 1000)
        metrics = trainer.train_steps(steps, stage=stage["name"])
        print(stage["name"], metrics)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

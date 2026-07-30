#!/usr/bin/env python3
"""Train Track A non-AR mel baseline (S1-scale path; no VibeVoice)."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from minispeech.data.mel_dataset import build_vocab_from_forced
from minispeech.frontend.g2p import text_to_phonemes
from minispeech.frontend.phone_vocab import PhoneVocab
from minispeech.models.mel_baseline import MelBaseline
from minispeech.train.mel_trainer import MelTrainer, make_mel_loaders
from minispeech.utils import expand, load_yaml, set_seed


def _build_vocab(
    data_cfg: dict,
    max_utts: int | None,
    *,
    gap_mode: str = "expand_neighbor",
    keep_stress: bool = False,
) -> PhoneVocab:
    from minispeech.data.manifest import filter_split, read_manifest

    forced = data_cfg.get("forced_path")
    text_by_utt: dict[str, str] | None = None
    if keep_stress:
        rows = filter_split(read_manifest(expand(data_cfg["manifest"])), "train")
        text_by_utt = {u.utt_id: (u.text_stressed or u.text_raw) for u in rows}
    if forced and expand(forced).is_file():
        return build_vocab_from_forced(
            expand(forced),
            max_utts=max_utts,
            gap_mode=gap_mode,  # type: ignore[arg-type]
            keep_stress=keep_stress,
            text_by_utt=text_by_utt,
        )
    # Fallback: scan manifest texts via G2P (grapheme fallback ok)
    rows = filter_split(read_manifest(expand(data_cfg["manifest"])), "train")
    if max_utts is not None:
        rows = rows[:max_utts]
    v = PhoneVocab()
    for u in rows:
        v.add_phones(text_to_phonemes(u.text_stressed or u.text_raw))
    return v


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/mel_baseline_v01.yaml")
    ap.add_argument("--data", default="configs/data_ruslan_local.yaml")
    ap.add_argument("--stage", default="S1", help="S1/S2 or all")
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument(
        "--allow-missing-mel",
        action="store_true",
        help="Dry-run with zero mels (smoke only; not for real training)",
    )
    ap.add_argument("--vocab-out", default=None, help="Save PhoneVocab JSON")
    ap.add_argument(
        "--run-dir",
        default=None,
        help="Override train.run_dir (e.g. runs/mel_baseline_v01_s1b)",
    )
    ap.add_argument(
        "--gap-mode",
        default=None,
        choices=["expand_neighbor", "sil_fill", "none", "raw"],
        help="CTC gap packing (none/raw = leave blank gaps; paper2 ablation)",
    )
    ap.add_argument(
        "--init-ckpt",
        default=None,
        help="Warm-start model(+EMA if present) from a prior mel best.pt (e.g. S2)",
    )
    args = ap.parse_args()

    cfg = load_yaml(args.config)
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
    mcfg = cfg.get("model", {}).get("mel", {})
    keep_stress = bool(
        cfg.get("train", {}).get("data", {}).get("keep_stress", mcfg.get("keep_stress", False))
    )
    cfg.setdefault("train", {}).setdefault("data", {})["keep_stress"] = keep_stress

    # Vocab from forced.json should cover the hour-cut; ignore max_utterances-only stages.
    vocab_cap = stages[0].get("max_utterances")
    vocab = _build_vocab(
        data_cfg, vocab_cap, gap_mode=gap_mode, keep_stress=keep_stress
    )
    vocab_path = expand(args.vocab_out or Path(cfg["train"]["run_dir"]) / "phone_vocab.json")
    vocab_path.parent.mkdir(parents=True, exist_ok=True)
    vocab.save(vocab_path)
    has_sil = "sil" in vocab.stoi
    print(
        f"vocab_size={len(vocab)} gap_mode={gap_mode} keep_stress={keep_stress} "
        f"sil_in_vocab={has_sil} saved={vocab_path}"
    )

    model = MelBaseline(
        vocab_size=max(len(vocab), int(mcfg.get("vocab_size", 512))),
        d_model=int(mcfg.get("d_model", 256)),
        n_mels=int(mcfg.get("n_mels", 80)),
        enc_layers=int(mcfg.get("enc_layers", 3)),
        dec_layers=int(mcfg.get("dec_layers", 4)),
        n_heads=int(mcfg.get("n_heads", 4)),
        ffn_dim=int(mcfg.get("ffn_dim", 1024)),
        dropout=float(mcfg.get("dropout", 0.1)),
        decoder=str(mcfg.get("decoder", "conv")),  # type: ignore[arg-type]
        upsample=str(mcfg.get("upsample", "hard")),  # type: ignore[arg-type]
        predict_duration=bool(mcfg.get("predict_duration", True)),
        predict_pitch=bool(mcfg.get("predict_pitch", False)),
        predict_energy=bool(mcfg.get("predict_energy", False)),
        use_postnet=bool(mcfg.get("use_postnet", False)),
        pad_id=vocab.pad_id,
    )

    init_ckpt = args.init_ckpt or cfg.get("train", {}).get("init_ckpt")
    if init_ckpt:
        ckpt = torch.load(expand(init_ckpt), map_location=device, weights_only=False)
        sd = ckpt.get("model") or ckpt
        if isinstance(sd, dict) and "state_dict" in sd and "model" not in ckpt:
            sd = sd["state_dict"]
        model_sd = model.state_dict()
        filtered = {
            k: v
            for k, v in sd.items()
            if k in model_sd and tuple(model_sd[k].shape) == tuple(v.shape)
        }
        missing, unexpected = model.load_state_dict(filtered, strict=False)
        print(
            f"init_ckpt={init_ckpt} loaded={len(filtered)}/{len(model_sd)} "
            f"missing={len(missing)} unexpected={len(unexpected)}"
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
        # Rebuild vocab from dataset scan (may grow on G2P path)
        if len(vocab) > model.emb.num_embeddings:
            raise SystemExit(
                f"vocab grew to {len(vocab)} > model emb {model.emb.num_embeddings}; "
                "rebuild with larger vocab_size"
            )
        trainer = MelTrainer(model, train_loader, val_loader, cfg, device)
        steps = args.steps or stage.get("steps", 1000)
        metrics = trainer.train_steps(steps, stage=stage["name"])
        print(stage["name"], metrics)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

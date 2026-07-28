#!/usr/bin/env python3
"""Offline encode RUSLAN with sample=False; write sharded latents + stats."""

from __future__ import annotations

import argparse
import json

import numpy as np
import soundfile as sf
import torch
from tqdm import tqdm

from minispeech.codec.stats import compute_latent_stats, save_stats
from minispeech.codec.wrapper import VibeVoiceCodec
from minispeech.data.manifest import read_manifest, write_manifest
from minispeech.data.shards import ShardWriter
from minispeech.utils import expand, load_yaml, set_seed


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/data_ruslan.yaml")
    ap.add_argument("--codec-config", default="configs/codec.yaml")
    ap.add_argument("--max-utts", type=int, default=None)
    ap.add_argument("--dtype", default="bfloat16")
    args = ap.parse_args()
    set_seed(42)
    dcfg = load_yaml(args.config)["data"]
    ccfg = load_yaml(args.codec_config)["codec"]
    rows = read_manifest(dcfg["manifest"])
    if args.max_utts:
        rows = rows[: args.max_utts]

    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    codec = VibeVoiceCodec(model_id=ccfg["model_id"], dtype=dtype)
    print(f"latent layout: {codec._latent_layout}")

    writer = ShardWriter(dcfg["latent_dir"], "latents", shard_size=dcfg.get("shard_size", 2000))
    train_latents = []
    for r in tqdm(rows, desc="encode"):
        audio, sr = sf.read(r.wav, dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=-1)
        # DC only
        audio = audio - float(audio.mean())
        enc = codec.encode(audio, sr, sample=False)
        z = enc.latents[0].float().cpu()  # [T, 64]
        # Trim to true frames if padded longer
        t_true = min(z.shape[0], r.n_frames or z.shape[0])
        z = z[:t_true]
        r.n_frames = int(z.shape[0])
        writer.add(r.utt_id, z.half(), {"n_frames": r.n_frames})
        if r.split == "train":
            train_latents.append(z)

    writer.close()
    stats = compute_latent_stats(train_latents)
    save_stats(stats, dcfg["stats_path"])
    write_manifest(dcfg["manifest"], rows)
    meta = {
        "n": len(rows),
        "ac1_mean": float(stats["latent_ac1"].float().mean()),
        "layout": codec._latent_layout,
    }
    print(json.dumps(meta, indent=2))
    expand(dcfg["latent_dir"]).joinpath("encode_meta.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

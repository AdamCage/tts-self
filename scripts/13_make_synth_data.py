#!/usr/bin/env python3
"""Build synthetic offline caches for pipeline smoke without full RUSLAN."""

from __future__ import annotations

import argparse

import numpy as np
import soundfile as sf
import torch

from minispeech.codec.stats import compute_latent_stats, save_stats
from minispeech.constants import LATENT_DIM, SR, n_frames_from_samples
from minispeech.data.alignment import monotonic_gaussian_prior
from minispeech.data.manifest import Utterance, contiguous_split, write_manifest
from minispeech.data.shards import ShardWriter
from minispeech.frontend.stress import to_canonical_stress
from minispeech.utils import expand, set_seed


TEXTS = [
    "Приве́т, как дела́?",
    "Сего́дня хоро́шая пого́да.",
    "Москва́ — столи́ца Росси́и.",
    "Ско́лько сто́ит биле́т?",
    "Я чита́ю интере́сную кни́гу.",
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="~/datasets/tts/ruslan_synth")
    ap.add_argument("--n", type=int, default=64)
    args = ap.parse_args()
    set_seed(0)
    root = expand(args.out)
    wav_dir = root / "wavs"
    wav_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    lat_w = ShardWriter(root / "latents", "latents", shard_size=1000)
    txt_w = ShardWriter(root / "text_hidden", "text_hidden", shard_size=1000)
    al_w = ShardWriter(root / "align", "align", shard_size=1000)
    train_lats = []

    for i in range(args.n):
        text = TEXTS[i % len(TEXTS)]
        dur = 1.5 + (i % 5) * 0.4
        n = int(dur * SR)
        t = np.linspace(0, dur, n, endpoint=False)
        audio = 0.1 * np.sin(2 * np.pi * 180 * t).astype(np.float32)
        wp = wav_dir / f"{i:04d}.wav"
        sf.write(str(wp), audio, SR)
        T = n_frames_from_samples(n)
        L = 8 + (i % 5)
        z = torch.randn(T, LATENT_DIM)
        # smooth latents a bit
        z = torch.nn.functional.avg_pool1d(z.T.unsqueeze(0), 3, stride=1, padding=1).squeeze(0).T
        th = torch.randn(3, L, 1024)
        prior = monotonic_gaussian_prior(T, L)
        uid = f"synth_{i:04d}"
        lat_w.add(uid, z.half(), {"n_frames": T})
        txt_w.add(uid, th.half(), {"text_len": L})
        al_w.add(uid, prior.half(), {"level": "C"})
        train_lats.append(z)
        rows.append(
            Utterance(
                utt_id=uid,
                wav=str(wp),
                text_raw=text.replace("\u0301", ""),
                text_stressed=to_canonical_stress(text),
                n_samples_24k=n,
                n_frames=T,
                dur_s=dur,
                input_ids=list(range(L)),
            )
        )

    rows = contiguous_split(rows)
    for r in rows:
        if r.split == "train":
            pass
    lat_w.close()
    txt_w.close()
    al_w.close()
    stats = compute_latent_stats(train_lats)
    save_stats(stats, root / "stats.safetensors")
    write_manifest(root / "manifest.jsonl", rows)
    print(f"Wrote synthetic dataset to {root} n={len(rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

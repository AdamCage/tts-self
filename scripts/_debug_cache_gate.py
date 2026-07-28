"""Diagnose Phase0 cached-vs-full decode differences (bf16 noise vs seam bug)."""

from __future__ import annotations

import argparse
import inspect
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from minispeech.codec.cache_check import cached_vs_full_stats, decode_chunked, evaluate_cache_gate
from minispeech.codec.wrapper import VibeVoiceCodec
from minispeech.utils import expand, load_yaml, set_seed


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="configs/data_ruslan_local.yaml")
    ap.add_argument("--ruslan-dir", default=None)
    ap.add_argument("--chunk", type=int, default=4)
    ap.add_argument("--dtype", choices=("bf16", "fp32"), default="bf16")
    args = ap.parse_args()

    from transformers import VibeVoiceAcousticTokenizerModel

    print("VibeVoiceAcousticTokenizerModel:", inspect.getfile(VibeVoiceAcousticTokenizerModel))

    dcfg = load_yaml(args.data)["data"]
    set_seed(42)
    ruslan = expand(args.ruslan_dir or dcfg["ruslan_dir"])
    wavs = sorted(Path(ruslan).expanduser().joinpath("wavs").glob("*.wav"))
    if not wavs:
        raise SystemExit(f"No wavs under {ruslan}/wavs")
    path = wavs[0]
    audio, sr = sf.read(str(path), dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(-1)

    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    codec = VibeVoiceCodec(dtype=dtype, device=device)
    print(f"layout={codec._latent_layout} dtype={dtype} device={device} wav={path.name}")

    enc = codec.encode(audio, sr, sample=False)
    z = enc.latents
    print(f"latents={tuple(z.shape)} n_true={enc.n_frames_true}")

    full = codec.decode(z).audio
    cached = decode_chunked(codec, z, chunk_frames=args.chunk)
    stats = cached_vs_full_stats(full, cached, chunk_frames=args.chunk)
    limit = 1e-3 if args.dtype == "fp32" else 2e-2
    ok, reason = evaluate_cache_gate(stats, max_abs_limit=limit, boundary_ratio_max=10.0)
    print("stats:", {k: stats[k] for k in stats})
    print(f"gate ok={ok} reason={reason} limit={limit}")


if __name__ == "__main__":
    main()

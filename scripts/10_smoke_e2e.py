#!/usr/bin/env python3
"""End-to-end smoke: codec roundtrip + model forward + short AR generate."""

from __future__ import annotations

import argparse
import time

import numpy as np
import torch

from minispeech.constants import LATENT_DIM, SR
from minispeech.models.minispeech import MiniSpeech
from minispeech.utils import set_seed


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-codec", action="store_true")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    set_seed(0)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"device={device}")

    if not args.skip_codec:
        try:
            from minispeech.codec.wrapper import VibeVoiceCodec

            codec = VibeVoiceCodec(device=device)
            audio = np.random.randn(SR).astype(np.float32) * 0.01
            enc = codec.encode(audio, SR, sample=False)
            assert enc.latents.shape[-1] == LATENT_DIM
            dec = codec.decode(enc.latents)
            print(f"codec ok layout={codec._latent_layout} z={tuple(enc.latents.shape)} audio={tuple(dec.audio.shape)}")
        except Exception as e:
            print(f"codec smoke skipped/failed: {e}")

    model = MiniSpeech(mode="regression").to(device)
    B, T, L, H = 2, 16, 12, 1024
    text_hidden = torch.randn(B, 3, L, H, device=device)
    z_target = torch.randn(B, T, LATENT_DIM, device=device)
    z_input = torch.zeros_like(z_target)
    z_input[:, 1:] = z_target[:, :-1]
    frame_mask = torch.ones(B, T, dtype=torch.bool, device=device)
    text_mask = torch.ones(B, L, dtype=torch.bool, device=device)
    t0 = time.perf_counter()
    out = model(z_input, text_hidden, frame_mask=frame_mask, text_mask=text_mask, z_target=z_target)
    loss = out.z_pred.pow(2).mean() + out.stop_logits.pow(2).mean()
    loss.backward()
    print(f"train step ok loss={float(loss):.4f} time={time.perf_counter()-t0:.3f}s")

    # short generate
    th = torch.randn(1, 3, 8, 1024, device=device)
    tm = torch.ones(1, 8, dtype=torch.bool, device=device)
    z, info = model.generate(th, tm, max_frames=8)
    print(f"generate ok z={tuple(z.shape)} info={info}")

    # flow mode smoke
    fmodel = MiniSpeech(mode="flow").to(device)
    outf = fmodel(z_input, text_hidden, frame_mask=frame_mask, text_mask=text_mask, z_target=z_target)
    floss = (outf.extras["v_pred"] - outf.extras["v_target"]).pow(2).mean()
    floss.backward()
    print(f"flow step ok loss={float(floss):.4f}")
    print("SMOKE_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Download SpeechBrain HiFi-GAN (22.05 kHz) into ~/models/hifigan/.

Prefer LibriTTS multi-speaker 22050 Hz weights — hop/n_fft/n_mels match mel80
geometry (fmax training=8 kHz vs mel80 nyquist is a known mild mismatch).

Does **not** auto-run from ``mel_to_wav``; explicit opt-in only.

Example (WSL)::

    python scripts/33_download_hifigan.py
    python scripts/33_download_hifigan.py --smoke-forward
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import torch

from minispeech.infer.hifigan import extract_generator_state_dict, load_hifigan_generator
from minispeech.infer.vocoder import NeuralVocoderStub, find_local_vocoder, vocoder_status
from minispeech.utils import expand

DEFAULT_REPO = "speechbrain/tts-hifigan-libritts-22050Hz"
DEFAULT_OUT = "~/models/hifigan"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo", default=DEFAULT_REPO)
    ap.add_argument("--out-dir", default=DEFAULT_OUT)
    ap.add_argument("--smoke-forward", action="store_true")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    out_dir = expand(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dest = out_dir / "generator.pt"
    meta_path = out_dir / "source.json"

    if dest.is_file() and not args.force:
        print(f"already exists: {dest}")
    else:
        from huggingface_hub import hf_hub_download

        ckpt = Path(hf_hub_download(args.repo, "generator.ckpt"))
        hp = Path(hf_hub_download(args.repo, "hyperparams.yaml"))
        raw = torch.load(ckpt, map_location="cpu", weights_only=False)
        state = extract_generator_state_dict(raw)
        if state is None:
            raise SystemExit(f"could not extract generator state from {ckpt}")
        # Smoke-build to validate keys, then persist the *weight_norm* state
        # (before remove_weight_norm) so reload matches HifiGanGenerator.
        _ = load_hifigan_generator(state, device=torch.device("cpu"))
        torch.save({"generator": state, "format": "hifigan_v1_weight_norm_state_dict"}, dest)
        # Also keep raw SpeechBrain ckpt for debugging
        shutil.copy2(ckpt, out_dir / "generator.ckpt")
        shutil.copy2(hp, out_dir / "hyperparams.yaml")
        meta = {
            "repo": args.repo,
            "files": ["generator.pt", "generator.ckpt", "hyperparams.yaml"],
            "format": "hifigan_v1_state_dict",
            "note": (
                "HiFi-GAN V1 LibriTTS 22050 Hz; mel80 hop/n_fft match; "
                "fmax train=8000 vs mel80 fmax=nyquist (listening OK, not bitexact)."
            ),
            "n_tensors": len(state),
        }
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        print(f"wrote {dest} ({dest.stat().st_size} bytes)")

    stub = NeuralVocoderStub(dest, device=torch.device("cpu"))
    print("loader:", stub.available, stub.note)
    print("status:", vocoder_status(dest))
    print("find_local:", find_local_vocoder())

    if args.smoke_forward:
        if not stub.available:
            raise SystemExit("vocoder not available for smoke")
        mel = torch.randn(1, 80, 40) * 0.5 - 2.0
        wav = stub(mel.squeeze(0).transpose(0, 1))  # [T, 80]
        print("smoke wav", tuple(wav.shape), float(wav.abs().mean()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

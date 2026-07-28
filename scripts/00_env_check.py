#!/usr/bin/env python3
"""Environment check for MiniSpeech-RU (WSL2 / CUDA)."""

from __future__ import annotations

import platform
import sys


def main() -> int:
    print("=== MiniSpeech-RU env check ===")
    print(f"python: {sys.version}")
    print(f"platform: {platform.platform()}")
    try:
        import torch

        print(f"torch: {torch.__version__}")
        print(f"cuda available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"gpu: {torch.cuda.get_device_name(0)}")
            props = torch.cuda.get_device_properties(0)
            print(f"vram_gb: {props.total_memory / 1e9:.2f}")
            x = torch.randn(1024, 1024, device="cuda", dtype=torch.bfloat16)
            y = x @ x
            print(f"bf16 matmul ok: {tuple(y.shape)}")
    except Exception as e:
        print(f"torch error: {e}")
        return 1

    try:
        import transformers

        print(f"transformers: {transformers.__version__}")
        from transformers import VibeVoiceAcousticTokenizerModel  # noqa: F401

        print("VibeVoiceAcousticTokenizerModel: OK")
    except Exception as e:
        print(f"transformers/VibeVoice: {e}")

    try:
        from transformers import Qwen3_5TextModel  # noqa: F401

        print("Qwen3_5TextModel: OK")
    except Exception as e:
        print(f"Qwen3_5TextModel: {e}")

    for pkg in ("causal_conv1d", "fla"):
        try:
            __import__(pkg)
            print(f"{pkg}: OK")
        except Exception as e:
            print(f"{pkg}: missing ({e.__class__.__name__}) — Qwen DeltaNet may use slow fallback")

    print("=== done ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

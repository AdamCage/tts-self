#!/usr/bin/env python3
"""sessionStart: inject MiniSpeech-RU agent context."""

from __future__ import annotations

import json
import sys


def main() -> int:
    try:
        json.load(sys.stdin)
    except Exception:
        pass
    ctx = (
        "MiniSpeech-RU agent context: read AGENTS.md first. "
        "GPU work runs in WSL2 (/mnt/c/projects/tts-self, HF_HOME=~/hf_cache). "
        "Use project skills under .cursor/skills/ for Phase 0, caches, train ladder, "
        "text-usage gate, streaming, WSL env, and eval gates. "
        "Never train the codec; encode with sample=False; do not treat vae_std as error budget."
    )
    json.dump({"additional_context": ctx}, sys.stdout)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

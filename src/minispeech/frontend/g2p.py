"""Optional phoneme channel (Phase 3+). Stub for v0.1."""

from __future__ import annotations


def text_to_phonemes(text: str) -> list[str]:
    """Placeholder G2P — returns characters as pseudo-phonemes until real G2P is wired."""
    from minispeech.frontend.stress import strip_stress

    return list(strip_stress(text).replace(" ", "_"))

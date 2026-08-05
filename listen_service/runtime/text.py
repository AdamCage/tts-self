"""Text frontend for free-text discrete listen (Wave-1 parity)."""

from __future__ import annotations


def prepare_text(raw: str) -> str:
    """TN → verbalize → stress (same order as ``scripts/05_cache_text_hidden``).

    Returns stressed natural Russian for Qwen3.5 tokenization.
    """
    text = (raw or "").strip()
    if not text:
        raise ValueError("empty text")
    from minispeech.frontend.stress import accentuate
    from minispeech.frontend.verbalize import normalize_for_tts

    return accentuate(normalize_for_tts(text))

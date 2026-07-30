"""Russian G2P for Track A phoneme channel.

**Order (mandatory):** accentuate / attach stress marks **before** G2P.
Vowel reduction (аканье / иканье) and related rules depend on stress position.
Pass stressed natural Russian (combining acute U+0301), never raw phoneme
strings as Qwen tokens.

Preferred backend: ``phonemizer`` + system ``espeak-ng`` language ``ru``.
If either is missing, fall back to grapheme tokens so unit tests and offline
scaffolding stay runnable without espeak.
"""

from __future__ import annotations

import logging
import re
from functools import lru_cache

from minispeech.constants import STRESS_CHAR
from minispeech.frontend.stress import to_canonical_stress

log = logging.getLogger(__name__)

_SPACE = "_"
_IPA_SPLIT = re.compile(r"\s+")


def _ensure_user_espeak_library() -> None:
    """Point phonemizer at a user-local libespeak-ng when apt/sudo is unavailable."""
    import os
    from pathlib import Path

    candidates = []
    env = os.environ.get("PHONEMIZER_ESPEAK_LIBRARY") or os.environ.get("ESPEAK_LIBRARY")
    if env:
        candidates.append(Path(env))
    home = Path.home()
    candidates.extend(
        [
            home / "local/espeak-ng/usr/lib/x86_64-linux-gnu/libespeak-ng.so.1",
            home / "local/lib/libespeak-ng.so.1",
            Path("/usr/lib/x86_64-linux-gnu/libespeak-ng.so.1"),
        ]
    )
    lib = next((p for p in candidates if p.is_file()), None)
    if lib is None:
        return
    # Shared deps (pulseaudio private dir) for user-extracted debs
    libdir = lib.parent
    pulse = libdir / "pulseaudio"
    ld = os.environ.get("LD_LIBRARY_PATH", "")
    parts = [str(libdir)]
    if pulse.is_dir():
        parts.append(str(pulse))
    for p in parts:
        if p not in ld.split(":"):
            ld = f"{p}:{ld}" if ld else p
    os.environ["LD_LIBRARY_PATH"] = ld
    data = libdir / "espeak-ng-data"
    if data.is_dir():
        os.environ.setdefault("ESPEAK_DATA_PATH", str(data))
    # Preload audio deps: changing LD_LIBRARY_PATH after process start does not
    # affect the dynamic linker on Linux, so CDLL-load dependents first.
    try:
        import ctypes

        for name in (
            "libpulse.so.0",
            "libpulse-simple.so.0",
            "libpcaudio.so.0",
            "libsonic.so.0",
            "libasound.so.2",
        ):
            for base in (libdir, pulse if pulse.is_dir() else None):
                if base is None:
                    continue
                cand = base / name
                if cand.is_file():
                    try:
                        ctypes.CDLL(str(cand), mode=ctypes.RTLD_GLOBAL)
                    except OSError:
                        pass
                    break
        # pulsecommon lives in pulseaudio/
        if pulse.is_dir():
            for cand in sorted(pulse.glob("libpulsecommon-*.so")):
                try:
                    ctypes.CDLL(str(cand), mode=ctypes.RTLD_GLOBAL)
                except OSError:
                    pass
        from phonemizer.backend.espeak.wrapper import EspeakWrapper

        EspeakWrapper.set_library(str(lib))
    except Exception as exc:  # pragma: no cover
        log.debug("EspeakWrapper.set_library failed: %s", exc)


@lru_cache(maxsize=1)
def _phonemizer_available() -> bool:
    try:
        _ensure_user_espeak_library()
        from phonemizer.backend import EspeakBackend  # noqa: F401

        backend = EspeakBackend(language="ru", preserve_punctuation=False, with_stress=True)
        _ = backend.phonemize(["привет"], strip=True)
        return True
    except Exception as exc:  # ImportError, RuntimeError (no espeak), etc.
        log.debug("phonemizer/espeak-ng unavailable: %s", exc)
        return False


def backend_name() -> str:
    """Return ``phonemizer_espeak_ru`` or ``grapheme_fallback``."""
    return "phonemizer_espeak_ru" if _phonemizer_available() else "grapheme_fallback"


def _grapheme_fallback(text: str) -> list[str]:
    """Character-level tokens; stress marks kept attached to vowels when present.

    Lowercases letters and drops punctuation so mel / forced-align vocabs stay
    compact when espeak is unavailable (still not true phones).
    """
    t = to_canonical_stress(text)
    out: list[str] = []
    i = 0
    while i < len(t):
        ch = t[i]
        if ch.isspace():
            if out and out[-1] != _SPACE:
                out.append(_SPACE)
            i += 1
            continue
        if ch == STRESS_CHAR:
            if out:
                out[-1] = out[-1] + STRESS_CHAR
            i += 1
            continue
        # Skip punctuation / symbols (keep letters, digits, combining marks)
        if not (ch.isalpha() or ch.isdigit()):
            i += 1
            continue
        # Keep letter + following stress as one token
        tok = ch.lower()
        if i + 1 < len(t) and t[i + 1] == STRESS_CHAR:
            tok += STRESS_CHAR
            i += 2
        else:
            i += 1
        out.append(tok)
    while out and out[-1] == _SPACE:
        out.pop()
    return out


def _espeak_phonemes(text: str) -> list[str]:
    _ensure_user_espeak_library()
    from phonemizer.backend import EspeakBackend
    from phonemizer.separator import Separator

    # espeak sees stress better with +Vowel or primary stress; keep combining acute
    # in the string — espeak-ng ru often preserves stress via ˈ in IPA.
    # Explicit phone/word separators: default EspeakBackend collapses to word IPA.
    backend = EspeakBackend(
        language="ru",
        preserve_punctuation=False,
        with_stress=True,
        language_switch="remove-flags",
    )
    sep = Separator(phone=" ", word="|", syllable="")
    ipa = backend.phonemize([text], separator=sep, strip=True)[0]
    # Example shape: "p r ʲ i v ʲ ˈ e t | m ʲ ˈ i r" (inventory varies by espeak)
    parts: list[str] = []
    for chunk in ipa.strip().split("|"):
        phones = [p for p in _IPA_SPLIT.split(chunk.strip()) if p]
        if not phones:
            continue
        if parts:
            parts.append(_SPACE)
        parts.extend(phones)
    if not parts:
        return _grapheme_fallback(text)
    return parts


def text_to_phonemes(
    text: str,
    *,
    assume_stressed: bool = True,
    prefer_phonemizer: bool = True,
) -> list[str]:
    """Convert Russian text to a phoneme (or grapheme) token list.

    Parameters
    ----------
    text:
        Preferably already stressed (combining acute). If ``assume_stressed`` is
        False, caller should run ``accentuate`` first; this function does not
        call ruaccent (keeps G2P free of heavy model loads in tests).
    prefer_phonemizer:
        Use espeak when available; otherwise grapheme fallback.
    """
    if not text or not text.strip():
        return []
    t = to_canonical_stress(text.strip())
    if prefer_phonemizer and _phonemizer_available():
        try:
            return _espeak_phonemes(t)
        except Exception as exc:
            log.warning("espeak G2P failed (%s); using grapheme fallback", exc)
    # Fallback: keep stress on graphemes when present; strip only if desired later
    _ = assume_stressed  # documented contract; accentuate is caller's job
    return _grapheme_fallback(t)


def text_to_phonemes_str(text: str, sep: str = " ") -> str:
    return sep.join(text_to_phonemes(text))


# Back-compat alias used in early stubs
def g2p(text: str) -> list[str]:
    return text_to_phonemes(text)

"""Telephone channel simulation for Track N (N-Lite / N08 probes).

Pipeline applied to wideband audio:

1. Band-pass 300–3400 Hz (PSTN voice band)
2. Resample to 8 kHz
3. Codec encode → decode (G.711 µ-law / A-law, or Opus narrowband)
4. Optional distortions (additive noise, level jitter) — off by default
5. Optionally upsample back to the original sample rate

Opus requires ``opuslib`` (and system ``libopus``). If unavailable, G.711
paths still work; requesting ``codec=\"opus\"`` raises
:class:`OpusUnavailableError`. See :func:`available_codecs`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
from numpy.typing import NDArray
from scipy import signal

TELEPHONY_SR = 8000
BANDPASS_LOW_HZ = 300.0
BANDPASS_HIGH_HZ = 3400.0

CodecName = Literal["mulaw", "alaw", "opus", "none"]


class OpusUnavailableError(RuntimeError):
    """Raised when Opus narrowband codec is requested but not installed."""


def opus_available() -> bool:
    """Return True if ``opuslib`` can be imported (system libopus present)."""
    try:
        import opuslib  # noqa: F401
    except Exception:
        return False
    return True


def available_codecs() -> list[str]:
    """Codecs usable in the current environment (always includes G.711)."""
    codecs = ["none", "mulaw", "alaw"]
    if opus_available():
        codecs.append("opus")
    return codecs


@dataclass(frozen=True)
class ChannelConfig:
    """Parameters for :func:`apply_telephony_channel`.

    Distortions are off when the corresponding field is ``None``.
    """

    codec: CodecName = "mulaw"
    restore_sr: bool = True
    noise_snr_db: float | None = None
    level_jitter_db: float | None = None
    seed: int | None = None


def apply_telephony_channel(
    waveform: NDArray[np.floating],
    sample_rate: int,
    *,
    codec: CodecName = "mulaw",
    restore_sr: bool = True,
    noise_snr_db: float | None = None,
    level_jitter_db: float | None = None,
    seed: int | None = None,
) -> tuple[NDArray[np.float32], int]:
    """Apply telephony channel simulation to a mono waveform.

    Parameters
    ----------
    waveform:
        Mono float audio, shape ``(T,)``. Stereo is averaged to mono.
    sample_rate:
        Input sample rate in Hz.
    codec:
        ``\"mulaw\"`` / ``\"alaw\"`` (G.711), ``\"opus\"`` (narrowband),
        or ``\"none\"`` (band-limit + resample only — simplified sim).
    restore_sr:
        If True, upsample back to ``sample_rate`` and match input length.
        If False, return 8 kHz audio.
    noise_snr_db:
        If set, add white noise at this SNR (dB) after decode. Default off.
    level_jitter_db:
        If set, apply a random gain in ``±level_jitter_db``. Default off.
    seed:
        RNG seed for optional distortions (deterministic when set).

    Returns
    -------
    out, out_sr
        Processed float32 mono waveform and its sample rate.
    """
    wav = _as_mono_float32(waveform)
    sr_in = int(sample_rate)
    if sr_in <= 0:
        raise ValueError(f"sample_rate must be positive, got {sr_in}")
    if wav.size == 0:
        out_sr = sr_in if restore_sr else TELEPHONY_SR
        return wav.copy(), out_sr

    n_in = int(wav.shape[0])
    filtered = _bandpass(wav, sr_in, BANDPASS_LOW_HZ, BANDPASS_HIGH_HZ)
    narrow = _resample(filtered, sr_in, TELEPHONY_SR)

    if codec == "none":
        coded = narrow
    elif codec == "mulaw":
        coded = _mulaw_roundtrip(narrow)
    elif codec == "alaw":
        coded = _alaw_roundtrip(narrow)
    elif codec == "opus":
        coded = _opus_roundtrip(narrow, TELEPHONY_SR)
    else:
        raise ValueError(f"unknown codec {codec!r}; choose from {available_codecs()}")

    coded = _apply_optional_distortions(
        coded,
        noise_snr_db=noise_snr_db,
        level_jitter_db=level_jitter_db,
        seed=seed,
    )

    if not restore_sr:
        return coded.astype(np.float32, copy=False), TELEPHONY_SR

    restored = _resample(coded, TELEPHONY_SR, sr_in)
    restored = _match_length(restored, n_in)
    return restored.astype(np.float32, copy=False), sr_in


def apply_telephony_file(
    path: str | Path,
    *,
    out_path: str | Path | None = None,
    codec: CodecName = "mulaw",
    restore_sr: bool = True,
    noise_snr_db: float | None = None,
    level_jitter_db: float | None = None,
    seed: int | None = None,
) -> tuple[NDArray[np.float32], int]:
    """Load a WAV, apply :func:`apply_telephony_channel`, optionally write out.

    Uses ``soundfile``. Returns ``(waveform, sample_rate)`` of the result.
    """
    import soundfile as sf

    path = Path(path)
    wav, sr = sf.read(str(path), always_2d=False)
    out, out_sr = apply_telephony_channel(
        np.asarray(wav),
        int(sr),
        codec=codec,
        restore_sr=restore_sr,
        noise_snr_db=noise_snr_db,
        level_jitter_db=level_jitter_db,
        seed=seed,
    )
    if out_path is not None:
        sf.write(str(out_path), out, out_sr)
    return out, out_sr


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _as_mono_float32(waveform: NDArray[np.floating] | NDArray[np.integer]) -> NDArray[np.float32]:
    x = np.asarray(waveform)
    if x.ndim == 2:
        x = x.mean(axis=-1) if x.shape[-1] <= 4 else x.mean(axis=0)
    elif x.ndim != 1:
        raise ValueError(f"waveform must be 1-D or 2-D, got shape {x.shape}")
    x = x.astype(np.float32, copy=False)
    peak = float(np.max(np.abs(x))) if x.size else 0.0
    if peak > 1.5:
        # Likely int PCM left as float; normalize to [-1, 1].
        x = x / 32768.0
    return x


def _bandpass(
    wav: NDArray[np.float32],
    sr: int,
    low_hz: float,
    high_hz: float,
) -> NDArray[np.float32]:
    nyq = 0.5 * float(sr)
    # If source is already narrow (e.g. 8 kHz), clamp the high edge.
    hi = min(high_hz, nyq * 0.95)
    lo = max(low_hz, 1.0)
    if lo >= hi:
        return wav.copy()
    sos = signal.butter(4, [lo / nyq, hi / nyq], btype="band", output="sos")
    # sosfiltfilt needs enough samples; fall back to sosfilt for short clips.
    if wav.shape[0] < 32:
        y = signal.sosfilt(sos, wav)
    else:
        y = signal.sosfiltfilt(sos, wav)
    return np.asarray(y, dtype=np.float32)


def _resample(wav: NDArray[np.float32], sr_from: int, sr_to: int) -> NDArray[np.float32]:
    if sr_from == sr_to:
        return wav.copy()
    g = int(np.gcd(sr_from, sr_to))
    up = sr_to // g
    down = sr_from // g
    y = signal.resample_poly(wav, up, down)
    return np.asarray(y, dtype=np.float32)


def _match_length(wav: NDArray[np.float32], n: int) -> NDArray[np.float32]:
    if wav.shape[0] == n:
        return wav
    if wav.shape[0] > n:
        return wav[:n].copy()
    out = np.zeros(n, dtype=np.float32)
    out[: wav.shape[0]] = wav
    return out


def _float_to_pcm16(wav: NDArray[np.float32]) -> NDArray[np.int16]:
    clipped = np.clip(wav, -1.0, 1.0)
    return (clipped * 32767.0).astype(np.int16)


def _pcm16_to_float(pcm: NDArray[np.int16]) -> NDArray[np.float32]:
    return (pcm.astype(np.float32) / 32768.0).astype(np.float32)


def _mulaw_encode_byte(pcm: NDArray[np.int16]) -> NDArray[np.uint8]:
    """ITU-T G.711 µ-law encode (bias 0x84), vectorized."""
    x = pcm.astype(np.int32)
    sign = np.where(x < 0, 0x80, 0).astype(np.int32)
    mag = np.minimum(np.abs(x) + 0x84, 0x7FFF)
    # exponent in [0, 7]: bit_length(mag) - 8
    bl = np.floor(np.log2(np.maximum(mag, 1).astype(np.float64))).astype(np.int32) + 1
    exp = np.clip(bl - 8, 0, 7)
    mant = (mag >> (exp + 3)) & 0x0F
    ulaw = ~(sign | (exp << 4) | mant) & 0xFF
    return ulaw.astype(np.uint8)


def _mulaw_decode_byte(ulaw: NDArray[np.uint8]) -> NDArray[np.int16]:
    u = (~ulaw.astype(np.int32)) & 0xFF
    sign = u & 0x80
    exp = (u >> 4) & 0x07
    mant = u & 0x0F
    x = (((mant << 3) + 0x84) << exp) - 0x84
    x = np.where(sign != 0, -x, x)
    return np.clip(x, -32768, 32767).astype(np.int16)


def _mulaw_roundtrip(wav: NDArray[np.float32]) -> NDArray[np.float32]:
    pcm = _float_to_pcm16(wav)
    return _pcm16_to_float(_mulaw_decode_byte(_mulaw_encode_byte(pcm)))


def _alaw_encode_byte(pcm: NDArray[np.int16]) -> NDArray[np.uint8]:
    """ITU-T G.711 A-law encode (sample-wise reference)."""
    out = np.empty(pcm.shape[0], dtype=np.uint8)
    for i, sample in enumerate(pcm.tolist()):
        sample = int(sample)
        if sample >= 0:
            mask = 0xD5
        else:
            mask = 0x55
            sample = 32767 if sample == -32768 else -sample
        if sample > 32767:
            sample = 32767
        if sample >= 256:
            exp = 7
            for e in range(1, 8):
                if sample < (256 << e):
                    exp = e
                    break
            mant = (sample >> (exp + 3)) & 0x0F
            aval = (exp << 4) | mant
        else:
            aval = sample >> 4
        out[i] = aval ^ mask
    return out


def _alaw_decode_byte(alaw: NDArray[np.uint8]) -> NDArray[np.int16]:
    # After XOR 0x55, SIGN_BIT set ⇒ positive (ITU-T G.711 / audioop convention).
    out = np.empty(alaw.shape[0], dtype=np.int16)
    for i, a in enumerate(alaw.tolist()):
        a = int(a) ^ 0x55
        sign = a & 0x80
        exp = (a & 0x70) >> 4
        mant = a & 0x0F
        if exp == 0:
            x = (mant << 4) + 8
        else:
            x = ((mant << 4) + 0x108) << (exp - 1)
        out[i] = x if sign else -x
    return out


def _alaw_roundtrip(wav: NDArray[np.float32]) -> NDArray[np.float32]:
    pcm = _float_to_pcm16(wav)
    return _pcm16_to_float(_alaw_decode_byte(_alaw_encode_byte(pcm)))


def _opus_roundtrip(wav: NDArray[np.float32], sr: int) -> NDArray[np.float32]:
    if not opus_available():
        raise OpusUnavailableError(
            "Opus narrowband requires the 'opuslib' package and system libopus. "
            "Install with: pip install opuslib  (and libopus via apt/brew). "
            "Fall back to codec='mulaw' or codec='none' (band-limit + resample only)."
        )
    import opuslib

    if sr != TELEPHONY_SR:
        raise ValueError(f"Opus path expects {TELEPHONY_SR} Hz, got {sr}")

    frame_samples = 160  # 20 ms @ 8 kHz
    # Pad to whole frames for encode, trim after decode.
    n = int(wav.shape[0])
    pad = (-n) % frame_samples
    if pad:
        wav_p = np.concatenate([wav, np.zeros(pad, dtype=np.float32)])
    else:
        wav_p = wav

    pcm = _float_to_pcm16(wav_p)
    enc = opuslib.Encoder(sr, 1, opuslib.APPLICATION_VOIP)
    enc.bitrate = 16000
    dec = opuslib.Decoder(sr, 1)

    decoded_chunks: list[NDArray[np.int16]] = []
    for start in range(0, pcm.shape[0], frame_samples):
        frame = pcm[start : start + frame_samples].tobytes()
        packet = enc.encode(frame, frame_samples)
        raw = dec.decode(packet, frame_samples)
        decoded_chunks.append(np.frombuffer(raw, dtype=np.int16))

    out_pcm = np.concatenate(decoded_chunks)[:n]
    return _pcm16_to_float(out_pcm)


def _apply_optional_distortions(
    wav: NDArray[np.float32],
    *,
    noise_snr_db: float | None,
    level_jitter_db: float | None,
    seed: int | None,
) -> NDArray[np.float32]:
    if noise_snr_db is None and level_jitter_db is None:
        return wav

    rng = np.random.default_rng(seed)
    y = wav.astype(np.float32, copy=True)

    if level_jitter_db is not None:
        # Uniform gain in ±level_jitter_db.
        db = float(rng.uniform(-abs(level_jitter_db), abs(level_jitter_db)))
        y = y * float(10.0 ** (db / 20.0))

    if noise_snr_db is not None:
        sig_pow = float(np.mean(y * y)) + 1e-12
        noise_pow = sig_pow / (10.0 ** (float(noise_snr_db) / 10.0))
        noise = rng.standard_normal(y.shape[0]).astype(np.float32) * np.sqrt(noise_pow)
        y = y + noise

    peak = float(np.max(np.abs(y))) if y.size else 0.0
    if peak > 1.0:
        y = y / peak
    return y

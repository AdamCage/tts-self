"""Optional acoustic / MOS metrics with graceful skip when deps missing."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

log = logging.getLogger(__name__)


def _skip(name: str, reason: str) -> dict[str, Any]:
    return {"metric": name, "available": False, "skipped": True, "reason": reason, "value": None}


def score_gigaam_ctc(wav: np.ndarray, sr: int, *, text: str | None = None) -> dict[str, Any]:
    """Hook for GigaAM-CTC CER; skipped unless a local backend is wired."""
    try:
        import gigaam  # type: ignore  # noqa: F401
    except ImportError:
        return _skip("gigaam_ctc", "gigaam package not installed")
    return _skip("gigaam_ctc", "gigaam imported but MiniSpeech adapter not wired yet")


def score_utmos_v2(wav: np.ndarray, sr: int) -> dict[str, Any]:
    """UTMOS22 strong (SpeechMOS / torch.hub) via ``minispeech.eval.utmos``."""
    try:
        from minispeech.eval.utmos import utmos_load_error, utmos_score_wav

        val = utmos_score_wav(np.asarray(wav, dtype=np.float32), int(sr))
        if val is None:
            reason = utmos_load_error() or "utmos backend unavailable"
            return _skip("utmos_v2", reason)
        return {
            "metric": "utmos_v2",
            "available": True,
            "skipped": False,
            "value": float(val),
            "backend": "speechmos_utmos22_strong",
        }
    except Exception as exc:  # noqa: BLE001
        return _skip("utmos_v2", str(exc))


def score_nisqa(wav: np.ndarray, sr: int) -> dict[str, Any]:
    """NISQA MOS — honest skip until a local adapter is wired."""
    try:
        import nisqa  # type: ignore  # noqa: F401
    except ImportError:
        return _skip("nisqa", "nisqa package not installed")
    return _skip(
        "nisqa",
        "nisqa imported but MiniSpeech adapter not wired yet "
        "(prefer UTMOS22 via score_utmos_v2)",
    )


def score_mcd_dtw(
    hyp_wav: np.ndarray,
    ref_wav: np.ndarray,
    *,
    sr: int,
) -> dict[str, Any]:
    """MCD-DTW vs reference wav; requires librosa (+ optional pysptk)."""
    try:
        import librosa
    except ImportError:
        return _skip("mcd_dtw", "librosa not installed")
    try:
        # Lightweight log-mel L2 after crude DTW on mel frames (no pysptk required)
        def _logmel(x: np.ndarray) -> np.ndarray:
            m = librosa.feature.melspectrogram(
                y=np.asarray(x, dtype=np.float32),
                sr=sr,
                n_fft=1024,
                hop_length=256,
                n_mels=80,
                fmin=0.0,
                fmax=sr / 2.0,
                power=1.0,
            )
            return np.log(np.maximum(m, 1e-5)).T  # [T, 80]

        a = _logmel(hyp_wav)
        b = _logmel(ref_wav)
        if a.size == 0 or b.size == 0:
            return _skip("mcd_dtw", "empty mel")
        # librosa DTW on mean-pooled band energy for alignment, then frame L2
        D, wp = librosa.sequence.dtw(X=a.T, Y=b.T, metric="euclidean")
        # path cost → mean frame distance (proxy MCD-like)
        path = np.asarray(wp)
        dists = [float(np.linalg.norm(a[i] - b[j])) for i, j in path]
        value = float(np.mean(dists)) if dists else None
        return {
            "metric": "mcd_dtw",
            "available": True,
            "skipped": False,
            "value": value,
            "note": "proxy mel-L2 after DTW (not classic MCD dB)",
            "path_len": int(len(dists)),
        }
    except Exception as exc:  # noqa: BLE001
        return _skip("mcd_dtw", str(exc))


def score_f0_rmse(
    hyp_wav: np.ndarray,
    ref_wav: np.ndarray,
    *,
    sr: int,
) -> dict[str, Any]:
    """F0 RMSE via librosa.pyin when available."""
    try:
        import librosa
    except ImportError:
        return _skip("f0_rmse", "librosa not installed")
    try:
        def _f0(x: np.ndarray) -> np.ndarray:
            f0, _, _ = librosa.pyin(
                np.asarray(x, dtype=np.float32),
                fmin=librosa.note_to_hz("C2"),
                fmax=librosa.note_to_hz("C7"),
                sr=sr,
            )
            return np.asarray(f0, dtype=np.float64)

        fa = _f0(hyp_wav)
        fb = _f0(ref_wav)
        n = min(len(fa), len(fb))
        if n == 0:
            return _skip("f0_rmse", "empty f0")
        fa, fb = fa[:n], fb[:n]
        mask = np.isfinite(fa) & np.isfinite(fb)
        if mask.sum() < 5:
            return _skip("f0_rmse", "too few voiced frames")
        rmse = float(np.sqrt(np.mean((fa[mask] - fb[mask]) ** 2)))
        return {
            "metric": "f0_rmse",
            "available": True,
            "skipped": False,
            "value": rmse,
            "n_voiced": int(mask.sum()),
        }
    except Exception as exc:  # noqa: BLE001
        return _skip("f0_rmse", str(exc))


OPTIONAL_SCORERS = {
    "gigaam_ctc": score_gigaam_ctc,
    "utmos_v2": score_utmos_v2,
    "nisqa": score_nisqa,
    "mcd_dtw": score_mcd_dtw,
    "f0_rmse": score_f0_rmse,
}


def run_optional_metrics(
    names: list[str],
    *,
    hyp_wav: np.ndarray | None,
    sr: int,
    ref_wav: np.ndarray | None = None,
    text: str | None = None,
) -> list[dict[str, Any]]:
    """Run named optional metrics; never raises for missing deps."""
    out: list[dict[str, Any]] = []
    if hyp_wav is None:
        return [_skip(n, "no hyp wav") for n in names]
    for name in names:
        fn = OPTIONAL_SCORERS.get(name)
        if fn is None:
            out.append(_skip(name, "unknown metric"))
            continue
        try:
            if name in ("mcd_dtw", "f0_rmse"):
                if ref_wav is None:
                    out.append(_skip(name, "no reference wav"))
                    continue
                out.append(fn(hyp_wav, ref_wav, sr=sr))  # type: ignore[misc]
            elif name == "gigaam_ctc":
                out.append(fn(hyp_wav, sr, text=text))  # type: ignore[misc]
            else:
                out.append(fn(hyp_wav, sr))  # type: ignore[misc]
        except Exception as exc:  # noqa: BLE001
            log.warning("optional metric %s failed: %s", name, exc)
            out.append(_skip(name, str(exc)))
    return out

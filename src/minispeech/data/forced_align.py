"""Forced-alignment packing for ``forced.json`` consumed by ``scripts/06_align.py``.

Schema per utterance id::

    {
      "word_spans_frames": [[start, end], ...],   # half-open frame indices @ FRAME_RATE
      "token_word_ids": [int, ...],               # length L (text tokens / BPE)
      "token_char_lens": [int, ...],              # optional, for level B
      "word_spans_seconds": [[t0, t1], ...],      # optional audit / A3 projection
      "words": ["...", ...],                      # optional labels for word spans
      "phones": [                                 # A3 / mel phone segments (preferred)
        {"start": 0.0, "end": 0.12, "phone": "p"},
        ...
      ],
      "phone_spans_seconds": [[t0, t1], ...],     # alt phone packing
      "phone_labels": ["p", ...],
      "granularity": "phone" | "word",
      "g2p_backend": "phonemizer_espeak_ru" | "grapheme_fallback",
      "method": "uniform" | "ctc" | "mfa" | ...,
      "defect_rate": 0.0
    }

Uniform method (Track A default for S1 smoke)
---------------------------------------------
Per utterance: TN (normalize/verbalize) → stress → G2P phones → equal-time
(or duration-proportional) spans across **phones**, not whole words.
Word spans are still written for level-A Talker priors; mel baseline / A3
prefer the ``phones`` list via ``segments_from_forced_entry``.

WSL packing path (corpus CTC)
-----------------------------
1. Author on Windows → commit/push → ``git pull`` in ``~/projects/tts-self``.
2. ``export HF_HOME=$HOME/hf_cache``; check free disk (≥15 GB recommended).
3. Prefer a **local** RU CTC checkpoint: ``--model /path/to/wav2vec2-ru``.
   Use ``discover_local_ctc_models()`` / ``--discover-ctc`` to scan
   ``~/datasets``, ``~/models``, and ``HF_HOME`` without Hub download.
4. Dry-run packing (no Hub)::

       python scripts/26_build_forced_align.py --config configs/data_ruslan_local.yaml \\
         --method uniform --max-utts 32 --out ~/datasets/tts/ruslan/align/forced.json

5. CTC (WSL + audio; char-CTC for typical RU wav2vec2)::

       python scripts/26_build_forced_align.py --config configs/data_ruslan_local.yaml \\
         --method ctc --model ~/models/wav2vec2-large-xlsr-53-russian \\
         --out ~/datasets/tts/ruslan/align/forced.json

6. Then ``python scripts/06_align.py --config configs/data_ruslan_local.yaml --level A``.

CTC path loads local ``Wav2Vec2ForCTC`` + processor, computes log-prob emissions,
maps TN/stress-stripped transcript to the model alphabet (characters + ``|`` for
space), runs ``torchaudio.functional.forced_align`` + ``merge_tokens``, and writes
second-level spans. Failed utterances are **skipped** (never silent-uniform
without labeling). MFA is optional — see ``mfa_crosscheck_stub``.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Iterable, Sequence

from minispeech.constants import FRAME_RATE
from minispeech.data.alignment import frames_from_seconds

log = logging.getLogger(__name__)


def split_words(text: str) -> list[str]:
    return [w for w in text.strip().split() if w]


def uniform_unit_spans_seconds(n_units: int, dur_s: float) -> list[tuple[float, float]]:
    """Equal-time spans over ``[0, dur_s]`` for ``n_units`` labels (phones or words)."""
    if n_units <= 0:
        return []
    dur_s = max(float(dur_s), 1e-3)
    step = dur_s / n_units
    return [(i * step, (i + 1) * step) for i in range(n_units)]


def uniform_word_spans_seconds(n_words: int, dur_s: float) -> list[tuple[float, float]]:
    """Equal-time word spans over ``[0, dur_s]`` (skeleton when CTC is unavailable)."""
    return uniform_unit_spans_seconds(n_words, dur_s)


def proportional_unit_spans_seconds(
    weights: Sequence[float],
    dur_s: float,
) -> list[tuple[float, float]]:
    """Allocate ``dur_s`` across units proportional to positive ``weights``."""
    if not weights:
        return []
    dur_s = max(float(dur_s), 1e-3)
    w = [max(float(x), 1e-6) for x in weights]
    total = sum(w)
    spans: list[tuple[float, float]] = []
    t = 0.0
    for i, wi in enumerate(w):
        step = dur_s * (wi / total)
        t1 = dur_s if i == len(w) - 1 else t + step
        spans.append((t, t1))
        t = t1
    return spans


def text_to_phones_for_align(
    text: str,
    *,
    already_stressed: bool = False,
    use_accentuator: bool = True,
) -> tuple[list[str], dict[str, Any]]:
    """TN → stress → G2P phone list for uniform / CTC packing.

    Returns ``(phones, meta)`` where meta records backend and intermediate text.
    Does not download models; accentuate / phonemizer fall back gracefully.
    """
    from minispeech.frontend.g2p import backend_name, text_to_phonemes
    from minispeech.frontend.normalize import has_stress_marks
    from minispeech.frontend.stress import accentuate, to_canonical_stress
    from minispeech.frontend.verbalize import normalize_for_tts

    raw = (text or "").strip()
    if not raw:
        return [], {"text_tn": "", "text_stressed": "", "g2p_backend": backend_name()}

    tn = normalize_for_tts(raw)
    if already_stressed or has_stress_marks(tn) or has_stress_marks(raw):
        # Prefer stress already on the utterance (RUSLAN text_stressed).
        stressed = to_canonical_stress(tn if has_stress_marks(tn) else raw)
        # Re-apply TN on stressed string only if TN wiped marks (rare); keep stressed.
        if not has_stress_marks(stressed):
            stressed = to_canonical_stress(raw)
            # Still run light TN pieces that preserve stress
            from minispeech.frontend.normalize import normalize_text

            stressed = to_canonical_stress(normalize_text(stressed))
    else:
        stressed = to_canonical_stress(accentuate(tn, use_model=use_accentuator))

    phones = [p for p in text_to_phonemes(stressed, assume_stressed=True) if p and p != "_"]
    meta = {
        "text_tn": tn,
        "text_stressed": stressed,
        "g2p_backend": backend_name(),
        "n_phones": len(phones),
    }
    return phones, meta


def phone_dicts_from_spans(
    phones: Sequence[str],
    spans_s: Sequence[tuple[float, float]],
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for i, phone in enumerate(phones):
        t0, t1 = spans_s[i] if i < len(spans_s) else (0.0, 0.0)
        out.append({"start": float(t0), "end": float(t1), "phone": str(phone)})
    return out


def discover_local_ctc_models(
    roots: Sequence[str | Path] | None = None,
    *,
    max_depth: int = 4,
) -> list[dict[str, Any]]:
    """Scan local dirs for CTC-like checkpoints (no Hub download).

    Looks under ``~/datasets``, ``~/models``, and ``HF_HOME`` / ``~/hf_cache``
    for folders with ``config.json`` mentioning CTC / Wav2Vec2ForCTC / etc.
    """
    if roots is None:
        hf = os.environ.get("HF_HOME") or os.environ.get("HUGGINGFACE_HUB_CACHE")
        roots = [
            Path.home() / "datasets",
            Path.home() / "models",
            Path(hf) if hf else Path.home() / "hf_cache",
        ]
    needles = (
        "Wav2Vec2ForCTC",
        "HubertForCTC",
        "Data2VecAudioForCTC",
        "UniSpeechSatForCTC",
        "ForCTC",
        '"architectures"',
    )
    found: list[dict[str, Any]] = []
    seen: set[str] = set()
    for root in roots:
        root_p = Path(root).expanduser()
        if not root_p.is_dir():
            continue
        # Prefer shallow named hits first
        candidates = [root_p]
        try:
            for p in root_p.rglob("config.json"):
                try:
                    rel_depth = len(p.relative_to(root_p).parts)
                except ValueError:
                    continue
                if rel_depth > max_depth + 1:
                    continue
                candidates.append(p.parent)
        except OSError as exc:
            log.debug("CTC scan skip %s: %s", root_p, exc)
            continue
        for d in candidates:
            key = str(d.resolve()) if d.exists() else str(d)
            if key in seen:
                continue
            cfg = d / "config.json"
            if not cfg.is_file():
                continue
            try:
                text = cfg.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue
            if not any(n in text for n in needles if n != '"architectures"'):
                # require an actual CTC architecture string
                continue
            if "ForCTC" not in text and "CTC" not in text:
                continue
            seen.add(key)
            found.append(
                {
                    "path": key,
                    "has_pytorch_model": (d / "pytorch_model.bin").is_file()
                    or (d / "model.safetensors").is_file()
                    or any(d.glob("model*.safetensors")),
                    "root": str(root_p),
                }
            )
    return found


def seconds_to_frame_spans(
    spans_s: Sequence[tuple[float, float]],
    *,
    frame_rate: float = FRAME_RATE,
    n_frames: int | None = None,
) -> list[tuple[int, int]]:
    """Convert second spans to half-open frame index spans."""
    out: list[tuple[int, int]] = []
    for t0, t1 in spans_s:
        s, e = frames_from_seconds(t0, t1, frame_rate=frame_rate)
        if e <= s:
            e = s + 1
        out.append((s, e))
    if n_frames is not None and out:
        # Clamp and cover trailing frames with the last word
        clamped: list[tuple[int, int]] = []
        for s, e in out:
            s = max(0, min(s, n_frames))
            e = max(s, min(e, n_frames))
            clamped.append((s, e))
        if clamped[-1][1] < n_frames:
            s0, _ = clamped[-1]
            clamped[-1] = (s0, n_frames)
        out = clamped
    return out


def token_word_ids_from_words(
    words: Sequence[str],
    n_tokens: int | None = None,
) -> tuple[list[int], list[int]]:
    """Map one text-token slot per word (identity packing).

    When real BPE ``input_ids`` exist, prefer ``token_word_ids_from_char_spans``.
    Returns ``(token_word_ids, token_char_lens)``.
    """
    ids = list(range(len(words)))
    lens = [max(1, len(w)) for w in words]
    if n_tokens is not None and n_tokens > 0:
        if n_tokens < len(ids):
            ids, lens = ids[:n_tokens], lens[:n_tokens]
        elif n_tokens > len(ids):
            # Pad with last word id so L matches text-hidden length
            last = ids[-1] if ids else 0
            last_len = lens[-1] if lens else 1
            while len(ids) < n_tokens:
                ids.append(last)
                lens.append(last_len)
    return ids, lens


def token_word_ids_from_char_spans(
    words: Sequence[str],
    token_strings: Sequence[str],
) -> tuple[list[int], list[int]]:
    """Assign each token to a word by walking the whitespace-joined character stream."""
    if not words:
        return [], []
    joined = " ".join(words)
    char_word: list[int] = []
    wi = 0
    pos = 0
    for w in words:
        for _ in w:
            char_word.append(wi)
            pos += 1
        if pos < len(joined) and joined[pos] == " ":
            char_word.append(wi)
            pos += 1
        wi += 1

    ids: list[int] = []
    lens: list[int] = []
    cursor = 0
    for tok in token_strings:
        while cursor < len(joined) and joined[cursor] == " ":
            cursor += 1
        n = max(1, len(tok))
        w_id = char_word[cursor] if cursor < len(char_word) else (len(words) - 1)
        ids.append(w_id)
        lens.append(n)
        cursor = min(len(char_word), cursor + n)
    return ids, lens


def pack_forced_entry(
    *,
    words: Sequence[str],
    dur_s: float,
    n_frames: int,
    token_word_ids: Sequence[int] | None = None,
    token_char_lens: Sequence[int] | None = None,
    word_spans_seconds: Sequence[tuple[float, float]] | None = None,
    phones: Sequence[str] | None = None,
    phone_spans_seconds: Sequence[tuple[float, float]] | None = None,
    phone_weights: Sequence[float] | None = None,
    method: str = "uniform",
    granularity: str | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build one ``forced.json`` value for an utterance.

    When ``phones`` is provided, equal-time (or ``phone_weights``-proportional)
    phone spans are written and ``granularity`` defaults to ``\"phone\"``.
    Word spans remain for Talker level-A priors.
    """
    if word_spans_seconds is None:
        word_spans_seconds = uniform_word_spans_seconds(len(words), dur_s)
    spans_f = seconds_to_frame_spans(word_spans_seconds, n_frames=n_frames)
    if token_word_ids is None:
        token_word_ids, auto_lens = token_word_ids_from_words(words)
        if token_char_lens is None:
            token_char_lens = auto_lens
    elif token_char_lens is None:
        token_char_lens = [max(1, len(w)) for w in words]
        # length may not match; pad/truncate to token_word_ids
        tw = list(token_word_ids)
        lens = list(token_char_lens)
        if len(lens) < len(tw):
            lens.extend([1] * (len(tw) - len(lens)))
        token_char_lens = lens[: len(tw)]

    phone_list = list(phones) if phones is not None else []
    if phone_list and phone_spans_seconds is None:
        if phone_weights is not None and len(phone_weights) == len(phone_list):
            phone_spans_seconds = proportional_unit_spans_seconds(phone_weights, dur_s)
        else:
            phone_spans_seconds = uniform_unit_spans_seconds(len(phone_list), dur_s)

    if granularity is None:
        granularity = "phone" if phone_list else "word"

    entry: dict[str, Any] = {
        "word_spans_frames": [list(x) for x in spans_f],
        "token_word_ids": list(token_word_ids),
        "token_char_lens": list(token_char_lens),
        "word_spans_seconds": [list(x) for x in word_spans_seconds],
        "words": list(words),
        "method": method,
        "granularity": granularity,
        "defect_rate": 0.0,
        "n_frames": int(n_frames),
        "n_words": len(words),
    }
    if phone_list:
        assert phone_spans_seconds is not None
        entry["phones"] = phone_dicts_from_spans(phone_list, phone_spans_seconds)
        entry["phone_spans_seconds"] = [list(x) for x in phone_spans_seconds]
        entry["phone_labels"] = list(phone_list)
        entry["n_phones"] = len(phone_list)
    if extra:
        entry.update(extra)
    return entry


def write_forced_json(path: str | Path, forced: dict[str, Any]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(forced, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def load_forced_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def merge_forced(existing: dict[str, Any], new_entries: dict[str, Any]) -> dict[str, Any]:
    out = dict(existing)
    out.update(new_entries)
    return out


def ctc_forced_align_available() -> bool:
    try:
        import torchaudio
        from torchaudio.functional import forced_align  # noqa: F401

        _ = torchaudio
        return True
    except Exception:
        return False


def _mfa_cli() -> list[str] | None:
    """Return argv prefix to invoke MFA, or None if unavailable.

    Prefers ``mfa`` on PATH (e.g. ``~/bin/mfa`` wrapper). Falls back to
    user-local micromamba env ``mfa`` without requiring sudo.
    """
    import os
    import shutil
    from pathlib import Path

    which = shutil.which("mfa")
    if which:
        return [which]
    home = Path.home()
    for candidate in (
        home / "bin" / "mfa",
        home / ".local" / "bin" / "mfa",
    ):
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return [str(candidate)]
    micromamba = shutil.which("micromamba") or str(home / "bin" / "micromamba")
    if Path(micromamba).is_file():
        # Existence of env is checked at run time; prefix is still useful.
        return [micromamba, "run", "-n", "mfa", "mfa"]
    return None


def mfa_available() -> bool:
    """True if Montreal Forced Aligner CLI appears runnable (optional)."""
    return _mfa_cli() is not None


def mfa_crosscheck_stub(
    *,
    corpus_dir: str | Path | None = None,
    dictionary: str | None = None,
    acoustic_model: str | None = None,
) -> dict[str, Any]:
    """Status dict for MFA (install + CLI presence). Prefer ``build_mfa_corpus_entries``."""
    return {
        "status": "not_run",
        "mfa_on_path": mfa_available(),
        "corpus_dir": None if corpus_dir is None else str(corpus_dir),
        "dictionary": dictionary,
        "acoustic_model": acoustic_model,
        "reason": (
            "Use scripts/26_build_forced_align.py --method mfa --mfa-run "
            "after installing MFA user-local (micromamba/conda-forge)."
        ),
        "suggested_cli": (
            "mfa align <corpus_dir> <dictionary> <acoustic_model> <output_dir>"
            if corpus_dir
            else None
        ),
    }


def _parse_textgrid_phones(path: Path) -> list[tuple[float, float, str]]:
    """Minimal TextGrid IntervalTier parser → ``(start, end, label)`` phones.

    Prefers tiers named ``phones`` / ``phone``; falls back to the last interval tier.
    Skips empty / silence-like labels that MFA marks as ``""`` / ``sil`` / ``sp``.
    """
    text = path.read_text(encoding="utf-8", errors="replace")
    # Detect short-text vs long-text format via presence of 'xmin =' lines.
    tiers: list[tuple[str, list[tuple[float, float, str]]]] = []
    name = ""
    intervals: list[tuple[float, float, str]] = []
    xmin = xmax = None
    text_val: str | None = None
    in_interval = False

    def _flush_interval() -> None:
        nonlocal xmin, xmax, text_val, in_interval
        if in_interval and xmin is not None and xmax is not None and text_val is not None:
            intervals.append((float(xmin), float(xmax), text_val))
        xmin = xmax = None
        text_val = None
        in_interval = False

    for raw in text.splitlines():
        line = raw.strip()
        if line.startswith("name =") or line.startswith("name="):
            _flush_interval()
            if intervals:
                tiers.append((name, intervals))
                intervals = []
            name = line.split("=", 1)[1].strip().strip('"')
        elif "intervals [" in line or line.startswith("intervals ["):
            _flush_interval()
            in_interval = True
        elif in_interval and (line.startswith("xmin") or line.startswith("xmin =")):
            try:
                xmin = float(line.split("=", 1)[1].strip())
            except ValueError:
                pass
        elif in_interval and (line.startswith("xmax") or line.startswith("xmax =")):
            try:
                xmax = float(line.split("=", 1)[1].strip())
            except ValueError:
                pass
        elif in_interval and (line.startswith("text") or line.startswith("text =")):
            text_val = line.split("=", 1)[1].strip().strip('"')
    _flush_interval()
    if intervals:
        tiers.append((name, intervals))

    if not tiers:
        return []
    preferred = None
    for n, iv in tiers:
        if n.lower() in ("phones", "phone", "phonemes"):
            preferred = iv
            break
    if preferred is None:
        preferred = tiers[-1][1]
    out: list[tuple[float, float, str]] = []
    for t0, t1, lab in preferred:
        lab_s = (lab or "").strip()
        if not lab_s:
            continue
        out.append((float(t0), float(t1), lab_s))
    return out


def prepare_mfa_corpus(
    rows: Iterable[Any],
    corpus_dir: str | Path,
    *,
    max_utts: int | None = None,
    text_attr: str = "text_stressed",
) -> dict[str, Any]:
    """Write MFA lab layout: ``<utt_id>.wav`` symlink/copy + ``<utt_id>.lab`` transcript."""
    import shutil

    corpus_dir = Path(corpus_dir).expanduser()
    corpus_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    written = 0
    for i, r in enumerate(rows):
        if max_utts is not None and i >= max_utts:
            break
        n += 1
        utt_id = str(getattr(r, "utt_id", f"utt_{i}"))
        text_stressed = getattr(r, text_attr, None) or ""
        text_raw = getattr(r, "text_raw", "") or ""
        text = str(text_stressed or text_raw).strip()
        if not text:
            continue
        from minispeech.frontend.stress import strip_stress

        lab = strip_stress(text).replace("\n", " ").strip()
        wav_src = Path(str(getattr(r, "wav", ""))).expanduser()
        if not wav_src.is_file():
            continue
        dst_wav = corpus_dir / f"{utt_id}.wav"
        dst_lab = corpus_dir / f"{utt_id}.lab"
        if not dst_wav.exists():
            try:
                dst_wav.symlink_to(wav_src.resolve())
            except OSError:
                shutil.copy2(wav_src, dst_wav)
        dst_lab.write_text(lab + "\n", encoding="utf-8")
        written += 1
    return {"corpus_dir": str(corpus_dir), "n_seen": n, "n_written": written}


def build_mfa_corpus_entries(
    rows: Iterable[Any],
    *,
    corpus_dir: str | Path,
    output_dir: str | Path,
    dictionary: str = "russian_mfa",
    acoustic_model: str = "russian_mfa",
    max_utts: int | None = None,
    run_align: bool = True,
    text_attr: str = "text_stressed",
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Run MFA (optional) and convert TextGrids into ``forced.json`` phone entries.

    Requires ``mfa`` on PATH. Does not use sudo. Returns ``(forced, stats)``.
    """
    import subprocess

    rows_list = list(rows)
    prep = prepare_mfa_corpus(
        rows_list, corpus_dir, max_utts=max_utts, text_attr=text_attr
    )
    corpus_dir = Path(prep["corpus_dir"])
    output_dir = Path(output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    stats: dict[str, Any] = {
        "method": "mfa",
        "mfa_on_path": mfa_available(),
        "dictionary": dictionary,
        "acoustic_model": acoustic_model,
        "prepare": prep,
        "output_dir": str(output_dir),
    }
    if run_align:
        cli = _mfa_cli()
        if not cli:
            stats["status"] = "mfa_not_installed"
            stats["reason"] = (
                "mfa not on PATH. Install user-local via micromamba: "
                "micromamba install -c conda-forge montreal-forced-aligner"
            )
            return {}, stats
        import os

        env = os.environ.copy()
        env.setdefault("MAMBA_ROOT_PREFIX", str(Path.home() / "micromamba"))
        # Ensure ~/bin wrappers resolve when subprocess PATH is minimal.
        home_bin = str(Path.home() / "bin")
        env["PATH"] = home_bin + os.pathsep + env.get("PATH", "")
        cmd = [
            *cli,
            "align",
            str(corpus_dir),
            dictionary,
            acoustic_model,
            str(output_dir),
            "--clean",
            "--overwrite",
            "--num_jobs",
            "1",
            "--single_speaker",
        ]
        stats["cli"] = cmd
        try:
            # Do not capture stdout/stderr into pipes (can SIGPIPE / OOM on large MFA logs).
            log_path = output_dir / "mfa_align_console.log"
            output_dir.mkdir(parents=True, exist_ok=True)
            with open(log_path, "w", encoding="utf-8") as logf:
                proc = subprocess.run(
                    cmd,
                    check=False,
                    stdout=logf,
                    stderr=subprocess.STDOUT,
                    timeout=7200,
                    env=env,
                )
            stats["log"] = str(log_path)
            stats["returncode"] = proc.returncode
            try:
                stats["log_tail"] = log_path.read_text(encoding="utf-8", errors="replace")[-2000:]
            except OSError:
                stats["log_tail"] = ""
            if proc.returncode != 0:
                stats["status"] = "mfa_align_failed"
                return {}, stats
        except Exception as exc:  # noqa: BLE001
            stats["status"] = "mfa_align_exception"
            stats["error"] = str(exc)
            return {}, stats

    # Index rows by utt_id for duration / words
    by_id = {str(getattr(r, "utt_id", "")): r for r in rows_list}
    forced: dict[str, Any] = {}
    n_ok = 0
    n_skip = 0
    coverages: list[float] = []
    for tg in sorted(output_dir.glob("**/*.TextGrid")):
        utt_id = tg.stem
        phones = _parse_textgrid_phones(tg)
        if not phones:
            n_skip += 1
            continue
        r = by_id.get(utt_id)
        dur_s = float(getattr(r, "dur_s", 0.0) or 0.0) if r is not None else 0.0
        if dur_s <= 0:
            dur_s = phones[-1][1]
        labels = [p for _, _, p in phones]
        spans = [(float(a), float(b)) for a, b, _ in phones]
        covered = sum(max(0.0, b - a) for a, b in spans)
        cover = covered / max(dur_s, 1e-6)
        coverages.append(cover)
        words = split_words(strip_stress_safe(getattr(r, text_attr, None) or getattr(r, "text_raw", "") or ""))
        if not words:
            words = ["_"]
        n_frames = int(getattr(r, "n_frames", 0) or 0) if r is not None else 0
        if n_frames <= 0:
            n_frames = max(1, int(round(dur_s * FRAME_RATE)))
        tw, tl = token_word_ids_from_words(words)
        forced[utt_id] = pack_forced_entry(
            words=words,
            dur_s=dur_s,
            n_frames=max(n_frames, 1),
            token_word_ids=tw,
            token_char_lens=tl,
            phones=labels,
            phone_spans_seconds=spans,
            method="mfa",
            granularity="phone",
            extra={
                "align_unit": "phone",
                "g2p_backend": "mfa",
                "mfa_textgrid": str(tg),
                "cover_ratio_vs_end": cover,
            },
        )
        n_ok += 1

    stats.update(
        {
            "status": "ok" if forced else "no_textgrids",
            "n_ok": n_ok,
            "n_skip_empty": n_skip,
            "success_rate": (n_ok / max(prep["n_written"], 1)),
            "mean_cover_ratio": float(sum(coverages) / len(coverages)) if coverages else None,
        }
    )
    return forced, stats


def strip_stress_safe(text: str) -> str:
    try:
        from minispeech.frontend.stress import strip_stress

        return strip_stress(text)
    except Exception:
        return text.replace("\u0301", "")


# Wav2Vec2 feature encoder: 320 samples @ 16 kHz → 20 ms / frame (≈50 Hz).
DEFAULT_W2V_SECONDS_PER_FRAME = 320.0 / 16_000.0


def transcript_for_ctc(
    text: str,
    *,
    already_stressed: bool = False,
) -> dict[str, str]:
    """TN → (optional stress) → strip stress → lowercase letters for char-CTC.

    Returns keys ``text_tn``, ``text_stressed``, ``text_ctc`` (no stress marks).
    """
    from minispeech.frontend.normalize import has_stress_marks, normalize_text
    from minispeech.frontend.stress import accentuate, strip_stress, to_canonical_stress
    from minispeech.frontend.verbalize import normalize_for_tts

    raw = (text or "").strip()
    if not raw:
        return {"text_tn": "", "text_stressed": "", "text_ctc": ""}

    tn = normalize_for_tts(raw)
    if already_stressed or has_stress_marks(tn) or has_stress_marks(raw):
        stressed = to_canonical_stress(tn if has_stress_marks(tn) else raw)
        if not has_stress_marks(stressed):
            stressed = to_canonical_stress(normalize_text(raw))
    else:
        stressed = to_canonical_stress(accentuate(tn, use_model=False))

    plain = strip_stress(stressed).lower()
    # Keep letters / digits / whitespace; drop punctuation (CTC alphabets are tight).
    cleaned_chars: list[str] = []
    for ch in plain:
        if ch.isspace():
            if cleaned_chars and cleaned_chars[-1] != " ":
                cleaned_chars.append(" ")
        elif ch.isalpha() or ch.isdigit() or ch in "-":
            cleaned_chars.append(ch)
    text_ctc = "".join(cleaned_chars).strip()
    return {"text_tn": tn, "text_stressed": stressed, "text_ctc": text_ctc}


def ctc_labels_from_transcript(
    text_ctc: str,
    processor: Any,
) -> tuple[list[int], list[str]]:
    """Map cleaned transcript to CTC target ids + printable labels (no blank).

    Uses the processor tokenizer alphabet. Spaces become the word-delimiter
    token when present (typically ``|`` for wav2vec2 char-CTC).
    """
    tok = processor.tokenizer
    delim = getattr(tok, "word_delimiter_token", None) or "|"
    # Prefer tokenizer.encode path (handles | substitution for space).
    enc = tok(text_ctc, return_tensors=None, add_special_tokens=False)
    ids = [int(x) for x in enc["input_ids"]]
    blank = int(getattr(tok, "pad_token_id", 0) or 0)
    # Drop blanks / specials that must not appear in forced_align targets.
    specials = {
        blank,
        int(getattr(tok, "bos_token_id", -1) or -1),
        int(getattr(tok, "eos_token_id", -1) or -1),
        int(getattr(tok, "unk_token_id", -1) or -1),
    }
    ids = [i for i in ids if i not in specials]
    id2tok = {int(v): str(k) for k, v in tok.get_vocab().items()}
    labels = [id2tok.get(i, delim if i == getattr(tok, "word_delimiter_token_id", None) else "?") for i in ids]
    return ids, labels


def _filter_ctc_chars_keep_stress(stressed: str) -> list[str]:
    """Same keep/collapse rules as ``transcript_for_ctc`` text_ctc, preserving U+0301.

    Returns one label per kept base character; stressed vowels are a base char
    plus combining acute (length of this list must equal ``len(text_ctc)``).
    """
    from minispeech.constants import STRESS_CHAR
    from minispeech.frontend.stress import to_canonical_stress

    s = to_canonical_stress(stressed or "").lower()
    labels: list[str] = []
    for ch in s:
        if ch == STRESS_CHAR:
            if labels and labels[-1] not in ("|", " ") and STRESS_CHAR not in labels[-1]:
                labels[-1] = labels[-1] + STRESS_CHAR
            continue
        if ch.isspace():
            if labels and labels[-1] != "|":
                labels.append("|")
            continue
        if ch.isalpha() or ch.isdigit() or ch in "-":
            labels.append(ch)
    # strip leading/trailing word delimiters to match text_ctc.strip()
    while labels and labels[0] == "|":
        labels.pop(0)
    while labels and labels[-1] == "|":
        labels.pop()
    return labels


def text_to_ctc_char_labels(
    text: str,
    *,
    already_stressed: bool = False,
    keep_stress: bool = False,
) -> tuple[list[str], dict[str, Any]]:
    """Infer-time / train-time char units matching RU wav2vec2 alphabet (no model load).

    Spaces → ``|``. When ``keep_stress=True``, stressed vowels keep the combining
    acute U+0301 as part of the label (``а́``), while still matching CTC span
    length 1:1 with ``text_ctc`` base characters. Default is False so legacy
    mel80 / mel80_hifigan checkpoints keep unstressed units at infer.
    """
    meta = transcript_for_ctc(text, already_stressed=already_stressed)
    text_ctc = meta["text_ctc"]
    if keep_stress:
        labels = _filter_ctc_chars_keep_stress(meta["text_stressed"])
        # Map spaces already emitted as ``|``; ensure 1:1 with text_ctc bases.
        plain = []
        for lab in labels:
            base = lab.replace("\u0301", "")
            plain.append(" " if base == "|" else base)
        plain_s = "".join(plain)
        if plain_s != text_ctc:
            # Fall back to unstressed labels if filters diverge (safety).
            log.warning(
                "stressed CTC labels length/filter mismatch "
                "(plain=%r text_ctc=%r); falling back to unstressed",
                plain_s[:80],
                text_ctc[:80],
            )
            labels = ["|" if ch == " " else ch for ch in text_ctc]
            meta = {**meta, "stress_labels": False}
        else:
            meta = {**meta, "stress_labels": True}
    else:
        labels = ["|" if ch == " " else ch for ch in text_ctc]
        meta = {**meta, "stress_labels": False}
    meta = {**meta, "granularity": "char", "n_labels": len(labels)}
    return labels, meta


def load_wav2vec2_ctc(
    model_path: str | Path,
    *,
    device: str | Any = "cpu",
    allow_download: bool = False,
) -> tuple[Any, Any, dict[str, Any]]:
    """Load local ``Wav2Vec2ForCTC`` + processor. No Hub unless ``allow_download``."""
    import torch
    from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

    path = Path(model_path).expanduser()
    local_only = not allow_download
    if local_only and not path.exists():
        raise FileNotFoundError(f"CTC model path missing (local_files_only): {path}")
    src = str(path if path.exists() else model_path)
    processor = Wav2Vec2Processor.from_pretrained(src, local_files_only=local_only)
    model = Wav2Vec2ForCTC.from_pretrained(src, local_files_only=local_only)
    model.eval()
    dev = torch.device(device)
    model.to(dev)
    blank = int(getattr(model.config, "pad_token_id", 0) or 0)
    sr = int(processor.feature_extractor.sampling_rate)
    info = {
        "model_path": src,
        "sample_rate": sr,
        "blank_id": blank,
        "vocab_size": int(getattr(model.config, "vocab_size", 0) or 0),
        "device": str(dev),
    }
    return model, processor, info


def compute_ctc_emissions(
    waveform: Any,
    model: Any,
    processor: Any,
    *,
    sample_rate: int | None = None,
    device: str | Any = "cpu",
) -> tuple[Any, float]:
    """Return ``(log_probs [T, C], seconds_per_frame)`` for one utterance.

    ``waveform`` may be a wav path or a 1-D float tensor/ndarray. When a path
    is given, audio is loaded and resampled to the processor sample rate.
    """
    import torch
    import torch.nn.functional as F

    from minispeech.data.mel import load_wav_mono

    target_sr = int(processor.feature_extractor.sampling_rate)
    if isinstance(waveform, (str, Path)):
        wav_np, sr = load_wav_mono(waveform, target_sr=target_sr)
        wav = torch.from_numpy(wav_np).float()
        sample_rate = sr
    else:
        wav = torch.as_tensor(waveform).float().reshape(-1)
        sr_in = int(sample_rate or target_sr)
        if sr_in != target_sr:
            try:
                import torchaudio.functional as AF

                wav = AF.resample(wav.unsqueeze(0), sr_in, target_sr).squeeze(0)
            except Exception:
                try:
                    import librosa

                    wav = torch.from_numpy(
                        librosa.resample(wav.numpy(), orig_sr=sr_in, target_sr=target_sr)
                    ).float()
                except Exception as exc:
                    raise RuntimeError(f"resample {sr_in}->{target_sr} failed: {exc}") from exc
        sample_rate = target_sr

    dur_s = float(wav.numel()) / float(sample_rate)
    inputs = processor(wav.numpy(), sampling_rate=int(sample_rate), return_tensors="pt", padding=False)
    input_values = inputs.input_values.to(device)
    with torch.no_grad():
        logits = model(input_values).logits  # [1, T, C]
    log_probs = F.log_softmax(logits, dim=-1)[0].cpu()
    t_frames = int(log_probs.shape[0])
    spf = (dur_s / t_frames) if t_frames > 0 else DEFAULT_W2V_SECONDS_PER_FRAME
    return log_probs, float(spf)


def try_ctc_align_emissions(
    emission: Any,
    targets: Any,
    *,
    blank: int = 0,
) -> list[tuple[int, int]] | None:
    """Thin wrapper around ``torchaudio.functional.forced_align`` (+ ``merge_tokens``).

    ``emission``: ``[T, C]`` log-probs; ``targets``: 1-D token ids (no blank).
    Returns list of half-open ``(frame_start, frame_end)`` per target token, or
    None on failure. Does not download models — caller supplies emissions.
    """
    try:
        import torch
        from torchaudio.functional import forced_align
    except Exception as exc:
        log.warning("torchaudio.forced_align unavailable: %s", exc)
        return None

    if not torch.is_tensor(emission):
        emission = torch.as_tensor(emission)
    if not torch.is_tensor(targets):
        targets = torch.as_tensor(targets)
    if emission.ndim != 2:
        raise ValueError("emission must be [T, C]")
    if targets.numel() == 0:
        return []
    if int(targets.min()) < 0 or int(targets.max()) >= int(emission.shape[-1]):
        log.warning("target ids out of emission vocab range")
        return None
    if blank in targets.tolist():
        log.warning("targets contain blank id=%s", blank)
        return None

    em = emission.unsqueeze(0).float()
    # torchaudio CUDA kernel historically prefers int32 targets
    tgt = targets.unsqueeze(0).to(dtype=torch.int32)
    try:
        out = forced_align(em, tgt, blank=blank)
        if isinstance(out, tuple) and len(out) == 2:
            alignments, scores = out
        else:
            alignments, scores = out, None
    except Exception as exc:
        log.warning("forced_align failed: %s", exc)
        return None

    path = alignments[0]
    # Prefer merge_tokens (handles repeated chars with intervening blanks).
    try:
        from torchaudio.functional import merge_tokens

        if scores is not None:
            token_spans = merge_tokens(path, scores[0], blank=blank)
            spans = [(int(s.start), int(s.end)) for s in token_spans]
            if len(spans) == int(targets.numel()):
                return spans
            log.warning(
                "merge_tokens span count %d != targets %d; falling back to path collapse",
                len(spans),
                int(targets.numel()),
            )
    except Exception as exc:
        log.debug("merge_tokens unavailable/failed: %s", exc)

    # Fallback: collapse non-blank runs (blank separates repeated labels).
    spans_fb: list[tuple[int, int]] = []
    cur_lab: int | None = None
    start = 0
    for t, lab in enumerate(path.tolist()):
        lab = int(lab)
        if lab == blank:
            if cur_lab is not None:
                spans_fb.append((start, t))
                cur_lab = None
            continue
        if cur_lab is None:
            cur_lab = lab
            start = t
        elif lab != cur_lab:
            spans_fb.append((start, t))
            cur_lab = lab
            start = t
    if cur_lab is not None:
        spans_fb.append((start, len(path)))
    if len(spans_fb) != int(targets.numel()):
        log.warning(
            "align span count %d != targets %d",
            len(spans_fb),
            int(targets.numel()),
        )
        return None
    return spans_fb


def _word_spans_from_char_labels(
    labels: Sequence[str],
    spans_s: Sequence[tuple[float, float]],
    words: Sequence[str],
    *,
    word_delim: str = "|",
) -> list[tuple[float, float]]:
    """Group char spans into word spans using ``|`` delimiters."""
    groups: list[list[int]] = []
    cur: list[int] = []
    for i, lab in enumerate(labels):
        if lab == word_delim:
            if cur:
                groups.append(cur)
                cur = []
            continue
        cur.append(i)
    if cur:
        groups.append(cur)

    if not groups:
        # No letters — fall back to uniform over words
        return uniform_word_spans_seconds(len(words), spans_s[-1][1] if spans_s else 1.0)

    out: list[tuple[float, float]] = []
    for gi, idxs in enumerate(groups):
        t0 = spans_s[idxs[0]][0]
        t1 = spans_s[idxs[-1]][1]
        out.append((float(t0), float(t1)))
    # If word count mismatches (punctuation dropped), pad/truncate to len(words)
    if words:
        if len(out) < len(words):
            last_t = out[-1][1] if out else 0.0
            while len(out) < len(words):
                out.append((last_t, last_t))
        elif len(out) > len(words):
            # Merge extras into last word
            head, rest = out[: len(words) - 1], out[len(words) - 1 :]
            out = head + [(rest[0][0], rest[-1][1])]
    return out


def build_ctc_corpus_entries(
    rows: Iterable[Any],
    *,
    model_path: str | Path,
    max_utts: int | None = None,
    device: str | None = None,
    allow_download: bool = False,
    text_attr: str = "text_stressed",
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Real CTC forced alignment over manifest rows (local Wav2Vec2ForCTC).

    Aligns on the model token alphabet (char-CTC for RU wav2vec2). Failed
    utterances are skipped — never rewritten as unlabeled uniform spans.

    Returns ``(forced_entries, stats)``.
    """
    import torch

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model, processor, model_info = load_wav2vec2_ctc(
        model_path, device=device, allow_download=allow_download
    )
    blank = int(model_info["blank_id"])
    delim = getattr(processor.tokenizer, "word_delimiter_token", None) or "|"

    forced: dict[str, Any] = {}
    n_seen = 0
    n_ok = 0
    n_skip = 0
    skip_reasons: dict[str, int] = {}
    sample_segments: list[dict[str, Any]] = []

    def _skip(reason: str) -> None:
        nonlocal n_skip
        n_skip += 1
        skip_reasons[reason] = skip_reasons.get(reason, 0) + 1

    for i, r in enumerate(rows):
        if max_utts is not None and i >= max_utts:
            break
        n_seen += 1
        utt_id = str(getattr(r, "utt_id", f"utt_{i}"))
        text_stressed = getattr(r, text_attr, None) or ""
        text_raw = getattr(r, "text_raw", "") or ""
        text = str(text_stressed or text_raw)
        if not text.strip():
            _skip("empty_text")
            continue
        already = bool(text_stressed and str(text_stressed).strip())
        tmeta = transcript_for_ctc(text, already_stressed=already)
        text_ctc = tmeta["text_ctc"]
        if not text_ctc:
            _skip("empty_ctc_transcript")
            continue
        try:
            target_ids, labels = ctc_labels_from_transcript(text_ctc, processor)
        except Exception as exc:
            log.warning("%s tokenize failed: %s", utt_id, exc)
            _skip("tokenize")
            continue
        if not target_ids:
            _skip("empty_targets")
            continue

        wav_path = getattr(r, "wav", None)
        if not wav_path:
            _skip("missing_wav")
            continue
        wav_p = Path(str(wav_path)).expanduser()
        if not wav_p.is_file():
            _skip("wav_not_found")
            continue

        try:
            emission, spf = compute_ctc_emissions(wav_p, model, processor, device=device)
            spans_f = try_ctc_align_emissions(
                emission,
                torch.tensor(target_ids, dtype=torch.int32),
                blank=blank,
            )
        except Exception as exc:
            log.warning("%s CTC align error: %s", utt_id, exc)
            _skip("align_exception")
            continue
        if spans_f is None or len(spans_f) != len(labels):
            _skip("align_failed")
            continue

        spans_s = [(float(s) * spf, float(e) * spf) for s, e in spans_f]
        # Cover full utterance duration when trailing silence exists
        dur_s = float(getattr(r, "dur_s", 0.0) or 0.0)
        if dur_s <= 0:
            dur_s = spans_s[-1][1] if spans_s else emission.shape[0] * spf
        if spans_s and spans_s[-1][1] < dur_s:
            s0, _ = spans_s[-1]
            spans_s[-1] = (s0, dur_s)

        words = split_words(tmeta["text_stressed"] or text_ctc)
        if not words:
            words = [w for w in text_ctc.split() if w] or ["_"]
        word_spans = _word_spans_from_char_labels(labels, spans_s, words, word_delim=delim)

        n_frames = int(getattr(r, "n_frames", 0) or 0)
        if n_frames <= 0 and dur_s > 0:
            n_frames = max(1, int(round(dur_s * FRAME_RATE)))
        L = len(getattr(r, "input_ids", None) or []) or None
        tw, tl = token_word_ids_from_words(words, n_tokens=L)

        forced[utt_id] = pack_forced_entry(
            words=words,
            dur_s=dur_s,
            n_frames=max(n_frames, 1),
            token_word_ids=tw,
            token_char_lens=tl,
            word_spans_seconds=word_spans,
            phones=labels,
            phone_spans_seconds=spans_s,
            method="ctc",
            granularity="char",
            extra={
                "g2p_backend": "wav2vec2_char_ctc",
                "align_unit": "char",
                "text_tn": tmeta["text_tn"],
                "text_stressed_align": tmeta["text_stressed"],
                "text_ctc": text_ctc,
                "ctc_model": model_info["model_path"],
                "ctc_blank_id": blank,
                "ctc_seconds_per_frame": spf,
                "ctc_n_emission_frames": int(emission.shape[0]),
                "ctc_vocab_size": model_info["vocab_size"],
            },
        )
        n_ok += 1
        if len(sample_segments) < 3:
            sample_segments.append(
                {
                    "utt_id": utt_id,
                    "n_labels": len(labels),
                    "labels_head": labels[:12],
                    "spans_head": [list(x) for x in spans_s[:12]],
                    "text_ctc": text_ctc[:80],
                }
            )

    stats = {
        "status": "ok" if n_ok else "no_success",
        "n_seen": n_seen,
        "n_ok": n_ok,
        "n_skip": n_skip,
        "success_rate": (n_ok / n_seen) if n_seen else 0.0,
        "skip_reasons": skip_reasons,
        "sample_segments": sample_segments,
        "model": model_info,
        "granularity": "char",
        "align_unit": "char",
    }
    return forced, stats


def build_uniform_corpus_entries(
    rows: Iterable[Any],
    *,
    text_attr: str = "text_stressed",
    max_utts: int | None = None,
    use_accentuator: bool = True,
    proportional: bool = False,
) -> dict[str, Any]:
    """Pack **phone-level** uniform forced entries from manifest-like rows.

    Pipeline per utterance: TN → stress → G2P → equal (or proportional) time
    across phones. Word spans are retained for Talker priors; mel / A3 consumers
    read ``phones`` via ``segments_from_forced_entry``.
    """
    forced: dict[str, Any] = {}
    for i, r in enumerate(rows):
        if max_utts is not None and i >= max_utts:
            break
        text_stressed = getattr(r, text_attr, None) or ""
        text_raw = getattr(r, "text_raw", "") or ""
        text = str(text_stressed or text_raw)
        if not text.strip():
            continue
        already = bool(text_stressed and str(text_stressed).strip())
        phones, g2p_meta = text_to_phones_for_align(
            text,
            already_stressed=already,
            use_accentuator=use_accentuator,
        )
        # Word labels from stressed TN text (whitespace split)
        words = split_words(str(g2p_meta.get("text_stressed") or text))
        if not phones and not words:
            continue
        if not phones:
            # Last-resort: treat words as units (should be rare)
            phones = list(words)
            g2p_meta = {**g2p_meta, "g2p_backend": "word_fallback", "n_phones": len(phones)}
        n_frames = int(getattr(r, "n_frames", 0) or 0)
        dur_s = float(getattr(r, "dur_s", 0.0) or 0.0)
        if n_frames <= 0 and dur_s > 0:
            n_frames = max(1, int(round(dur_s * FRAME_RATE)))
        if dur_s <= 0 and n_frames > 0:
            dur_s = n_frames / FRAME_RATE
        L = len(getattr(r, "input_ids", None) or []) or None
        if not words:
            words = ["_"]
        tw, tl = token_word_ids_from_words(words, n_tokens=L)
        weights = [max(1.0, float(len(p))) for p in phones] if proportional else None
        forced[r.utt_id] = pack_forced_entry(
            words=words,
            dur_s=dur_s,
            n_frames=max(n_frames, 1),
            token_word_ids=tw,
            token_char_lens=tl,
            phones=phones,
            phone_weights=weights,
            method="uniform",
            granularity="phone",
            extra={
                "g2p_backend": g2p_meta.get("g2p_backend"),
                "text_tn": g2p_meta.get("text_tn"),
                "text_stressed_align": g2p_meta.get("text_stressed"),
            },
        )
    return forced

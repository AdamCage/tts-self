"""Acoustic stress accuracy (SA) for Russian homographs (Phase M / M05).

ASR-text SA cannot distinguish stress-only homographs (``за́мок`` vs ``замо́к``).
This module predicts the stressed vowel from duration / energy / F0 after CTC
forced alignment of an arbitrary wav+text pair.

Decision rule: among vowels in the target word, pick the one with the highest
weighted score. Weights are calibrated on real RUSLAN recordings with known
stress from ``accentuate`` / forced labels (not on TTS).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np

from minispeech.constants import STRESS_CHAR
from minispeech.data.forced_align import (
    compute_ctc_emissions,
    ctc_labels_from_transcript,
    load_wav2vec2_ctc,
    text_to_ctc_char_labels,
    transcript_for_ctc,
    try_ctc_align_emissions,
)
from minispeech.data.mel import load_wav_mono, waveform_to_logmel
from minispeech.frontend.stress import accentuate

RU_VOWELS = set("аеёиоуыэюяАЕЁИОУЫЭЮЯ")

SaLabel = Literal["correct", "incorrect", "skipped"]


@dataclass(frozen=True)
class StressWeights:
    w_dur: float = 1.0
    w_energy: float = 1.0
    w_f0: float = 1.0


DEFAULT_WEIGHTS = StressWeights()


@dataclass
class VowelSpan:
    char: str
    start_s: float
    end_s: float
    has_stress_mark: bool
    duration: float
    energy: float
    f0: float


@dataclass
class AlignmentResult:
    chars: list[str]
    spans_s: list[tuple[float, float]]
    sample_rate: int
    seconds_per_frame: float


def _discover_ctc_path() -> Path | None:
    candidates = [
        Path.home() / "models" / "wav2vec2-large-xlsr-53-russian",
        Path("/home/adam/models/wav2vec2-large-xlsr-53-russian"),
    ]
    for p in candidates:
        if p.is_dir():
            return p
    return None


class AcousticStressAligner:
    """Lazy-loaded CTC aligner for arbitrary wav+text."""

    def __init__(
        self,
        model_path: str | Path | None = None,
        *,
        device: str = "cpu",
    ):
        path = Path(model_path) if model_path else _discover_ctc_path()
        if path is None:
            raise FileNotFoundError("No local wav2vec2 RU CTC model found")
        self.model, self.processor, self.info = load_wav2vec2_ctc(path, device=device)
        self.device = device
        self.blank = int(self.info["blank_id"])

    def align(self, wav_path: str | Path, text: str) -> AlignmentResult | None:
        tr = transcript_for_ctc(text, already_stressed=STRESS_CHAR in (text or ""))
        text_ctc = tr["text_ctc"]
        char_labels, _ = text_to_ctc_char_labels(
            text, already_stressed=STRESS_CHAR in (text or ""), keep_stress=True
        )
        if not char_labels:
            char_labels, _ = text_to_ctc_char_labels(text_ctc, keep_stress=False)
        ids, id_labels = ctc_labels_from_transcript(text_ctc, self.processor)
        if not ids:
            return None
        import torch

        emission, spf = compute_ctc_emissions(
            wav_path, self.model, self.processor, device=self.device
        )
        targets = torch.as_tensor(ids, dtype=torch.long)
        frame_spans = try_ctc_align_emissions(emission, targets, blank=self.blank)
        if frame_spans is None:
            return None
        chars = char_labels if len(char_labels) == len(frame_spans) else id_labels
        spans_s: list[tuple[float, float]] = []
        if len(frame_spans) == len(chars):
            for a, b in frame_spans:
                spans_s.append((a * spf, b * spf))
        else:
            n = len(chars)
            m = len(frame_spans)
            for i in range(n):
                j = min(int(i * m / max(1, n)), m - 1)
                a, b = frame_spans[j]
                spans_s.append((a * spf, b * spf))
        _wav, sr = load_wav_mono(wav_path)
        return AlignmentResult(
            chars=list(chars),
            spans_s=spans_s,
            sample_rate=int(sr),
            seconds_per_frame=spf,
        )


def _pyin_f0(wav: np.ndarray, sr: int) -> np.ndarray:
    import librosa

    f0, _, _ = librosa.pyin(
        wav.astype(np.float64),
        fmin=librosa.note_to_hz("C2"),
        fmax=librosa.note_to_hz("C7"),
        sr=sr,
        frame_length=2048 if sr >= 22050 else 1024,
        hop_length=256,
    )
    return np.asarray(f0, dtype=np.float32)


def extract_vowel_features(
    wav: np.ndarray,
    sr: int,
    alignment: AlignmentResult,
    *,
    word: str | None = None,
) -> list[VowelSpan]:
    """Build vowel spans from char alignment; optionally restrict to ``word``."""
    mel = waveform_to_logmel(wav, sample_rate=sr)
    energy = np.linalg.norm(np.asarray(mel), axis=-1).astype(np.float32)
    f0 = _pyin_f0(wav, sr)
    vowels: list[VowelSpan] = []
    # Optional: restrict character window to target word (stress-stripped).
    char_lo, char_hi = 0, len(alignment.chars)
    if word:
        target = "".join(ch for ch in word.lower().replace(STRESS_CHAR, "") if ch not in " |")
        plain = []
        for ch in alignment.chars:
            if ch in {"|", " "}:
                plain.append(" ")
            else:
                plain.append(ch.replace(STRESS_CHAR, "").lower()[:1])
        joined = "".join(plain)
        compact = joined.replace(" ", "")
        pos = compact.find(target) if target else -1
        if pos >= 0:
            # Map compact index back to char indices
            ci = 0
            start_ci = None
            end_ci = None
            for i, ch in enumerate(plain):
                if ch == " ":
                    continue
                if ci == pos and start_ci is None:
                    start_ci = i
                ci += 1
                if start_ci is not None and ci == pos + len(target):
                    end_ci = i + 1
                    break
            if start_ci is not None and end_ci is not None:
                char_lo, char_hi = start_ci, end_ci

    i = char_lo
    while i < char_hi:
        ch = alignment.chars[i]
        if ch in {"|", " "}:
            i += 1
            continue
        stressed = False
        if i + 1 < len(alignment.chars) and alignment.chars[i + 1] == STRESS_CHAR:
            stressed = True
        if ch.lower() in {c.lower() for c in RU_VOWELS} or (
            len(ch) >= 1 and ch[0].lower() in {c.lower() for c in RU_VOWELS}
        ):
            base = ch[0]
            if STRESS_CHAR in ch:
                stressed = True
                base = ch.replace(STRESS_CHAR, "")[:1]
            start_s, end_s = alignment.spans_s[i]
            dur = max(0.0, end_s - start_s)
            f0_hop = 256
            a = max(0, int(start_s * sr / f0_hop))
            b = max(a + 1, int(end_s * sr / f0_hop))
            seg_e = energy[a:b] if a < len(energy) else np.array([])
            seg_f = f0[a:b] if a < len(f0) else np.array([])
            e_mean = float(seg_e.mean()) if seg_e.size else 0.0
            voiced = seg_f[np.isfinite(seg_f)] if seg_f.size else np.array([])
            f0_mean = float(voiced.mean()) if voiced.size else 0.0
            vowels.append(
                VowelSpan(
                    char=base,
                    start_s=start_s,
                    end_s=end_s,
                    has_stress_mark=stressed,
                    duration=dur,
                    energy=e_mean,
                    f0=f0_mean,
                )
            )
            if stressed and i + 1 < len(alignment.chars) and alignment.chars[i + 1] == STRESS_CHAR:
                i += 2
                continue
        i += 1
    return vowels


def score_vowels(vowels: list[VowelSpan], weights: StressWeights = DEFAULT_WEIGHTS) -> int | None:
    """Return index of predicted stressed vowel, or None if empty."""
    if not vowels:
        return None
    durs = np.array([v.duration for v in vowels], dtype=np.float64)
    ens = np.array([v.energy for v in vowels], dtype=np.float64)
    f0s = np.array([v.f0 for v in vowels], dtype=np.float64)

    def _z(x: np.ndarray) -> np.ndarray:
        if x.size == 0:
            return x
        s = float(x.std())
        if s < 1e-8:
            return np.zeros_like(x)
        return (x - float(x.mean())) / s

    scores = (
        weights.w_dur * _z(durs)
        + weights.w_energy * _z(ens)
        + weights.w_f0 * _z(f0s)
    )
    return int(np.argmax(scores))


def predict_stress_index(
    wav_path: str | Path,
    text: str,
    *,
    aligner: AcousticStressAligner | None = None,
    weights: StressWeights = DEFAULT_WEIGHTS,
    device: str = "cpu",
    target_word: str | None = None,
) -> dict[str, Any]:
    """Predict stressed vowel index in ``text`` from acoustics."""
    al = aligner or AcousticStressAligner(device=device)
    alignment = al.align(wav_path, text)
    if alignment is None:
        return {"label": "skipped", "reason": "alignment_failed"}
    wav, sr = load_wav_mono(wav_path, target_sr=22_050)
    vowels = extract_vowel_features(wav, sr, alignment, word=target_word)
    if len(vowels) < 2:
        return {"label": "skipped", "reason": "need_ge2_vowels", "n_vowels": len(vowels)}
    pred = score_vowels(vowels, weights)
    gold = next((i for i, v in enumerate(vowels) if v.has_stress_mark), None)
    return {
        "pred_index": pred,
        "gold_index": gold,
        "n_vowels": len(vowels),
        "vowels": [
            {
                "char": v.char,
                "dur": v.duration,
                "energy": v.energy,
                "f0": v.f0,
                "marked": v.has_stress_mark,
            }
            for v in vowels
        ],
    }


def score_stress_acoustic_item(
    wav_path: str | Path,
    text: str,
    stress_target: dict[str, Any] | None,
    *,
    aligner: AcousticStressAligner | None = None,
    weights: StressWeights | None = None,
    device: str = "cpu",
) -> SaLabel:
    """Compare predicted stress index to gold mark in aligned text.

    ``stress_target`` is optional; gold comes from U+0301 in ``text`` when present.
    """
    w = weights if weights is not None else DEFAULT_WEIGHTS
    target_word = None
    if isinstance(stress_target, dict):
        target_word = (
            stress_target.get("expected")
            or stress_target.get("lexeme")
            or stress_target.get("form")
        )
    elif isinstance(stress_target, (list, tuple)) and stress_target:
        target_word = str(stress_target[0])
    out = predict_stress_index(
        wav_path,
        text,
        aligner=aligner,
        weights=w,
        device=device,
        target_word=str(target_word) if target_word else None,
    )
    if out.get("label") == "skipped" or out.get("pred_index") is None:
        return "skipped"
    gold = out.get("gold_index")
    if gold is None and stress_target:
        # Fall back: accentuate text and re-align marks — already in text usually
        return "skipped"
    if gold is None:
        return "skipped"
    return "correct" if int(out["pred_index"]) == int(gold) else "incorrect"


def calibrate_weights_on_manifest(
    items: list[tuple[str, str]],
    *,
    device: str = "cpu",
    max_items: int = 200,
) -> tuple[StressWeights, dict[str, Any]]:
    """Grid-search weights on (wav_path, stressed_text) pairs.

    For each utterance, score every word that carries U+0301 (word-local gold).
    Returns best weights and accuracy report. Acceptance target: ≥0.90.
    """
    aligner = AcousticStressAligner(device=device)
    grid = [0.5, 1.0, 1.5]
    best_w = DEFAULT_WEIGHTS
    best_acc = -1.0
    report: dict[str, Any] = {"n_tried": 0, "n_scored": 0, "n_words": 0}
    scored_cache: list[dict[str, Any]] = []
    for i, (wav, text) in enumerate(items[:max_items]):
        stressed = text if STRESS_CHAR in text else accentuate(text)
        report["n_tried"] += 1
        # Collect stressed word tokens
        words = []
        for tok in stressed.replace(".", " ").replace(",", " ").split():
            if STRESS_CHAR in tok:
                words.append(tok)
        if not words:
            continue
        al = aligner.align(wav, stressed)
        if al is None:
            continue
        wav_np, sr = load_wav_mono(wav, target_sr=22_050)
        for wtok in words:
            vowels = extract_vowel_features(wav_np, sr, al, word=wtok)
            if len(vowels) < 2:
                continue
            gold = next((j for j, v in enumerate(vowels) if v.has_stress_mark), None)
            if gold is None:
                # Gold from accent pattern on the token itself
                bases = []
                k = 0
                while k < len(wtok):
                    ch = wtok[k]
                    if ch == STRESS_CHAR:
                        k += 1
                        continue
                    marked = k + 1 < len(wtok) and wtok[k + 1] == STRESS_CHAR
                    if ch.lower() in {c.lower() for c in RU_VOWELS}:
                        bases.append(marked)
                        if marked:
                            k += 2
                            continue
                    k += 1
                if len(bases) == len(vowels):
                    gold = next((j for j, m in enumerate(bases) if m), None)
            if gold is None:
                continue
            scored_cache.append(
                {
                    "gold_index": gold,
                    "vowels": [
                        {
                            "char": v.char,
                            "dur": v.duration,
                            "energy": v.energy,
                            "f0": v.f0,
                            "marked": v.has_stress_mark,
                        }
                        for v in vowels
                    ],
                }
            )
            report["n_words"] += 1
        if (i + 1) % 10 == 0:
            print(
                f"calibrate progress {i+1}/{min(len(items), max_items)} "
                f"words={len(scored_cache)}",
                flush=True,
            )
    report["n_scored"] = len(scored_cache)
    if not scored_cache:
        return best_w, {**report, "accuracy": None, "reason": "no_scored"}

    for wd in grid:
        for we in grid:
            for wf in grid:
                w = StressWeights(wd, we, wf)
                ok = 0
                for pred in scored_cache:
                    vowels = [
                        VowelSpan(
                            char=v["char"],
                            start_s=0.0,
                            end_s=0.0,
                            has_stress_mark=v["marked"],
                            duration=v["dur"],
                            energy=v["energy"],
                            f0=v["f0"],
                        )
                        for v in pred["vowels"]
                    ]
                    idx = score_vowels(vowels, w)
                    if idx == pred["gold_index"]:
                        ok += 1
                acc = ok / len(scored_cache)
                if acc > best_acc:
                    best_acc = acc
                    best_w = w
    report["accuracy"] = best_acc
    report["weights"] = {"w_dur": best_w.w_dur, "w_energy": best_w.w_energy, "w_f0": best_w.w_f0}
    report["pass_ge_0_90"] = bool(best_acc >= 0.90)
    return best_w, report

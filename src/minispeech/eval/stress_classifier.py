"""Supervised acoustic stress detector for the P1/B05 validity check.

The classifier ranks vowel candidates within each stressed Russian word.  It
uses forced-alignment duration, three-bin RMS-energy and F0 contours, vowel
position, and syllable count.  Speaker groups are held out when possible,
otherwise real container IDs are held out.  Legacy RUSLAN has neither, so the
CLI reports an explicit contiguous-source-block proxy instead of claiming a
true container holdout.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from minispeech.constants import STRESS_CHAR
from minispeech.data.mel import load_wav_mono
from minispeech.frontend.stress import accentuate
from minispeech.utils import expand, load_yaml

RU_VOWELS = frozenset("аеёиоуыэюя")
FEATURE_NAMES = (
    "log_duration",
    "log_rms",
    "rms_start",
    "rms_middle",
    "rms_end",
    "rms_slope",
    "log_f0",
    "f0_start",
    "f0_middle",
    "f0_end",
    "f0_slope",
    "voiced_fraction",
    "vowel_position",
    "syllable_count",
    "is_first_vowel",
    "is_last_vowel",
    "duration_within_word_z",
    "rms_within_word_z",
    "f0_within_word_z",
)


@dataclass(frozen=True)
class StressWord:
    utt_id: str
    group: str
    word: str
    features: np.ndarray
    gold_index: int


@dataclass(frozen=True)
class LogisticRanker:
    mean: np.ndarray
    scale: np.ndarray
    coef: np.ndarray
    intercept: float

    def scores(self, features: np.ndarray) -> np.ndarray:
        x = (features - self.mean) / self.scale
        return x @ self.coef + self.intercept

    def predict(self, features: np.ndarray) -> int:
        return int(np.argmax(self.scores(features)))


def _word_gold_index(word: str) -> int | None:
    vowel_index = -1
    gold: int | None = None
    for index, char in enumerate(word):
        if char.lower() in RU_VOWELS:
            vowel_index += 1
            if index + 1 < len(word) and word[index + 1] == STRESS_CHAR:
                if gold is not None:
                    return None
                gold = vowel_index
    return gold


def _clean_word(word: str) -> str:
    return "".join(
        char for char in word if char.isalpha() or char == STRESS_CHAR
    )


def _z(values: np.ndarray) -> np.ndarray:
    std = float(values.std())
    if std < 1e-8:
        return np.zeros_like(values)
    return (values - float(values.mean())) / std


def _three_bins(values: np.ndarray) -> tuple[float, float, float]:
    if values.size == 0:
        return 0.0, 0.0, 0.0
    chunks = np.array_split(values, 3)
    means = [float(chunk.mean()) if chunk.size else 0.0 for chunk in chunks]
    return means[0], means[1], means[2]


def _f0_track(wav: np.ndarray, sample_rate: int) -> tuple[np.ndarray, int]:
    import librosa

    hop = 256
    frame_length = 2048 if sample_rate >= 22_050 else 1024
    f0, _, _ = librosa.pyin(
        wav.astype(np.float64),
        fmin=librosa.note_to_hz("C2"),
        fmax=librosa.note_to_hz("C7"),
        sr=sample_rate,
        frame_length=frame_length,
        hop_length=hop,
    )
    return np.asarray(f0, dtype=np.float32), hop


def _phone_groups(entry: dict[str, Any]) -> list[list[dict[str, Any]]]:
    groups: list[list[dict[str, Any]]] = [[]]
    for phone in entry.get("phones") or []:
        if str(phone.get("phone")) in {"|", " "}:
            if groups[-1]:
                groups.append([])
            continue
        groups[-1].append(phone)
    return [group for group in groups if group]


def _expanded_phone_span(
    group: list[dict[str, Any]],
    index: int,
    *,
    word_start: float,
    word_end: float,
) -> tuple[float, float]:
    phone = group[index]
    start = float(phone["start"])
    end = float(phone["end"])
    if index == 0:
        left = word_start
    else:
        left = 0.5 * (float(group[index - 1]["end"]) + start)
    if index + 1 == len(group):
        right = word_end
    else:
        right = 0.5 * (end + float(group[index + 1]["start"]))
    return max(word_start, left), min(word_end, max(left, right))


def _features_for_word(
    *,
    wav: np.ndarray,
    sample_rate: int,
    f0: np.ndarray,
    f0_hop: int,
    word: str,
    phones: list[dict[str, Any]],
    word_span: Sequence[float],
) -> tuple[np.ndarray, int] | None:
    clean = _clean_word(word)
    gold = _word_gold_index(clean)
    if gold is None:
        return None
    vowel_phones = [
        index
        for index, phone in enumerate(phones)
        if str(phone.get("phone") or "").lower()[:1] in RU_VOWELS
    ]
    n_syllables = sum(char.lower() in RU_VOWELS for char in clean)
    if n_syllables < 2 or len(vowel_phones) != n_syllables:
        return None

    rows: list[list[float]] = []
    for vowel_position, phone_index in enumerate(vowel_phones):
        start_s, end_s = _expanded_phone_span(
            phones,
            phone_index,
            word_start=float(word_span[0]),
            word_end=float(word_span[1]),
        )
        start_sample = max(0, int(round(start_s * sample_rate)))
        end_sample = min(len(wav), max(start_sample + 1, int(round(end_s * sample_rate))))
        segment = wav[start_sample:end_sample]
        energy_frames = np.sqrt(
            np.maximum(
                0.0,
                np.asarray(
                    [
                        np.mean(chunk.astype(np.float64) ** 2)
                        for chunk in np.array_split(segment, 3)
                        if chunk.size
                    ],
                    dtype=np.float64,
                ),
            )
        )
        e0, e1, e2 = _three_bins(energy_frames)
        rms = float(np.sqrt(np.mean(segment.astype(np.float64) ** 2)))

        f0_start = max(0, int(math.floor(start_s * sample_rate / f0_hop)))
        f0_end = min(
            len(f0),
            max(f0_start + 1, int(math.ceil(end_s * sample_rate / f0_hop))),
        )
        f0_segment = f0[f0_start:f0_end]
        voiced = f0_segment[np.isfinite(f0_segment)]
        voiced_fraction = (
            float(voiced.size) / float(f0_segment.size)
            if f0_segment.size
            else 0.0
        )
        f0_fill = np.where(np.isfinite(f0_segment), f0_segment, 0.0)
        p0, p1, p2 = _three_bins(f0_fill)
        f0_mean = float(voiced.mean()) if voiced.size else 0.0
        position = (
            float(vowel_position) / float(n_syllables - 1)
            if n_syllables > 1
            else 0.0
        )
        rows.append(
            [
                math.log(max(end_s - start_s, 1e-5)),
                math.log(max(rms, 1e-8)),
                e0,
                e1,
                e2,
                e2 - e0,
                math.log(max(f0_mean, 1e-5)),
                p0,
                p1,
                p2,
                p2 - p0,
                voiced_fraction,
                position,
                float(n_syllables),
                float(vowel_position == 0),
                float(vowel_position + 1 == n_syllables),
            ]
        )
    matrix = np.asarray(rows, dtype=np.float64)
    relative = np.stack(
        (
            _z(matrix[:, 0]),
            _z(matrix[:, 1]),
            _z(matrix[:, 6]),
        ),
        axis=1,
    )
    matrix = np.concatenate((matrix, relative), axis=1)
    if not (0 <= gold < matrix.shape[0]):
        return None
    return matrix, gold


def _raw_manifest(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _numeric_source_block(utt_id: str, block_size: int) -> str:
    match = re.search(r"(\d+)", utt_id)
    number = int(match.group(1)) if match else 0
    return f"source_block_{number // block_size:05d}"


def assign_holdout_groups(
    rows: list[dict[str, Any]],
    *,
    proxy_block_size: int,
) -> tuple[dict[str, str], dict[str, Any]]:
    speakers = {
        str(row.get("speaker_id") or row.get("speaker") or "")
        for row in rows
        if row.get("speaker_id") or row.get("speaker")
    }
    containers = {
        str(row.get("container_id"))
        for row in rows
        if row.get("container_id")
    }
    if len(speakers) > 1:
        return (
            {
                str(row["utt_id"]): str(
                    row.get("speaker_id") or row.get("speaker")
                )
                for row in rows
            },
            {
                "holdout_unit": "speaker",
                "n_groups": len(speakers),
                "is_required_holdout": True,
            },
        )
    if len(containers) > 1:
        return (
            {
                str(row["utt_id"]): str(row["container_id"])
                for row in rows
            },
            {
                "holdout_unit": "container",
                "n_groups": len(containers),
                "is_required_holdout": True,
            },
        )
    groups = {
        str(row["utt_id"]): _numeric_source_block(
            str(row["utt_id"]),
            proxy_block_size,
        )
        for row in rows
    }
    return groups, {
        "holdout_unit": "contiguous_source_block_proxy",
        "proxy_block_size": proxy_block_size,
        "n_groups": len(set(groups.values())),
        "is_required_holdout": False,
        "blocker": (
            "The legacy single-speaker RUSLAN manifest exposes no distinct "
            "container_id values; a true container-held-out estimate is unavailable."
        ),
    }


def split_groups(
    groups: Sequence[str],
    *,
    test_fraction: float,
    seed: int,
) -> tuple[set[str], set[str]]:
    unique = sorted(set(groups))
    if len(unique) < 2:
        raise ValueError("at least two groups are required for held-out evaluation")

    def order_key(group: str) -> str:
        return hashlib.sha256(f"{seed}:{group}".encode("utf-8")).hexdigest()

    shuffled = sorted(unique, key=order_key)
    n_test = min(len(unique) - 1, max(1, round(len(unique) * test_fraction)))
    test = set(shuffled[:n_test])
    return set(unique) - test, test


def train_logistic_ranker(
    words: Sequence[StressWord],
    *,
    iterations: int = 1500,
    learning_rate: float = 0.05,
    l2: float = 0.01,
) -> LogisticRanker:
    if not words:
        raise ValueError("no training words")
    features = np.concatenate([word.features for word in words], axis=0)
    labels = np.concatenate(
        [
            np.arange(word.features.shape[0], dtype=np.int64)
            == word.gold_index
            for word in words
        ]
    ).astype(np.float64)
    mean = features.mean(axis=0)
    scale = features.std(axis=0)
    scale[scale < 1e-8] = 1.0
    x = (features - mean) / scale
    coef = np.zeros(x.shape[1], dtype=np.float64)
    intercept = 0.0
    n_pos = max(1.0, float(labels.sum()))
    n_neg = max(1.0, float(len(labels) - labels.sum()))
    weights = np.where(labels > 0.5, len(labels) / (2.0 * n_pos), len(labels) / (2.0 * n_neg))
    norm = float(weights.sum())
    for _ in range(iterations):
        logits = np.clip(x @ coef + intercept, -30.0, 30.0)
        probabilities = 1.0 / (1.0 + np.exp(-logits))
        residual = (probabilities - labels) * weights
        grad = x.T @ residual / norm + l2 * coef
        grad_intercept = float(residual.sum() / norm)
        coef -= learning_rate * grad
        intercept -= learning_rate * grad_intercept
    return LogisticRanker(
        mean=mean,
        scale=scale,
        coef=coef,
        intercept=intercept,
    )


def wilson_interval(correct: int, total: int, z: float = 1.96) -> tuple[float, float]:
    if total <= 0:
        return 0.0, 0.0
    p = correct / total
    denominator = 1.0 + z * z / total
    centre = (p + z * z / (2.0 * total)) / denominator
    margin = (
        z
        * math.sqrt(p * (1.0 - p) / total + z * z / (4.0 * total * total))
        / denominator
    )
    return max(0.0, centre - margin), min(1.0, centre + margin)


def evaluate_ranker(
    model: LogisticRanker,
    words: Sequence[StressWord],
) -> dict[str, Any]:
    predictions = [model.predict(word.features) for word in words]
    correct = sum(
        prediction == word.gold_index
        for prediction, word in zip(predictions, words)
    )
    low, high = wilson_interval(correct, len(words))
    return {
        "n_words": len(words),
        "n_correct": correct,
        "accuracy": correct / len(words) if words else None,
        "accuracy_ci95_low": low if words else None,
        "accuracy_ci95_high": high if words else None,
    }


def _select_evenly(rows: list[dict[str, Any]], limit: int | None) -> list[dict[str, Any]]:
    if limit is None or limit <= 0 or len(rows) <= limit:
        return rows
    indices = np.linspace(0, len(rows) - 1, num=limit, dtype=np.int64)
    return [rows[int(index)] for index in indices]


def build_stress_words(
    rows: list[dict[str, Any]],
    forced: dict[str, Any],
    groups: dict[str, str],
    *,
    max_utts: int | None,
) -> tuple[list[StressWord], dict[str, int]]:
    eligible = [
        row
        for row in rows
        if str(row.get("utt_id")) in forced
        and row.get("wav")
        and Path(str(row["wav"])).is_file()
        and str(row.get("qc_tag") or "clean") not in {"exclude", "duplicate"}
    ]
    selected = _select_evenly(eligible, max_utts)
    words: list[StressWord] = []
    stats = {
        "n_manifest": len(rows),
        "n_eligible": len(eligible),
        "n_selected_utts": len(selected),
        "n_audio_errors": 0,
        "n_alignment_mismatches": 0,
    }
    for index, row in enumerate(selected, start=1):
        utt_id = str(row["utt_id"])
        entry = forced[utt_id]
        aligned_words = list(entry.get("words") or [])
        spans = list(entry.get("word_spans_seconds") or [])
        phone_groups = _phone_groups(entry)
        if not (len(aligned_words) == len(spans) == len(phone_groups)):
            stats["n_alignment_mismatches"] += 1
            continue
        try:
            wav, sample_rate = load_wav_mono(str(row["wav"]))
            f0, f0_hop = _f0_track(wav, sample_rate)
        except Exception:  # noqa: BLE001
            stats["n_audio_errors"] += 1
            continue
        for word, span, phone_group in zip(aligned_words, spans, phone_groups):
            stressed = (
                word
                if STRESS_CHAR in word
                else accentuate(_clean_word(str(word)))
            )
            result = _features_for_word(
                wav=wav,
                sample_rate=sample_rate,
                f0=f0,
                f0_hop=f0_hop,
                word=stressed,
                phones=phone_group,
                word_span=span,
            )
            if result is None:
                continue
            features, gold = result
            words.append(
                StressWord(
                    utt_id=utt_id,
                    group=groups[utt_id],
                    word=stressed,
                    features=features,
                    gold_index=gold,
                )
            )
        if index % 25 == 0:
            print(
                f"stress-classifier features {index}/{len(selected)} "
                f"words={len(words)}",
                flush=True,
            )
    stats["n_words"] = len(words)
    return words, stats


def run(args: argparse.Namespace) -> dict[str, Any]:
    data_cfg = load_yaml(expand(args.data)).get("data", {})
    manifest_path = expand(data_cfg["manifest"])
    forced_path = expand(data_cfg["forced_path"])
    rows = _raw_manifest(manifest_path)
    forced = json.loads(forced_path.read_text(encoding="utf-8"))
    groups, holdout = assign_holdout_groups(
        rows,
        proxy_block_size=args.proxy_container_size,
    )
    words, extraction = build_stress_words(
        rows,
        forced,
        groups,
        max_utts=args.max_utts,
    )
    train_groups, test_groups = split_groups(
        [word.group for word in words],
        test_fraction=args.test_fraction,
        seed=args.seed,
    )
    train_words = [word for word in words if word.group in train_groups]
    test_words = [word for word in words if word.group in test_groups]
    model = train_logistic_ranker(
        train_words,
        iterations=args.iterations,
        learning_rate=args.learning_rate,
        l2=args.l2,
    )
    train_metrics = evaluate_ranker(model, train_words)
    test_metrics = evaluate_ranker(model, test_words)
    accuracy = test_metrics["accuracy"]
    passed = bool(accuracy is not None and accuracy >= 0.90)
    verdict = (
        "SA detector valid on held-out data"
        if passed
        else "измеритель SA по-прежнему невалиден, гейт N5 остаётся неизмеримым"
    )
    return {
        "schema_version": 1,
        "task": "P1-B05-supervised-stress-classifier",
        "data": {
            "manifest": str(manifest_path),
            "forced_alignment": str(forced_path),
            **extraction,
        },
        "features": list(FEATURE_NAMES),
        "model": {
            "type": "weighted_binary_logistic_vowel_ranker",
            "iterations": args.iterations,
            "learning_rate": args.learning_rate,
            "l2": args.l2,
            "coef": model.coef.tolist(),
            "intercept": model.intercept,
            "mean": model.mean.tolist(),
            "scale": model.scale.tolist(),
        },
        "holdout": {
            **holdout,
            "seed": args.seed,
            "test_fraction": args.test_fraction,
            "train_groups": sorted(train_groups),
            "test_groups": sorted(test_groups),
            "n_train_groups": len(train_groups),
            "n_test_groups": len(test_groups),
            "group_overlap": sorted(train_groups & test_groups),
        },
        "train": train_metrics,
        "held_out": test_metrics,
        "threshold": 0.90,
        "pass_ge_0_90": passed,
        "verdict": verdict,
        "gate_n5_measurable": passed and bool(holdout["is_required_holdout"]),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="configs/data_ruslan_local.yaml")
    parser.add_argument("--out", default="out/stress_classifier_eval.json")
    parser.add_argument("--max-utts", type=int, default=500)
    parser.add_argument("--proxy-container-size", type=int, default=250)
    parser.add_argument("--test-fraction", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--iterations", type=int, default=1500)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--l2", type=float, default=0.01)
    args = parser.parse_args(argv)
    payload = run(args)
    output = expand(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "held_out": payload["held_out"],
                "threshold": payload["threshold"],
                "pass_ge_0_90": payload["pass_ge_0_90"],
                "verdict": payload["verdict"],
                "holdout_unit": payload["holdout"]["holdout_unit"],
                "is_required_holdout": payload["holdout"][
                    "is_required_holdout"
                ],
                "out": str(output),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

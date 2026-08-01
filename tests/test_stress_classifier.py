from __future__ import annotations

import numpy as np

from minispeech.eval.stress_classifier import (
    FEATURE_NAMES,
    StressWord,
    assign_holdout_groups,
    evaluate_ranker,
    split_groups,
    train_logistic_ranker,
    wilson_interval,
)


def test_holdout_prefers_speakers_then_containers() -> None:
    rows = [
        {"utt_id": "a", "speaker_id": "s1", "container_id": "c1"},
        {"utt_id": "b", "speaker_id": "s2", "container_id": "c1"},
    ]
    groups, meta = assign_holdout_groups(rows, proxy_block_size=100)
    assert groups == {"a": "s1", "b": "s2"}
    assert meta["holdout_unit"] == "speaker"
    assert meta["is_required_holdout"] is True

    rows = [
        {"utt_id": "a", "speaker": "s1", "container_id": "c1"},
        {"utt_id": "b", "speaker": "s1", "container_id": "c2"},
    ]
    groups, meta = assign_holdout_groups(rows, proxy_block_size=100)
    assert groups == {"a": "c1", "b": "c2"}
    assert meta["holdout_unit"] == "container"


def test_legacy_manifest_reports_proxy_holdout() -> None:
    rows = [
        {"utt_id": "ruslan_000001_RUSLAN", "speaker": "ruslan"},
        {"utt_id": "ruslan_000250_RUSLAN", "speaker": "ruslan"},
    ]
    groups, meta = assign_holdout_groups(rows, proxy_block_size=250)
    assert len(set(groups.values())) == 2
    assert meta["holdout_unit"] == "contiguous_source_block_proxy"
    assert meta["is_required_holdout"] is False
    assert "blocker" in meta


def test_logistic_ranker_learns_vowel_ranking() -> None:
    words = []
    for index in range(40):
        gold = index % 3
        features = np.zeros((3, len(FEATURE_NAMES)), dtype=np.float64)
        features[:, 0] = [-1.0, -1.0, -1.0]
        features[gold, 0] = 2.0
        words.append(
            StressWord(
                utt_id=f"u{index}",
                group=f"g{index // 10}",
                word="пример",
                features=features,
                gold_index=gold,
            )
        )
    train_groups, test_groups = split_groups(
        [word.group for word in words],
        test_fraction=0.25,
        seed=42,
    )
    train = [word for word in words if word.group in train_groups]
    test = [word for word in words if word.group in test_groups]
    model = train_logistic_ranker(train, iterations=300)
    metrics = evaluate_ranker(model, test)
    assert metrics["accuracy"] == 1.0
    assert not (train_groups & test_groups)


def test_wilson_interval_contains_observed_accuracy() -> None:
    low, high = wilson_interval(80, 100)
    assert low < 0.8 < high

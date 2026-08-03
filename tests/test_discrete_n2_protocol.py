"""CPU tests for D1b FR checkpoint selection + N2 protocol helpers."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from minispeech.eval.discrete_n2 import (
    FR_CANDIDATE_STEPS_BASE,
    assert_dev_confirm_disjoint,
    fr_candidate_steps,
    is_catastrophic,
    items_sha256,
    load_utt_ids_from_subset,
    select_fr_checkpoint,
    selection_sort_key,
    stability_acceptance,
    text_usage_acceptance,
    word_token_jaccard,
)

REPO = Path(__file__).resolve().parents[1]
DEV32 = REPO / "configs" / "train_subsets" / "d1b_dev32.json"
CONFIRM32 = REPO / "configs" / "train_subsets" / "d1b_confirm32.json"
FROZEN_HASHES = REPO / ".cursor" / "results" / "d1b_n2_reliability" / "artifacts" / "frozen_hashes.json"


def test_catastrophic_cer_ge_050():
    assert is_catastrophic(0.50, "привет мир", "xxxxxx")
    assert is_catastrophic(0.91, "a", "b")
    assert not is_catastrophic(0.49, "привет мир", "привет мир")


def test_catastrophic_cer_and_low_jaccard():
    # CER high enough and almost no word overlap
    assert is_catastrophic(0.35, "красная машина едет быстро", "синий дом стоит далеко")
    # Same CER but high overlap → not catastrophic
    assert not is_catastrophic(0.35, "красная машина едет быстро", "красная машина едет медленно")
    # Below CER threshold regardless of jaccard
    assert not is_catastrophic(0.34, "aaaa bbbb cccc", "xxxx yyyy zzzz")


def test_word_token_jaccard_basic():
    assert word_token_jaccard("a b c", "a b c") == 1.0
    assert word_token_jaccard("a b", "c d") == 0.0
    assert word_token_jaccard("", "") == 1.0


def test_selection_lexicographic_ordering():
    candidates = [
        {
            "ckpt": "cand_high_cer.pt",
            "step": 250,
            "macro_cer": 0.10,
            "n_catastrophic": 0,
            "mean_dur_ratio": 1.0,
            "runaway_or_repetition": False,
            "teacher_forced_ce": 0.1,
        },
        {
            "ckpt": "cand_best.pt",
            "step": 500,
            "macro_cer": 0.04,
            "n_catastrophic": 0,
            "mean_dur_ratio": 1.02,
            "runaway_or_repetition": False,
            "teacher_forced_ce": 1.5,
        },
        {
            "ckpt": "cand_runaway.pt",
            "step": 750,
            "macro_cer": 0.01,
            "n_catastrophic": 0,
            "mean_dur_ratio": 1.0,
            "runaway_or_repetition": True,
            "teacher_forced_ce": 0.01,
        },
    ]
    result = select_fr_checkpoint(candidates, selection_subset="select8")
    assert result["selected"]["ckpt"] == "cand_best.pt"
    assert result["selection_subset"] == "select8"
    # Runaway ranks after eligible despite lower CER
    ranks = {c["ckpt"]: c["rank"] for c in result["candidates"]}
    assert ranks["cand_best.pt"] < ranks["cand_high_cer.pt"] < ranks["cand_runaway.pt"]


def test_selection_tiebreak_catastrophic_then_duration_then_tf():
    base = {
        "runaway_or_repetition": False,
        "macro_cer": 0.05,
        "mean_dur_ratio": 1.0,
        "teacher_forced_ce": 1.0,
    }
    a = {**base, "ckpt": "more_cat.pt", "step": 250, "n_catastrophic": 2}
    b = {**base, "ckpt": "less_cat.pt", "step": 375, "n_catastrophic": 0}
    assert select_fr_checkpoint([a, b])["selected"]["ckpt"] == "less_cat.pt"

    c = {
        **base,
        "ckpt": "dur_far.pt",
        "step": 500,
        "n_catastrophic": 0,
        "mean_dur_ratio": 1.5,
    }
    d = {
        **base,
        "ckpt": "dur_near.pt",
        "step": 625,
        "n_catastrophic": 0,
        "mean_dur_ratio": 1.05,
    }
    assert select_fr_checkpoint([c, d])["selected"]["ckpt"] == "dur_near.pt"

    e = {
        **base,
        "ckpt": "tf_high.pt",
        "step": 250,
        "n_catastrophic": 0,
        "mean_dur_ratio": 1.0,
        "teacher_forced_ce": 2.0,
    }
    f = {
        **base,
        "ckpt": "tf_low.pt",
        "step": 375,
        "n_catastrophic": 0,
        "mean_dur_ratio": 1.0,
        "teacher_forced_ce": 0.5,
    }
    assert select_fr_checkpoint([e, f])["selected"]["ckpt"] == "tf_low.pt"


def test_selection_key_tuple_shape():
    key = selection_sort_key(
        {
            "macro_cer": 0.1,
            "n_catastrophic": 1,
            "mean_dur_ratio": 1.0,
            "runaway_or_repetition": False,
            "teacher_forced_ce": 0.2,
        }
    )
    assert len(key) == 5
    assert key[0] == 0  # eligible


def test_stability_acceptance():
    good = {
        "runaway_or_repetition": False,
        "cb0": {"cb0_max_run": 8},
        "rows": [
            {"n_frames": 100, "gt_frames": 100},
            {"n_frames": 105, "gt_frames": 100},
        ],
    }
    st = stability_acceptance(good)
    assert st["pass"] is True

    bad_run = {
        "runaway_or_repetition": False,
        "cb0": {"cb0_max_run": 13},
        "rows": [{"n_frames": 100, "gt_frames": 100}],
    }
    assert stability_acceptance(bad_run)["pass"] is False

    bad_ratio = {
        "runaway_or_repetition": False,
        "cb0": {"cb0_max_run": 4},
        "rows": [{"n_frames": 300, "gt_frames": 100}],
    }
    assert stability_acceptance(bad_ratio)["pass"] is False


def test_text_usage_acceptance():
    model_rows = [{"hyp": f"hyp{i}"} for i in range(8)]
    shuffle_rows = [{"hyp": f"other{i}"} for i in range(8)]
    ok = text_usage_acceptance(
        model_cer=0.05,
        shuffle_cer=0.20,
        mask_cer=0.25,
        model_rows=model_rows,
        shuffle_rows=shuffle_rows,
        paired_ci95_high=-0.1,
    )
    assert ok["pass"] is True

    fail = text_usage_acceptance(
        model_cer=0.05,
        shuffle_cer=0.06,  # < model+0.05
        mask_cer=0.25,
        model_rows=model_rows,
        shuffle_rows=model_rows,  # no content change
        paired_ci95_high=0.1,
    )
    assert fail["pass"] is False
    assert fail["shuffle_ok"] is False
    assert fail["prefix_cf_ok"] is False
    assert fail["paired_ci_lt_zero"] is False


@pytest.mark.skipif(not DEV32.is_file() or not CONFIRM32.is_file(), reason="frozen subsets missing")
def test_dev_confirm_disjoint_and_split_integrity():
    dev = load_utt_ids_from_subset(DEV32)
    confirm_payload = json.loads(CONFIRM32.read_text(encoding="utf-8"))
    confirm = [str(x) for x in confirm_payload["utt_ids"]]
    assert len(dev) == 32
    assert len(confirm) == 32
    assert_dev_confirm_disjoint(dev, confirm)

    splits = confirm_payload["splits"]
    select8 = [str(x) for x in splits["select8"]]
    gate24 = [str(x) for x in splits["gate24"]]
    full32 = [str(x) for x in splits["full32"]]
    assert len(select8) == 8
    assert len(gate24) == 24
    assert set(select8).isdisjoint(set(gate24))
    assert sorted(select8 + gate24) == sorted(full32)
    assert full32 == confirm


@pytest.mark.skipif(not DEV32.is_file() or not FROZEN_HASHES.is_file(), reason="frozen hashes missing")
def test_frozen_hash_stability():
    frozen = json.loads(FROZEN_HASHES.read_text(encoding="utf-8"))
    dev = load_utt_ids_from_subset(DEV32)
    confirm_payload = json.loads(CONFIRM32.read_text(encoding="utf-8"))
    confirm = [str(x) for x in confirm_payload["utt_ids"]]

    assert items_sha256(dev) == frozen["dev32"]["items_sha256"]
    assert items_sha256(confirm) == frozen["confirm32"]["items_sha256"]

    split_hashes = frozen["confirm32"]["split_hashes"]
    for name in ("select8", "gate24", "full32"):
        ids = [str(x) for x in confirm_payload["splits"][name]]
        assert items_sha256(ids) == split_hashes[name]


def test_fr_candidate_schedule():
    assert FR_CANDIDATE_STEPS_BASE == (250, 375, 500, 625, 750)
    assert fr_candidate_steps() == FR_CANDIDATE_STEPS_BASE
    assert fr_candidate_steps(include_dev_extension=True)[-2:] == (1000, 1250)


def test_confirm_selection_subset_label():
    cands = [
        {
            "ckpt": "a.pt",
            "step": 500,
            "macro_cer": 0.05,
            "n_catastrophic": 0,
            "mean_dur_ratio": 1.0,
            "runaway_or_repetition": False,
            "teacher_forced_ce": 1.0,
        }
    ]
    out = select_fr_checkpoint(cands, selection_subset="select8")
    assert out["selection_subset"] == "select8"
    assert "best.pt" not in (out["selected"]["ckpt"] or "")

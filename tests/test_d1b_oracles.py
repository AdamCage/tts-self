"""Frozen D1b oracle loading + eval-contract wiring."""

from __future__ import annotations

from pathlib import Path

import pytest

from minispeech.data.d1b_oracles import (
    TRACKN_V2_SPEAKER_SHA256,
    enforce_eval_contract,
    expand_path_template,
    items_sha256,
    load_eval_contract,
    load_frozen_utt_ids,
)
from minispeech.utils import load_yaml

ROOT = Path(__file__).resolve().parents[1]
DEV32 = ROOT / "configs" / "train_subsets" / "d1b_dev32.json"
CONFIRM32 = ROOT / "configs" / "train_subsets" / "d1b_confirm32.json"
EVAL_CONTRACT = ROOT / "configs" / "eval_contracts" / "d1b_n2_reliability.json"
D1B_YAML = ROOT / "configs" / "discrete_speech_lm_d1b_443m.yaml"


@pytest.mark.skipif(not DEV32.is_file(), reason="dev32 missing")
def test_dev32_alias_ordered_and_hashed():
    ids, payload, digest = load_frozen_utt_ids("dev32", verify_hash=True)
    assert len(ids) == 32
    assert ids == [str(x) for x in payload["utt_ids"]]
    assert digest == payload["items_sha256"]
    assert "ba1a4089af87b3ebc90db07d" in ids[22] or any("ba1a4089" in u for u in ids)


@pytest.mark.skipif(not CONFIRM32.is_file(), reason="confirm32 missing")
def test_select8_split_hash():
    ids, payload, digest = load_frozen_utt_ids("select8", verify_hash=True)
    assert len(ids) == 8
    assert digest == payload["split_hashes"]["select8"]
    assert ids == [str(x) for x in payload["splits"]["select8"]]


@pytest.mark.skipif(not EVAL_CONTRACT.is_file() or not D1B_YAML.is_file(), reason="contract/yaml missing")
def test_eval_contract_speaker_wired_and_enforced():
    contract = load_eval_contract(EVAL_CONTRACT)
    assert contract["speaker_map"]["sha256"] == TRACKN_V2_SPEAKER_SHA256
    assert contract["speaker_map"]["status"] == "wired"
    assert contract["infer"]["stop_overrun_bias"] == 0.0
    cfg = load_yaml(D1B_YAML)
    ids, _, digest = load_frozen_utt_ids("dev32", verify_hash=True)
    enforce_eval_contract(
        cfg,
        contract_path=EVAL_CONTRACT,
        subset_items_sha256=digest,
        subset_label="dev32",
        require_clean_sha=False,
    )
    assert len(ids) == 32


def test_expand_run_name_template(tmp_path: Path):
    p = expand_path_template("runs/${run_name}", {"run_name": "discrete_speech_lm_d1b_dev_seed42"})
    assert p.name == "discrete_speech_lm_d1b_dev_seed42"


def test_items_sha256_stable():
    assert items_sha256(["a", "b"]) == items_sha256(["a", "b"])
    assert items_sha256(["a", "b"]) != items_sha256(["b", "a"])

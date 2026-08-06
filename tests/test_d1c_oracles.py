from __future__ import annotations

import hashlib
import json

import pytest

from minispeech.data.discrete_oracles import items_sha256, load_formal_subset


def test_formal_oracle_refuses_missing_declared_hash(tmp_path):
    path = tmp_path / "subset.json"
    payload = {"utt_ids": ["u1"], "items_sha256": items_sha256(["u1"]), "splits": {"select8": ["u1"]}}
    path.write_text(json.dumps(payload), encoding="utf-8")
    contract = {"subsets": {"select8": {"path": str(path), "items_sha256": items_sha256(["u1"]), "file_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}}}
    with pytest.raises(ValueError, match="split_hashes"):
        load_formal_subset("select8", contract=contract, split="select8")


def test_formal_oracle_refuses_unknown_alias():
    with pytest.raises(ValueError, match="unknown"):
        load_formal_subset("unknown", contract={"subsets": {}})


def test_d1c_dev32_path_full32_not_contradiction_with_shared_hash():
    """D1c contract registers both ``dev32`` and ``full32`` with the same items hash.

    Passing the JSON path + full32 split must not false-positive as a label/hash
    contradiction (Wave 4 train blocker).
    """
    from pathlib import Path

    from minispeech.data.d1b_oracles import enforce_eval_contract, load_frozen_utt_ids
    from minispeech.utils import load_yaml

    root = Path(__file__).resolve().parents[1]
    contract = root / "configs" / "eval_contracts" / "d1c_semantic_eos_v2_0_0.json"
    yaml_path = root / "configs" / "discrete_speech_lm_d1c_semantic_eos_w1.yaml"
    if not contract.is_file() or not yaml_path.is_file():
        pytest.skip("D1c contract/yaml missing")
    cfg = load_yaml(yaml_path)
    _, _, digest = load_frozen_utt_ids(
        "configs/train_subsets/d1c_dev32.json", split="full32", verify_hash=True
    )
    enforce_eval_contract(
        cfg,
        contract_path=contract,
        subset_label="configs/train_subsets/d1c_dev32.json",
        subset_items_sha256=digest,
        require_clean_sha=False,
    )
    # True contradictions still fail (select8 label + full32 digest).
    with pytest.raises(RuntimeError, match="label/hash contradiction"):
        enforce_eval_contract(
            cfg,
            contract_path=contract,
            subset_label="select8",
            subset_items_sha256=digest,
            require_clean_sha=False,
        )


def test_d1c_dev32_path_select8_split_rebinds_without_contradiction():
    """Path to d1c_dev32.json + select8 items hash must rebind to select8."""
    from pathlib import Path

    from minispeech.data.d1b_oracles import enforce_eval_contract, load_frozen_utt_ids
    from minispeech.utils import load_yaml

    root = Path(__file__).resolve().parents[1]
    contract = root / "configs" / "eval_contracts" / "d1c_semantic_eos_v2_0_0.json"
    yaml_path = root / "configs" / "discrete_speech_lm_d1c_semantic_eos_w1.yaml"
    if not contract.is_file() or not yaml_path.is_file():
        pytest.skip("D1c contract/yaml missing")
    cfg = load_yaml(yaml_path)
    _, _, select8 = load_frozen_utt_ids(
        "configs/train_subsets/d1c_dev32.json", split="select8", verify_hash=True
    )
    enforce_eval_contract(
        cfg,
        contract_path=contract,
        subset_label="configs/train_subsets/d1c_dev32.json",
        subset_items_sha256=select8,
        require_clean_sha=False,
    )

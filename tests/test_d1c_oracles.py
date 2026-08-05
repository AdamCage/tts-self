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

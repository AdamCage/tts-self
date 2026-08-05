"""REAUDIT ROUND2 probe G — published-number consistency + no-invented-seed check."""

from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

ROOT = Path("/mnt/c/projects/tts-self-d1c")
sys.path.insert(0, str(ROOT / "src"))
os.chdir(ROOT)

from minispeech.data.d1b_oracles import file_sha256  # noqa: E402
from minispeech.data.discrete_oracles import (  # noqa: E402
    load_formal_subset,
    payload_content_file_sha256,
)
from minispeech.data.d1b_oracles import load_eval_contract  # noqa: E402

sub = ROOT / "configs/train_subsets/d1c_dev32.json"
raw_bytes_sha = file_sha256(sub)
payload = json.loads(sub.read_text(encoding="utf-8"))
content_sha = payload_content_file_sha256(payload)
print(f"G1 d1c_dev32.json raw-bytes sha256      = {raw_bytes_sha}")
print(f"G1 d1c_dev32.json content sha (declared)= {content_sha}")
print(f"G1 payload.file_sha256 field            = {payload.get('file_sha256')}")
print(f"G1 WAVE1_REPORT publishes               = 4f57a222f47384642f34acbc1c6d29623d96aa3fc39f1c9152c14ab5ba2a83eb")
print(f"G1 raw==WAVE1  : {raw_bytes_sha == '4f57a222f47384642f34acbc1c6d29623d96aa3fc39f1c9152c14ab5ba2a83eb'}")
print(f"G1 content==WAVE1: {content_sha == '4f57a222f47384642f34acbc1c6d29623d96aa3fc39f1c9152c14ab5ba2a83eb'}")

contract = load_eval_contract(ROOT / "configs/eval_contracts/d1c_semantic_eos_v2_0_0.json")
sel = load_formal_subset("select8", contract=contract)
print(f"G2 load_formal_subset('select8').file_sha256 = {sel['file_sha256']}")
print(f"G2 contract.subsets entries:")
for k, v in (contract.get("subsets") or {}).items():
    print(f"   {k}: items_sha256={v.get('items_sha256')} file_sha256={v.get('file_sha256')} path={v.get('path')}")

print("\nG3 published split hashes vs live:")
for split, published in (
    ("full32", "501330481c07d928d0fe6b86395355aea1410a680a4020c73b602df5cc3d9b40"),
    ("select8", "e32d70aebe34284eb015413b86a4bc4d0e654b848d98aeec5854dc616600e123"),
    ("gate24", "832ce5fafe2cf9eed9bff7b940cca1fbb8f9e54a21ad41807618f804cec6b57a"),
):
    s = load_formal_subset(split if split != "full32" else "full32", contract=contract)
    live = s.get("items_sha256")
    print(f"   {split}: published={published[:24]} live={str(live)[:24]} match={published == live}")

commit = ROOT / "configs/train_subsets/d1c_confirm_seed_commitment.json"
print("\nG4 confirmation-seed commitment (must NOT be invented):")
print("   " + json.dumps(json.loads(commit.read_text(encoding="utf-8")), ensure_ascii=False, sort_keys=True))
print(f"   contract.confirm_seed_commitment={json.dumps(contract.get('confirm_seed_commitment'), sort_keys=True)}")
print(f"   contract.repro.clean_sha_available={(contract.get('repro') or {}).get('clean_sha_available')}")
print(f"   contract.repro={json.dumps(contract.get('repro'), sort_keys=True)}")

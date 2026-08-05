"""Post-audit integrity check: prove the re-audit probes mutated no attested source."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "src"))

from minispeech.data.d1b_oracles import (  # noqa: E402
    eval_contract_content_sha256,
    load_eval_contract,
)
from minispeech.data.discrete_oracles import file_sha256, load_formal_subset  # noqa: E402

contract = load_eval_contract(ROOT / "configs/eval_contracts/d1c_semantic_eos_v2_0_0.json")
manifest = contract["recipe_source_manifest"]
bad = [k for k, v in manifest.items() if file_sha256(ROOT / k) != v]
print(f"post_probe_manifest_mismatches={bad}")
print(f"post_probe_contract_selfcheck_ok={contract['sha256'] == eval_contract_content_sha256(contract)}")
for key in ("dev32", "select8", "gate24", "full32"):
    att = load_formal_subset(key, contract=contract)
    print(f"post_probe_formal_load {key}: n={len(att['utt_ids'])} ok=True")

from pathlib import Path
from minispeech.data.d1b_oracles import (
    verify_recipe_source_manifest,
    load_eval_contract,
    eval_contract_content_sha256,
)

root = Path(".")
c = load_eval_contract("configs/eval_contracts/d1c_semantic_eos_v2_0_0.json")
m = verify_recipe_source_manifest(c["recipe_source_manifest"], root=root)
live = eval_contract_content_sha256(c)
print("sha", c["sha256"])
print("live", live)
print("match", c["sha256"] == live)
print("manifest", c["recipe_source_manifest_sha256"])
print("mismatches", m)

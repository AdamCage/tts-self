#!/usr/bin/env python3
from pathlib import Path
from minispeech.data.d1b_oracles import (
    recipe_source_file_sha256,
    verify_recipe_source_manifest,
    load_eval_contract,
    eval_contract_content_sha256,
)

c = load_eval_contract("configs/eval_contracts/d1c_semantic_eos_v2_0_0.json")
m = verify_recipe_source_manifest(c["recipe_source_manifest"], root=Path("."))
print("live", eval_contract_content_sha256(c))
print("decl", c["sha256"])
print("match", c["sha256"] == eval_contract_content_sha256(c))
print("mismatches", m)
for rel in (
    "scripts/74_eval_discrete_lm.py",
    "scripts/76_select_discrete_checkpoint.py",
    "src/minispeech/eval/discrete_report_schema.py",
):
    print(rel, recipe_source_file_sha256(rel)[:16], c["recipe_source_manifest"][rel][:16])

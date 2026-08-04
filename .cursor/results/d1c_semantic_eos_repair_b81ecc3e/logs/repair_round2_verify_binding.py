from pathlib import Path

from minispeech.data.d1b_oracles import (
    eval_contract_content_sha256,
    load_eval_contract,
    verify_recipe_source_manifest,
)
from minispeech.train.discrete_lm_trainer import resolve_repro_hashes
from minispeech.utils import load_yaml

ROOT = Path(".")
c = load_eval_contract(ROOT / "configs/eval_contracts/d1c_semantic_eos_v2_0_0.json")
print("selfcheck", c["sha256"] == eval_contract_content_sha256(c))
print("mismatches", verify_recipe_source_manifest(c["recipe_source_manifest"], root=ROOT))
for name in ("w1", "w2"):
    cfg = load_yaml(ROOT / f"configs/discrete_speech_lm_d1c_semantic_eos_{name}.yaml")
    h = resolve_repro_hashes(cfg)
    inert = [k for k in ("recipe", "recipe_source_manifest", "run_manifest") if h.get(k) is None]
    print(name, "hash_match", cfg["train"]["eval_contract_hash"] == c["sha256"], "inert", inert)

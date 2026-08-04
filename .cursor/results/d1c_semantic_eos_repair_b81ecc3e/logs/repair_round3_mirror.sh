#!/usr/bin/env bash
set -euo pipefail
SRC=/mnt/c/projects/tts-self-d1c/.cursor/results/d1c_semantic_eos_repair_b81ecc3e
DST=/mnt/c/projects/tts-self/.cursor/results/d1c_semantic_eos_repair_b81ecc3e
mkdir -p "$DST/logs" "$DST/artifacts"
cp -v "$SRC/REPAIR_ROUND3.md" "$DST/"
cp -v "$SRC/WAVE1_REPORT.md" "$DST/"
cp -v "$SRC/artifacts/oracle_overlap_report.json" "$DST/artifacts/"
cp -v "$SRC/artifacts/repair_round2_hashes.json" "$DST/artifacts/repair_round3_hashes.json"
cp -v "$SRC/artifacts/repair_round2_hashes.json" "$SRC/artifacts/repair_round3_hashes.json"
if [[ -f "$SRC/artifacts/d1c_semantic_eos_v2_0_0.json" ]]; then
  cp -v "$SRC/artifacts/d1c_semantic_eos_v2_0_0.json" "$DST/artifacts/"
fi
cp -v "$SRC"/logs/repair_round3_* "$DST/logs/"
python3 - <<'PY'
import json
from pathlib import Path

paths = [
    Path("/mnt/c/projects/tts-self-d1c/.cursor/results/d1c_semantic_eos_repair_b81ecc3e/d1c_reliability_gate.json"),
    Path("/mnt/c/projects/tts-self/.cursor/results/d1c_semantic_eos_repair_b81ecc3e/d1c_reliability_gate.json"),
]
payload = None
src = paths[0]
d = json.loads(src.read_text(encoding="utf-8"))
d["formal_verdict"] = "INFRA_REPAIR_ROUND3_COMPLETE_AWAITING_HUMAN_REAUDIT"
d["reason"] = (
    "REAUDIT_ROUND2 found FAIL #4 still open on real evaluator path. "
    "Round-3 repaired aggregation→emitter plumbing, score-only contract, "
    "HIGH hash docs, and MEDIUM gaps; live probe + pytest 398 passed / 3 skipped. "
    "Waves 0-2 not PASS; Wave 3 still NO-GO."
)
d["contract_2_0_0_sha256"] = "25c370a94f589127913623cc04db61d1fd593d4b4158f692ded9e72408006f82"
d["recipe_source_manifest_sha256"] = "2d31bb9b0b37d548eb5be2e16083fbf497064232d3053d683d85235329b8de08"
d["pytest_full"] = "398 passed, 3 skipped"
d["pytest_log"] = "logs/repair_round3_full_pytest.log"
d["repair_audit"] = "REPAIR_ROUND3.md"
d["prior_reaudit"] = "REAUDIT_ROUND2.md"
d["required_to_resume"] = [
    "human_reaudit_of_infra_repair_round3",
    "user_authorize_commit_push_d1c_branch",
    "external_confirm_seed_commitment",
]
text = json.dumps(d, indent=2, ensure_ascii=False) + "\n"
for p in paths:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")
print("gate updated")
PY

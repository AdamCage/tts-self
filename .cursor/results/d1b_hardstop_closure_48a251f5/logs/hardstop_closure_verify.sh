#!/usr/bin/env bash
# D1b hardstop closure — clean-SHA verify (fail-closed v2).
# Normative: WSL clone after git pull; NEVER set MINISPEECH_ALLOW_DIRTY=1.
# Does NOT treat superseded v1 PASS JSON as success.
set -euo pipefail

WIN=/mnt/c/projects/tts-self
WSL="${HOME}/projects/tts-self"
LOGDIR_WIN="${WIN}/.cursor/results/d1b_hardstop_closure_48a251f5/logs"
LOGDIR_WSL="${WSL}/.cursor/results/d1b_hardstop_closure_48a251f5/logs"
RELIAB_WIN="${WIN}/.cursor/results/d1b_n2_reliability"
RELIAB_WSL="${WSL}/.cursor/results/d1b_n2_reliability"

mkdir -p "$LOGDIR_WIN" "$LOGDIR_WSL"

run_body() {
  echo "=== hardstop_closure_verify start $(date -Iseconds) ==="
  echo "hostname=$(hostname)"
  if [[ "${MINISPEECH_ALLOW_DIRTY:-}" == "1" ]]; then
    echo "FATAL: MINISPEECH_ALLOW_DIRTY=1 is forbidden for normative verify"
    return 1
  fi

  cd "$WSL"
  export PATH="${HOME}/.local/bin:${PATH}"
  # shellcheck disable=SC1091
  source .venv/bin/activate
  export HF_HOME="${HOME}/hf_cache"
  export PYTHONPATH=src
  unset MINISPEECH_ALLOW_DIRTY || true

  WSL_SHA=$(git rev-parse HEAD)
  WSL_SHORT=$(git rev-parse --short HEAD)
  WSL_PORCELAIN=$(git status --porcelain | wc -l | tr -d ' ')
  echo "WSL_SHA=${WSL_SHA}"
  echo "WSL_SHORT=${WSL_SHORT}"
  echo "WSL_PORCELAIN_COUNT=${WSL_PORCELAIN}"
  git status -sb | head -20
  if [[ "${WSL_PORCELAIN}" != "0" ]]; then
    echo "FATAL: WSL working tree not clean (porcelain=${WSL_PORCELAIN}); refuse normative verify"
    return 1
  fi

  echo "=== editable reinstall (no-deps) ==="
  uv pip install -e . --no-deps -q
  python -c "import sys; print(sys.version)"

  FOCUS_TESTS=(
    tests/test_d1b_oracles.py
    tests/test_discrete_n2_protocol.py
    tests/test_discrete_speech_lm.py
    tests/test_speaker_registry.py
    tests/test_discrete_lm_trainer_steps.py
    tests/test_discrete_lm_resume.py
    tests/test_discrete_stop_config.py
    tests/test_n2_collate_latent_seq.py
  )

  echo "=== focused D1b suite ==="
  set +e
  PYTHONPATH=src python -m pytest -q "${FOCUS_TESTS[@]}" 2>&1 | tee "${LOGDIR_WSL}/hardstop_closure_focused.log"
  FOCUSED_EXIT=${PIPESTATUS[0]}
  set -e
  cp -f "${LOGDIR_WSL}/hardstop_closure_focused.log" "${LOGDIR_WIN}/hardstop_closure_focused.log"
  echo "FOCUSED_EXIT=${FOCUSED_EXIT}"

  echo "=== full pytest -q ==="
  set +e
  PYTHONPATH=src python -m pytest -q 2>&1 | tee "${LOGDIR_WSL}/hardstop_closure_pytest.log"
  FULL_PYTEST_EXIT=${PIPESTATUS[0]}
  set -e
  cp -f "${LOGDIR_WSL}/hardstop_closure_pytest.log" "${LOGDIR_WIN}/hardstop_closure_pytest.log"
  echo "FULL_PYTEST_EXIT=${FULL_PYTEST_EXIT}"

  echo "=== real torchrun --standalone --nproc_per_node=1 smoke ==="
  SMOKE_PY="${LOGDIR_WSL}/torchrun_ddp_smoke.py"
  if [[ ! -f "$SMOKE_PY" ]]; then
    SMOKE_PY="${LOGDIR_WIN}/torchrun_ddp_smoke.py"
    cp -f "$SMOKE_PY" "${LOGDIR_WSL}/torchrun_ddp_smoke.py"
    SMOKE_PY="${LOGDIR_WSL}/torchrun_ddp_smoke.py"
  fi
  set +e
  PYTHONPATH=src torchrun --standalone --nproc_per_node=1 "$SMOKE_PY" \
    2>&1 | tee "${LOGDIR_WSL}/hardstop_closure_torchrun.log"
  TORCHRUN_EXIT=${PIPESTATUS[0]}
  set -e
  cp -f "${LOGDIR_WSL}/hardstop_closure_torchrun.log" "${LOGDIR_WIN}/hardstop_closure_torchrun.log"
  if ! grep -q "TORCHRUN_SMOKE_OK" "${LOGDIR_WSL}/hardstop_closure_torchrun.log"; then
    echo "FATAL: TORCHRUN_SMOKE_OK not found in torchrun log"
    TORCHRUN_EXIT=1
  fi
  echo "TORCHRUN_EXIT=${TORCHRUN_EXIT}"

  echo "=== frozen hash + HARD-STOP provenance (v1 PASS must not count as OK) ==="
  HASH_LOG="${LOGDIR_WSL}/hardstop_closure_hashes.log"
  set +e
  PYTHONPATH=src python - <<'PY' 2>&1 | tee "$HASH_LOG"
import json
from pathlib import Path

from minispeech.data.d1b_oracles import (
    items_sha256,
    load_eval_contract,
    load_subset_payload,
    utt_ids_from_payload,
    verify_items_sha256,
)

root = Path(".")
errors: list[str] = []

frozen = json.loads(
    (root / ".cursor/results/d1b_n2_reliability/artifacts/frozen_hashes.json").read_text(
        encoding="utf-8"
    )
)

ec = frozen["eval_contract"]
ec_path = root / ec["path"]
contract = load_eval_contract(ec_path)
got = str(contract.get("_content_sha256") or contract.get("sha256"))
print(f"eval_contract.path={ec_path}")
print(f"eval_contract.version={contract.get('version')}")
print(f"eval_contract.expected_sha256={ec['sha256']}")
print(f"eval_contract.got_sha256={got}")
art = json.loads(
    (
        root / ".cursor/results/d1b_n2_reliability/artifacts/eval_contract.sha256.json"
    ).read_text(encoding="utf-8")
)
if contract.get("version") != ec["version"]:
    errors.append(
        f"eval_contract version mismatch: {contract.get('version')} != {ec['version']}"
    )
if got != ec["sha256"]:
    errors.append(f"eval_contract sha mismatch: got={got} expected={ec['sha256']}")
if art.get("sha256") != ec["sha256"]:
    errors.append(
        f"artifact sha record mismatch: {art.get('sha256')} != {ec['sha256']}"
    )

for key in ("dev32", "confirm32"):
    block = frozen[key]
    p = root / block["path"]
    payload = load_subset_payload(p)
    ids = utt_ids_from_payload(payload, split="full32")
    digest = verify_items_sha256(ids, payload, split="full32")
    expected = block["items_sha256"]
    print(f"{key}.path={p}")
    print(f"{key}.items_sha256.expected={expected}")
    print(f"{key}.items_sha256.got={digest}")
    if digest != expected:
        errors.append(f"{key} items_sha256 mismatch")
    if key == "confirm32":
        for sk, sv in (block.get("split_hashes") or {}).items():
            split_ids = utt_ids_from_payload(payload, split=sk)
            got_split = items_sha256(split_ids)
            print(f"confirm32.split.{sk}.expected={sv}")
            print(f"confirm32.split.{sk}.got={got_split}")
            if got_split != sv:
                errors.append(f"confirm32 split {sk} hash mismatch")

# Formal gate — HARD-STOP only (never promote on v1 PASS).
formal_path = root / ".cursor/results/d1b_n2_reliability/d1b_reliability_gate.json"
formal = json.loads(formal_path.read_text(encoding="utf-8"))
print(f"formal_gate.path={formal_path}")
print(f"formal_gate.terminal_result={formal.get('terminal_result')!r}")
print(f"formal_gate.formal_verdict={formal.get('formal_verdict')!r}")
print(f"formal_gate.prior_pass_status={formal.get('prior_pass_status')!r}")
if formal.get("terminal_result") != "HARD-STOP":
    errors.append(f"formal terminal_result != HARD-STOP: {formal.get('terminal_result')!r}")
if formal.get("formal_verdict") != "HARD-STOP":
    errors.append(f"formal_verdict != HARD-STOP: {formal.get('formal_verdict')!r}")
if formal.get("prior_pass_status") not in ("INVALID_WITHDRAWN", "SUPERSEDED", "INVALID"):
    errors.append(
        f"prior_pass_status not withdrawn: {formal.get('prior_pass_status')!r}"
    )

c42 = (formal.get("confirm_v2") or {}).get("42") or {}
c43 = (formal.get("confirm_v2") or {}).get("43") or {}
print(f"confirm_v2.43.classification={c43.get('classification')!r}")
print(f"confirm_v2.43.git_dirty={c43.get('git_dirty')!r}")
print(f"confirm_v2.43.gate24_verdict={c43.get('gate24_verdict')!r}")
print(f"confirm_v2.43.full32_verdict={c43.get('full32_verdict')!r}")
print(f"confirm_v2.42.classification={c42.get('classification')!r}")
print(f"confirm_v2.42.git_dirty={c42.get('git_dirty')!r}")

if c43.get("classification") != "FORMAL_HARD_STOP_BASIS":
    errors.append("confirm_v2 seed43 classification must be FORMAL_HARD_STOP_BASIS")
if c43.get("git_dirty") is not False:
    errors.append(f"confirm_v2 seed43 must be clean git_dirty=false; got {c43.get('git_dirty')!r}")
if c43.get("gate24_verdict") != "FAIL":
    errors.append(f"confirm_v2 seed43 gate24 must FAIL; got {c43.get('gate24_verdict')!r}")
if c43.get("full32_verdict") != "FAIL":
    errors.append(f"confirm_v2 seed43 full32 must FAIL; got {c43.get('full32_verdict')!r}")
if c42.get("classification") != "DIAGNOSTIC":
    errors.append("confirm_v2 seed42 must remain DIAGNOSTIC")

summary_path = root / ".cursor/results/d1b_n2_reliability/metrics/confirm_v2_summary.json"
summary = json.loads(summary_path.read_text(encoding="utf-8"))
print(f"confirm_v2_summary.formal_verdict={summary.get('formal_verdict')!r}")
if summary.get("formal_verdict") != "HARD-STOP":
    errors.append(f"confirm_v2_summary.formal_verdict != HARD-STOP: {summary.get('formal_verdict')!r}")
s43 = (summary.get("seeds") or {}).get("43") or {}
if s43.get("gate24_verdict") != "FAIL" or s43.get("full32_verdict") != "FAIL":
    errors.append(
        "confirm_v2_summary seed43 must FAIL gate24+full32; "
        f"got gate24={s43.get('gate24_verdict')!r} full32={s43.get('full32_verdict')!r}"
    )

# Superseded v1 PASS artifacts must NOT establish success.
v1_note = root / ".cursor/results/d1b_n2_reliability/artifacts/V1_SUPERSEDED.md"
if not v1_note.is_file():
    errors.append("missing artifacts/V1_SUPERSEDED.md")
else:
    print(f"v1_superseded_note={v1_note}")

v1_gates = [
    root / ".cursor/results/d1b_n2_reliability/artifacts/confirm_seed43_full32_n2_gate.json",
    root / ".cursor/results/d1b_n2_reliability/artifacts/confirm_seed43_gate24_n2_gate.json",
]
for gp in v1_gates:
    if not gp.exists():
        print(f"v1_artifact_absent_ok={gp.name}")
        continue
    g = json.loads(gp.read_text(encoding="utf-8"))
    verdict = g.get("verdict")
    print(f"v1_artifact={gp.name} verdict={verdict!r} (SUPERSEDED; must not count as OK)")
    if verdict == "PASS":
        # Informational only — do not treat as HASH_CHECK success signal.
        print(f"NOTE: ignoring superseded v1 PASS in {gp.name}")

print("---")
if errors:
    print("HASH_CHECK_FAIL")
    for e in errors:
        print(f"ERROR: {e}")
    raise SystemExit(1)
print("HASH_CHECK_OK")
print("HARDSTOP_PROVENANCE_OK")
PY
  HASH_EXIT=${PIPESTATUS[0]}
  set -e
  cp -f "$HASH_LOG" "${LOGDIR_WIN}/hardstop_closure_hashes.log"
  echo "HASH_EXIT=${HASH_EXIT}"

  echo "=== verify end $(date -Iseconds) ==="
  echo "WSL_SHA=${WSL_SHA}"
  echo "FOCUSED_LOG=${LOGDIR_WIN}/hardstop_closure_focused.log"
  echo "PYTEST_LOG=${LOGDIR_WIN}/hardstop_closure_pytest.log"
  echo "TORCHRUN_LOG=${LOGDIR_WIN}/hardstop_closure_torchrun.log"
  echo "HASH_LOG=${LOGDIR_WIN}/hardstop_closure_hashes.log"

  FAIL=0
  for code in "${FOCUSED_EXIT}" "${FULL_PYTEST_EXIT}" "${TORCHRUN_EXIT}" "${HASH_EXIT}"; do
    if [[ "${code}" != "0" ]]; then FAIL=1; fi
  done
  echo "AGGREGATE_FAIL=${FAIL}"
  echo "CLEAN_SHA=${WSL_SHA}"
  return "${FAIL}"
}

set +e
run_body 2>&1 | tee "${LOGDIR_WIN}/hardstop_closure_verify_master.log"
MASTER_EXIT=${PIPESTATUS[0]}
set -e
cp -f "${LOGDIR_WIN}/hardstop_closure_verify_master.log" \
  "${LOGDIR_WSL}/hardstop_closure_verify_master.log" || true
# Mirror into reliability logs for continuity.
mkdir -p "${RELIAB_WIN}/logs" "${RELIAB_WSL}/logs"
cp -f "${LOGDIR_WIN}/hardstop_closure_verify.sh" \
  "${RELIAB_WIN}/logs/hardstop_closure_verify.sh" || true
exit "${MASTER_EXIT}"

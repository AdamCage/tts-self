# REPAIR_ROUND6 — D1c INFRA repair (payload-step + run-hashes identity)

**Plan:** `d1c_semantic_eos_repair_b81ecc3e`  
**Worktree:** `C:\projects\tts-self-d1c` (`d1c/semantic-eos`)  
**Input:** `REAUDIT_ROUND5.md` (verdict **INFRA_READY_FOR_HUMAN_AUDIT**; MEDIUM: filename-only step invent via rename/re-save; MEDIUM–LOW: stale `WAVE1_REPORT.md`)  
**Date:** 2026-08-04  
**Constraints:** no commit, no push, no GPU (`CUDA_VISIBLE_DEVICES=`), no seed invented, no methodology edit, no Wave 0–2 PASS claim.

---

## Verdict

> **INFRA_REPAIR_ROUND6_COMPLETE_AWAITING_HUMAN_REAUDIT**
>
> The Round-5 MEDIUM rename/re-save gap is closed fail-closed: D1c formal
> `resolve_checkpoint_step(require=True)` now requires trainer-stamped
> `payload['step']` (filename may only confirm). Formal eval also requires
> `payload['hashes']` (eval_contract / recipe / recipe_source_manifest /
> run_manifest), stamps them into `checkpoint.hashes`, verifies
> `eval_contract` against the live contract, and the selector refuses mixed or
> missing run identities. Basename `candidate_step_<step:04d>.pt` is covered by
> a dedicated negative test. `WAVE1_REPORT.md` publishes the live round-6
> digests (not superseded round-4/5 values). Live probe: rename attack 0/5
> reports; legitimate five-candidate set accepted. Pytest
> `406 passed, 3 skipped` (409 collected).
>
> Waves 0–2 remain **not PASS**. Wave 3 remains **NO-GO**.

---

## Contract decisions (documented)

### Payload step (evaluator)

**Chosen: payload is authoritative on formal path (`require=True`).**

| Situation | Behavior |
|---|---|
| `payload['step']` = S; filename agrees or absent | OK → S |
| `payload['step']` = S; CLI = S | OK (confirm) |
| `payload['step']` missing; filename alone; `require=True` | refuse (`payload['step']`) |
| `payload['step']` missing; CLI set; `require=True` | refuse (`cannot invent`) |
| Filename alone; `require=False` | still allowed (non-formal / diagnostic) |

### Payload hashes (evaluator + schema + selector)

Trainer already stamps `payload["hashes"] = dict(self.repro_hashes)`. Formal path now:

1. `require_checkpoint_payload_hashes` — fail-closed if any of
   `eval_contract` / `recipe` / `recipe_source_manifest` / `run_manifest` is
   missing; optionally verify `eval_contract` against the enforced contract;
2. emitter stamps `checkpoint.hashes` into the v2 report;
3. schema requires those four keys;
4. selector requires all five candidates share one identical
   `checkpoint.hashes` fingerprint (one sealed run).

Five candidates must be distinct **training steps** (payload-backed) from one
sealed run — not merely five distinct files.

---

## Fixes

### 1. Filename-only step invent closed — **FIXED**

`scripts/74_eval_discrete_lm.py` `resolve_checkpoint_step(..., require=True)`:
payload step required; filename confirms only.

### 2. Payload hashes binding — **FIXED**

- `require_checkpoint_payload_hashes` / `checkpoint_hashes_from_payload`
- `main()` stamps `provenance["checkpoint_hashes"]` on D1c contracts
- `_emit_d1c_v2_from_gate_impl` requires + verifies + stamps `checkpoint.hashes`
- `validate_formal_report` requires `checkpoint.hashes.*`
- selector `_require_distinct_checkpoint_identities` requires shared run identity

### 3. Basename rule unit coverage — **FIXED**

`test_selector_refuses_checkpoint_basename_mismatch` (mutation Y3 hole closed).

### 4. WAVE1 digest sync — **FIXED**

`WAVE1_REPORT.md` (worktree) updated to live round-6 digests; mirrored to
authoring results path.

### 5. Negative / positive tests — **ADDED/UPDATED**

- rename/filename-only refuse in `test_resolve_checkpoint_step_payload_filename_cli`
- `test_require_checkpoint_payload_hashes_fail_closed`
- `test_selector_refuses_checkpoint_basename_mismatch`
- e2e main/selector path carries live hashes; rename attack resolve refuse

---

## Live probe

`logs/repair_round6_probe_rename.py` → `logs/repair_round6_probe_rename.log`:

```text
A1 PASS filename-only require=True refused
B1 PASS missing hashes refused
B2 PASS eval_contract mismatch refused
B3 PASS legit hashes accepted
C1 PASS rename attack emitted 0/5 reports
C2 PASS no reports to select (evaluator refused)
D1 PASS legit main() stamped steps=[250, 375, 500, 625, 750]
D2 PASS selector ACCEPTS legit set selected_step=250
D3 PASS legit distinct sha count=5
E1 PASS basename refused (best.pt vs candidate_step_0250.pt)
SUMMARY pass=10 fail=0
```

---

## Sealed digests (post-edit reseal)

| Field | Value |
|---|---|
| `contract_sha256` / `recipe_sha256` | `c9fd265ac715b3d600dab1fcdf3d0b941d8db6b6fb3d9c9495728c0c55e124cf` |
| `recipe_source_manifest_sha256` | `85b896a907769af467f9e76d11b2347849b1c7fb776fcd7a6104d1885877ebbd` |
| Locked sources | 19 (resealed after `74` + `76` + `discrete_report_schema`) |
| Manifest mismatches | `[]` |

Artifacts: `artifacts/repair_round6_hashes.json`, updated W1/W2 YAML binding hashes,
`configs/eval_contracts/d1c_semantic_eos_v2_0_0.json` (+ v2_0_1 parent pointer).
Round-4 digests (`9922ce4d…` / `62eeeae2…`) and round-5 digests (`cff9e9f5…` /
`e505acf9…`) are superseded.

---

## Pytest

| Check | Result |
|---|---|
| Full suite | `406 passed, 3 skipped, 2 warnings in 56.05s` |
| Collected | 409 (= 406 + 3) |
| Focused | `32 passed` (`tests/test_d1c_infra_repair.py` + schema) |
| Log | `logs/repair_round6_full_pytest.log` |
| GPU | masked (`CUDA_VISIBLE_DEVICES=`) |
| GIT_DIR | worktree override (same caveat as round 5) |

Delta vs round 5 (407 collected / 404 passed): +2 tests
(`test_require_checkpoint_payload_hashes_fail_closed`,
`test_selector_refuses_checkpoint_basename_mismatch`).

---

## Remaining (not closed this round)

From `REAUDIT_ROUND5.md` §12, still open / deferred:

1. **[LOW]** `--promote-best` D1c branch lacks symlink guard (pre-existing)
2. **[LOW]** Redundant full-file hashing across schema/selector/emit
3. **[LOW]** Cover real `_load_model` returns payload with `step` in unit suite
4. **[LOW]** Weight EOS accuracy aggregate by utterance count
5. **[LOW]** Pin `verify_recipe_source_manifest` to repo root
6. **[LOW]** Assert rather than overwrite `recipe_hash` in `scripts/72`
7. **[LOW]** Non-D1c resume fail-open
8. **[LOW]** CER margin tolerance at selector

Standing non-code blockers: no clean committed D1c SHA; no confirmation-seed
commitment; Wave 3/4 NO-GO pending human sign-off.

---

## Confirmations

- No commit / no push / no branch mutation
- No GPU / no training / no synthesis / no seed invented
- No methodology edits
- No Wave 0–2 PASS claim; Wave 3 **NO-GO**

## Key files changed

- `scripts/74_eval_discrete_lm.py`
- `scripts/76_select_discrete_checkpoint.py`
- `src/minispeech/eval/discrete_report_schema.py`
- `tests/test_d1c_infra_repair.py`
- `tests/test_discrete_eval_report_schema.py`
- `configs/eval_contracts/d1c_semantic_eos_v2_0_0.json` (+ v2_0_1, W1/W2 YAML)
- `.cursor/results/.../WAVE1_REPORT.md`, `d1c_reliability_gate.json`,
  `REPAIR_ROUND6.md`, `logs/repair_round6_*`, `artifacts/repair_round6_hashes.json`

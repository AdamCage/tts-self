# REPAIR_ROUND4 — D1c INFRA repair (checkpoint.step + main()-level coverage)

**Plan:** `d1c_semantic_eos_repair_b81ecc3e`  
**Worktree:** `C:\projects\tts-self-d1c` (`d1c/semantic-eos`)  
**Input:** `REAUDIT_ROUND3.md` (verdict **STILL_INVALID_RUN**; FAIL #4/#2/#7 PASS; NEW BLOCKER `checkpoint.step=0`)  
**Date:** 2026-08-04  
**Constraints:** no commit, no push, no GPU (`CUDA_VISIBLE_DEVICES=`), no seed invented, no methodology edit, no Wave 0–2 PASS claim.

---

## Verdict

> **INFRA_REPAIR_ROUND4_COMPLETE_AWAITING_HUMAN_REAUDIT**
>
> The Round-3 NEW BLOCKER is fixed on the production path: `scripts/74::main()` now
> resolves the candidate step from checkpoint payload and/or `candidate_step_XXXX.pt`,
> stamps `provenance.checkpoint_step` / `gate.step` / formal `checkpoint.step`, and
> fail-closes on zero/missing step. Live probe proves evaluator→selector §8.4 with
> steps `{250,375,500,625,750}`. `evaluate_n2_gate_d1c` is wired (no longer dead).
> A `main()`-level regression test catches discarded-metrics / missing-step class bugs.
>
> Waves 0–2 remain **not PASS**. Wave 3 remains **NO-GO**.

---

## Blocker fixes

### 1. Stamp `checkpoint.step` on production path — **FIXED**

| Mechanism | Detail |
|---|---|
| `_load_model` | Returns `(model, cfg, payload)` — no longer discards trainer `"step"` |
| `resolve_checkpoint_step` | Cross-checks payload ↔ filename; optional `--checkpoint-step`; fail-closed for D1c against `contract.train.candidate_steps` |
| `main()` | Stamps `provenance["checkpoint_step"]` and `gate["step"]` before emit |
| Emitter | Refuses formal report when step missing/≤0 or outside candidate_steps |
| Score-only | Recovers step from decode provenance or checkpoint filename |

**Live probe** (`logs/repair_round4_probe_step.log`): all 17 checks PASS, including:

```text
S2f PASS report checkpoint.step=750
S2c PASS gate id=N2_D1c_semantic_eos_v1
S4a PASS selector accepted steps=[250, 375, 500, 625, 750]
S4b PASS selected step=250
```

### 2. Tighten formal validation — **FIXED**

`validate_formal_report` now rejects `checkpoint.step` that is missing, non-int, bool, or ≤0
(`"checkpoint.step must be a positive integer (not 0/missing)"`). Optional
`allowed_candidate_steps=` kwarg for membership checks. Probe S3a confirms step=0 fail-closed.

### 3. Promote regression to `main()` level — **FIXED**

New `test_main_stamps_checkpoint_step_and_selector_e2e`:

- AST: `main()` calls `resolve_checkpoint_step`, `evaluate_n2_gate_d1c`, aggregation helpers
- Drives real `main()` with GPU/codec/ASR stubbed (reaudit3 template)
- Asserts gate=`N2_D1c_semantic_eos_v1`, `checkpoint.step=750`, formal emit
- Builds five-step report set → `select_d1c_formal_reports` succeeds
- Also covers bare-float soft-refuse and schema step=0 rejection helpers

### 4. `evaluate_n2_gate_d1c` — **WIRED** (not deleted)

`main()` and `_score_n2_bundle` call `evaluate_n2_gate_d1c` when the active contract is
`semantic_eos_v1`; otherwise keep `evaluate_n2_gate_d1b`. Published `n2_gate.json` now carries
`gate: "N2_D1c_semantic_eos_v1"` with `criteria.termination_d1c` on D1c runs.

---

## Sealed digests (post-edit reseal)

| Field | Value |
|---|---|
| `contract_sha256` / `recipe_sha256` | `9922ce4d75932dcaff255dd20601f731f48c369647677c68249b9ff652d938af` |
| `recipe_source_manifest_sha256` | `62eeeae2c6360e88923dd6416e8c5a22403d869cbbdf806aabce31e6a512b81d` |
| Locked sources | 19 (resealed after `74` + `discrete_report_schema` edits) |
| Manifest mismatches | `[]` |

Artifacts: `artifacts/repair_round4_hashes.json`, updated W1/W2 YAML binding hashes,
`configs/eval_contracts/d1c_semantic_eos_v2_0_0.json` (+ v2_0_1 parent pointer).

---

## LOW items

| Item | Decision |
|---|---|
| Soft-refuse beyond `ValueError` | **Fixed (cheap)** — `_emit_d1c_v2_from_gate` catches `Exception` |
| Unweighted EOS batch mean | **Deferred** — selector uses strict `==1.0`; corpus mean skew is diagnostic only |
| `cwd`-rooted manifest verify | **Deferred** — fail-closed (safe); pin to repo root later |
| `recipe_hash` overwrite in scripts/72 | **Deferred** — contained; same pattern as round 2/3 residual |
| Non-D1c resume fail-open | **Deferred** — D1c path already fail-closed |
| CER margin tolerance at scripts/76 | **Deferred** — unchanged from round 2 |

---

## Pytest

| Check | Result |
|---|---|
| Full suite | **401 passed, 3 skipped, 2 warnings in 52.51s** |
| Collected | **404** (= round-3 401 + 3 new tests) |
| Log | `logs/repair_round4_full_pytest.log` |
| Environment | WSL Ubuntu, `CUDA_VISIBLE_DEVICES=`, `GIT_DIR`/`GIT_WORK_TREE` worktree override |
| New tests | `test_validate_formal_report_rejects_checkpoint_step_zero`, `test_resolve_checkpoint_step_payload_filename_cli`, `test_main_stamps_checkpoint_step_and_selector_e2e` |

---

## Remaining blockers (non-code / standing)

1. No clean committed D1c source SHA (`require_clean_sha` still correctly refuses). Needs user commit/push auth.
2. External confirmation-seed commitment still absent (not invented).
3. Wave 3 GPU smoke / Wave 4 DEV remain NO-GO pending human re-audit of this round.
4. Human adversarial re-audit of Round 4 (especially production `checkpoint.step` + selector §8.4).

---

## Key files changed

- `scripts/74_eval_discrete_lm.py` — step resolve/stamp, D1c gate wire, `--checkpoint-step`, soft-refuse broaden
- `src/minispeech/eval/discrete_report_schema.py` — reject `checkpoint.step≤0`
- `tests/test_d1c_infra_repair.py` — main()-level + step unit tests
- `configs/eval_contracts/d1c_semantic_eos_v2_0_0.json` (+ v2_0_1) — resealed
- `configs/discrete_speech_lm_d1c_semantic_eos_w1.yaml` / `w2.yaml` — rebound hashes

## Confirmations

- No commit / no push
- No GPU work / no training / no inference seed invent
- No methodology edit
- No Waves 0–2 PASS claim

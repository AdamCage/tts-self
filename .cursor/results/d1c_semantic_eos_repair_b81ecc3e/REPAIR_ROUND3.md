# REPAIR_ROUND3 — D1c INFRA re-audit FAIL #4 plumbing

**Plan:** `d1c_semantic_eos_repair_b81ecc3e`  
**Worktree:** `C:\projects\tts-self-d1c` / branch `d1c/semantic-eos`  
**Date:** 2026-08-04  
**Input:** independent `REAUDIT_ROUND2.md` verdict **STILL_INVALID_RUN** (FAIL #4 still live; #2/#7 PASS)  
**Scope:** Fix production evaluator→emitter plumbing + score-only contract + HIGH docs. **No GPU. No commit. No push. No seed invented.**

## Overall status after round 3

**INFRA round-3 repair complete for human/adversarial re-audit.**  
Waves 0–2 remain **not PASS** until re-audit accepts this pack.  
**Wave 3 / Wave 4 DEV / 1h: still NO-GO.**

Fresh full pytest (WSL Ubuntu, `GIT_DIR`/`GIT_WORK_TREE` worktree override, `CUDA_VISIBLE_DEVICES=` masked, header recorded):

```text
398 passed, 3 skipped
```

Log: `.cursor/results/d1c_semantic_eos_repair_b81ecc3e/logs/repair_round3_full_pytest.log`  
(401 collected = round-2 399 + 2 new tests; CUDA unit → skip under no-GPU mask.)

Sealed hashes after round 3 (`artifacts/repair_round3_hashes.json`):

| Artifact | SHA |
|----------|-----|
| Contract 2.0.0 `sha256` / `recipe_sha256` | `25c370a94f589127913623cc04db61d1fd593d4b4158f692ded9e72408006f82` |
| `recipe_source_manifest_sha256` | `2d31bb9b0b37d548eb5be2e16083fbf497064232d3053d683d85235329b8de08` |
| Locked sources | 19 (resealed after `74`/`76`/`d1b_oracles` edits) |

## Per-FAIL verdicts

### FAIL #4 — evaluator→emitter EOS plumbing — **FIXED**

| | |
|--|--|
| Before | `teacher_forced_ce()` returned EOS metrics; aggregation kept only `{loss,ce_cb0,ce_residual}`; `tf_ce=summary['loss']` bare float → emitter raised unhandled `ValueError` after writing `n2_gate.json` |
| After | `aggregate_teacher_forced_batches` + `teacher_forced_metrics_for_emitter` preserve EOS fields; main + decode-bundle paths pass the full dict; `_emit_d1c_v2_from_gate` soft-refuses (returns `None`, prints reason) instead of crashing mid-flight |
| Evidence | Live probe `logs/repair_round3_probe_realpath.log` H4–H7; `test_production_tf_aggregation_to_emitter_survives` |

Live probe (decisive):

```text
H1 per-batch keys include ce_semantic_eos + tf_eos_* + teacher_forced_ce_source
H3 aggregated teacher_forced_ce_source=eval_computed (EOS metrics survive)
H4 MAIN measured path → wrote discrete_eval_report_v2.json
H5 bare-float soft-refuse → None (no crash)
H7 score-only + measured TF → OK
```

### `--score-only` contract decision — **(a)/(b) hybrid**

**Decision:** `--score-only` may emit a **formal** v2 report **only** when the decode bundle already carries measured `teacher_forced_ce_source="eval_computed"` metrics (written by `--decode-only` after TF aggregation). Otherwise it **skips** formal emit with a clear message and leaves gate JSON intact — never stamps `decode_bundle` as a formal-eligible provenance source.

| Case | Behaviour |
|------|-----------|
| score-only + measured TF dict from decode bundle | formal emit succeeds |
| score-only + bare float / missing EOS | skip formal (`None`); no traceback |
| main eval + measured TF | formal emit succeeds |
| main eval + bare float | soft-refuse (`None`); no traceback |

### FAIL #2 / #7 — **still PASS** (resealed; not re-broken)

Contract/recipe binding and resume hash fail-closed behaviour preserved; seal regenerated as a set after locked-source edits.

## HIGH — stale DEV32 provenance docs — **FIXED**

| Artifact | Fix |
|----------|-----|
| `WAVE1_REPORT.md` lines 29–31 | Live `file_sha256` / select8 / gate24; note that `file_sha256` is payload-content digest |
| `WAVE1_REPORT.md` contract block | Updated to round-3 seal digests |
| `artifacts/oracle_overlap_report.json` | Regenerated to live hashes |

## MEDIUM — fixed in same round

| Item | Status |
|------|--------|
| §11 text-usage / phrase-loop / runaway in emitter | **Fixed** — `_emit_d1c_v2_from_gate` stamps `gates.text_usage`, `summaries.phrase_loop_repetition`, `summaries/gates.runaway_or_repetition` from gate acceptance blocks |
| Selector vacuous absent `text_usage` | **Fixed** — absent block → not eligible |
| `recipe_source_manifest` runtime verify | **Fixed** — `enforce_eval_contract` verifies on `semantic_eos_v1` contracts |
| D1c `--subset` aliases | **Fixed** — `d1c_dev32` / `d1c_full32` / `d1c_select8` / `d1c_gate24` (D1b aliases unchanged) |

## Remaining blockers (not code)

1. Human/adversarial **re-audit** of this round-3 pack before any Wave 0–2 PASS claim.  
2. Clean **committed** D1c source SHA — still requires explicit user commit/push authorization.  
3. External confirmation-seed commitment — still awaiting custodian; **not invented**.  
4. Wave 3 GPU smoke / Wave 4 DEV — NO-GO until (1–3) clear.

## Confirmations

- No commit created  
- No push  
- No GPU train/decode/synth (CUDA masked for pytest; probe is CPU tiny-model only)  
- No seed invented  
- No methodology edits  
- Waves 0–2 **not** claimed PASS  

## Key files changed

- `scripts/74_eval_discrete_lm.py` — aggregation, emitter soft-refuse, §11 fields, score-only contract  
- `scripts/76_select_discrete_checkpoint.py` — refuse absent text_usage  
- `src/minispeech/data/d1b_oracles.py` — D1c aliases + runtime manifest verify  
- `tests/test_d1c_infra_repair.py` — production-path integration + alias tests  
- `configs/eval_contracts/d1c_semantic_eos_v2_0_0.json` (+ v2_0_1 parent) — resealed  
- `configs/discrete_speech_lm_d1c_semantic_eos_w{1,2}.yaml` — rebound hashes  
- `.cursor/results/.../WAVE1_REPORT.md`, `artifacts/oracle_overlap_report.json`  
- `.cursor/results/.../logs/repair_round3_*`, `artifacts/repair_round3_hashes.json`

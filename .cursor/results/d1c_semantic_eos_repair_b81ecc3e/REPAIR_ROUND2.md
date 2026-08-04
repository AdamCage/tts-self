# REPAIR_ROUND2 — D1c INFRA re-audit FAILs 2/4/7

**Plan:** `d1c_semantic_eos_repair_b81ecc3e`  
**Worktree:** `C:\projects\tts-self-d1c` / branch `d1c/semantic-eos`  
**Date:** 2026-08-04  
**Input:** independent `REAUDIT.md` verdict **STILL_INVALID_RUN** (3 FAILs remaining)  
**Scope:** Fix remaining INFRA blockers only. **No GPU. No commit. No push. No seed invented.**

## Overall status after round 2

**INFRA round-2 repair complete for human/adversarial re-audit.**  
Waves 0–2 remain **not PASS** until re-audit accepts this pack.  
**Wave 3 / Wave 4 DEV / 1h: still NO-GO.**

Fresh full pytest (WSL Ubuntu, `GIT_DIR`/`GIT_WORK_TREE` worktree override, header recorded):

```text
397 passed, 2 skipped
```

Log: `.cursor/results/d1c_semantic_eos_repair_b81ecc3e/logs/repair_round2_full_pytest.log`

Sealed hashes after round 2 (`artifacts/repair_round2_hashes.json`):

| Artifact | SHA |
|----------|-----|
| Contract 2.0.0 `sha256` / `recipe_sha256` | `32f0ea2b16fabb0a16086754d28e4f8986efa78735658de43769c172d13aac5c` |
| `recipe_source_manifest_sha256` | `cb697d4f35f6a25066d0c71bb8c280045721189a50a21450d69415a9467e3111` |
| Locked sources | 19 (was 14; added discrete_oracles, d1b_oracles, asr, codec_seq_protocol, constants) |
| W1/W2 `eval_contract_hash` | equals live contract sha |
| v2_0_1 `parent_sha256` / `parent_recipe_sha256` | updated to live 2.0.0 digests |

Recipe YAML binding uses **stripped hashing** for W1/W2 in `recipe_source_manifest` (train binding-hash fields excluded) so declaring `eval_contract_hash` cannot create a cyclic seal. Other locked files use raw bytes. Helper: `recipe_source_file_sha256` / `verify_recipe_source_manifest`.

## Per-FAIL verdicts

### FAIL #2 — Contract / recipe hash binding — **FIXED**

| | |
|--|--|
| Before | W1/W2 declared stale `a59ada9…`; v2_0_1 parent stale; `72` silently overwrote declared hash |
| After | W1/W2 + v2_0_1 rebound to `32f0ea2b…`; manifest regenerated as a set; `72` asserts equality and refuses silent overwrite / missing declared hash on D1c |
| Evidence | `logs/repair_round2_verify_binding.py` → `hash_match True`; `test_w1_w2_bound_to_live_contract_and_resolve_populates_hashes`; `test_stale_eval_contract_hash_fails_closed`; `test_v2_0_1_parent_points_at_live_contract` |

### FAIL #4 — CLI/v2 report emitter — **FIXED**

| | |
|--|--|
| Before | `_emit_d1c_v2_from_gate` raised on missing `ordinal`; fabricated TF provenance / EOS accuracy / `formal_eligible=True`; `pcm_sha256=file_sha256`; no `catastrophic` |
| After | ordinal stamped 0-based; catastrophic via `annotate_catastrophic_rows`; PCM hashed from float32 samples; missing wav/placeholder hashes refuse; TF source + EOS metrics measured or marked `unavailable` (formal emit refuses fabrication) |
| Evidence | `test_emit_d1c_v2_from_evaluator_shaped_rows`; `test_emit_refuses_fabricated_eos_tf_defaults`; `test_teacher_forced_ce_returns_eos_metrics` |

### FAIL #7 — Resume hash checks no-ops — **FIXED**

| | |
|--|--|
| Before | `resolve_repro_hashes(W1/W2)` → `recipe/recipe_source_manifest/run_manifest = null`; `_validate_resume` skipped `None` |
| After | Always backfills recipe/manifest from contract; verifies declared contract hash; derives stable D1c `run_manifest` identity; D1c resume refuses missing required keys |
| Evidence | verify probe `inert []`; `test_resume_fail_closed_on_run_manifest_hash`; `test_d1c_resume_fails_when_required_hash_missing` |

## Additional gaps

| Gap | Status |
|-----|--------|
| Selector §11 eligibility (early_eos, cb0 run, empty gen, duration, CER vs copy) | **Fixed** — `scripts/76`; negative tests added |
| Source manifest coverage (5 critical modules) | **Fixed** — locked set 14→19; verify empty mismatches |
| `72` D1c awareness (silent first-N, skip-contract, dirty warn, require_clean_sha) | **Fixed** — `is_d1c_cfg` wired |
| Pytest attestation header / GIT_DIR note | **Fixed** — round2 log records rootdir, SHA, timestamp, GIT_* |

## Remaining blockers (not code)

1. Human/adversarial **re-audit** of this round-2 pack before any Wave 0–2 PASS claim.  
2. Clean **committed** D1c source SHA — still requires explicit user commit/push authorization.  
3. External confirmation-seed commitment — still awaiting custodian; **not invented**.  
4. Wave 3 GPU smoke / Wave 4 DEV — NO-GO until (1–3) clear.

## Confirmations

- No commit created  
- No push  
- No GPU train/infer/smoke  
- No plaintext seed invented  

## Key files changed

- `src/minispeech/data/d1b_oracles.py` (stripped recipe YAML hashing + verify)
- `src/minispeech/train/discrete_lm_trainer.py` (resolve backfill + D1c resume fail-closed)
- `scripts/72_train_discrete_lm.py` (D1c guards + hash assert)
- `scripts/74_eval_discrete_lm.py` (emitter/enrich/TF CE)
- `scripts/76_select_discrete_checkpoint.py` (§11 eligibility)
- `configs/discrete_speech_lm_d1c_semantic_eos_w1.yaml` / `w2.yaml`
- `configs/eval_contracts/d1c_semantic_eos_v2_0_0.json` / `v2_0_1.json`
- `tests/test_d1c_infra_repair.py`, `tests/test_discrete_lm_resume.py`
- `.cursor/results/d1c_semantic_eos_repair_b81ecc3e/**` (this pack)

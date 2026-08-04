# REPAIR_AUDIT — D1c Semantic EOS INFRA-STOP

**Plan:** `d1c_semantic_eos_repair_b81ecc3e`  
**Worktree:** `C:\projects\tts-self-d1c` / branch `d1c/semantic-eos`  
**Date:** 2026-08-04  
**Scope:** Fix INVALID_RUN / INFRA blockers only. **No GPU. No commit. No push. No seed invented.**

## Overall status after repair

**INFRA repair complete for human re-audit.**  
Waves 0–2 remain **not PASS** until human re-audit accepts this repair pack.  
**Wave 3 / Wave 4 DEV / 1h: still NO-GO.**

Fresh pytest (WSL Ubuntu, GIT_DIR worktree override):

```text
387 passed, 2 skipped
```

Log: `.cursor/results/d1c_semantic_eos_repair_b81ecc3e/logs/repair_full_pytest.log`

Frozen hashes after repair (see `artifacts/repair_oracle_contract_hashes.json`):

| Artifact | SHA |
|----------|-----|
| Contract 2.0.0 `sha256` / `recipe_sha256` | `c636562ee0f158f722dc02d2161ea3dac6424c58f5404e7f346cc10f9cc76f71` |
| `recipe_source_manifest_sha256` | `15869c58ecd1f01dc33a3c8b5f8432c8c303c47d34243cbb0bc2fbc3e76ec261` |
| DEV32 content `file_sha256` | `f9f82b708b23ab61af1ee3ffc493b1c064998a14eb2edcab4d7020483bb3c574` |
| full32 `items_sha256` (unchanged) | `501330481c07d928d0fe6b86395355aea1410a680a4020c73b602df5cc3d9b40` |
| select8 `items_sha256` | `882691c9fa7e799b3e49e8dbde6ef8cfa7226aecdf18bf77110ebee424b62e05` |
| gate24 `items_sha256` | `5485d37be12342d187cef264fa5626e58edee3c821737dada26418142223b1ad` |

## Per-blocker verdicts

### 1. Residual loss detached — **PASS (fixed)**

| | |
|--|--|
| Before | `MultiCodebookHead.loss()` returned `ce_residual.detach()`; semantic-EOS composed that into `total` → 0 residual grads |
| After | `ce_cb0` / `ce_residual` stay live; semantic path still detaches only logging fields |
| Evidence | `tests/test_d1c_infra_repair.py::test_residual_heads_receive_gradients_in_semantic_eos_mode` |

### 2. Contract 2.0.0 self-check — **PASS (fixed)**

| | |
|--|--|
| Before | Loader excluded only `sha256` → declared `a59ada9…` ≠ computed `4e4f4f0…`; `contract_infer_from_eval_contract` re-injected five legacy stop knobs |
| After | `eval_contract_content_sha256` excludes `{sha256, recipe_sha256}`; semantic infer uses `SEMANTIC_EOS_INFER_DEFAULTS`; enforce skips WAVE3c equality for `semantic_eos_v1` |
| Evidence | Contract self-check `match True`; infer keys only `{max_frames,temperature,top_k,min_frames,max_cb0_repeat,repetition_penalty}`; `test_contract_2_0_0_self_check_and_no_legacy_stop_in_infer` |

### 3. DEV oracle formal load — **PASS (fixed)**

| | |
|--|--|
| Before | Declared file SHA = body-without-field, loader compared full-file bytes; select8/gate24/full32 missing `file_sha256`; select8 was 4 RUSLAN + 4 CV |
| After | `payload_content_file_sha256` semantics; all four registry entries carry `file_sha256`; `split_dev` enforces **2 RUSLAN + 6 CV**; select8 composition locked in loader |
| Evidence | Formal load of `dev32/select8/gate24/full32` OK; select8 composition `{ruslan:2, cv:6}`; negative tests for missing sha / wrong composition |

### 4. Semantic generation CLI wiring — **PASS (fixed)**

| | |
|--|--|
| Before | Script 73 treated `DiscreteGenerationOutput` as tensor; 74 TF CE omitted semantic targets; `estimate_stop_reason` on semantic path; `emit_d1c_v2_report` never called |
| After | 73 unwraps generation + metadata; 74 passes semantic targets/masks, uses row `termination_reason` for semantic mode, forbids legacy stop estimator there, emits v2 report on D1c contracts |
| Evidence | `test_eval_teacher_forced_requires_semantic_targets`, `test_eval_unwrap_forbids_tensor_confusion`, `test_emit_d1c_v2_report_is_exported_and_callable` |

### 5. Immediate EOS phantom frame — **PASS (fixed)**

| | |
|--|--|
| Before | `length=0` but `codes.shape=(1,1,16)` |
| After | Trim codes to `max(lengths)`; empty → `(B,0,n_codebooks)` |
| Evidence | `test_immediate_eos_has_no_phantom_codec_frame` |

### 6. Selector NameError + eligibility — **PASS (fixed)**

| | |
|--|--|
| Before | `_sha256_file` NameError (`chunk` unbound); eligibility trusted `gates.eligible` |
| After | Correct chunked hash; eligibility recomputed from bound rows (`learned_eos`, CER, catastrophic, duration) |
| Evidence | `test_selector_sha256_file_and_eligibility_from_rows` |

### 7. Checkpoint resume hash checks — **PASS (fixed)**

| | |
|--|--|
| Before | Resume checked only `model_config/speaker_map/subset/eval_contract` |
| After | Also stores/checks `config`, `recipe`, `recipe_source_manifest`, `run_manifest` when stamped; stable config hash excludes volatile run paths / separately tracked hashes |
| Evidence | New resume mismatch tests in `tests/test_discrete_lm_resume.py` |

### 8. Frozen source manifest mismatches — **PASS (fixed)**

| | |
|--|--|
| Before | W1/W2 YAMLs + `discrete_n2.py` drifted vs declared SHAs |
| After | Manifest regenerated for full locked source set; `manifest_mismatches []` |
| Evidence | Live verify after repair; `artifacts/repair_oracle_contract_hashes.json` |

### 9. Stale pytest attestation — **PASS (refreshed)**

| | |
|--|--|
| Before | “373 passed” older than W1/W2 / `discrete_n2.py` edits |
| After | Fresh full suite on final tree: **387 passed, 2 skipped** |
| Evidence | `logs/repair_full_pytest.log` |

## Remaining blockers (not code)

1. **Human re-audit** of this repair pack before any Wave 0–2 PASS claim.  
2. **Clean committed D1c source SHA** — still requires explicit user commit/push authorization.  
3. **External confirmation-seed commitment** — still awaiting custodian; **not invented**.  
4. **Wave 3 GPU smoke / Wave 4 DEV** — NO-GO until (1–3) clear.

## Confirmations

- No commit created  
- No push  
- No GPU train/infer/smoke  
- No plaintext seed invented  

## Key files changed

- `src/minispeech/models/multi_codebook_head.py`
- `src/minispeech/models/discrete_speech_lm.py`
- `src/minispeech/data/d1b_oracles.py`
- `src/minispeech/data/discrete_oracles.py`
- `src/minispeech/infer/discrete_config.py`
- `src/minispeech/train/discrete_lm_trainer.py`
- `scripts/73_infer_discrete_lm.py`
- `scripts/74_eval_discrete_lm.py`
- `scripts/76_select_discrete_checkpoint.py`
- `scripts/77_freeze_d1c_oracles.py`
- `configs/train_subsets/d1c_dev32.json`
- `configs/eval_contracts/d1c_semantic_eos_v2_0_0.json`
- `tests/test_d1c_infra_repair.py` (new)
- `tests/test_discrete_lm_resume.py`
- `.cursor/results/d1c_semantic_eos_repair_b81ecc3e/**` (audit + artifacts + logs)

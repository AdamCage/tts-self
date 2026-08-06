# WAVE3_REPORT — CPU/full tests and 1-utterance GPU smoke

**Plan:** `.cursor/plans/d1c_semantic_eos_repair_b81ecc3e.plan.md`  
**Slug:** `d1c_semantic_eos_repair_b81ecc3e`  
**Date:** 2026-08-06  
**Worktree:** `C:\projects\tts-self-d1c` (`d1c/semantic-eos`)  
**WSL runtime:** `~/projects/tts-self`  
**SHA used:** `27164affb20df9092d04fd397bf9e26334ad5f27` (`27164af`)

## Overall verdict: **HARD-STOP (GPU blocked by null confirmation commitment)**

Wave 3 **CPU slice PASS**. Wave 3 **GPU slice NOT RUN** (fail-closed).

| Gate | Verdict | Evidence |
|------|---------|----------|
| Clean committed SHA for formal run | **PASS** | WSL `git status` clean @ `27164af` |
| Round-7 infra commit + push | **PASS** | `ff9d8c7` then LF reseal `27164af` |
| LF contract reseal (CRLF poison fix) | **PASS** | `artifacts/repair_wave3_lf_reseal_hashes.json` |
| Focused unit gates (plan §15) | **PASS** | `logs/wave3_focused_pytest.log` |
| Full pytest | **PASS** | `408 passed, 3 skipped` — `logs/wave3_full_pytest.log` |
| `torchrun` world-size-1 smoke | **NOT RUN** | blocked by §18 commitment gate |
| 1-utt EOS overfit + WAVs | **NOT RUN** | blocked by §18 commitment gate |
| Checkpoint-listen / `listen_index` | **N/A** | no new checkpoint |
| Wave 4 / DEV / 1h | **NOT STARTED** | hard-stop before DEV |

## Why GPU hard-stopped

Plan **§18 Immediate stop matrix** — *Остановиться до GPU, если: … external confirmation commitment отсутствует*.

Supporting gates (same conclusion):

- `PROTOCOL.md` resume step 2: custodian fills commitment **before** Wave 3 smoke / 1-utt.
- `RESULTS.md` prior NO-GO note: Wave 3 needs seed commitment.
- Stub `configs/train_subsets/d1c_confirm_seed_commitment.json` has `"commitment": null`.

**Nuance (documented, not used to bypass):** §9.4 and the stub note say stop before **DEV** model output; Wave 3 text itself does not name commitment. Fail-closed prefers §18 + PROTOCOL → no inventing a seed, no GPU smoke/1-utt until custodian supplies a non-null commitment.

## Commits made this wave

| SHA | Message |
|-----|---------|
| `ff9d8c7` | `WIP(d1c): close round-7 identical-weights selector gap.` |
| `27164af` | `WIP(d1c): reseal contract digests on WSL LF bytes.` |

## LF reseal (blocking finding during Wave 3 CPU)

Round-7 seals (`181acd02…` / `ebb91df5…`) were computed on **Windows CRLF** working copies. Clean WSL LF checkout failed `verify_recipe_source_manifest` (17 mismatches). Resealed on WSL LF:

| Field | Value |
|-------|-------|
| `contract_sha256` / `recipe_sha256` | `fbd0f38cae71922bd71eb4a5a6dfb10d8257bce29da625f1c08e016cbfe5a4b9` |
| `recipe_source_manifest_sha256` | `7676b8efa9f5db4b1449d8b096041fe4f3eacab766885031756971ab01e24b54` |
| Artifact | `artifacts/repair_wave3_lf_reseal_hashes.json` |
| Log | `logs/wave3_lf_reseal.log` |

## CPU test evidence

Focused sequence (CUDA masked `CUDA_VISIBLE_DEVICES=`):

```text
tests/test_d1c_semantic_eos.py                 5 passed
tests/test_discrete_eval_report_schema.py      3 passed
tests/test_d1c_oracles.py                      2 passed
tests/test_d1c_selector_integration.py         1 passed
tests/test_discrete_stop_config.py            13 passed
tests/test_discrete_speech_lm.py              15 passed, 1 skipped
tests/test_discrete_lm_trainer_steps.py
  + test_discrete_lm_resume.py                23 passed, 2 skipped
tests/test_d1c_infra_repair.py                31 passed
```

Full suite:

```text
408 passed, 3 skipped, 2 warnings in 13.46s
```

Logs: `logs/wave3_focused_pytest.log`, `logs/wave3_full_pytest.log`.

Env check: CUDA available (RTX 4080 SUPER), used only for preflight; CPU gates ran with GPU masked.

## WAVs / listen paths

**None.** No smoke/1-utt checkpoint was produced.  
Expected path when unblocked: `out/d1c_semantic_eos_repair_b81ecc3e/smoke_1utt/listen_demo/`.

## Confirm-seed status

| Field | Value |
|-------|-------|
| Path | `configs/train_subsets/d1c_confirm_seed_commitment.json` |
| `commitment` | `null` |
| `plaintext_seed_present` | `false` |
| Blocked Wave 3 GPU? | **YES** (§18 / PROTOCOL) |
| Blocked Wave 4 DEV? | **YES** (§9.4) |

### Custodian must supply (do **not** invent)

1. Generate secret unsigned 64-bit seed **outside** the repository; keep plaintext private.  
2. Compute:
   `SHA256(UTF8("d1c-confirm-v1\n" + decimal_seed + "\n"))`  
3. Fill into `configs/train_subsets/d1c_confirm_seed_commitment.json`:
   - `commitment` (hex digest string)
   - `generator_version_or_hash` (how the seed was generated)
   - leave `plaintext_seed_present: false`  
4. Commit/push the commitment-only update (no plaintext seed).  
5. Re-run Wave 3 GPU: `torchrun` smoke → 1-utt overfit → WAVs under `out/d1c_semantic_eos_repair_b81ecc3e/smoke_1utt/listen_demo/`.

## Next action

1. Custodian provides non-null commitment (steps above).  
2. Resume Wave 3 GPU slice only on clean SHA after that commit.  
3. Do **not** start Wave 4 / DEV / 1h until Wave 3 GPU exit criteria pass.

## Explicit non-claims

- Waves 0–2 are **not** re-declared PASS.  
- Wave 3 is **not** PASS (GPU incomplete).  
- No N2 claim. No recipe lock. No CONFIRM32.

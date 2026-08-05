# RESULTS — D1c Semantic EOS Repair

**Plan:** `.cursor/plans/d1c_semantic_eos_repair_b81ecc3e.plan.md`  
**Slug:** `d1c_semantic_eos_repair_b81ecc3e`  
**Authoring worktree:** `C:\projects\tts-self-d1c` on branch `d1c/semantic-eos`  
**Baseline tip:** `0661d10eb616362e74dc1750e03cf25c33a7943e`

## Overall status: **INVALID_RUN / INFRA-STOP → INFRA repair complete (awaiting human re-audit)**

Prior Wave 0–2 “PASS” claims are **withdrawn / invalid** for promotion purposes.  
This directory now contains an INFRA repair pack. **Do not treat Waves 0–2 as PASS until human re-audit.**  
**Wave 3 still NO-GO. Wave 4 / DEV / 1h still NO-GO.** No GPU runs in this repair.

See **[REPAIR_AUDIT.md](REPAIR_AUDIT.md)** for per-blocker before/after evidence.

## Per-stage verdicts (post-repair)

| Stage | Verdict | Notes |
|-------|---------|-------|
| Wave 0 isolation + preflight | **INVALID → repair pending re-audit** | Isolation still valid; prior PASS claim not authoritative |
| Wave 1 eval/provenance/oracles | **INVALID → infra repaired** | Contract + oracle hashes regenerated; formal load OK |
| Wave 2 semantic EOS model | **INVALID → infra repaired** | Residual grads, empty EOS codes, CLI wiring fixed |
| Wave 3 CPU/GPU smoke | **NO-GO** | Needs human re-audit + commit SHA + seed commitment |
| Waves 4–9 DEV/CONFIRM | **NO-GO / NOT STARTED** | |

## Key frozen hashes (post-repair)

| Artifact | SHA |
|----------|-----|
| D1c DEV32 items/full32 | `501330481c07d928d0fe6b86395355aea1410a680a4020c73b602df5cc3d9b40` |
| DEV32 content file_sha256 | `f9f82b708b23ab61af1ee3ffc493b1c064998a14eb2edcab4d7020483bb3c574` |
| select8 | `882691c9fa7e799b3e49e8dbde6ef8cfa7226aecdf18bf77110ebee424b62e05` (2 RUSLAN + 6 CV) |
| gate24 | `5485d37be12342d187cef264fa5626e58edee3c821737dada26418142223b1ad` |
| Contract 2.0.0 content | `c636562ee0f158f722dc02d2161ea3dac6424c58f5404e7f346cc10f9cc76f71` |
| recipe_source_manifest | `15869c58ecd1f01dc33a3c8b5f8432c8c303c47d34243cbb0bc2fbc3e76ec261` |

Artifacts: `artifacts/repair_oracle_contract_hashes.json`, `artifacts/d1c_semantic_eos_v2_0_0.json`, `artifacts/d1c_dev32.json`

## Tests

```text
387 passed, 2 skipped   (fresh full suite after INFRA repair)
```

Log: `logs/repair_full_pytest.log`  
(WSL Ubuntu + `GIT_DIR`/`GIT_WORK_TREE` worktree override; prior “373 passed” is stale and must not be cited.)

## Inference / WAVs

None (no GPU). `listen_index.json` not created.

## Blockers remaining for parent

1. Human re-audit of `REPAIR_AUDIT.md` + fresh pytest attestation.  
2. Authorize commit + push of `d1c/semantic-eos` from `C:\projects\tts-self-d1c` only if/when ready (not done here).  
3. Custodian confirmation-seed commitment (no plaintext seed; not invented).  
4. Only then Wave 3 smoke under WSL Linux FS.

## Confirmations

- **No commit / no push**  
- **No GPU**  
- **No seed invented**

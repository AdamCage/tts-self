# RESULTS — D1c Semantic EOS Repair

**Plan:** `.cursor/plans/d1c_semantic_eos_repair_b81ecc3e.plan.md`  
**Slug:** `d1c_semantic_eos_repair_b81ecc3e`  
**Authoring worktree:** `C:\projects\tts-self-d1c` on branch `d1c/semantic-eos`  
**Baseline tip:** `0661d10eb616362e74dc1750e03cf25c33a7943e`  
**Formal SHA (Wave 3 CPU):** `27164affb20df9092d04fd397bf9e26334ad5f27`

## Overall status: **Wave 3 HARD-STOP (CPU PASS; GPU blocked by null commitment)**

Prior Wave 0–2 “PASS” claims remain **withdrawn / invalid** for promotion.  
INFRA round-7 closed and pushed; LF reseal landed.  
**Wave 3 CPU gates PASS. Wave 3 GPU / torchrun / 1-utt NOT RUN.**  
**Wave 4 / DEV / CONFIRM / 1h still NO-GO.**

See **[WAVE3_REPORT.md](WAVE3_REPORT.md)** for Wave 3 evidence.  
See **[REAUDIT_ROUND7.md](REAUDIT_ROUND7.md)** / **[REPAIR_ROUND7.md](REPAIR_ROUND7.md)** for infra audit.

## Per-stage verdicts

| Stage | Verdict | Notes |
|-------|---------|-------|
| Wave 0 isolation + preflight | **INVALID → repair pending re-audit** | Isolation still valid; prior PASS claim not authoritative |
| Wave 1 eval/provenance/oracles | **INVALID → infra repaired** | Contract LF-resealed @ `27164af` |
| Wave 2 semantic EOS model | **INVALID → infra repaired** | Residual grads, empty EOS codes, CLI wiring fixed |
| Wave 3 CPU gates | **PASS** | Focused + full pytest `408 passed, 3 skipped` |
| Wave 3 GPU smoke / 1-utt | **HARD-STOP** | `commitment: null` — plan §18 / PROTOCOL |
| Waves 4–9 DEV/CONFIRM | **NO-GO / NOT STARTED** | Blocked until Wave 3 GPU completes |

## Key frozen hashes (post Wave-3 LF reseal)

| Artifact | SHA |
|----------|-----|
| D1c DEV32 items/full32 | `501330481c07d928d0fe6b86395355aea1410a680a4020c73b602df5cc3d9b40` |
| DEV32 content file_sha256 | `f9f82b708b23ab61af1ee3ffc493b1c064998a14eb2edcab4d7020483bb3c574` |
| select8 | `882691c9fa7e799b3e49e8dbde6ef8cfa7226aecdf18bf77110ebee424b62e05` (2 RUSLAN + 6 CV) |
| gate24 | `5485d37be12342d187cef264fa5626e58edee3c821737dada26418142223b1ad` |
| Contract 2.0.0 / recipe | `fbd0f38cae71922bd71eb4a5a6dfb10d8257bce29da625f1c08e016cbfe5a4b9` |
| recipe_source_manifest | `7676b8efa9f5db4b1449d8b096041fe4f3eacab766885031756971ab01e24b54` |

Supersedes round-7 CRLF seals `181acd02…` / `ebb91df5…`.  
Artifacts: `artifacts/repair_wave3_lf_reseal_hashes.json`, `artifacts/d1c_semantic_eos_v2_0_0.json`

## Tests (Wave 3)

```text
408 passed, 3 skipped, 2 warnings in 13.46s
```

- `logs/wave3_focused_pytest.log`
- `logs/wave3_full_pytest.log`
- WSL clean tree @ `27164af`, `CUDA_VISIBLE_DEVICES=` masked for CPU gates

## Inference / WAVs

None (GPU hard-stopped). No `listen_index.json` entries for smoke_1utt.

Expected when unblocked: `out/d1c_semantic_eos_repair_b81ecc3e/smoke_1utt/listen_demo/`

## Confirm-seed status

`configs/train_subsets/d1c_confirm_seed_commitment.json`: **`commitment: null`** (blocks GPU / DEV).  
Plaintext seed was **not** invented.

## Blockers for parent

1. Custodian fills non-null `commitment` (+ `generator_version_or_hash`) per plan §9.4 algorithm; commit/push.  
2. Resume Wave 3 GPU: torchrun smoke → 1-utt overfit → WAVs / listen_index.  
3. Only then Wave 4 DEV W1 seed101.

## Commits

| SHA | Note |
|-----|------|
| `ff9d8c7` | Round-7 identical-weights selector fix + evidence |
| `27164af` | LF reseal after CRLF poison |

## Confirmations

- Round-7 + LF reseal **committed and pushed** to `origin/d1c/semantic-eos`  
- **No GPU** train/infer/smoke in this Wave 3 attempt  
- **No seed invented**  
- **No Wave 4 / 1h**

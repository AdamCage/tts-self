# RESULTS — D1c Semantic EOS Repair

**Plan:** `.cursor/plans/d1c_semantic_eos_repair_b81ecc3e.plan.md`  
**Slug:** `d1c_semantic_eos_repair_b81ecc3e`  
**Authoring worktree:** `C:\projects\tts-self-d1c` on branch `d1c/semantic-eos`  
**Baseline tip:** `0661d10eb616362e74dc1750e03cf25c33a7943e`  
**Wave 3 GPU tip:** `74d4caa535fb34e12a804f07a09308151c1f7a19`

## Overall status: **Wave 3 PASS — hard-stop before Wave 4**

Prior Wave 0–2 “PASS” claims remain **withdrawn / invalid** for promotion.  
INFRA round-7 + LF reseal remain landed.  
**Wave 3 CPU + GPU (torchrun + 1-utt + WAVs) PASS.**  
**Wave 4 / DEV / CONFIRM / 1h still NOT STARTED** (plan default hard-stop after Wave 3).

See **[WAVE3_REPORT.md](WAVE3_REPORT.md)** for full Wave 3 evidence.

## Per-stage verdicts

| Stage | Verdict | Notes |
|-------|---------|-------|
| Wave 0 isolation + preflight | **INVALID → repair pending re-audit** | Isolation still valid; prior PASS claim not authoritative |
| Wave 1 eval/provenance/oracles | **INVALID → infra repaired** | Contract LF-resealed @ `27164af` |
| Wave 2 semantic EOS model | **INVALID → infra repaired** | Residual grads, empty EOS codes, CLI wiring fixed |
| Wave 3 CPU gates | **PASS** | Focused + full pytest `408 passed, 3 skipped` |
| Wave 3 GPU smoke / 1-utt | **PASS** | tip `74d4caa`; `learned_eos`; wav mirrored |
| Waves 4–9 DEV/CONFIRM | **NOT STARTED** | Next: Wave 4 DEV W1 seed101 |

## Wave 3 GPU snapshot

| Item | Value |
|------|-------|
| Commitment | `4c46d418caf9bcabc088864da5bf0b3de0475a9d9225f50383314a2c5e3bb3f2` |
| Train loss | `22.9 → 0.000565` (80 steps, 1 utt) |
| FR termination | `learned_eos` @ step 22 |
| `codes_max` | 2034 (&lt; 2048; no EOS to decoder) |
| Wav | `out/d1c_semantic_eos_repair_b81ecc3e/smoke_1utt/listen_demo/0000_ruslan_000842_RUSLAN.wav` |

Play (Windows):  
`C:\projects\tts-self\out\d1c_semantic_eos_repair_b81ecc3e\smoke_1utt\listen_demo\0000_ruslan_000842_RUSLAN.wav`

## Key frozen hashes (post Wave-3 LF reseal)

| Artifact | SHA |
|----------|-----|
| D1c DEV32 items/full32 | `501330481c07d928d0fe6b86395355aea1410a680a4020c73b602df5cc3d9b40` |
| Contract 2.0.0 / recipe | `fbd0f38cae71922bd71eb4a5a6dfb10d8257bce29da625f1c08e016cbfe5a4b9` |
| recipe_source_manifest | `7676b8efa9f5db4b1449d8b096041fe4f3eacab766885031756971ab01e24b54` |

## Confirm-seed status

`configs/train_subsets/d1c_confirm_seed_commitment.json`: **commitment set** (`c15ed40`).  
`plaintext_seed_present: false`. Plaintext seed is **not** in git.

## Blockers / next for parent

1. Wave 3 complete → authorize Wave 4 DEV W1 seed101.  
2. Before formal DEV: set `eval_contract.repro.clean_sha_available=true` and reseal (smoke used `--allow-dirty` only to bypass missing flag).  
3. Do **not** start 1h/5h/full.

## Commits (Wave 3 resume)

| SHA | Note |
|-----|------|
| `c15ed40` | Confirm-seed commitment mint |
| `c71b453` | smoke_1utt subset + GPU runner |
| `a602593` / `74d4caa` | External log dir + probe fixes |
| (this pack) | WAVE3_REPORT / RESULTS / listen_index / logs |

## Explicit non-claims

- No N2 PASS. No recipe lock. No CONFIRM32. No 1h.

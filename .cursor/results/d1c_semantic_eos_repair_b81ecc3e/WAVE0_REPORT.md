# WAVE0_REPORT — D1c Semantic EOS Repair

**Plan slug:** `d1c_semantic_eos_repair_b81ecc3e`  
**Date:** 2026-08-03 (updated 2026-08-04 INFRA repair)  
**Verdict:** **INVALID_RUN for promotion** — isolation facts remain useful; prior PASS claim withdrawn pending human re-audit after INFRA repair (`REPAIR_AUDIT.md`).

## Isolation

| Item | Value |
|------|-------|
| Baseline tip | `0661d10eb616362e74dc1750e03cf25c33a7943e` (= `origin/master`) |
| Clean worktree | `C:\projects\tts-self-d1c` |
| Branch | `d1c/semantic-eos` (tracks `origin/master` at creation) |
| Authoring tree | `C:\projects\tts-self` left dirty; foreign inventory saved |
| Foreign inventory | `.cursor/results/d1c_semantic_eos_repair_b81ecc3e/logs/wave0_foreign_inventory.txt` |

No `git reset --hard` / checkout of unrelated paths. Methodology not edited.

## Preflight evidence

| Check | Result | Evidence |
|-------|--------|----------|
| Linux free disk | **691G** avail on `/home/adam` (≥200G required) | WSL `df` |
| CUDA / bf16 | RTX 4080 SUPER 16GB; bf16 matmul OK; torch 2.13+cu130 | `scripts/00_env_check.py` |
| WSL clone SHA | `0661d10` | `~/projects/tts-self` |
| Qwen TTS venv | present (`.venv_qwen_tts`) | WSL ls |
| Codec caches | ruslan + CV shard files present | `~/datasets/tts/*/codec_tokens` |
| D1b gate | **HARD-STOP** (seed43 formal) | `.cursor/results/d1b_n2_reliability/d1b_reliability_gate.json` |
| Historical hashes | **ALL_OK** | `logs/verify_d1b_hashes.py` |

### Historical hash lock (unchanged)

| Artifact | Expected items/content SHA | Status |
|----------|---------------------------|--------|
| contract 1.2.0 | `d04295d3…ada92` | OK |
| DEV32 | `c886ba31…172c` | OK |
| CONFIRM v1 | `232df278…265c` | OK |
| CONFIRM v2 | `63d7e351…46b9` | OK |

## Blockers deferred (must clear before GPU)

1. **Human re-audit** of INFRA repair (`REPAIR_AUDIT.md`).  
2. **Clean committed D1c source SHA** — commit/push only with explicit user authorization.  
3. **External confirmation-seed commitment** — custodian hash only; plaintext seed not invented.

## Exit criteria (Wave 0)

| Criterion | Status |
|-----------|--------|
| Clean isolation without touching foreign changes | retained |
| No D1c GPU run | **PASS** |
| Clean committed source path available | **PENDING** |
| External confirm-seed commitment | **PENDING** |
| Promotion PASS claim | **INVALID until re-audit** |

## Next

After human re-audit of INFRA repair: commit/push authorization + seed commitment, then Wave 3 only. No model training until then.

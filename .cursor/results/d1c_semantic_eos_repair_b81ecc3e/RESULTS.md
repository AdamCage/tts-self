# RESULTS — D1c Semantic EOS Repair

**Plan:** `.cursor/plans/d1c_semantic_eos_repair_b81ecc3e.plan.md`  
**Slug:** `d1c_semantic_eos_repair_b81ecc3e`  
**Authoring worktree:** `C:\projects\tts-self-d1c` on branch `d1c/semantic-eos`  
**Wave 4 formal tip:** `c7e3ee0019d975269655be4a46d6ed00cdebe1e9`  
**Wave 5 formal tip:** `5bf319b9389750ee252f2a5c62be70708e621fc0`  
**Contract 2.0.0 content SHA:** `fc07b643ba366a1dbe9fad294590fa5bd67288b7be98aecac8a6cd2297550865`

## Overall status: **HARD-STOP**

D1c stage closed as **scientific HARD-STOP** after Wave 5 DEV seed102 failed formal gate24/full32 (CER ceiling + catastrophic).  
Wave 4 W1 seed101 remains a historical PASS. Waves 6–8 (recipe lock / CONFIRM) were **not** run. **1h/5h/full were not started.**

Gate file: [`d1c_reliability_gate.json`](d1c_reliability_gate.json).

## Per-wave table

| Wave | Verdict | Evidence |
|------|---------|----------|
| 0–2 | PASS after INFRA repair (ROUND7) | `REAUDIT_ROUND7.md`, prior WAVE reports |
| 3 | PASS (CPU + GPU smoke/1-utt; user listen OK) | `WAVE3_REPORT.md` |
| 4 | **PASS** W1 seed101 gate24+full32 | `WAVE4_REPORT.md`, `artifacts/gate_w1_seed101_*.json` |
| 5 | **FAIL → HARD-STOP** seed102 | `WAVE5_REPORT.md`, `artifacts/gate_seed102_*.json` |
| 6–8 | **SKIPPED** | require both DEV seeds PASS |
| 9 | **CLOSED** as HARD-STOP | this pack |

## Key metrics (from artifacts)

| Run | Split | macro CER | ceiling (copy+0.03) | catastrophic | learned_eos | verdict |
|-----|-------|-----------|---------------------|--------------|-------------|---------|
| seed101 | gate24 | 0.04911 | — | 0 | 24/24 | PASS |
| seed101 | full32 | 0.04276 | — | 0 | 32/32 | PASS |
| seed102 | select8 | 0.02373 | — | 0 | (eligible 5/5) | selection OK |
| seed102 | gate24 | **0.11756** | 0.07911 | **2** | 24/24 | **FAIL** |
| seed102 | full32 | **0.09410** | 0.07276 | **2** | 32/32 | **FAIL** |

EOS / text-usage / stability passed on seed102; content CER did not replicate.

## Listen paths (Windows)

Wave 3 smoke:

- `C:\projects\tts-self\out\d1c_semantic_eos_repair_b81ecc3e\smoke_1utt\listen_demo\0000_ruslan_000842_RUSLAN.wav`

Wave 4 seed101:

- `C:\projects\tts-self\out\d1c_semantic_eos_repair_b81ecc3e\dev\w1_seed101\best_listen_demo\`
- `C:\projects\tts-self\out\d1c_semantic_eos_repair_b81ecc3e\dev\w1_seed101\last_listen_demo\`

Wave 5 seed102:

- `C:\projects\tts-self\out\d1c_semantic_eos_repair_b81ecc3e\dev\seed102\best_listen_demo\`
- `C:\projects\tts-self\out\d1c_semantic_eos_repair_b81ecc3e\dev\seed102\last_listen_demo\`

See [`listen_index.json`](listen_index.json).

## Confirm-seed status

Commitment only (no plaintext in git):  
`4c46d418caf9bcabc088864da5bf0b3de0475a9d9225f50383314a2c5e3bb3f2`  
Reveal / CONFIRM32 **not** materialized (HARD-STOP before Wave 6).

## Explicit non-claims / follow-ups outside this stage

- No D1c N2 PASS. No recipe lock. No CONFIRM32. No 1h/5h/full.
- Next work (if any) requires a **new plan**; this stage may not retune W1/W2 or invent extra seeds.
- Historical note: seed101 alone is insufficient under the two-seed DEV rule.

## Commits (recent formal path)

| SHA | Note |
|-----|------|
| `c7e3ee0` | score-only semantic infer keys (Wave 4 formal eval tip) |
| `5bf319b` | Wave 4 PASS report + Wave 5 runner |
| (closure) | WAVE5 / RESULTS / reliability gate / listen_index |

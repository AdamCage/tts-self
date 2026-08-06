# WAVE5_REPORT — DEV seed102 replication

**Plan:** `d1c_semantic_eos_repair_b81ecc3e`  
**Status:** **FAIL → D1c HARD-STOP** (model-result)  
**Formal tip:** `5bf319b9389750ee252f2a5c62be70708e621fc0`  
**Contract:** `fc07b643ba366a1dbe9fad294590fa5bd67288b7be98aecac8a6cd2297550865`  
**Run:** `runs/d1c_eos_dev_locked_seed102/`  
**Recipe:** W1 unchanged (`configs/discrete_speech_lm_d1c_semantic_eos_w1.yaml`)  
**Train seed:** `102`

## Verdict

Seed102 **does not** replicate Wave 4 W1 seed101 PASS on gate24/full32.  
Per plan §13 / Wave 5: **terminal HARD-STOP**. No W2 retune, no recipe lock, no CONFIRM seeds, no 1h/5h/full.

Defect-first re-audit: **MODEL_FAIL_HARD_STOP** (not INFRA_BLOCK / INVALID_RUN) — [Wave5 HARD-STOP reaudit](a7fdc191-c918-49f9-8143-89d421bd70df).

## Selection (select8)

| Item | Value | Evidence |
|------|-------|----------|
| Eligible | **5/5** | `artifacts/selection_seed102.json` |
| Selected step | **750** | same |
| `best.pt` sha256 | `9f18a899418e7abc77fa69fcd3624be279bd75e981d6a42c391a8664a1ab57aa` | `promotion.verified=true` |
| select8 macro CER | ≈0.02373 | same (all candidates tied) |
| select8 catastrophic | 0 | same |

## Formal gates

| Split | Verdict | macro CER | copy+0.03 ceiling | catastrophic | termination | text-usage | stability |
|-------|---------|-----------|-------------------|--------------|-------------|------------|-----------|
| gate24 | **FAIL** | 0.11756 | 0.07911 | **2** | PASS (24/24 learned_eos) | PASS | PASS |
| full32 | **FAIL** | 0.09410 | 0.07276 | **2** | PASS (32/32 learned_eos) | PASS | PASS |

Fail drivers (both splits): `cer_pass=false` (macro CER above copy_synth+0.03) and `zero_catastrophic=false`.

Provenance (both gates): `git_dirty=false`, `git_dirty_score=false`, `score_only=true`, `git_sha=5bf319b…`.

Artifacts:

- `artifacts/gate_seed102_gate24.json`
- `artifacts/gate_seed102_full32.json`
- `artifacts/selection_seed102.json`
- `artifacts/wave5_seed102_summary.json`
- Console: `logs/wave5_seed102_console.log` (+ external `~/datasets/tts/d1c_wave5_logs/`)

## Contrast vs Wave 4 seed101 (historical PASS)

| Metric | W1 seed101 (Wave 4) | seed102 (Wave 5) |
|--------|---------------------|------------------|
| gate24 macro CER | ≈0.04911 | ≈0.11756 **FAIL** |
| full32 macro CER | ≈0.04276 | ≈0.09410 **FAIL** |
| catastrophic | 0 / 0 | 2 / 2 |
| learned_eos | 24+32 | 24+32 |
| TF EOS / early FP | 1.0 / 0 | 1.0 / 0 |

EOS learning and stability transferred; **content CER / catastrophic did not**.

## Listen WAVs

Windows:

- `C:\projects\tts-self\out\d1c_semantic_eos_repair_b81ecc3e\dev\seed102\best_listen_demo\`
- `C:\projects\tts-self\out\d1c_semantic_eos_repair_b81ecc3e\dev\seed102\last_listen_demo\`

(Also mirrored under `C:\projects\tts-self-d1c\out\...` when present.)

## Exit actions taken

- Waves 6–8 **skipped** (require both DEV seeds PASS).
- Wave 9 closure pack written as **HARD-STOP**.
- 1h/5h/full **not started**.

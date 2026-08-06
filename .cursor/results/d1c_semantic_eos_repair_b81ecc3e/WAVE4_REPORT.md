# WAVE4_REPORT — DEV W1 seed101

**Plan:** `.cursor/plans/d1c_semantic_eos_repair_b81ecc3e.plan.md`  
**Slug:** `d1c_semantic_eos_repair_b81ecc3e`  
**Formal tip:** `c7e3ee0019d975269655be4a46d6ed00cdebe1e9` (`c7e3ee0`)  
**Contract 2.0.0:** `fc07b643ba366a1dbe9fad294590fa5bd67288b7be98aecac8a6cd2297550865`  
**Date:** 2026-08-06

## Overall verdict: **PASS (Wave 4 W1 seed101)**

Prior dirty-tree formal reports are **INVALIDATED** and replaced by a clean-tree decode+score re-eval on the same checkpoints (`WAVE4_CLEAN_EVAL_DONE`). W2 not required (not pure-EOS under-learning).

| Gate | Verdict | Evidence |
|------|---------|----------|
| Clean SHA / `clean_sha_available` | PASS | tip `c7e3ee0`; contract `fc07b643…` |
| Train 750 steps + 5 candidates | PASS | `runs/d1c_eos_dev_w1_seed101/candidate_step_{0250..0750}.pt` |
| select8 formal reports (5/5) | PASS | `out/.../select8/cand_*/discrete_eval_report_v2.json` |
| Selection eligible + promote | PASS | step **750**; 5/5 eligible; `best.pt` hash `f51adca9…` |
| gate24 N2_D1c | PASS | `artifacts/gate_w1_seed101_gate24.json` |
| full32 N2_D1c | PASS | `artifacts/gate_w1_seed101_full32.json` |
| Clean provenance (`git_dirty=false`) | PASS | re-eval console `logs/wave4_w1_clean_eval.console.log` |
| Mandatory listen WAVs | PASS | `out/.../best_listen_demo/`, `last_listen_demo/` |

## Key metrics (clean formal)

From `artifacts/wave4_w1_seed101_summary.json`:

| Split | Verdict | macro CER | copy CER | learned_eos | safety | TF EOS top1 | text-usage | stability |
|-------|---------|-----------|----------|-------------|--------|-------------|------------|-----------|
| select8 (selected 750) | eligible | 0.0237 | — | 8/8 | 0 | 1.0 | (in report) | (in report) |
| gate24 | PASS | 0.04911 | 0.04911 | 24/24 | 0 | 1.0 | PASS | PASS (`hard_cap=0`, mean dur≈0.998) |
| full32 | PASS | 0.04276 | 0.04276 | 32/32 | 0 | 1.0 | PASS | PASS (`hard_cap=0`, mean dur≈0.998) |

Termination: all model rows `learned_eos`; `early_eos_suppressed` not used as PASS path; TF early-EOS FP = 0.

## Infra notes (resolved before PASS)

1. Subset path+split alias contradiction (`dev32`/`full32`/`select8`) — fixed in `d1b_oracles.py`, resealed.
2. Score-only required legacy stop keys — fixed in `74_eval_discrete_lm.py`, resealed.
3. First formal pack recorded `git_dirty=true` because the runner copied logs into `.cursor/results/` mid-run — **INVALID**; clean re-eval done without mid-run results writes.

## WAVs to play (Windows)

- `C:\projects\tts-self\out\d1c_semantic_eos_repair_b81ecc3e\dev\w1_seed101\best_listen_demo\`
- `C:\projects\tts-self-d1c\out\d1c_semantic_eos_repair_b81ecc3e\dev\w1_seed101\best_listen_demo\`

## Decision

- **Provisional recipe:** W1 (`eos_loss_weight=1.0`)
- **W2:** not launched
- **Next:** Wave 5 seed102 replication (unchanged recipe)

## Commits

| SHA | Note |
|-----|------|
| `8fdb3c4` | clean_sha_available reseal |
| `bbe4421` | shared items_sha256 alias fix |
| `715055c` | parent/split rebind |
| `c7e3ee0` | score-only semantic infer keys |

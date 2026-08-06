# WAVE1_REPORT — D1c eval / provenance / oracle infrastructure

**Plan slug:** `d1c_semantic_eos_repair_b81ecc3e`  
**Date:** 2026-08-03 (updated 2026-08-06 INFRA repair round 7 + Wave3 LF reseal)  
**Verdict:** **INVALID_RUN → infra repaired (round 7; LF reseal)** (not PASS until human/adversarial re-audit)

Prior PASS claim withdrawn: contract self-check, oracle `file_sha256`, and select8 composition (2+6) were broken. See `REPAIR_AUDIT.md` blockers 2–3.

## Exit criteria

| Criterion | Status | Evidence |
|-----------|--------|----------|
| Report schema v2 + row binding | PASS | `src/minispeech/eval/discrete_report_schema.py`, schema JSON |
| Generic formal oracles + burn registry | PASS | `discrete_oracles.py`, `d1_oracle_burn_registry.json` |
| Fresh DEV32 freeze + disjointness | PASS | `configs/train_subsets/d1c_dev32.json`, `artifacts/oracle_overlap_report.json` |
| Contract 2.0.0 hashes | PASS | `configs/eval_contracts/d1c_semantic_eos_v2_0_0.json` |
| ASR provenance helper | PASS | `asr_provenance()` in `eval/asr.py` |
| Model-vs-shuffle ID pairing | PASS | `paired_bootstrap_model_vs_shuffle` / D1c gate helper |
| Evaluator→selector formal path | PASS | scripts 74/76 v2 modes |
| Wave-1 focused tests | PASS | 40 focused tests incl. EOS (see logs) |
| No model training | PASS | — |

## Fresh DEV32 (seed `20260805`)

| Field | Value |
|-------|-------|
| path | `configs/train_subsets/d1c_dev32.json` |
| items_sha256 / full32 | `501330481c07d928d0fe6b86395355aea1410a680a4020c73b602df5cc3d9b40` |
| file_sha256 | `f9f82b708b23ab61af1ee3ffc493b1c064998a14eb2edcab4d7020483bb3c574` (payload content digest with self-referential `file_sha256` field stripped; raw file bytes differ) |
| select8 | `882691c9fa7e799b3e49e8dbde6ef8cfa7226aecdf18bf77110ebee424b62e05` |
| gate24 | `5485d37be12342d187cef264fa5626e58edee3c821737dada26418142223b1ad` |
| composition | 6 RUSLAN + 26 CV; buckets 10/11/11; ratio extremes low=6 high=9 |
| overlap vs burns | **empty** (`artifacts/oracle_overlap_report.json`) |
| eligible pool | 51129 after exclusions |

## Contract 2.0.0

| Field | Value |
|-------|-------|
| path | `configs/eval_contracts/d1c_semantic_eos_v2_0_0.json` |
| sha256 | `fbd0f38cae71922bd71eb4a5a6dfb10d8257bce29da625f1c08e016cbfe5a4b9` |
| recipe_sha256 | `fbd0f38cae71922bd71eb4a5a6dfb10d8257bce29da625f1c08e016cbfe5a4b9` |
| recipe_source_manifest_sha256 | `7676b8efa9f5db4b1449d8b096041fe4f3eacab766885031756971ab01e24b54` |
| seal note | Round-7 `181acd02…`/`ebb91df5…` were Windows-CRLF poisoned; Wave 3 resealed on WSL LF (`artifacts/repair_wave3_lf_reseal_hashes.json`) |
| ASR raw sha | `f5d2b1b8ae80e76f4c585cc5dd3411732617ebc539fb8a97b975f025efbcfdb9` (unchanged vs D1b) |

## Remaining (not Wave-1 blockers)

1. External confirmation-seed **commitment** still absent (`d1c_confirm_seed_commitment.json` stub with `commitment: null`). Required before first DEV model output.
2. Clean **committed** D1c source SHA required before GPU (user authorization).

## Logs

- `logs/w1_freeze_dev32.log`
- `logs/w1_build_contract.log`
- `logs/w1w2_focused_pytest.log`

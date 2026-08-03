# D1b protocol (closed)

Plan slug: `d1b_n2_reliability`  
Phase: **closed — HARD-STOP**  
Date: 2026-08-03 (confirm_v2 closure)  
Prior PASS: **INVALID_WITHDRAWN** (contaminated CONFIRM v1; do not cite as authority)

## Formal gate

`d1b_reliability_gate.json` → **HARD-STOP**

| Seed | Classification | git_dirty | Role |
|------|----------------|-----------|------|
| 43 | `FORMAL_HARD_STOP_BASIS` | `false` (clean) | Alone establishes HARD-STOP |
| 42 | `DIAGNOSTIC` | `true` | Corroborating only; not sole basis |

- `next_stage`: `D1c_termination_repair_only`
- **Do not proceed to** `1h` / `5h` / `full`
- No document may claim D1b RELIABILITY PASS

## GPU code SHAs (confirm_v2)

| Role | SHA | Tree |
|------|-----|------|
| Selection/provenance/stability fixes | `bee3f009661bb30472fc02682f429787c38b46cb` | clean at train/select |
| Atomic `torch.save` + seed43 train/eval | `0db69d608eb3d3d2f8fd9ba535fb92eb457fc06b` | seed43 eval **clean**; seed42 eval recorded **dirty** |
| Stale v1 “canonical” SHA (superseded) | `dc7493766409c69d09365418dc42f280be4eacdb` | **INVALID** for PASS claims |

## Frozen subsets (canonical = confirm32_v2)

| Set | Path | n | items_sha256 | Status |
|-----|------|--:|--------------|--------|
| DEV32 | `configs/train_subsets/d1b_dev32.json` | 32 | `c886ba31f3305546d9b4938e1cd0b7a2fa4fce738d7da342e9da2a30a28b172c` | retained (dev) |
| CONFIRM32 v2 | `configs/train_subsets/d1b_confirm32_v2.json` | 32 | `63d7e351602e62d26196d944cd1f7a9e928a241e70d968f6eb58c0d1b7a746b9` | **canonical** |
| select8 (v2) | CONFIRM32 v2 split | 8 | `6667da2f8fc2c3706deff48a79489103f739419041f3fb492f5822647075a0ed` | **canonical** |
| gate24 (v2) | CONFIRM32 v2 split | 24 | `68f8d98738114761cee86bc336be5313f3ad550519a4b6764adbdc0c4b32eda8` | **canonical** |
| CONFIRM32 v1 | `configs/train_subsets/d1b_confirm32.json` | 32 | `232df2786859820298271a4845006bbcd4f9b78cfd9fb37872da90d0eee7265c` | **SUPERSEDED / INVALID_contaminated** |
| select8/gate24 v1 | v1 splits | 8/24 | `fa083ed5…` / `611762cc…` | **SUPERSEDED** — see `artifacts/V1_SUPERSEDED.md` |

## Eval contract

- `configs/eval_contracts/d1b_n2_reliability.json` version **1.2.0**
- sha256: `d04295d3ac9864afe5002a8470baead8f5f57a3c8f8d38831fe134aa7feada92`
- Infer: WAVE3c block (`stop_overrun_bias=0.0`, `stop_threshold=0.5`, `stop_consecutive=1`, …)
- Catastrophic Jaccard uses `normalize_for_asr`
- Primary confirm_v2 metric artifacts under this contract are **frozen** (do not mutate)

## Execution rules observed

1. No `MINISPEECH_ALLOW_DIRTY=1` on canonical gates; seed42 confirm_v2 eval nonetheless recorded `git_dirty=true` → diagnostic only.
2. Train/eval via frozen `--subset` (fail-closed missing utts / hash mismatch).
3. CONFIRM selection **select8 only**; gate24/full32 scored once after promote.
4. No length-prior / overrun retune on confirmation.
5. **No 1h/5h/full** — HARD-STOP; next stage is D1c termination-repair only.

## Superseded v1 artifacts

Burned CONFIRM v1 gates/selections under `.cursor/results/d1b_n2_reliability/artifacts/confirm_seed*` and `dev_seed42_*` (and mirrored `out/.../metrics/confirm_seed*`) remain on disk for audit but are **not** a PASS basis. Index: `artifacts/V1_SUPERSEDED.md`.

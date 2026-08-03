# CONFIRM v1 / prior PASS artifacts — SUPERSEDED / INVALID

**Status:** `SUPERSEDED` · `INVALID_WITHDRAWN` · not a D1b PASS basis  
**Canonical confirmation oracle:** `configs/train_subsets/d1b_confirm32_v2.json`  
**Formal terminal result:** HARD-STOP (`../d1b_reliability_gate.json`)  
**Do not proceed to:** 1h / 5h / full

These files are retained for audit. Do **not** delete. Do **not** cite their `verdict: PASS` as current D1b status.

## Contaminated oracle

| Field | Value |
|-------|-------|
| Path | `configs/train_subsets/d1b_confirm32.json` |
| items_sha256 | `232df2786859820298271a4845006bbcd4f9b78cfd9fb37872da90d0eee7265c` |
| Status | `INVALID_contaminated` (superseded by confirm32_v2) |

## Artifacts in this directory (v1 / pre-remediation)

| File | Note |
|------|------|
| `confirm_seed42_selection.json` | Burned CONFIRM v1 selection |
| `confirm_seed42_select8_candidate_rows.json` | Burned CONFIRM v1 |
| `confirm_seed42_gate24_n2_gate.json` | Reports PASS — **INVALID** as formal D1b |
| `confirm_seed42_full32_n2_gate.json` | Reports PASS — **INVALID** as formal D1b |
| `confirm_seed43_selection.json` | Burned CONFIRM v1 selection |
| `confirm_seed43_select8_candidate_rows.json` | Burned CONFIRM v1 |
| `confirm_seed43_gate24_n2_gate.json` | Reports PASS — **INVALID** as formal D1b |
| `confirm_seed43_full32_n2_gate.json` | Reports PASS — **INVALID** as formal D1b |
| `dev_seed42_selection.json` | Pre-confirm_v2 DEV selection (protocol `d1b_fr_lexicographic_v1`) |
| `dev_seed42_candidate_rows.json` | Pre-confirm_v2 DEV rows |
| `dev_seed42_n2_gate.json` | Reports PASS — withdrawn with prior formal PASS |

## Why withdrawn

Prior formal RELIABILITY PASS was invalidated for null TF-CE selection, score-only contract bypass, GT max_frames clamp, missing hard_cap/phrase-loop stability fails, and burned CONFIRM v1. confirm_v2 under contract 1.2.0 did **not** restore PASS (seed43 clean FAIL → HARD-STOP).

## Canonical replacements

| Role | Path |
|------|------|
| Gate | `../d1b_reliability_gate.json` |
| Summary | `../metrics/confirm_v2_summary.json` |
| Hashes | `frozen_hashes.json` (confirm32_v2 + contract 1.2.0) |
| Out metrics (frozen primaries) | `out/d1b_n2_reliability/metrics/confirm_v2_seed{42,43}_*` |

Mirrored v1 copies under `out/d1b_n2_reliability/metrics/confirm_seed*` and `out/d1b_n2_reliability/confirm_seed{42,43}/` are likewise superseded.

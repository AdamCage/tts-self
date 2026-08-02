# Paper 1 — Figures (measured artifacts only)

**Status:** rendered for S4 (`n2_first_execution`).  
**Output dir:** `out/paper1_figures/`  
**Builder:** `python scripts/68_paper1_figures.py --out-dir out/paper1_figures`  
**MOS:** **unmeasured** (do not treat UTMOS as MOS).

Each figure has a machine-readable sidecar JSON with the same stem and a
`source` / `sources` field pointing at the on-disk artifact(s). No synthetic
curves were invented; missing inputs would mark `status: missing_source`.

| Figure | PNG | Source artifact(s) |
|--------|-----|--------------------|
| fig01 CER(σ)/CER(α)/smooth | `out/paper1_figures/fig01_cer_sigma_alpha.png` | `out/latent_error_calibration.json` |
| fig02 R2 three systems | `out/paper1_figures/fig02_r2_three_systems.png` | `out/external_baseline_silero_v3.json` |
| fig03 bank 2×2 delta | `out/paper1_figures/fig03_bank_delta_2x2.png` | `out/bank_delta_2x2.json` |
| fig04 Gate 0 + E0 | `out/paper1_figures/fig04_gate0_panel.png` | `out/head_diagnostics_v01_best.json`, `out/trivial_baselines.json` |
| fig05 E2 vs E4 gap | `out/paper1_figures/fig05_e2_vs_e4_gap.png` | `out/latent_error_calibration.json`, `out/intelligibility_S3.json` |
| fig06 diagonality | `out/paper1_figures/fig06_diagonality.png` | `out/align_audit.json` |
| fig07 telephony | `out/paper1_figures/fig07_telephony_channels.png` | `out/p1_telephony_channels.json` |
| fig08 Track A ladder | `out/paper1_figures/fig08_track_a_ladder.png` | `out/eval_mel_suite_s4_baseline_full.json`, `out/eval_mel_suite_stress_s4.json`, `out/eval_mel_suite_variance_s4.json`, `out/gate_v4_v5_summary.json` |

Manifest: `out/paper1_figures/manifest.json`.

## Key numbers (from sources, not re-estimated)

- Gate 0: `r2_overall = −0.110` (`out/head_diagnostics_v01_best.json`); `gate0_r2_ge_0.28 = false`.
- E2 topline CER ≈ **0.0725** vs E4 S3 synth CER ≈ **0.889** (`latent_error_calibration` / `intelligibility_S3`).
- Silero+frontend bank DER **0.103** vs Track A **0.624**; frontend Δ vs raw **−0.870** (`external_baseline_silero_v3.json`).
- Telephony paired test: `difference_detected_any = false` at n=30 (`p1_telephony_channels.json`).

Related: [paper1_proxy_metrics_trap.md](paper1_proxy_metrics_trap.md), [paper1_preference_status.md](paper1_preference_status.md).

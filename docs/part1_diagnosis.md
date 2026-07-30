# Part 1 diagnosis — measure the truth

Status: **complete** (2026-07-30). Do not re-run heavy ASR for these toplines unless the protocol or ASR backend changes.

Artifacts (WSL / authoring `out/`):

| File | Experiment |
|------|------------|
| `out/trivial_baselines.json` | E0 |
| `out/latent_error_calibration.json` | E1 + E2 GT-latent topline |
| `out/head_diagnostics_*.json` | E3 |
| `out/intelligibility_S3.json` (+ S4 scan) | E4 |
| `out/align_audit.json` | E5 |
| `out/qc_revision.json` | E6 |

Runbook / commands: [next_intelligibility.md](next_intelligibility.md). Promotion gates: [experiment_protocol.md](experiment_protocol.md). Recovery plan: Track A next.

---

## Verdict

**Gate 0 FAIL.** Teacher-forced error matches a constant-mean predictor; the Talker did not learn text→latent content. Legacy free-running stop/duration gates are content-blind. Synth CER is near floor (~0.86–0.89). Do **not** restart S4 / Flow / LoRA until Gate 0 and a Track A intelligibility baseline exist.

---

## Gate 0 / E3 — head diagnostics

| Metric | Value | Interpretation |
|--------|------:|----------------|
| `r2_overall` | ≈ **−0.11** | Worse than / no better than predicting the mean |
| `tf_l1` | ≈ **0.798** | ≈ \(\sqrt{2/\pi}\) constant-mean MAE on unit-variance latents |
| `var_ratio` (`Var(z_pred)/Var(z_gt)`) | ≈ **0.319** | Strong variance collapse |
| `hf_power_ratio` | ≈ **0.311** | High-frequency temporal dynamics attenuated |

Gate 0 requires \(R^2 \ge 0.28\) (TF MAE ≤ 0.85 × constant MAE). **FAIL.**

---

## E0 — trivial baselines

All four baselines (constant mean, copy-previous-frame, random corpus latent, random other utterance of same length) **pass** legacy free-running duration/stop gates when given **oracle length**.

Conclusion: `duration_in_band` / `stop_success` alone must not promote checkpoints.

---

## E5 — alignment audit

| Metric | Value |
|--------|------:|
| `mean_diagonality` | ≈ **0.092** |
| chance (same T, L) | ≈ **0.077** |

Cross-attention is near chance → no usable learned text↔frame alignment (level C Gaussian priors only; `forced.json` was never produced).

---

## E2 — copy-synthesis topline (GT latents)

| Path | CER | n |
|------|----:|--:|
| GT latent → frozen VibeVoice decode → ASR | ≈ **0.073** | 50 |

Latent→wave path is intelligible when latents are correct. Mel→vocoder topline deferred to Track A.

---

## E1 — latent error calibration (normalized units)

**White noise** on GT latents (then denorm → decode → ASR):

- CER flat through σ ≈ **0.2**
- Rise near σ ≈ **0.3**
- Steep after σ ≈ **0.6** (CER roughly **0.22 → 0.82**)

**Variance collapse** \(\hat z' = \alpha \hat z\):

- Cliff between α = **0.7 → 0.5** (CER ≈ **0.076 → 0.439**)

**Temporal smoothing** destroys intelligibility (matches oversmoothed L1 regression).

Head budget: mean-seeking / HF attenuation is far more damaging than modest additive noise. Do not use `vae_std=0.625` as an error budget.

---

## E4 — synth intelligibility (demo set)

| Checkpoint | Synth CER | Notes |
|------------|----------:|-------|
| S3 calibrated | ≈ **0.889** | Unintelligible |
| S4 FR-best | ≈ **0.860** | Also unintelligible; demos flagged **NEAR_SILENT** |

NEAR_SILENT: peak/RMS near zero — check denorm `stats` path vs training `stats_path`, not only MAE.

---

## E6 — QC revision (verbalize rescore)

Rescoring existing Whisper hyps against **verbalized** references (`scripts/25_qc_revision.py --from-qc`):

- Hours recovered from `exclude` by reference verbalization alone: **0**
- Usable corpus (clean + review under current thresholds): ≈ **17.8 h**

Full CTC-ASR re-decode (GigaAM-CTC) may still recover hours lost to Whisper insertions — Track A / C follow-up.

---

## Implications for Track A

1. Codec + GT latents are fine (E2); the failure is Talker / alignment / regression head.
2. Need real TN → stress → G2P, forced alignment → `forced.json`, and a modular mel baseline with explicit durations before scaling data or restarting latent AR.
3. Keep S1–S4 / Flow / TTFB artifacts for Article 1 (proxy-metric trap).

**Paper 1 draft:** working notes for a methods / negative-results article on the proxy-metrics trap (timing gates pass, synthesis fails) are in [paper1_proxy_metrics_trap.md](paper1_proxy_metrics_trap.md). Use Part 1 numbers only; no new training required to cite that doc.

# Paper 1 (working draft) — The Proxy-Metrics Trap in Neural TTS

**Status:** internal notes only — not submission LaTeX.  
**Project:** MiniSpeech-RU (Russian TTS research prototype).  
**Evidence date:** 2026-07-30. Sources: [part1_diagnosis.md](part1_diagnosis.md), frozen JSON under `out/`.

---

## Title (sketch)

**When Timing Looks Right but Speech Does Not: Proxy Metrics in Latent-Space TTS**

Alternate: *The Proxy-Metrics Trap — Duration, Stop, and Teacher-Forced Loss Can Pass While Synthesis Fails*

---

## Abstract (sketch, ~150 words)

Neural TTS pipelines often promote checkpoints using proxy gates: teacher-forced latent L1, free-running duration ratio, stop-head success, and DTW alignment. We document a **methodological trap** on MiniSpeech-RU: a regression Talker over frozen VibeVoice latents with cross-attention to offline Qwen text hiddens **passes legacy free-running gates** under oracle length while **Gate 0 (learned content)** and **synthesis CER** fail decisively. Teacher-forced \(R^2 \approx -0.11\); trivial content-blind baselines pass stop/duration; cross-attention diagonality is indistinguishable from chance; GT-latent copy-synthesis achieves CER \(\approx 0.073\) but model synthesis \(\approx 0.889\). Latent perturbation curves show **variance collapse** and temporal oversmoothing cliffs that mirror the failed head. We argue proxy gates must be demoted to diagnostics and replaced by a ladder anchored on \(R^2\), copy-synthesis toplines, and ASR CER — as implemented in our Gate 0–6 rewrite.

---

## 1. Setup (minimal)

- **Task:** stressed Russian text → offline Qwen3.5-0.8B hidden states → causal Talker + cross-attn → 64-D VibeVoice latents @ 7.5 Hz → frozen decoder.
- **Head:** direct L1 regression (not flow); training ladder S1–S3 completed; S4 FR abandoned.
- **Checkpoint evaluated:** `runs/v01_regression_direct/best.pt` (S3 calibrated).
- **Corpus slice:** RUSLAN val; ASR via frozen config (`configs/asr.yaml`).

This paper is **not** a new model claim — it is a **negative result + measurement protocol** paper.

---

## 2. Core claim — the trap

> **Proxy gates that measure timing, stability, or teacher-forced reconstruction can pass (or look acceptable) while the model has not learned text→acoustic content and synthesized speech is unintelligible.**

Three decoupled facts must be reported together:

| Layer | What passed | What failed |
|-------|-------------|-------------|
| Legacy FR | stop, duration-in-band, DTW (with oracle length) | — |
| Teacher-forced | L1 numerically “reasonable” | \(R^2 < 0\), var collapse |
| End-to-end | GT-latent decode intelligible | Synth CER \(\approx 0.89\) |

---

## 3. Measured evidence (Part 1 only)

All numbers from committed diagnosis artifacts; do not extrapolate beyond these runs.

### 3.1 Gate 0 FAIL — head diagnostics (E3)

Source: `out/head_diagnostics_v01_best.json` (val, n=64, normalized latents).

| Metric | Value | Note |
|--------|------:|------|
| `r2_overall` | **−0.110** | Below zero ⇒ worse than corpus mean |
| `tf_l1_mean` | **0.798** | `tf_l1_over_constant_mae` = **0.9998** ≈ 1.0 |
| `var_ratio` | **0.319** | Var(pred)/Var(gt); strong collapse |
| `hf_power_ratio` | **0.311** | HF temporal dynamics attenuated |
| `gate0_r2_ge_0.28` | **false** | Promotion threshold not met |

Gate 0 criterion: TF MAE ≤ 0.85 × constant-mean MAE (\(R^2 \ge 0.28\)). **FAIL.**

Interpretation: the head predicts near the **corpus mean latent**, not text-conditioned acoustics.

### 3.2 E0 — trivial baselines pass legacy FR gates

Source: `out/trivial_baselines.json` (val, n=32, oracle length).

All four content-blind predictors **`passes_legacy_fr_gates`: true**:

- constant mean (`stop_success_rate` 1.0, `duration_in_band_rate` 1.0)
- copy-previous-frame (mean DTW L1 **0.023** — timing-only artifact)
- random corpus latent
- random other utterance (same length)

**Conclusion:** `duration_in_band`, `stop_success`, and DTW L1 **do not prove** text→latent learning when length is oracle-given.

### 3.3 E5 — alignment audit (cross-attention)

Source: `out/align_audit.json` (n=16).

| Metric | Value |
|--------|------:|
| `mean_diagonality` | **0.0922** |
| `mean_chance` | **0.0770** |
| `mean_vs_chance` | **+0.015** |

Cross-attention mass on the diagonal is **near chance** — no usable monotonic text↔frame alignment. Align prior was level-C Gaussian only (`forced.json` never produced).

### 3.4 E2 — copy-synthesis topline (GT latents)

Source: `out/latent_error_calibration.json` (n=50).

| Path | mean CER | mean WER |
|------|----------|----------|
| GT latent → frozen decode → ASR | **0.0725** | **0.113** |

**Codec + decode path is fine** when latents are correct. Failure is upstream (Talker / alignment / head).

### 3.5 E1 — latent error calibration (cliffs)

Same source; perturbations in **normalized** latent units.

**White noise** (\(\sigma\) on GT latents):

- CER flat through \(\sigma \le 0.2\) (CER ≈ 0.07–0.07)
- Rise at \(\sigma = 0.3\) (CER **0.087**)
- Steep after \(\sigma = 0.6\) (CER **0.222** → **0.817** at \(\sigma=1.0\))

**Variance collapse** (\(\hat z' = \alpha \hat z\)):

- \(\alpha = 0.7\): CER **0.074** (still intelligible)
- \(\alpha = 0.5\): CER **0.439** (cliff)
- \(\alpha = 0.3\): CER **0.858**

**Temporal smoothing** (box kernel width):

- kernel = 3 frames: CER **0.892** (immediate collapse)

Head pathology (`var_ratio ≈ 0.32`, HF attenuation) sits in the **collapse/smooth** regime — far more damaging than modest additive noise. Do **not** use `vae_std=0.625` as an error budget.

### 3.6 E4 — synthesis intelligibility

Source: `out/intelligibility_S3.json` (5-phrase demo set, Whisper ASR).

| Checkpoint | mean CER | mean WER | Sanity |
|------------|----------|----------|--------|
| S3 calibrated | **0.889** | **1.0** | peak/RMS OK |
| S4 FR-best | **0.860** | **1.0** | 2× NEAR_SILENT |

S3 emits audible but **lexically wrong** speech (e.g. ref “привет…” → hyp “дитя”). S4 FR-best adds near-silent outputs — likely denorm/stats mismatch, not intelligibility improvement.

Gap vs E2 topline: **ΔCER ≈ +0.82** (0.889 vs 0.073).

### 3.7 E6 — QC revision (data, not model)

Source: `out/qc_revision.json` (Whisper hyps rescored with verbalized refs).

- **`exclude_hours_delta`: 0.0** — verbalization alone recovered **0 h** from exclude
- Usable corpus (clean + review): **≈ 17.78 h**

Data QC is not the primary bottleneck for the observed synthesis failure.

---

## 4. Implications

### 4.1 For TTS research practice

1. **Never promote on stop/duration/DTW alone** — demonstrate with trivial baselines (E0).
2. **Teacher-forced L1 is insufficient** — report \(R^2\), variance ratio, and HF power vs constant-mean baseline.
3. **Always report a copy-synthesis topline** (GT latents or GT mel) on the same ASR backend.
4. **Calibrate latent sensitivity** (CER vs \(\sigma\), \(\alpha\), smooth) before trusting codec noise budgets.
5. **Audit cross-attention** (diagonality vs chance) when alignment is learned implicitly.
6. **Flag NEAR_SILENT** separately from high CER — amplitude collapse can hide behind timing metrics.

### 4.2 For MiniSpeech-RU — Gate 0–6 rewrite

Legacy FR metrics → **diagnostics only**. New ladder ([experiment_protocol.md](experiment_protocol.md)):

| Gate | Role |
|------|------|
| **0** | \(R^2 \ge 0.28\) — *failed here* |
| **1** | Single-utt overfit + audible words |
| **2** | S1 train-set CER |
| **3** | Val CER vs E2 topline |
| **4–6** | Naturalness, robustness, TTFB |

**Stop rule validated:** do not restart S4 / Flow / LoRA until Gate 0 passes. Track A (TN → stress → G2P → forced align → mel baseline) replaces implicit alignment.

---

## 5. Related work (placeholder bullets)

- Autoregressive / non-AR TTS with auxiliary duration and stop losses (FastSpeech, VITS, etc.) — proxy metric culture.
- Neural codec / latent TTS (VibeVoice, SoundStream, EnCodec) — sensitivity of decode to latent statistics.
- Alignment supervision: CTC, MFA, monotonic aligners vs soft cross-attention.
- Teacher-forcing gap / exposure bias in acoustic models.
- ASR-based TTS evaluation (CER/WER) vs MOS-only reporting.
- Negative results and reproducibility in ML speech ( cite Interspeech / ASRU workshop norms ).

*Fill with proper citations before submission.*

---

## 6. Figures to produce later

1. **Trap schematic** — three-layer diagram: FR gates ✓ / TF L1 misleading / CER ✗.
2. **Gate 0 panel** — bar: constant MAE vs model TF L1; inset \(R^2\), var_ratio, hf_power_ratio.
3. **E0 baseline table** — four trivial predictors vs legacy gate checklist (all pass).
4. **E1 calibration curves** — CER vs \(\sigma\), CER vs \(\alpha\), CER vs smooth kernel (three subplots).
5. **E2 vs E4 gap** — paired bar: GT latent CER 0.073 vs S3 synth 0.889.
6. **E5 attention heatmaps** — 2–3 utterances: learned weights vs chance diagonal (optional if plots regenerated).
7. **Waveform + spec** — one S3 demo: audible but wrong ASR hyp (qualitative).
8. **Gate ladder** — old vs new promotion criteria flowchart.

Audio supplementary: `out/demo_s3_calibrated_20260730/` vs GT copy for same phrases.

---

## 7. Limitations (for discussion section)

- Single speaker corpus (RUSLAN); Russian only.
- One Talker architecture and regression head; flow head not re-evaluated post-diagnosis.
- ASR backend (Whisper) may substitute; CTC re-score deferred.
- Demo set n=5 for E4 — report as illustrative; val CER at scale is future work.
- Align audit n=16; forced alignment not yet built.

---

## 8. Artifact index

| File | Experiment |
|------|------------|
| `out/head_diagnostics_v01_best.json` | E3 / Gate 0 |
| `out/trivial_baselines.json` | E0 |
| `out/latent_error_calibration.json` | E1 + E2 |
| `out/align_audit.json` | E5 |
| `out/intelligibility_S3.json` | E4 |
| `out/qc_revision.json` | E6 |

Full narrative: [part1_diagnosis.md](part1_diagnosis.md).

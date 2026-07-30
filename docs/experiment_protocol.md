# Experiment protocol — MiniSpeech-RU

Gate definitions and frozen evaluation contracts. Fill dates/results as runs complete.

Recovery Part 1 **results**: [docs/part1_diagnosis.md](part1_diagnosis.md) (Gate 0 FAIL). Tooling / Track A next: [docs/next_intelligibility.md](next_intelligibility.md).

## Frozen artifacts

| Artifact | Location / rule |
|----------|-----------------|
| Test set | `manifest.meta.json` → `test_set_sha256` (do not change after Gate B) |
| ASR config | `configs/asr.yaml` (+ hash in run manifests) |
| Listening set | challenge rows in manifest; wav copies under `out/listen/<stage>/` |

## New ladder gates (0–6)

These **replace** stop/duration/DTW as primary promotion criteria. Legacy FR metrics remain diagnostics only.

| Gate | Name | Criterion |
|------|------|-----------|
| **0** | Learned something | Teacher-forced MAE ≤ 0.85 × constant-mean MAE on normalized latents ⇒ \(R^2 \ge 0.28\). Fail ⇒ do not scale data/stage. (`scripts/22_head_diagnostics.py`, `scripts/20_trivial_baselines.py`) |
| **1** | Single-utt overfit | 1 utterance, ~2k steps, GT durations; audible words; CER ≤ 20%. Fail ⇒ pipeline bug, not HPs. |
| **2** | S1 (32 utts) | CER ≤ 10% on the **training** 32 utterances. |
| **3** | Val intelligibility | CER (prefer CTC ASR) ≤ 15% on 100 held-out, reported vs E2 copy-synthesis topline. |
| **4** | Naturalness | Objective MOS predictor within 0.3 of copy-synthesis; human MOS ≥ 3.5 (95% CI) when run. |
| **5** | Robustness | Word repeats/skips ≤ 2%; CER on 6–10 s bucket ≤ CER(2–4 s) + 3 pp. |
| **6** | Latency | Warm TTFB ≤ 200 ms; RTF ≤ 0.3 on 4080 (`scripts/18_measure_ttfb.py`). |

**Stop:** do not restart abandoned S4 / Flow / LoRA until Gate 0 passes. Part 1 curves/toplines exist and show Gate 0 FAIL (`r2_overall≈−0.11`); proceed with Track A (TN/G2P/forced align → mel baseline), not latent ladder scale-up.

## Phase 0 — Codec (ASR)

Structural (all required): `sample=False` determinism; cached-vs-full decode gate in **float32** (`cache_max_abs_diff` ≤ 1e-3) with boundary-spike check; decoder RTF < 0.3. bf16 max-abs is reported separately (soft ≤ 2e-2) — global bf16 noise without a seam spike is not a structural fail.

Intelligibility exit codes from `scripts/11_phase0_feasibility.py`:

| Decision | Exit | Criteria |
|----------|------|----------|
| GO | 0 | ΔWER ≤ 2 pp **and** 95% CI high ≤ 4 pp |
| REVIEW | 3 | ΔWER in 2–4 pp **or** CI crosses 4 pp; or ASR unavailable (without `--skip-asr`) |
| NO-GO | 2 | ΔWER > 4 pp **and** CI low > 4 pp; or structural fail |

ΔCER is auxiliary: if ΔWER is borderline but ΔCER < 1 pp, document as Whisper substitution noise for REVIEW→GO judgment.

Sensitivity σ is in **normalized** latent units when `stats_path` loads. Do **not** treat `vae_std=0.625` as a head error budget — use `scripts/21_latent_error_calibration.py` CER(σ)/CER(α).

## Gate B — Full caches + QC

- 100% of `qc_tag != exclude` rows have latent + text_hidden (index join).
- `stats.safetensors` from train only; align C cached.
- Meta includes filter counters, hours/split, `qc_tag` counts, frozen `test_set_sha256`.

QC tags (`scripts/03b_apply_qc.py`, thresholds configurable):

| Tag | Default CER |
|-----|-------------|
| clean | ≤ 0.15 |
| review | (0.15, 0.25] |
| exclude | > 0.25 |

Revise with verbalized references / CTC ASR via `scripts/25_qc_revision.py` before trusting exclude hours.

## Text-usage (S1/S2)

1. Shuffle texts → relative loss rise ≥ 0.20 (S1); keep logging after.
2. Zero text memory → rise ≥ 0.20 (S1), ≥ 0.35 (S2).
3. Shared GT prefix + two texts → continuation L1 ratio ≥ 3 (S2).

Necessary but **not sufficient** — also require Gate 0 R² and synth CER.

## Free-running (diagnostic)

DTW L1, duration ratio, stop success remain logged (`scripts/09b_eval_free_running.py`) but are **not** promotion gates alone: trivial predictors with oracle length pass them (`scripts/20_trivial_baselines.py`). Prefer Gates 0 and 3.

## Flow A/B + S4 (historical)

- Flow A/B on S1–S3: **keep regression** (`docs/phase3_ab.md`). Flow S4 not started.
- Regression S4 **abandoned** after FR-best `duration_in_band=0.844` FAIL under the old gate; no restart until Gate 0.
  Production snap remains S3 calibrated: `runs/v01_regression_direct/best.pt`
  (= `S3_calibrated_best.pt`; see `docs/phase2_train.md`).

## TTFB (Phase 4 / Gate 6)

Real marks via streaming path (`synthesize_streaming`, chunk_latents=4):
`qwen_ms` / `text_encode_ms`, `first_latent_ms`, `first_pcm_ms` (≥40 ms PCM), plus
`talker_ms` / `sampler_ms` / `codec_first_ms`. Report cold and warm start separately
(`scripts/18_measure_ttfb.py` → `docs/phase4_latency.md`).

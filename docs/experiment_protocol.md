# Experiment protocol — MiniSpeech-RU

Stub for gate definitions and frozen evaluation contracts. Fill dates/results as runs complete.

## Frozen artifacts

| Artifact | Location / rule |
|----------|-----------------|
| Test set | `manifest.meta.json` → `test_set_sha256` (do not change after Gate B) |
| ASR config | `configs/asr.yaml` (+ hash in run manifests) |
| Listening set | challenge rows in manifest; wav copies under `out/listen/<stage>/` |

## Phase 0 — Codec (ASR)

Structural (all required): `sample=False` determinism; cached-vs-full decode gate in **float32** (`cache_max_abs_diff` ≤ 1e-3) with boundary-spike check; decoder RTF < 0.3. bf16 max-abs is reported separately (soft ≤ 2e-2) — global bf16 noise without a seam spike is not a structural fail.

Intelligibility exit codes from `scripts/11_phase0_feasibility.py`:

| Decision | Exit | Criteria |
|----------|------|----------|
| GO | 0 | ΔWER ≤ 2 pp **and** 95% CI high ≤ 4 pp |
| REVIEW | 3 | ΔWER in 2–4 pp **or** CI crosses 4 pp; or ASR unavailable (without `--skip-asr`) |
| NO-GO | 2 | ΔWER > 4 pp **and** CI low > 4 pp; or structural fail |

ΔCER is auxiliary: if ΔWER is borderline but ΔCER < 1 pp, document as Whisper substitution noise for REVIEW→GO judgment.

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

## Text-usage (S1/S2)

1. Shuffle texts → relative loss rise ≥ 0.20 (S1); keep logging after.
2. Zero text memory → rise ≥ 0.20 (S1), ≥ 0.35 (S2).
3. Shared GT prefix + two texts → continuation L1 ratio ≥ 3 (S2).

## Free-running (S3)

Primary: DTW-aligned L1, duration ratio ∈ [0.8, 1.2] on ≥90% val, ASR CER/WER, stop success ≥95%, runaway ≤2%, no NaN/silence collapse. Raw frame L1 is diagnostic only.

## Flow A/B + S4

- Flow A/B on S1–S3: **keep regression** (`docs/phase3_ab.md`). Flow S4 not started.
- Regression S4 **abandoned** after FR-best `duration_in_band=0.844` FAIL (gate ≥0.90); no restart.
  Production snap remains S3 calibrated: `runs/v01_regression_direct/best.pt`
  (= `S3_calibrated_best.pt`; see `docs/phase2_train.md`).
- Gate (val32, calibrated infer): stop≥0.95, runaway≤0.02, duration_in_band≥0.90.

## TTFB (Phase 4)

Real marks via streaming path (`synthesize_streaming`, chunk_latents=4):
`qwen_ms` / `text_encode_ms`, `first_latent_ms`, `first_pcm_ms` (≥40 ms PCM), plus
`talker_ms` / `sampler_ms` / `codec_first_ms`. Report cold and warm start separately
(`scripts/18_measure_ttfb.py` → `docs/phase4_latency.md`).

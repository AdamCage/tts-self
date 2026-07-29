# Phase 4 — Streaming TTFB / RTF (Workstream E)

Date: 2026-07-29  
Runtime: WSL2 + RTX 4080 Super · `HF_HOME=$HOME/hf_cache`  
Code base (pre-commit authoring): `dc5028a` + streaming TTFB patches  
Artifact: `runs/v01_regression_direct/ttfb_phase4.json`

## Inference snapshot

| Item | Value |
|------|-------|
| Checkpoint | `runs/v01_regression_direct/S3_calibrated_best.pt` (= restored `best.pt`) |
| Head | regression (Flow not used) |
| Stop | `thr=0.45`, `consec=3`, `min_frac=0.94`, `max_frac=1.6`, `duration_scale=1.04` |
| Stream | cached causal decode, `chunk_latents=4`, `min_pcm_ms=40` |
| Gate proof | val32 `duration_in_band=0.906`, stop=1.0 (`eval_free_running_S3_calibrated.json`) |

S4 FR-best fails duration gate (in_band=0.844) — see [`phase2_train.md`](phase2_train.md) §C4. Latency measured on S3 production snap; no Flow.

## Method

- Clock starts **after** text normalize + stress (protocol).
- Real marks from `synthesize_streaming` / `iter_generate` (not estimated):
  - **Qwen** `qwen_ms` / `text_encode_ms`
  - **Talker first latent** `talker_ms` (wall to first latent − Qwen)
  - **Sampler** `sampler_ms` (0 for regression)
  - **Codec first PCM** `codec_first_ms` (first cached decode flush ≥40 ms PCM)
- TTFB = wall time to first PCM buffer ≥ 40 ms.
- Cold = first utterance after model load (includes CUDA / cudnn warmup).
- Warm = subsequent utterances (3 reps on phrase 0 + 1 each on phrases 1–2).
- Script: `scripts/18_measure_ttfb.py`.

## Results (measured)

### Summary

| Mode | TTFB (ms) | RTF | estimated? |
|------|----------:|----:|:----------:|
| Cold (post-load, first utt) | **770** | 0.344 | no |
| Warm mean (n=5) | **132** | **0.086** | no |
| Warm min | 129 | — | no |
| Long-form (~8.0 s audio) | TTFB 141 | **0.057** | no |

Cold is dominated by first Qwen forward + CUDA warmup; **warm TTFB ~130 ms** is the production-relevant figure.

### Breakdown (ms)

| Run | Qwen | Talker→1st latent | Sampler | Codec 1st PCM | TTFB | audio_s | RTF |
|-----|-----:|------------------:|--------:|--------------:|-----:|--------:|----:|
| Cold | 658 | 74 | 0 | 29 | 770 | 2.53 | 0.344 |
| Warm 0.0 | 107 | 4.4 | 0 | 12 | 132 | 2.53 | 0.086 |
| Warm 0.1 | 106 | 4.1 | 0 | 12 | 130 | 2.53 | 0.086 |
| Warm 0.2 | 109 | 3.9 | 0 | 12 | 133 | 2.53 | 0.086 |
| Warm 1.0 | 105 | 4.1 | 0 | 11 | 129 | 2.13 | 0.094 |
| Warm 2.0 | 113 | 4.2 | 0 | 12 | 138 | 4.00 | 0.076 |
| Long-form | 116 | 4.6 | 0 | 16 | 141 | 8.00 | 0.057 |

Warm stack is ~**Qwen 80% / Talker 3% / Codec first chunk 9%** of TTFB; regression sampler = 0.

### Long-form note

Multi-sentence probe (60 frames, stop triggered, peak abs 0.34, not silence-like). Cached decode with `chunk=4` held RTF **0.057** with no seam/silence collapse observed in the automated checks. Subjective seam listening left as optional follow-up.

## Commands (WSL)

```bash
cd ~/projects/tts-self && source .venv/bin/activate && export HF_HOME=$HOME/hf_cache

python scripts/18_measure_ttfb.py \
  --ckpt runs/v01_regression_direct/best.pt \
  --config configs/infer.yaml \
  --out runs/v01_regression_direct/ttfb_phase4.json

python scripts/17_demo_phrases.py \
  --ckpt runs/v01_regression_direct/best.pt \
  --out-dir out/demo_phrases
```

## Demo paths

- WSL: `~/projects/tts-self/out/demo_phrases/`
- Windows mirror: `C:\projects\tts-self\out\demo_phrases\`

## MVP ladder status (regression path)

| Stage | Status |
|-------|--------|
| S1 text-usage | PASS |
| S2 / S2b stop | PASS |
| S3 + calibrated infer | **PASS** — **production snap** |
| S4 full corpus | **FAIL gate** (FR-best in_band 0.844; MVP stays on S3) |
| Flow A/B | Done → keep regression |
| Phase 4 streaming TTFB | **DONE** (this doc) |

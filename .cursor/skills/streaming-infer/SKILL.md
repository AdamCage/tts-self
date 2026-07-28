---
name: streaming-infer
description: >-
  Runs MiniSpeech-RU inference and cached causal streaming decode, measures
  TTFB decomposition. Use when synthesizing audio, debugging streaming seams,
  measuring latency/TTFB/RTF, or Phase 4 streaming work.
---

# Streaming & inference

## Batch infer

```bash
python scripts/08_infer.py \
  --text "Привет, это тест." \
  --ckpt runs/v01_regression_direct/best.pt \
  --config configs/infer.yaml \
  --out out/synth.wav
```

Stress: accentuator via `frontend/stress.py` before Qwen (train/infer parity).

## Streaming decode

Use `stream_decode_latents(codec, z_raw, chunk=4)` — **cached causal decode**, not overlap-add by default.

TTFB definition: time from finished text normalization to first PCM buffer ≥ 20–40 ms.

Measure components separately: Qwen forward, first latent, flow steps, decoder chunk.

## Stop rule (infer)

Stop only when all hold:

1. `p(stop) > θ` for N consecutive frames
2. `t ≥ dur_pred * 0.5`
3. hard cap `t ≥ dur_pred * 1.5`

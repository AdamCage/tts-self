---
name: offline-caches
description: >-
  Builds MiniSpeech-RU offline caches: RUSLAN manifest, VibeVoice latents
  (sample=False), latent stats, Qwen text-hidden shards, and alignment priors
  A/B/C. Use when preparing datasets, encoding latents, caching Qwen hiddens,
  building manifests, or when the user mentions Phase 1, shards, or precompute.
---

# Offline caches (Phase 1)

## Order

1. Manifest → 2. Latents → 3. Stats (inside encode) → 4. Text hidden → 5. Align (optional)

## Commands

```bash
# Extract subset from tar if needed
python scripts/16_extract_ruslan_subset.py --ruslan-dir $HOME/datasets/tts/ruslan --max-wavs 400

python scripts/02_build_manifest.py --config configs/data_ruslan_local.yaml \
  --ruslan-dir $HOME/datasets/tts/ruslan

# Optional ASR filter (slow)
# python scripts/03_verify_text_asr.py --config configs/data_ruslan_local.yaml

python scripts/04_encode_latents.py --config configs/data_ruslan_local.yaml
python scripts/05_cache_text_hidden.py --config configs/data_ruslan_local.yaml \
  --qwen-config configs/qwen_local.yaml --batch-size 4

# Non-blocking; C is fine to start
python scripts/06_align.py --config configs/data_ruslan_local.yaml --level C
```

## Rules

- Resample 22k→24k **in memory**; no second WAV corpus; no custom loudness (FE does −25 dBFS).
- Always `sample=False`.
- Shard with safetensors (`ShardWriter`), not tens of thousands of `.pt` files.
- Cache Qwen layers indices `[11, 17, 23]` with learned `LayerMix` at train time.
- `--max-utts` truncates the written manifest — re-split or use full encode before claiming val/test coverage.
- Synth path for CI: `python scripts/13_make_synth_data.py` + `configs/data_synth.yaml`.

## Verify

- Random 32: decode(cached latent) ≈ decode(fresh encode)
- `stats.safetensors` exists; normalized mean≈0 / std≈1
- Align meta records level A/B/C

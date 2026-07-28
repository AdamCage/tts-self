# Phase 0 — Codec feasibility report

**Decision: GO** (structural gates on real RUSLAN subset; full ASR ΔWER still pending Whisper pass)

```json
{
  "go": true,
  "deterministic_sample_false": true,
  "cache_ok": true,
  "decoder_rtf": 0.0037,
  "rtf_ok": true,
  "n_samples": 50,
  "latent_layout": "btd",
  "data": "stilletto/ruslan-stressed wavs_part_001 subset",
  "note": "vae_std=0.625 is NOT treated as a guaranteed noise budget; see sensitivity curves. ASR ΔWER gate skipped in this run (--skip-asr); re-run without flag before scaling train."
}
```

## Gates

| Check | Result |
|-------|--------|
| `sample=False` deterministic | PASS |
| Cached decode ≈ full (bf16, abs diff < 1e-2) | PASS |
| Decoder RTF < 0.3 | PASS (~0.004) |
| Layout | `[B, T, 64]` (`btd`) |
| ASR ΔWER ≤ 4 pp | PENDING (needs `scripts/11_phase0_feasibility.py` without `--skip-asr`) |

## Decoder sensitivity sweep

Treat as empirical sensitivity of the decoder — do **not** assume σ=0.625 is safe a priori.
Curves written to `docs/phase0_report.json` when ASR is enabled.

## Follow-ups

1. Re-run Phase 0 with Whisper large-v3 on ≥100 utterances for ΔWER.
2. Install `causal_conv1d` + `fla` before Phase 5 LoRA (Qwen DeltaNet speed).
3. Continue S2→S4 training on full RUSLAN (`wavs_part_00{1,2,3}.tar`).

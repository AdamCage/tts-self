# Phase 0 — Codec feasibility report

**Decision: GO (structural only)** — full ASR ΔWER gate not yet re-run with the upgraded three-level script.

```json
{
  "go": true,
  "decision_note": "structural gates on real RUSLAN subset; ASR gate pending",
  "deterministic_sample_false": true,
  "cache_ok": true,
  "decoder_rtf": 0.0037,
  "rtf_ok": true,
  "n_samples": 50,
  "latent_layout": "btd",
  "data": "stilletto/ruslan-stressed wavs_part_001 subset",
  "note": "vae_std=0.625 is NOT a guaranteed noise budget. Re-run without --skip-asr using configs/asr.yaml before scaling."
}
```

## Gates

| Check | Result |
|-------|--------|
| `sample=False` deterministic | PASS |
| Cached decode ≈ full (bf16, abs diff < 1e-2) | PASS |
| Decoder RTF < 0.3 | PASS (~0.004) |
| Layout | `[B, T, 64]` (`btd`) |
| ASR ΔWER (GO/REVIEW/NO-GO) | PENDING |

## Three-level ASR gate (new)

See [experiment_protocol.md](experiment_protocol.md). Script: `scripts/11_phase0_feasibility.py`.

```bash
cd ~/projects/tts-self && source .venv/bin/activate
export HF_HOME=$HOME/hf_cache
python scripts/11_phase0_feasibility.py \
  --config configs/codec.yaml \
  --data configs/data_ruslan_local.yaml \
  --asr-config configs/asr.yaml \
  --n 120 --sweep-n 10 \
  --report docs/phase0_report.md
```

Per-utterance records: `docs/phase0_asr_records.jsonl`.

## Decoder sensitivity sweep

Treat as empirical sensitivity of the decoder — do **not** assume σ=0.625 is safe a priori.
Curves written to `docs/phase0_report.json` when ASR is enabled (`--sweep-n` utterances).

## Follow-ups

1. Re-run Phase 0 with Whisper large-v3 on ≥120 utterances (expand to 300 on REVIEW).
2. Install `causal_conv1d` + `fla` before Phase 5 LoRA.
3. After GO: full RUSLAN extract → corpus QC → caches → S2+.

# T11 — HiFi-GAN fine-tune (deferred)

**Status:** deferred pending Gate V2 (T07 A/B).

## Why wait

Gate V0 already showed that matching mel `fmax=8000` to SpeechBrain HiFi-GAN lifts UTMOS (~+0.26 on GT mel copy-synth). T07 fine-tunes the acoustic model in that domain. Only if V2 still leaves a large gap to real-audio UTMOS (~3.2 on the domain probe) is speaker-adapted vocoder FT the next bottleneck.

## Acceptance (Gate V3) when unblocked

- Fine-tune LibriTTS HiFi-GAN on RUSLAN with MPD/MSD, same mel domain as the acoustic model (`mel80_hifigan`).
- Evaluate **copy-synthesis only** (GT mel → vocoder).
- Pass if UTMOS(copy-synth FT) > UTMOS(copy-synth LibriTTS baseline) on the same val subset.

## Suggested entry point (not implemented yet)

Reuse `~/models/hifigan` SpeechBrain weights; do not train from scratch. Keep pairs under a dedicated dir (e.g. `~/datasets/tts/ruslan/vocoder_pairs_hifigan/`) so acoustic caches stay untouched.

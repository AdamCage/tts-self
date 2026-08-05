# Wave 0 — Tokenizer GO/NO-GO + external baselines

**Plan:** `.cursor/plans/d1-discrete-speech-lm_860c29d5.plan.md`  
**Todo:** `probe-tokenizer-baselines`  
**Date:** 2026-08-02  
**WSL HEAD:** `dce4f45` (clean clone); authoring code via `PYTHONPATH=/mnt/c/projects/tts-self/src`  
**git_dirty:** true on D0 probe (methodology + Wave 0 WIP uncommitted); `MINISPEECH_ALLOW_DIRTY=1`  
**Methodology:** read-only (not edited)

## Verdict

**D0 PASS** — frozen Qwen3-TTS Tokenizer-12Hz is GO for discrete Speech LM training.

Artifact: `out/d1_discrete_speech_lm/d0_gate.json`

| Criterion | Result | Value |
|-----------|--------|------:|
| Copy-synth CER ≤ 0.073 (E2-compatible n=50) | **PASS** | **0.0678** |
| Bit-identical re-encode | **PASS** | true |
| bank_entity DER ≤ continuous VibeVoice anchor | **PASS** | **0.325** ≤ 0.446 |

## Key metrics vs anchors

| Metric | Qwen3-TTS 12Hz | Continuous VibeVoice anchor | Notes |
|--------|---------------:|----------------------------:|-------|
| E2 copy-synth mean CER | 0.0678 | 0.0725 (live) / 0.073 (historical E2) | Qwen better than ceiling |
| E2 telephony CER (µ-law) | 0.0956 | — | ΔCER +0.0279 vs clean |
| bank_entity DER (n=40, Track A src) | 0.325 | 0.446 | Same Track A sources |
| bank_entity EER | 0.45 | 0.525 | |
| Mean codebook utilization | 0.817 | — | 16×2048 |

Geometry (measured + registered as `qwen3tts_12hz`):

- SR 24000, hop 1920 → **12.5 Hz**, 16 codebooks, vocab 2048, `torch.int64`
- Model id: `Qwen/Qwen3-TTS-Tokenizer-12Hz` (Apache-2.0)

## External baselines (demo5)

| System | demo5 mean CER | Wavs |
|--------|---------------:|------|
| Silero raw | 0.0305 | `out/d1_discrete_speech_lm/silero_raw/demo5/` |
| Silero + frontend | **0.0169** | `out/d1_discrete_speech_lm/silero_frontend/demo5/` |
| Track A mel S4 | 0.0935 | `out/d1_discrete_speech_lm/track_a/demo5/` |
| Qwen3-TTS 0.6B CustomVoice (speaker=ryan) | **0.0169** | `out/d1_discrete_speech_lm/qwen3_tts_0.6b/demo5/` |
| CosyVoice3 | **blocked** | not installed (`cosyvoice3_metrics.json`) |
| Qwen3-TTS 1.7B | not run (time); API same as 0.6B | — |

Bank/latin expansion for full external systems deferred (codec bank DER already measured for D0).

## Files added

- `src/minispeech/constants.py` — `qwen3tts_12hz` + `QWEN3_TTS_TOKENIZER_ID`
- `src/minispeech/codec/qwen3_tts_wrapper.py` — frozen encode/decode
- `src/minispeech/codec/__init__.py` — exports
- `scripts/70_probe_qwen3_tts_tokenizer.py` — D0 probe + continuous-anchor pass
- `scripts/70b_external_baselines_d1.py` — Silero / Track A / Qwen3-TTS / CosyVoice stub
- `tests/test_qwen3_tts_geometry.py`
- `pyproject.toml` — optional `[qwen_tts]` extra + pin warning

## Artifacts

```
out/d1_discrete_speech_lm/
  d0_gate.json
  d0_probe_report.json
  continuous_codec_anchor.json
  qwen3tts_12hz_copysynth/{e2,e2_telephony,bank_entity}/
  vibevoice_continuous_{copysynth,bank}/
  bank_src_track_a/
  silero_{raw,frontend}/demo5/
  track_a/demo5/
  qwen3_tts_0.6b/demo5/
  *_metrics.json

.cursor/results/d1_discrete_speech_lm/
  WAVE0_REPORT.md
  listen_index.json
  logs/d0_*.log
```

## Env notes / dual transformers pin

`qwen-tts==0.1.1` requires `transformers~=4.57` and **conflicts** with VibeVoice (`transformers>=5`). Wave 0 used two passes:

1. Continuous anchor with transformers 5.14.1  
2. Qwen D0 / Qwen3-TTS baseline with transformers 4.57.3  

After probes, restore `transformers==5.14.1` for Track A / VibeVoice work. Documented in `pyproject.toml` optional extra.

Runtime: `PYTHONPATH=/mnt/c/projects/tts-self/src` + `MINISPEECH_ALLOW_DIRTY=1` (no commit of methodology).

## Listen paths (Windows)

```
C:\projects\tts-self\out\d1_discrete_speech_lm\qwen3tts_12hz_copysynth\e2\
C:\projects\tts-self\out\d1_discrete_speech_lm\silero_frontend\demo5\
C:\projects\tts-self\out\d1_discrete_speech_lm\silero_raw\demo5\
C:\projects\tts-self\out\d1_discrete_speech_lm\track_a\demo5\
C:\projects\tts-self\out\d1_discrete_speech_lm\qwen3_tts_0.6b\demo5\
```

## Blockers for next todo `build-discrete-caches`

1. **None for D0** — gate PASS; encoding may proceed.
2. Implement `scripts/71_encode_codec_tokens.py` + shard I/O; do **not** reuse VibeVoice latent caches.
3. Encode under **qwen-tts transformers pin** (or isolated venv) — keep VibeVoice pin separate.
4. Full Track N train encode ~72.98 h / 51234 utts — schedule GPU time; record codec hash/fingerprint in shard provenance.
5. CosyVoice3 / Qwen 1.7B full-suite baselines still open (not blocking caches).
6. Optional: `sox` missing in WSL (warning only from `qwen_tts` import).

## Methodology alignment

Read `.cursor/methodology/{README,04_tokenizers,08_evaluation,09_roadmap}.md` — D0 criteria match; no methodology edits during this todo.

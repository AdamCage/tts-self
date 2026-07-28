---
name: phase0-feasibility
description: >-
  Runs MiniSpeech-RU Phase 0 codec GO/NO-GO on VibeVoice Acoustic Tokenizer
  (roundtrip, cache-equivalence, RTF, sensitivity sweep, optional ASR ΔWER).
  Use when starting the project, validating the codec on Russian audio,
  changing tokenizer settings, or when the user mentions Phase 0, feasibility,
  codec roundtrip, or noise/sensitivity sweep.
---

# Phase 0 — Codec feasibility

## Preconditions

- WSL2, repo venv activated, `HF_HOME=$HOME/hf_cache`
- Tokenizer at `~/datasets/tts/models/VibeVoice-AcousticTokenizer` (or Hub id)
- Prefer real RUSLAN wavs under `~/datasets/tts/ruslan/wavs`

## Steps

1. `python scripts/00_env_check.py`
2. Structural + RTF (fast):

```bash
python scripts/11_phase0_feasibility.py \
  --config configs/codec.yaml \
  --data configs/data_ruslan_local.yaml \
  --ruslan-dir $HOME/datasets/tts/ruslan \
  --n 50 --skip-asr \
  --report docs/phase0_report.md
```

3. Full gate (before scaling train):

```bash
python scripts/11_phase0_feasibility.py \
  --data configs/data_ruslan_local.yaml \
  --ruslan-dir $HOME/datasets/tts/ruslan \
  --n 100 \
  --report docs/phase0_report.md
```

## Acceptance

- `sample=False` deterministic
- Cached vs full decode: structural gate in **float32** (`cache_max_abs_diff` default `1e-3`)
  plus boundary-spike check (`boundary_max > ratio × interior_p95` → fail).
  bf16 max-abs alone is diagnostic (`cache_max_abs_diff_bf16`, ~`2e-2`); global bf16
  noise without a seam spike is not a NO-GO.
- Decoder RTF < 0.3
- If ASR enabled: `ΔWER ≤ 4` pp
- Exit `0` = GO, `2` = NO-GO

## Interpretation

- Do **not** assume σ=`vae_std` (0.625) is a safe head RMSE.
- Sensitivity curves are empirical only.
- On NO-GO: isolate codec fallback in `constants.py` + `codec/wrapper.py` only.

## Outputs

- `docs/phase0_report.md`
- `docs/phase0_report.json`

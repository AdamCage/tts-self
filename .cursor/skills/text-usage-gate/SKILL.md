---
name: text-usage-gate
description: >-
  Diagnoses and fixes MiniSpeech-RU S1 text-usage failures (shuffle/mask tests,
  residual head ignoring text, acoustic dropout). Use when text-usage fails,
  WER is high despite low teacher-forced loss, attention is non-diagonal, or
  the user mentions shuffle test, mask test, or Talker ignoring text.
---

# Text-usage gate (S1)

## Run

```bash
python scripts/09_eval.py --ckpt runs/<run>/best.pt --data configs/data_ruslan_local.yaml
```

Pass when both shuffle and mask `relative_rise ≥ 0.20` (`s1_text_usage_pass: true`).

## Failure modes

| Symptom | Likely cause | Fix |
|---------|--------------|-----|
| Rise ≈ 0 with residual head | Copying `z_(t-1)` | Switch to `residual: false` |
| Rise small with direct head | Acoustics dominate | Raise `train.acoustic_dropout` (0.25–0.4) |
| Low diagonality | No/weak prior | Enable align C + guided loss early |
| Empty val loader | Manifest truncated to train-only | Eval falls back to train; re-encode full split |

## Required mitigations (already in code)

- `acoustic_dropout` in `MiniSpeech.forward` / trainer
- Direct-head config: `configs/talker_v01_direct.yaml`
- Do not trust teacher-forced L1 alone

## After pass

Re-introduce residual as an ablation only; keep text-usage checks on every stage.

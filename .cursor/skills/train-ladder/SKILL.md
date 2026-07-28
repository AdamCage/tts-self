---
name: train-ladder
description: >-
  Runs the MiniSpeech-RU training ladder S1→S4 (32 utts → 1h → 5h → full),
  regression then flow, with auto batch and run manifests. Use when training,
  fine-tuning Talker/heads, switching residual vs direct head, starting flow
  matching, or when the user mentions S1/S2/S3/S4 or Phase 2/3 train.
---

# Train ladder

## Defaults

- Early S1: `configs/talker_v01_direct.yaml` (`residual: false`, `acoustic_dropout: 0.35`)
- Residual head only after text-usage gate passes
- Flow: `configs/flow_v02.yaml`, init Talker from regression ckpt when available
- Data: `configs/data_ruslan_local.yaml` (or synth)

## S1 smoke

```bash
python scripts/07_train.py \
  --config configs/talker_v01_direct.yaml \
  --data configs/data_ruslan_local.yaml \
  --mode regression --stage S1 --batch-size 8 --steps 2000

python scripts/09_eval.py \
  --ckpt runs/v01_regression_direct/best.pt \
  --data configs/data_ruslan_local.yaml
```

Require `s1_text_usage_pass: true` before S2.

## Scale

Edit/use stages in config:

| Stage | Data | Goal |
|-------|------|------|
| S1 | 32 utts | Overfit + text-usage |
| S2 | 1 hour | Generalize inside speaker |
| S3 | 5 hours | Stability / stop / duration |
| S4 | full | Scale |

If S3 free-running L1 > 3× teacher-forced and speech collapses after ~3s despite noise+SS → enable non-AR fallback (`scripts/14_enable_nonar_fallback.py`).

## Flow

```bash
python scripts/07_train.py \
  --config configs/flow_v02.yaml \
  --data configs/data_ruslan_local.yaml \
  --mode flow --stage S1 --batch-size 8
```

Expect smaller auto batch (often 8–32) than regression.

## Checklist

- [ ] Caches exist for every train utt
- [ ] Normalized latents in train loop
- [ ] `run_manifest.json` written
- [ ] Text-usage gate green on S1
- [ ] Free-running metric logged (not only TF loss)

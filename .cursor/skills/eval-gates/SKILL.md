---
name: eval-gates
description: >-
  Runs MiniSpeech-RU evaluation gates: text-usage, smoke e2e, optional WER/TTFB,
  and interprets pass/fail. Use when evaluating checkpoints, before promoting a
  stage, measuring quality, or when the user mentions eval, WER, UTMOS, or gates.
---

# Eval gates

## Always (cheap)

```bash
pytest -q
python scripts/10_smoke_e2e.py --skip-codec
python scripts/09_eval.py --ckpt runs/<run>/best.pt --data configs/data_ruslan_local.yaml
```

S1 promote only if `s1_text_usage_pass: true`.

## Phase 0 quality (before scaling data)

```bash
python scripts/11_phase0_feasibility.py \
  --data configs/data_ruslan_local.yaml \
  --ruslan-dir $HOME/datasets/tts/ruslan \
  --n 100
```

Need ASR ΔWER ≤ 4 pp (do not rely on `--skip-asr` for go-to-scale).

## Metrics map

| Metric | Script / module | Notes |
|--------|-----------------|-------|
| Text-usage | `09_eval.py` | Gate S1 |
| Smoke | `10_smoke_e2e.py` | Structural |
| WER/CER | `eval/asr_wer.py` | Compare to codec ceiling |
| Diagonality | `eval/alignment_metrics.py` | Alignment health |
| TTFB/RTF | `infer/stream.py` | Phase 4 |
| Stress | `eval/stress_acc.py` | Challenge set |

## Reporting

Write JSON next to the checkpoint (`eval_*.json`) and mention `run_manifest.json` git sha + config hash in any summary to the user.

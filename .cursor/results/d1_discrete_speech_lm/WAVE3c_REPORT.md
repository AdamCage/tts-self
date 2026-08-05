# Wave 3c — Final N2 push (infer-only)

**Plan:** `.cursor/plans/d1-discrete-speech-lm_860c29d5.plan.md`  
**Todo:** `pass-oracle-n2` (final S1-only push; **no** 1h/5h/full)  
**Date:** 2026-08-02  
**Prior:** WAVE3b FR CER **0.0641** / ceiling **0.0528**  
**Ckpt (unchanged):** `~/projects/tts-self/runs/discrete_speech_lm_300m_s1_stop_3k/last.pt` @ 3000  
**Methodology / plan:** read-only

## Verdict

**N2 FAIL** — miss ≈ **0.0005** CER. Ladder stays **blocked**.

| | WAVE3b | WAVE3c (canonical) | Ceiling |
|--|--:|--:|--:|
| FR CER | 0.0641 | **0.05335** | 0.05284 |
| Δ vs WAVE3b | — | **−0.0108** | — |
| Runaway / CB0 | PASS | PASS (`max_run=6`) | — |
| Text-usage / baselines | PASS | PASS | — |
| Mean dur ratio | 0.911 | **1.000** | — |

Gate: `out/d1_discrete_speech_lm/n2_gate.json`  
Eval: `out/d1_discrete_speech_lm/n2_eval_stop_3c/`  
Rows: `out/d1_discrete_speech_lm/n2_eval_stop_3c/n2_model_rows.json`

## Diagnosis (before change)

Per-utt WAVE3b rows:

- Duration `<0.9×GT`: **12** utts, mean CER **0.140**
- Duration OK: **20** utts, mean CER **0.018**
- Failures = truncated endings on slower CV speech after `expected` frames
- `stop_overrun_bias` adds `+bias * max(0, t−expected)` to stop logits → early cut once past the text-length prior

## What changed (one focused lever)

Infer-only (no retrain):

| Knob | WAVE3b | WAVE3c |
|------|--------|--------|
| `stop_overrun_bias` | 0.5 | **0.0** |
| `frames_per_text_token` | 2.5 | 2.5 (kept) |
| `stop_earliest_frac` | 0.7 (hardcoded) | **0.7** (now config; exposed) |
| High-conf early stop `@0.85` before earliest | yes | **removed** (caused min-frame risk) |

Also: CLI `--config` infer block now **overrides** frozen ckpt snapshot (`73_infer` / `74_eval`).

### Rejected attempts (same iteration)

| Attempt | FR CER | Notes |
|---------|-------:|-------|
| Raise prior + earliest 0.95 / thr 0.65 / consec 2 | **0.341** | Forced past GT → garbage tail / runaway FAIL |
| Mild fpt=3.2 + earliest 0.85 | **0.202** | Still overshoot on Ruslan (earliest > GT) |
| Zero overrun bias only | **0.05335** | Best; near miss |

## Remaining error mass

| Bucket | n | mean CER |
|--------|--:|---------:|
| dur OK [0.9, 1.1] | 27 | 0.0138 |
| short `<0.9` | 2 | 0.4545 |
| long `>1.1` | 3 | 0.1421 |

Worst utt: `common_voice_ru_17_hf_mirror__ba1a4089…` CER **0.909**  
REF «Подать, что обыкновенно.» → HYP «Краснодар – прекрасное место.» (content hallucination, not truncation).  
Zero that single utt → mean **0.0249** ≪ ceiling.

## Classification

**Almost** — not architectural for the codec/LM stack. Residual is:

1. One catastrophic content failure on a short CV utt (~half the remaining gap),
2. A few lexical/ASR diffs at matched duration,
3. Tiny stop-tail overshoot on 3 utts.

Next (later, still S1): FR-aware checkpoint / targeted stop calibration — **not** data scale. Do **not** raise length prior above GT for Ruslan.

## Wavs

Listen (ASCII):

- `out/d1_discrete_speech_lm/s1_listen_demo/0000_ruslan_000418_RUSLAN.wav` … `0007_*.wav`  
  (e.g. 0000 frames=**29** = GT)

N2 suite: `out/d1_discrete_speech_lm/s1_n2_wavs_stop_3c/model/`

## Ladder

**`train-current-ladder` blocked.** Do not start 1h / 5h / full.

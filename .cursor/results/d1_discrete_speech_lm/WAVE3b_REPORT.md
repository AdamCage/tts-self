# Wave 3b — Stop/EOS + repetition fix after N2 FAIL

> **Superseded for CER:** see [`WAVE3c_REPORT.md`](WAVE3c_REPORT.md)  
> (FR CER 0.0641 → **0.05335**; still FAIL by ≈0.0005; ladder still blocked.)

**Plan:** `.cursor/plans/d1-discrete-speech-lm_860c29d5.plan.md`  
**Todo:** `pass-oracle-n2` (continuation; do **not** scale ladder)  
**Date:** 2026-08-02  
**Prior:** `WAVE3_REPORT.md` — FR CER **0.451**, runaway `cb0_max_run=140`  
**git_sha:** `dce4f456` (dirty; `MINISPEECH_ALLOW_DIRTY=1`)  
**Methodology / plan file:** read-only (not edited)

## Verdict

**N2 FAIL** — CER still above ceiling; **`train-current-ladder` remains blocked**.

Canonical gate: `out/d1_discrete_speech_lm/n2_gate.json`  
Canonical ckpt: `~/projects/tts-self/runs/discrete_speech_lm_300m_s1_stop_3k/last.pt` @ **3000** steps  
(`ckpt_role: last.pt@3000_stop_retrain`)

| Criterion | WAVE3 (prior) | WAVE3b (this) | Result |
|-----------|--------------:|--------------:|--------|
| FR CER | **0.451** | **0.0641** | still FAIL vs ceiling **0.0528** |
| Ceiling (copy+0.03) | 0.0528 | 0.0528 | — |
| Subset copy CER | 0.0228 | 0.0228 | — |
| Runaway / max CB0 run | 140 FAIL | **5 PASS** | fixed |
| CB0 healthy | FAIL | **PASS** | fixed |
| Text-usage (shuffle/mask/prefix) | PASS | **PASS** | held |
| Bootstrap vs wrong/random | PASS | **PASS** | held |
| Mean dur ratio (model) | ~2.7 | **0.911** | fixed |

Miss margin: **0.0641 − 0.0528 ≈ 0.011** CER (near miss).  
Blind extension to 5000 steps previously **regressed** FR CER to **0.107** while TF CE kept falling — do not chase TF CE.

## Root cause (localized)

1. Collate already emitted `eos_targets` / `SPEECH_EOS_ID=2048`, but the model had **no stop head / stop loss**.
2. `infer.eos_cb0: null` → `generate()` always ran to `max_frames` → duration runaway + CB0 loops (`cb0_max_run=140`).
3. Teacher-forced CE looked healthy while free-run collapsed → **infer/stop bug**, not data scaling.

## What changed

| Area | Change |
|------|--------|
| `src/minispeech/models/discrete_speech_lm.py` | `stop_head` MLP; BCE stop loss on last valid frame; `generate()` stop threshold / consecutive fires / length prior / overrun bias / hard cap / CB0 repetition penalty + `max_cb0_repeat` |
| `src/minispeech/train/discrete_lm_trainer.py` | Pass `stop_weight` / `stop_pos_weight`; log `stop` |
| `configs/discrete_speech_lm_300m.yaml` | `loss.stop_weight: 1.0`, `stop_pos_weight: 20.0`; greedy infer + stop/repeat knobs |
| `scripts/73_infer_discrete_lm.py`, `74_eval_discrete_lm.py` | Forward new `generate` kwargs |
| `tests/test_discrete_speech_lm.py` | Stop targets / stop loss / forced early stop |

## Oracle re-run

Caches: `codec_tokens_s1_32` (Ruslan + CV).  
Train: batch 2, grad_accum 4, bf16, 3000 steps, ~**240 s**, params **442.7M** (stop head ~+263k vs Wave3).  
Log: `.cursor/results/d1_discrete_speech_lm/logs/w3b_train_s1_3k.log`

Listen frames (8 demo): `[29,66,88,78,98,70,29,36]` — stop fires; no 750-frame runaway.

N2 pipeline: decode in `.venv_qwen_tts` → score in `.venv`  
Logs: `logs/w3b_n2_decode_3k.log`, `logs/w3b_n2_score_3k.log`

### Also observed (non-canonical)

| Ckpt | FR CER | Notes |
|------|-------:|-------|
| first stop run @3000 | 0.0641 | matched this retrain |
| same run continued @5000 | 0.107 | TF overfit; overwritten ckpts — discard |
| Wave3 last.pt (no stop) | 0.451 | runaway |

## Wav paths (Windows / Cursor)

Listen demo:

- `C:\projects\tts-self\out\d1_discrete_speech_lm\s1_listen_demo\0000_ruslan_000418_RUSLAN.wav`
- … through `0007_*.wav` (8 demos; ASCII names)

Full N2 suite (32 utts / condition):

- `C:\projects\tts-self\out\d1_discrete_speech_lm\s1_n2_wavs_stop_3k\model\`
- `...\copy_synth\`, `wrong_utt\`, `random_codes\`, `shuffle_text\`, `mask_text\`

Index: `.cursor/results/d1_discrete_speech_lm/listen_index.json`

## Ladder status

**Blocked.** Do **not** start 1h / 5h / full until FR CER ≤ copy_synth + 0.03 on S1 oracle.

Remaining gap is no longer stop/runaway; next focused work (still S1-only) would be FR-aware selection / residual intelligibility — not data scale.

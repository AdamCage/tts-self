# Wave 3 — Oracle ladder / N2 hard-stop

> **Superseded for stop/runaway:** see [`WAVE3b_REPORT.md`](WAVE3b_REPORT.md)  
> (FR CER 0.451 → **0.064**; runaway fixed; N2 still FAIL on CER; ladder still blocked.)

**Plan:** `.cursor/plans/d1-discrete-speech-lm_860c29d5.plan.md`  
**Todo:** `pass-oracle-n2`  
**Date:** 2026-08-02  
**WSL / authoring:** code via `PYTHONPATH=/mnt/c/projects/tts-self/src`; train artifacts under `~/projects/tts-self/runs/`  
**git_sha:** `dce4f456` (dirty; `MINISPEECH_ALLOW_DIRTY=1`)  
**Methodology:** read-only (not edited)

## Verdict

**N2 FAIL** — do **not** start 1h / 5h / full ladder (`train-current-ladder` blocked).

Artifact: `out/d1_discrete_speech_lm/n2_gate.json`  
Canonical checkpoint scored: `~/projects/tts-self/runs/discrete_speech_lm_300m_s1/last.pt` (step 3000).

| Criterion | Result | Value |
|-----------|--------|------:|
| Free-running CER ≤ copy_synth + 0.03 | **FAIL** | **0.451** vs ceiling **0.0528** (subset copy CER **0.0228**) |
| Bootstrap / CI vs wrong-utt | **PASS** | paired Δ −73.1 pp, CI95 entirely &lt; 0 |
| Bootstrap / CI vs random codes | **PASS** | model CI upper &lt; random CI lower (soft/hard) |
| No runaway / repetition collapse | **FAIL** | `cb0_max_run=140`, mean dur ratio ≈ 2.7, always hits `max_frames` |
| CB0 entropy / utilization healthy | **FAIL** | util 0.36 / H≈4.7 but max_run≥64 → unhealthy |
| Text-usage shuffle hurts FR CER | **PASS** | 1.457 &gt; 0.451 + 0.02 |
| Text-usage mask hurts FR CER | **PASS** | 1.740 &gt; 0.451 + 0.02 |
| Prefix counterfactual changes hyp | **PASS** | content differs on enough utts |

Teacher-forced CE (diagnostic only, 32 utts): loss≈0.33 / ce_cb0≈0.13 / ce_residual≈0.20  
(`out/d1_discrete_speech_lm/n2_eval_last/eval_summary.json` and train final loss≈0.009 on last step).

## Unit gates (pre-GPU)

```text
pytest tests/test_discrete_speech_lm.py tests/test_codec_tokens_pack.py
18 passed in 3.51s
```

Log: `.cursor/results/d1_discrete_speech_lm/logs/w3_unit_gates.log`  
Covers: AR shift `c_t→c_{t+1}`, causality, padding invariance, pack train/infer equality, synthetic overfit, shapes.

## Oracle ladder

| Stage | Result | Notes |
|-------|--------|-------|
| Synthetic (CPU tiny) | PASS | unit `test_tiny_overfit_synthetic_codes` |
| 1-utt smoke | PASS (TF) | 200 steps, loss 15.4→3.9e-4; decode needs `.venv_qwen_tts` |
| 32-utt S1 | TRAIN OK / N2 FAIL | 3000 steps, batch=2, grad_accum=4, bf16 |

### Critical bugs found and fixed in this todo

1. **Text tokenizer vocab mismatch (Wave-1 vs model config)**  
   Caches use `Qwen3.5-0.8B-Base` (`len(tok)=248077`, `pad=248044`, text_ids up to ~212k).  
   Config had `vocab_size=151936` / `pad=0` → OOB embeddings.  
   **Fix:** `configs/discrete_speech_lm_300m.yaml` → `vocab_size: 248077`, `pad_token_id: 248044`, text-init kept **off** (Qwen3-0.6B vocab ≠ Wave-1 ids).  
   Side effect: param count **442.4M** (above 400M band) due to larger text embedding.

2. **Safetensors loader missing in train CLI**  
   Wave-1 writes `codec_tokens_*.safetensors` + `codec_tokens_index.json`; trainer expected `.pt`/`index.jsonl`.  
   **Fix:** `scripts/72_train_discrete_lm.py` multi-root safetensors reader.

3. **Collate → trainer key mismatch**  
   Wave-1 collate exposed `codes_target` / string `speaker_ids`; trainer needs `codes` + int `speaker_id`.  
   **Fix:** adapter in `codec_seq_protocol.resolve_collate_codec_seq` + aliases in `collate_codec_seq.py`.

4. **`best.pt` frozen at step 500**  
   `fr_checkpoint` with empty `val_loader` and no `fr_eval_fn` produced identical selection keys → first save never displaced.  
   **Fix (post-hoc):** `DiscreteLMTrainer._maybe_fr_select` probes train TF CE when val empty.  
   **N2 scoring** therefore uses **`last.pt`** (true overfit). `best.pt` eval kept as secondary (FR CER 0.388, still FAIL).

5. **Dual transformers pin for decode**  
   Main `.venv` = transformers 5.14 (VibeVoice); codec decode requires `.venv_qwen_tts` (4.57.3 + qwen-tts).  
   N2 pipeline: decode-only in qwen venv → ASR score in main venv (`--decode-only` / `--score-only` on `74_eval_discrete_lm.py`).

## Throughput / hardware

| Item | Value |
|------|------:|
| GPU | RTX 4080 SUPER 16 GB |
| Params | 442,391,552 |
| S1 wall time | **275.2 s** / 3000 steps |
| Approx opt-step rate | **~10.9 steps/s** (batch 2 × accum 4) |
| Peak host RSS (time) | ~2.8 GB (GPU mem separate; job fit in 16 GB) |
| Decode env | `.venv_qwen_tts` |

Logs: `.cursor/results/d1_discrete_speech_lm/logs/w3_s1_train.log`

## Free-running metrics (last.pt, n=32 S1 subset)

| Condition | mean CER | CI95 |
|-----------|---------:|------|
| model (FR) | **0.451** | [~0.15, ~0.69] |
| copy_synth (GT codes) | **0.0228** | [~0.008, ~0.040] |
| wrong_utt text | 1.919 | — |
| random_codes | 0.911 | — |
| shuffle_text | 1.457 | — |
| mask_text | 1.740 | — |

Copy-synth topline on this subset (**0.0228**) is *better* than D0 E2 (0.0678); ceiling for N2 = 0.0228+0.03 = **0.0528**. Model misses by a wide margin.

Localization signals:

- No EOS / stop: every FR utt emits exactly `max_frames` (165) → duration runaway vs GT.
- CB0 max run 140 → repetition collapse under free-running.
- Text-usage **does** hurt CER → conditioning path is not dead; failure is closer to **sampler / EOS / AR stability** than “ignored text”.

## Listen wavs

Prefer capped N2 wavs (not the early 750-frame runaway demos):

```text
out/d1_discrete_speech_lm/s1_listen_demo/last_*.wav          # model FR (last.pt)
out/d1_discrete_speech_lm/s1_listen_demo/copysynth_*.wav     # GT codes decode
out/d1_discrete_speech_lm/s1_n2_wavs_last/model/             # full 32
out/d1_discrete_speech_lm/s1_n2_wavs_last/copy_synth/
```

Index: `.cursor/results/d1_discrete_speech_lm/listen_index.json`

## Next-step recommendation

**Debug — do not scale.** Suggested order:

1. Wire a real **EOS / stop** for CB0 (or length head) and re-run 1-utt → 32-utt with stop; confirm duration ratio ≈ 1.
2. Add repetition / loop penalties or lower temperature + forced max≈GT for oracle; measure CB0 max_run.
3. Keep text-only base; **do not** enable RU channels or Qwen text-init until FR CER approaches copy+0.03.
4. Optionally shrink text vocab / rematerialize text_ids with a smaller matched tokenizer so param count returns to ~300M (current 442M is vocab-driven).
5. Only after N2 PASS: start `train-current-ladder` (1h→5h→full).

## Files touched (implementation)

- `configs/discrete_speech_lm_300m.yaml` — vocab/pad match Wave-1 Qwen3.5 tokenizer  
- `scripts/72_train_discrete_lm.py` — safetensors multi-root dataset  
- `scripts/73_infer_discrete_lm.py` — batch `--codec-tokens-dir` listen  
- `scripts/74_eval_discrete_lm.py` — FR ASR suite, `--decode-only` / `--score-only`, N2 gate  
- `src/minispeech/data/codec_seq_protocol.py` — collate adapter  
- `src/minispeech/data/collate_codec_seq.py` — `codes` / `speaker_id` aliases  
- `src/minispeech/train/discrete_lm_trainer.py` — best.pt selection when no val  

## Hard-stop

Per plan Wave 3 / Wave 4: **1h / 5h / full training is forbidden** until N2 PASS.

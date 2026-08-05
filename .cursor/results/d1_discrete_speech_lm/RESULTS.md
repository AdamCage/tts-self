# Results — `d1_discrete_speech_lm`

**Plan:** `.cursor/plans/d1-discrete-speech-lm_860c29d5.plan.md`  
**Slug:** `d1_discrete_speech_lm`  
**Date:** 2026-08-02  
**Authoring HEAD:** `dce4f45` (`dce4f456b0757fb63da4cf014188ff4e474bdbe9`)  
**Runtime note:** WSL code via `PYTHONPATH=/mnt/c/projects/tts-self/src`; `MINISPEECH_ALLOW_DIRTY=1` (methodology + D1 WIP uncommitted)  
**Status:** **HARD-STOP / N2 FAIL (near miss)** — D0 PASS, caches complete, Speech LM + oracle ladder measured; 1h/5h/full ladder and ablations **not run**; **D2 (500h+) = NO-GO**

Methodology was read (README, strategy, tokenizers, evaluation, roadmap, compute) and **not edited** during packaging.

Wave reports (detail): `WAVE0_REPORT.md`, `WAVE1_REPORT.md`, `WAVE3_REPORT.md`, `WAVE3b_REPORT.md`, `WAVE3c_REPORT.md`.

---

## 1. Executive verdict

| Decision | Result |
|----------|--------|
| **D0 (tokenizer GO/NO-GO)** | **PASS** |
| **Wave 1 caches** | **DONE** — 51234 utts / 72.9836 h |
| **Wave 2 model + trainer** | **LANDED** — measured **442.4M** params (vocab-inflated vs ~300M target) |
| **N2 (32-utt free-running)** | **FAIL** — FR CER **0.05335** vs ceiling **0.05284** (miss ≈ **0.00051**) |
| **Ladder 1h → 5h → full** | **BLOCKED** (not run) |
| **Ablations** | **BLOCKED** (not run) |
| **D2 GO/NO-GO (500h+)** | **NO-GO** until N2 PASS |

Canonical gates:

- `out/d1_discrete_speech_lm/d0_gate.json` (copy: `artifacts/d0_gate.json`)
- `out/d1_discrete_speech_lm/n2_gate.json` (copy: `artifacts/n2_gate.json`) — WAVE3c canonical

---

## 2. Task status matrix

| Plan todo | Status | Notes / artifacts |
|-----------|--------|-------------------|
| `sync-methodology` | **DONE** | Pre-Agent methodology sync (not re-edited here) |
| `land-foundations` | **DONE** | Frontend/eval/multicorpus/repro foundations landed earlier |
| `probe-tokenizer-baselines` | **PASS** | D0; `WAVE0_REPORT.md`; `d0_gate.json` |
| `build-discrete-caches` | **DONE** | 51234 / 72.98 h; `WAVE1_REPORT.md`; `encode_meta_train_full.json` |
| `implement-speech-lm` | **DONE** | Model/trainer/CLIs; unit tests; `configs/discrete_speech_lm_300m.yaml` |
| `pass-oracle-n2` | **FAIL (near miss)** | WAVE3 → 3b → 3c; runaway fixed; CER miss ~5e-4 |
| `train-current-ladder` | **CANCELLED / BLOCKED** | Plan forbids scale on N2 FAIL |
| `run-ablation-suite` | **CANCELLED / BLOCKED** | Requires base PASS |
| `package-demo-results` | **DONE** | This pack: `RESULTS.md` + `listen_index.json` + gate copies |

---

## 3. Key metrics (with artifact paths)

### 3.1 D0 — frozen Qwen3-TTS Tokenizer-12Hz

Source: `out/d1_discrete_speech_lm/d0_gate.json`, `WAVE0_REPORT.md`

| Criterion | Value | Result |
|-----------|------:|--------|
| Copy-synth CER (E2-compatible n=50) | **0.0678** (≤ 0.073) | **PASS** |
| Bit-identical re-encode | true | **PASS** |
| bank_entity DER vs VibeVoice continuous | **0.325** ≤ **0.446** | **PASS** |
| bank_entity EER (report) | 0.45 vs continuous 0.525 | — |
| Mean codebook utilization | 0.817 | — |

Geometry registered as `qwen3tts_12hz`: SR 24000, hop 1920 → **12.5 Hz**, 16×2048, Apache-2.0 `Qwen/Qwen3-TTS-Tokenizer-12Hz`.

### 3.2 External baselines (demo5)

Source: `WAVE0_REPORT.md` (+ per-system metric JSONs under `out/d1_discrete_speech_lm/`)

| System | demo5 mean CER | Wav dir |
|--------|---------------:|---------|
| Silero raw | 0.0305 | `out/d1_discrete_speech_lm/silero_raw/demo5/` |
| Silero + frontend | 0.0169 | `out/d1_discrete_speech_lm/silero_frontend/demo5/` |
| Track A mel S4 | 0.0935 | `out/d1_discrete_speech_lm/track_a/demo5/` |
| Qwen3-TTS 0.6B CustomVoice | 0.0169 | `out/d1_discrete_speech_lm/qwen3_tts_0.6b/demo5/` |
| CosyVoice3 | **blocked** (not installed) | — |
| Qwen3-TTS 1.7B | not run | — |

### 3.3 Wave 1 — discrete codec-token caches

Source: `WAVE1_REPORT.md`, `out/d1_discrete_speech_lm/encode_verify.json`, `encode_meta_*.json`

| Subset | n utts | hours | Meta |
|--------|------:|------:|------|
| `s1_32` | 32 | 0.0494 | `out/d1_discrete_speech_lm/encode_meta_s1_32.json` |
| `h1` | 691 | 0.9990 | `out/d1_discrete_speech_lm/encode_meta_h1.json` |
| **train_full** | **51234** | **72.9836** | `out/d1_discrete_speech_lm/encode_meta_train_full.json` |

| Integrity check | Result |
|-----------------|--------|
| train∩val / train∩test / val∩test | **0 / 0 / 0** |
| Speaker leaks | **none** |
| Codec hash (partial) | `7030d98a18ae5968` |
| Pack unit tests | 7 passed (`tests/test_codec_tokens_pack.py`) |

Corpus split (full train): CV17 **39199** / 57.9533 h + RUSLAN **12035** / 15.0303 h.

### 3.4 Wave 2 — own Speech LM (~300M target)

Measured from `out/d1_discrete_speech_lm/s1_run_manifest.json` / `artifacts/s1_run_manifest.json`:

| Item | Value |
|------|------:|
| total params | **442,391,552** |
| trainable | 442,391,552 |
| text_blocks_and_emb | 373,665,792 |
| speech_heads_adapters | 68,725,760 |
| text_init | disabled (vocab mismatch Qwen3-0.6B vs Wave-1 Qwen3.5 ids) |
| frozen | Qwen3-TTS tokenizer (external) |

Config target band was 250–400M / ~300M; inflation is from `vocab_size: 248077` to match Wave-1 text_ids (`configs/discrete_speech_lm_300m.yaml`). Later stop-head retrain ≈ **442.7M** (`WAVE3b_REPORT.md`).

Code landings (representative):

- `src/minispeech/models/discrete_speech_lm.py`, `multi_codebook_head.py`, `russian_conditioning.py`, `text_init.py`
- `src/minispeech/train/discrete_lm_trainer.py`
- `src/minispeech/data/codec_tokens.py`, `collate_codec_seq.py`
- `src/minispeech/codec/qwen3_tts_wrapper.py`
- `scripts/70_*.py` … `74_eval_discrete_lm.py`
- `configs/discrete_speech_lm_300m.yaml`

Unit gates (Wave 3): `pytest tests/test_discrete_speech_lm.py tests/test_codec_tokens_pack.py` → **18 passed** (`logs/w3_unit_gates.log`).

### 3.5 N2 oracle ladder — progression to near-miss

Canonical gate (WAVE3c): `out/d1_discrete_speech_lm/n2_gate.json`  
Eval dir: `out/d1_discrete_speech_lm/n2_eval_stop_3c/`  
Rows: `out/d1_discrete_speech_lm/n2_eval_stop_3c/n2_model_rows.json`  
Ckpt: `~/projects/tts-self/runs/discrete_speech_lm_300m_s1_stop_3k/last.pt` @ 3000  
(`ckpt_role`: last.pt@3000_stop + WAVE3c `stop_overrun_bias=0`)

| Wave | FR CER | Ceiling (copy+0.03) | Runaway / CB0 | Mean dur ratio | Report |
|------|-------:|--------------------:|---------------|---------------:|--------|
| WAVE3 | **0.45137** | 0.05284 | FAIL (`max_run=140`) | ~2.71 | `WAVE3_REPORT.md` |
| WAVE3b | **0.06415** | 0.05284 | PASS (`max_run=5`) | 0.911 | `WAVE3b_REPORT.md` |
| **WAVE3c (canonical)** | **0.05335** | **0.05284** | PASS (`max_run=6`) | **1.000** | `WAVE3c_REPORT.md` |

Subset copy-synth CER (n=32): **0.02284** → ceiling **0.05284** (`n2_gate.json`).

#### WAVE3c N2 criteria matrix

| Criterion | Value | Result |
|-----------|------:|--------|
| FR CER ≤ copy + 0.03 | 0.05335 vs 0.05284 | **FAIL** |
| Bootstrap sep vs wrong-utt | paired Δ −94.3 pp; CI95 entirely &lt; 0 | **PASS** |
| Bootstrap sep vs random codes | model CI upper &lt; random CI lower | **PASS** |
| No runaway / repetition | `cb0_max_run=6`, `runaway_or_repetition=false` | **PASS** |
| CB0 healthy | util 0.386 / H≈6.25 / unique 791 | **PASS** |
| Text-usage shuffle hurts | shuffle CER 1.395 | **PASS** |
| Text-usage mask hurts | mask CER 1.006 | **PASS** |
| Prefix counterfactual changes hyp | true | **PASS** |

Teacher-forced CE (diagnostic only): loss≈0.047 / ce_cb0≈0.025 / ce_residual≈0.022  
(`out/d1_discrete_speech_lm/n2_eval_stop_3c/eval_summary.json`).

#### Residual error mass (WAVE3c)

From `WAVE3c_REPORT.md` + `n2_model_rows.json`:

| Duration bucket | n | mean CER |
|-----------------|--:|---------:|
| OK [0.9, 1.1]×GT | 27 | 0.0138 |
| short &lt;0.9×GT | 2 | 0.4545 |
| long &gt;1.1×GT | 3 | 0.1421 |

Worst utterance (dominates residual gap):

- utt: `common_voice_ru_17_hf_mirror__ba1a4089af87b3ebc90db07d`
- CER **0.9091**; frames 37/44
- REF «Подать, что обыкновенно.» → HYP «Краснодар – прекрасное место.» (content hallucination, not truncation)
- wav: `out/d1_discrete_speech_lm/s1_n2_wavs_stop_3c/model/0022_common_voice_ru_17_hf_mirror__ba1a4089af87b3ebc9.wav`
- Zero that single utt → mean CER **≈0.0249** ≪ ceiling (report)

Miss margin: **0.05335 − 0.05284 ≈ 0.00051** CER (**almost**).

Rejected WAVE3c infer knobs (same ckpt): aggressive length prior → FR CER **0.341**; mild fpt=3.2 → **0.202**; zero overrun bias only → **0.05335** (best).

---

## 4. Formal GO / NO-GO for D2 (500h+)

### Verdict: **NO-GO**

Plan Wave-6 rule: GO to D2 only if D0 PASS **and** 32-utt N2 PASS **and** full-current N3 PASS **and** no sampler/EOS collapse **and** legally admissible 500h corpus.

| Prerequisite | Status |
|--------------|--------|
| D0 PASS | **yes** |
| N2 PASS (32-utt FR CER ≤ copy+0.03) | **no** (near miss 0.05335 &gt; 0.05284) |
| Full-current N3 | **not run** (blocked) |
| Sampler / EOS collapse | fixed at WAVE3b/3c (healthy) |
| Legal 500h corpus | **not cleared** (Golos/OpenSTT/YODAS still blocked) |

**Almost, but not GO.** The stack is not architecturally dead: text-usage and CB0 health pass; the residual is dominated by a single short CV hallucination plus a few lexical/ASR diffs at matched duration. Scaling hours will not clear an S1 oracle miss.

### Recommended next debug (still S1 — **not** more hours)

1. Per-utt triage of the CER-0.909 hallucination (`ba1a4089…`) — conditioning / speaker / short-utt packing, not data scale.
2. FR-aware checkpoint selection (do not chase teacher-forced CE; WAVE3b @5000 regressed to 0.107).
3. Targeted stop/tail calibration **without** raising length prior above GT for Ruslan.
4. Optional later: shrink text vocab / rematerialize text_ids so param count returns toward ~300M band.
5. Only after N2 PASS: unlock 1h → 5h → 72.98h ladder, then ablations, then revisit D2 legal/data gate.

---

## 5. Inference / listen paths (Windows)

Index: `.cursor/results/d1_discrete_speech_lm/listen_index.json`

### Primary (play these)

| Role | Folder |
|------|--------|
| S1 listen demo (ASCII) | `C:\projects\tts-self\out\d1_discrete_speech_lm\s1_listen_demo\` |
| N2 WAVE3c model suite (32) | `C:\projects\tts-self\out\d1_discrete_speech_lm\s1_n2_wavs_stop_3c\model\` |
| Hallucination worst utt | `...\s1_n2_wavs_stop_3c\model\0022_common_voice_ru_17_hf_mirror__ba1a4089af87b3ebc9.wav` |

### Related / baselines

| Role | Folder |
|------|--------|
| N2 WAVE3b (prior stop fix) | `...\s1_n2_wavs_stop_3k\model\` |
| Qwen tokenizer copy-synth E2 | `...\qwen3tts_12hz_copysynth\e2\` |
| Silero + frontend demo5 | `...\silero_frontend\demo5\` |
| Silero raw demo5 | `...\silero_raw\demo5\` |
| Track A demo5 | `...\track_a\demo5\` |
| Qwen3-TTS 0.6B demo5 | `...\qwen3_tts_0.6b\demo5\` |

---

## 6. Code / config changes (authoring tree; not committed in this pack)

Uncommitted D1 work lives in the Windows authoring tree (dirty relative to `dce4f45`). Representative paths listed in §3.4. **No commit** was created for this packaging todo.

Env reminder: dual transformers pin — encode/decode under `.venv_qwen_tts` (`transformers==4.57.3` + `qwen-tts`); main `.venv` keeps `transformers==5.14.1` for VibeVoice/Track A ASR scoring.

---

## 7. Logs

Under `.cursor/results/d1_discrete_speech_lm/logs/`:

| Wave | Examples |
|------|----------|
| D0 | `d0_qwen_probe.log`, `d0_continuous_anchor.log`, `d0_baselines_*.log` |
| W1 | `w1_encode_*.log`, `w1_verify.log` |
| W3 | `w3_unit_gates.log`, `w3_s1_train*.log`, `w3_n2_*.log` |
| W3b | `w3b_*.log` |
| W3c | `w3c_unit.log`, `w3c_listen*.log`, `w3c_n2_decode*.log`, `w3c_n2_score*.log` |

---

## 8. Artifacts copied into results dir

```text
.cursor/results/d1_discrete_speech_lm/artifacts/
  d0_gate.json
  n2_gate.json                 # WAVE3c canonical (FR CER 0.05335)
  continuous_codec_anchor.json
  encode_verify.json
  encode_meta_{s1_32,h1,train_full}.json
  s1_run_manifest.json
  s1_train_summary.json
  n2_eval_stop_3c_eval_summary.json
  n2_model_rows_stop_3c.json
```

Large audio and codec shards stay under `out/d1_discrete_speech_lm/` and `~/datasets/tts/.../codec_tokens*` (not duplicated here).

---

## 9. Methodology alignment

Read-only during implementation/packaging. D1 remains the research national-component discrete path; frozen Qwen3-TTS 12Hz is the only external acoustic component; own codec remains a later independence track. Promotion still requires free-running CER/text-usage — teacher-forced CE alone is insufficient (WAVE3 runaway lesson).

---

## 10. Follow-ups / open blockers

1. **Hard blocker for ladder & D2:** N2 CER FAIL by ≈0.0005 — fix S1 residual (hallucination + stop), re-score gate.
2. CosyVoice3 / Qwen 1.7B full external suite still open (not blocking N2).
3. Text-init still off until vocab alignment with Wave-1 text_ids.
4. RU conditioning channels deferred until text-only base PASS.
5. Legal/provenance gate for any 500h+ corpus remains open regardless of N2.
6. Management demo side-by-side is partially available via listen folders; full 10–20 phrase bank + DER/EER board deferred until N2 PASS.

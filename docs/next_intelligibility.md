# Next: intelligibility & Track A

Part 1 (**measure the truth**) is **done**. Full write-up: [part1_diagnosis.md](part1_diagnosis.md). For a paper-oriented synthesis of the same measured facts (proxy gates vs CER, Gate 0–6 rewrite), see [paper1_proxy_metrics_trap.md](paper1_proxy_metrics_trap.md).

Do **not** restart S4 / Flow / LoRA until Gate 0 (R²) passes and Track A has an intelligibility baseline.

Authoring: `C:\projects\tts-self` → commit/push → WSL `~/projects/tts-self` `git pull`.

```bash
cd ~/projects/tts-self && source .venv/bin/activate
export HF_HOME=$HOME/hf_cache
git pull && git rev-parse --short HEAD
uv pip install -e ".[dev]" --no-deps   # after src/ changes; add phonemizer/espeak for G2P
```

## Part 1 results (summary)

| ID | Result |
|----|--------|
| Gate 0 / E3 | **FAIL** — `r2_overall≈−0.11`, `tf_l1≈0.798` ≈ constant MAE, `var_ratio≈0.319`, `hf_power_ratio≈0.311` |
| E0 | All 4 trivial baselines pass legacy FR duration/stop with oracle length |
| E5 | `mean_diagonality≈0.092` vs chance≈0.077 |
| E2 | GT latent decode CER≈0.073 (n=50) |
| E1 | Noise flat to σ≈0.2; rise at 0.3; steep after 0.6. Collapse cliff α=0.7→0.5. Temporal smooth kills CER |
| E4 | Synth CER S3≈0.889, S4≈0.860 (NEAR_SILENT on S4 demos) |
| E6 | Verbalize rescore: **0 h** recovered from exclude; usable ≈**17.8 h** |

Artifacts: `out/latent_error_calibration.json`, `out/intelligibility_S3.json`, `out/trivial_baselines.json`, `out/head_diagnostics_*.json`, `out/align_audit.json`, `out/qc_revision.json`.

## Track A (weeks 1–2) — frontend + forced align + mel baseline

In-repo scaffolding:

| Piece | Location |
|-------|----------|
| RU TN | `frontend/normalize.py`, `frontend/verbalize.py` |
| G2P | `frontend/g2p.py` (phonemizer+espeak-ng `ru` if present; grapheme fallback) |
| Forced pack + CTC align | `data/forced_align.py`, `scripts/26_build_forced_align.py` → `forced.json` (uniform phones or real char-CTC) |
| A6 target registry | `constants.py` — `TARGETS` / `get_target`; `vibevoice64` + `mel80` (hop=256 @ 22.05 kHz ≈86.13 Hz) |
| A3 phoneme→frame | `data/frame_projection.py`, `frontend/durations.py` (mean stub from `forced.json`) |
| A4 non-AR mel | `models/mel_baseline.py`, `data/mel.py`, `data/mel_dataset.py`, `configs/mel_baseline_v01.yaml`, `scripts/27_train_mel.py` / `28_infer_mel.py` / `29_cache_mels.py` |
| A5 CFM mel | `models/mel_cfm.py`, `train/mel_cfm_trainer.py`, `configs/mel_cfm_v01.yaml`, `scripts/31_train_mel_cfm.py`; infer via `28_infer_mel.py` (`--flow-steps`) |
| A7 trainer fixes | LR warmup+cosine, bf16 autocast, `sample_weight`, token-duration loss hook, CER selection interface, lazy shards, `git_sha` walk-up — in `train/trainer.py` + `train/mel_trainer.py` |

```bash
# Unit tests (Windows or WSL, no GPU / no espeak required)
pytest -q tests/test_frontend_track_a.py tests/test_track_a_projection.py \
  tests/test_mel_baseline.py tests/test_recovery_metrics.py

# Dry-run packing only (no ASR model download)
python scripts/26_build_forced_align.py \
  --config configs/data_ruslan_local.yaml --method uniform --max-utts 32 \
  --out ~/datasets/tts/ruslan/align/forced.json

# Optional MFA status only (no MFA install required)
python scripts/26_build_forced_align.py --method mfa --config configs/data_ruslan_local.yaml

# Full CTC forced align — WSL only; prefer local --model; check HF_HOME disk
python scripts/26_build_forced_align.py \
  --config configs/data_ruslan_local.yaml --method ctc \
  --model ~/models/ru-ctc \
  --out ~/datasets/tts/ruslan/align/forced.json

# Then rebuild priors that consume forced.json
python scripts/06_align.py --config configs/data_ruslan_local.yaml --level A
```

### Mel S1 (WSL, once data ready)

Requires: manifest + wavs, mel cache, and `forced.json` (uniform OK for S1 smoke; CTC preferred for quality).
No VibeVoice. No Hub vocoder download — place local HiFi-GAN/UnivNet under `~/models/…` or use griffin-lim / save-mel.

```bash
cd ~/projects/tts-self && source .venv/bin/activate
export HF_HOME=$HOME/hf_cache
git pull && git rev-parse --short HEAD
uv pip install -e ".[dev]" --no-deps

# 1) Forced durations (uniform S1 smoke, or CTC when local model exists)
python scripts/26_build_forced_align.py \
  --config configs/data_ruslan_local.yaml --method uniform --max-utts 32 \
  --out ~/datasets/tts/ruslan/align/forced.json

# 2) Cache mel80 from wavs (S1 subset)
python scripts/29_cache_mels.py \
  --config configs/data_ruslan_local.yaml --split train --max-utts 32

# 3) Train mel baseline S1
python scripts/27_train_mel.py \
  --config configs/mel_baseline_v01.yaml \
  --data configs/data_ruslan_local.yaml \
  --stage S1

# 4) Infer (neural vocoder if VOCODER_CKPT / ~/models/hifigan/generator.pt exists;
#    else griffin-lim; or --save-mel-only)
python scripts/28_infer_mel.py \
  --ckpt runs/mel_baseline_v01/best.pt \
  --text "приве́т мир" \
  --out out/mel_s1_demo
```

**Status:** Mel **S4 full clean** (≈15.4 h, expand_neighbor) free val100 CER≈0.048 / demo≈0.048 — **production = S4** (`runs/mel_baseline_v01_s4/best.pt`). S3 remains Gate 3 reference (val100≈0.057, demo≈0.041). **CFM parked** after S2 A/B loss (demo CER≈0.699 ≫ regression) — do **not** train CFM further. **HiFi-GAN neural vocoder** available at `~/models/hifigan/generator.pt` (listening path `out/mel_s4_demo_wavs_neural/`; demo CER≈0.042 vs griffin-lim≈0.048). Live suite (neural): demo5 CER≈0.099 / homograph≈0.034; **bank_entity** CER≈0.458→**0.093** after TN/ASR-score fix (`out/eval_mel_suite_s4_bankfix.json`). Eval suite entrypoint: `scripts/32_eval_mel_suite.py`. Do **not** restart latent Talker S4/Flow/LoRA.

**Latent Talker note:** `acoustic_dropout` stays in `talker_v01_direct.yaml` for S3 resume. Mel baseline never uses it. Remove from latent configs only after forced-align + explicit durations replace that crutch.

### A3 / A6 quick API

```python
from minispeech.constants import get_target, MEL80, VIBEVOICE64
from minispeech.data.frame_projection import (
    load_forced_segments, soft_overlap_projection, hard_overlap_projection,
)
from minispeech.frontend.durations import MeanDurationPredictor

fr_vv = get_target("vibevoice64").frame_rate   # 7.5
fr_mel = MEL80.frame_rate                      # 22050/256 ≈ 86.13
segs = load_forced_segments("forced.json", "utt_id")
W_soft = soft_overlap_projection(segs, frame_rate="mel80")   # [P, T]
W_hard = hard_overlap_projection(segs, frame_rate=7.5)
pred = MeanDurationPredictor.from_forced_json("forced.json")
durs_s = pred.predict_seconds(["a", "b"])  # seconds; frames via predict_frames(..., frame_rate=...)
```

**WSL required for:** CTC forced-align over the corpus (audio + optional HF model), mel cache from wavs, mel S1 train/infer, any GT-mel vocoder topline. Uniform packing + TN/G2P / A3–A7 unit tests run on Windows. MFA is optional and not required.

## Part 1 commands (reference; do not re-run heavy ASR casually)

```bash
pytest -q tests/test_recovery_metrics.py
python scripts/20_trivial_baselines.py --data configs/data_ruslan_local.yaml --split val --n 32 --out out/trivial_baselines.json
python scripts/21_latent_error_calibration.py --data configs/data_ruslan_local.yaml --asr-config configs/asr.yaml --n 50 --out out/latent_error_calibration.json
python scripts/22_head_diagnostics.py --ckpt runs/v01_regression_direct/best.pt --data configs/data_ruslan_local.yaml --split val --n 64
python scripts/23_eval_intelligibility.py --infer configs/infer.yaml --asr-config configs/asr.yaml --out out/eval_intelligibility.json
python scripts/24_align_audit.py --ckpt runs/v01_regression_direct/best.pt --data configs/data_ruslan_local.yaml --n 16 --out out/align_audit.json
python scripts/25_qc_revision.py --from-qc --config configs/data_ruslan_local.yaml --out out/qc_revision.json
```

## NEAR_SILENT / CTC QC notes

- `scripts/23_eval_intelligibility.py` flags `peak < 1e-3` or `rms < 1e-4` — usually wrong denorm stats or collapsed latents.
- Full QC revision still wants deterministic RU CTC ASR; Whisper-only rescore did not return exclude hours (E6).


## Mel S1 smoke follow-up (2026-07-30)

- **Ckpt:** `runs/mel_baseline_v01/best.pt` (uniform forced align; phone vocab size **17** — weak / near word-level).
- **Vocoder:** no local HiFi-GAN under `~/models` / `VOCODER_CKPT` → **griffin-lim** via `scripts/28_infer_mel.py`.
- **Bugfix (synced WSL→Windows):** `mel_to_wav` short-circuited to save-mel whenever `save_mel_path` was set without a neural ckpt; now only when `prefer=="save_mel"`. File: `src/minispeech/infer/vocoder.py`. The five mel train/cache files were already identical on `/mnt/c`.
- **Wavs:** `out/mel_s1_demo_wavs/*.wav` (also mirrored under Windows `out/`).
- **ASR (`out/intelligibility_mel_s1.json`):** mean **CER≈0.836**, **WER=1.0**. No NEAR_SILENT (peaks ok); hyps are **garbage** Whisper fillers (`продолжение следует`, `субтитры сделал dimatorzok`) — unintelligible S1 mel, not silent PCM.
- Do **not** scale mel ladder / restart S4/Flow until Track A phone inventory + align improve and CER moves.

## Mel S1b — phone-level uniform forced (2026-07-30)

**Root cause of S1 vocab=17:** `scripts/26_build_forced_align.py --method uniform` packed **word** spans; `segments_from_forced_entry` fell through to `word_spans_seconds` + `words`, so `PhoneVocab` learned whole words.

**Fix (Windows → rsync WSL):** uniform packing is now **TN → stress → G2P → equal-time phone spans** (`phones` / `phone_labels` / `granularity=phone`). Word spans kept for Talker level-A. Grapheme fallback lowercases and drops punct when espeak is missing.

| Run | Forced | Vocab | Val mel_l1 (best) | CER | WER |
|-----|--------|------:|------------------:|----:|----:|
| S1 | word (bug) | **17** | — | **0.836** | 1.0 |
| S1b | phone uniform (grapheme) | **45** | **1.987** | **1.797** | 4.47 |
| S1c | **char CTC** (wav2vec2-ru) | **38** | **1.969** | **0.860** | 1.0 |

- **Ckpt / run:** `runs/mel_baseline_v01_s1b/` (`best.pt`, `phone_vocab.json`, `best_metrics.json`).
- **Forced:** `~/datasets/tts/ruslan/align/forced.json` (32 train + 32 val; mean ≈47 phones/utt; 42 unique labels in forced; vocab 45 with specials).
- **G2P (S1b):** `grapheme_fallback` at the time — espeak later installed user-local (see Env status).
- **Vocoder:** griffin-lim (no local HiFi-GAN).
- **Wavs / ASR:** `out/mel_s1b_demo_wavs/`, `out/intelligibility_mel_s1b.json` (mirrored to Windows `out/`).
- **Verdict:** granularity fix is correct (17 → 45), but CER did **not** improve — uniform grapheme timing + griffin-lim on 32 utts is still unintelligible.

## Mel S1c — real CTC forced align (2026-07-30)

**Wired:** `forced_align.py` + `scripts/26_build_forced_align.py --method ctc` load local `Wav2Vec2ForCTC`, emit log-probs, run `torchaudio.functional.forced_align` + `merge_tokens`, write char spans (`granularity=char`, `method=ctc`). Failures are skipped (no silent `uniform_pending_ctc`). Infer matches train via `PhoneVocab.unit=char` + `text_to_ctc_char_labels` (space→`|`).

| Piece | Result |
|-------|--------|
| CTC model | `~/models/wav2vec2-large-xlsr-53-russian` (char CTC, model vocab **39**, blank=`pad`=0, space=`\|`) |
| Align success | **64/64** (32 train + 32 val), success_rate **1.0** |
| Forced labels | mean ≈**53.9** chars/utt; **35** unique labels in forced; mel vocab **38** (specials) |
| Sample | `с \| т р е в о ж н ы м \| …` with non-uniform spans (e.g. 20–60 ms) |
| Ckpt / run | `runs/mel_baseline_v01_s1c/` |
| Val mel_l1 | **1.969** (best) |
| CER / WER | **0.860** / **1.0** (`out/intelligibility_mel_s1c.json`) |
| Vocoder | griffin-lim |
| Demos | `out/mel_s1c_demo_wavs/` (mirrored to Windows) |

**API quirks:** torchaudio **2.11** — `forced_align(log_probs[B,T,C], targets[B,L], blank=0) → (alignments, scores)`; prefer `merge_tokens` for repeated chars. Emission rate ≈50 Hz (`seconds_per_frame ≈ dur/T`). Targets must be `int32` and must not contain blank. Host RAM/swap thrash mid-train near step ~2300; resumed from `S1_last.pt` (step 2000) to finish 3000.

**Verdict (pre-diagnosis):** CTC spans exist and val mel_l1 inches past S1b, but free demo CER stays ~S1 (~0.86).

### Mel S1c diagnosis — GT mel vs oracle-dur (2026-07-30)

Script: `scripts/30_mel_copy_synth_eval.py` (WSL; mirrored to Windows). Artifacts: `out/intelligibility_gt_mel_griffin.json`, `out/intelligibility_mel_s1c_oracle_dur.json`.

| Condition | n | mean CER | mean WER | notes |
|-----------|--:|---------:|---------:|-------|
| **A) GT cached mel → griffin-lim → Whisper** | 20 train | **0.027** | 0.076 | mel-path vocoder topline |
| **B) S1c + oracle forced durations → griffin-lim** | 5 train | **0.611** | 0.844 | same gap-truncated duration path as train |
| **C) S1c free synth (demo phrases)** | 5 | **0.860** | 1.0 | `out/intelligibility_mel_s1c.json` |

**Interpretation**

1. **Griffin-lim is not the bottleneck.** GT mel invert CER≈0.027 ≪ 0.3 — neural vocoder is nice for listening, not required to judge Talker/mel model.
2. **Acoustic / align packing bug dominates.** CTC `phones[].start/end` leave large blank gaps (~60% of utt time). `MelSpeechDataset` uses `end−start` only → `sum(durs) ≈ 0.38×` mel length, then **crops** `mel[:n_frames]`. Hard upsample packs phones into the *prefix* of the spectrogram → systematic misalignment. Oracle-dur hyps are truncated prefixes (e.g. «с тревожным чувством» for a longer sentence); `mean_cover_ratio_vs_end≈0.384`.
3. Free CER (0.86) > oracle (0.61): duration head compounds a broken teacher; scaling to S2 on the same packing will not fix intelligibility.

**Recommendation (priority order)**

1. **Fix CTC gap → frame coverage** before any S2 scale: fill blanks / merge gaps into neighboring phones or `_` silence tokens; or train with **soft** `proj_W` over full `utt_end` (no mel crop to gappy sum). Re-train S1c after cover_ratio≈1.
2. Re-run A/B after the fix; expect oracle CER to drop toward GT topline if the decoder can fit.
3. Neural vocoder optional for demos; do **not** restart S4/Flow/LoRA; do **not** scale full corpus yet.

Still do **not** restart S4/Flow/LoRA.

## Mel S1d — CTC gap-fill packing (2026-07-30)

**Fix (load path, no CTC re-run):** `fill_segment_gaps` / `phones_and_frame_durs_from_forced` in `frame_projection.py` insert explicit `sil` over `[0, utt_end]` for CTC blank gaps; `MelSpeechDataset` snaps `sum(durs)` to mel `T`. Unit test: `test_ctc_gap_fill_cover_ratio_full_timeline`.

| Metric | S1c (gappy) | S1d (sil fill) |
|--------|------------:|---------------:|
| cover_ratio mean (64 utts) | **0.318** | **1.000** |
| mean `#sil` / utt | 0 | **44.0** |
| Vocab | 38 | **39** (+`sil`) |
| Val mel_l1 (best) | 1.969 | **1.939** |
| Free demo CER | 0.860 | **0.874** |
| Oracle-dur CER (n=5 train) | **0.611** | **0.052** |
| Oracle cover_ratio | 0.384 | **1.000** |

- **Ckpt / run:** `runs/mel_baseline_v01_s1d/`
- **Artifacts:** `out/cover_ratio_s1d.json`, `out/intelligibility_mel_s1d.json`, `out/intelligibility_mel_s1d_oracle_dur.json`, `out/mel_s1d_demo_wavs/`, `out/mel_s1d_oracle_dur_wavs/` (mirrored to Windows `out/`).
- **Verdict:** Packing bug confirmed fixed — oracle CER **0.61 → 0.052** (near GT mel griffin 0.027). Free CER stays ~0.87 because free infer has no oracle/`sil` timeline (duration head + char-only sequence). Next: improve free duration (insert/`sil` policy or expand-neighbor at infer) before S2 scale.

Still do **not** restart S4/Flow/LoRA; do **not** scale S2 until free CER moves with a duration policy.

## Mel S1d free-infer sil/duration fix (2026-07-30)

**No retrain (S1e not needed):** vocab + dur head unchanged; infer-only packing on `runs/mel_baseline_v01_s1d/best.pt`.

**Root cause (ablation):** content labels already match train; **sil mask placement** dominates. Oracle sil mask + mean durs → CER≈0.08; naive char-only / wrong sil → CER≈0.9.

**Implement (default free path):**
- `frontend/durations.py`: `SilGapModel` (bigram P(sil|a→b) + unigram backoff), `pack_infer_phones_and_durs`, sil-aware snap, optional **forced content lookup**
- `scripts/28_infer_mel.py`: `--duration-mode {mean_stats,oracle_forced,uniform,pred}`, `--sil-policy {bigram,interleave,word,none}`, `--forced-lookup` / `--no-forced-lookup` (default **on**)
- Unit test: `test_free_infer_sil_pack_full_timeline`

| Condition | mean CER | notes |
|-----------|---------:|-------|
| Before free demo (char-only + dur_head) | **0.874** | `out/intelligibility_mel_s1d.json` |
| Oracle-dur (script 30, n=5 train) | **0.052** | regression unchanged; cover=1.0 |
| Free train + **forced_lookup** (content match → sil timeline) | **0.008** | = oracle path; no `--utt-id` needed |
| Free train + bigram only (`--no-forced-lookup`) | **0.808** | ~82% sil-mask agree; still far from oracle |
| Free OOD demos + bigram | **1.031** | S1d 32-utt overfit; demos not in forced |

**Artifacts:** `out/intelligibility_mel_s1d_free_fixed.json` (summary + nested), `out/intelligibility_mel_s1d_free_fixed_bigram.json`, `out/intelligibility_mel_s1d_free_fixed_demo.json`, `out/mel_s1d_free_fixed_wavs/`, `out/mel_s1d_free_fixed_train_wavs/` (mirrored to Windows).

**Verdict:** Infer packing now emits train-consistent `sil`/`|`/char timelines. Corpus texts that match `forced.json` content recover oracle CER via lookup. **True text-only free** still fails because CTC gap positions are not predictable from text at S1 scale; OOD demos need data scale / a learned duration+sil model (future S1e+), not more griffin tweaks.

```bash
# Free infer (default: mean_stats + bigram + forced_lookup)
python scripts/28_infer_mel.py \
  --ckpt runs/mel_baseline_v01_s1d/best.pt --demo \
  --duration-mode mean_stats --data configs/data_ruslan_local.yaml \
  --out out/mel_s1d_free_fixed_wavs

# Honest text-only free (no forced match)
python scripts/28_infer_mel.py \
  --ckpt runs/mel_baseline_v01_s1d/best.pt --text "..." \
  --duration-mode mean_stats --no-forced-lookup --sil-policy bigram
```

Still do **not** restart S4/Flow/LoRA; do **not** scale S2 until true free (no forced lookup) CER moves or a learned sil/duration head is trained.

## Mel S1e — expand-neighbor (no sil) retrain (2026-07-30)

**Goal:** train without CTC `sil` tokens so free infer is content labels + mean/pred durs only.

**Fix:** `fill_segment_gaps(..., mode="expand_neighbor")` absorbs lead/trail/inter gaps into neighboring phone/char boundaries (mid-gap split). `MelSpeechDataset` / vocab default `gap_mode=expand_neighbor`; `sil_fill` kept for S1d ablation. Free infer defaults: `--sil-policy none --no-forced-lookup`.

| Metric | S1d (sil_fill) | S1e (expand_neighbor) |
|--------|---------------:|----------------------:|
| cover_ratio mean (64) | **1.000** | **1.000** |
| mean `#sil` / utt | 44.0 | **0** |
| Vocab | 39 (+`sil`) | **38** (no `sil`) |
| Val mel_l1 (best) | 1.939 | **2.069** (ckpt @ step 2000) |
| Oracle-dur CER (n=5) | 0.052 | **0.031** |
| Free train text-only CER (n=5, no lookup) | ~0.81 (bigram) | **0.211** |
| Free OOD demo CER (no lookup) | ~1.03 (bigram) / 0.87 (char-only) | **0.690** |

- **Ckpt / run:** `runs/mel_baseline_v01_s1e/` (train stalled ~step 2250 twice — likely I/O/hang in trainer loop; eval used `best.pt` @ step 2000).
- **Artifacts:** `out/cover_ratio_s1e.json`, `out/intelligibility_mel_s1e.json`, `out/intelligibility_mel_s1e_oracle_dur.json`, `out/intelligibility_mel_s1e_train_free.json`, `out/mel_s1e_demo_wavs/`, `out/mel_s1e_oracle_dur_wavs/` (mirrored to Windows `out/`).
- **Verdict:** Sil-mask bottleneck removed for **in-domain** free: train free CER **0.81 → 0.21**. Oracle still excellent (0.031). OOD demos remain **0.69** (≫0.2) — residual is **mean-duration mismatch + S1 capacity/OOD**, not sil placement. Next: learned duration head (or scale past 32 utts) before S2; optional finish/restart S1e to 3000 if hang is fixed.

```bash
# S1e train (expand_neighbor, no sil in vocab)
python scripts/27_train_mel.py \
  --config configs/mel_baseline_v01.yaml \
  --data configs/data_ruslan_local.yaml --stage S1 \
  --gap-mode expand_neighbor --run-dir runs/mel_baseline_v01_s1e

# Free text-only demos (default: no sil, no forced-lookup)
python scripts/28_infer_mel.py \
  --ckpt runs/mel_baseline_v01_s1e/best.pt --demo \
  --duration-mode mean_stats --gap-mode expand_neighbor \
  --sil-policy none --no-forced-lookup \
  --data configs/data_ruslan_local.yaml --out out/mel_s1e_demo_wavs

# Oracle-dur regression
python scripts/30_mel_copy_synth_eval.py --skip-gt \
  --ckpt runs/mel_baseline_v01_s1e/best.pt --gap-mode expand_neighbor \
  --n-oracle 5 --out-oracle out/intelligibility_mel_s1e_oracle_dur.json
```

Still do **not** restart S4/Flow/LoRA.

## Mel S2 — 1h scale, expand_neighbor (2026-07-30)

**Goal:** scale past 32-utt S1e after packing/sil fixed; more data for OOD free CER.

| Piece | Result |
|-------|--------|
| Stage | `max_hours: 1.0`, 12000 steps (`configs/mel_baseline_v01.yaml`) |
| Align | CTC merge → **1936** utts in `forced.json` (all `method=ctc`) |
| S2 train cut | **1236** utts / **0.999 h** after `exclude_tags`; forced+mel cover **1236/1236** |
| Val | 32 utts forced+mel (loader cap) |
| Mel cache | index **1478** (train buffer 1414 + val 64; resume skips prior 64) |
| Gap mode | `expand_neighbor` (no `sil`); vocab **39** (char CTC + specials) |
| Ckpt / run | `runs/mel_baseline_v01_s2/best.pt` (best @ step **6500**) |
| Val mel_l1 | **1.381** |
| Vocoder | griffin-lim |
| RAM | ~2.5 GiB peak during train (lazy shards; no S1c-style thrash) |

| Metric | S1e (32 utt) | S2 (≈1 h) |
|--------|-------------:|----------:|
| cover_ratio | 1.000 | 1.000 |
| Val mel_l1 (best) | 2.069 | **1.381** |
| Oracle-dur CER (n=5 train) | **0.031** | 0.074 |
| Free train text-only CER (n=5, no lookup) | 0.211 | **0.021** |
| Free OOD demo CER (no lookup) | 0.690 | **0.079** |
| Free OOD demo WER | — | **0.086** |

- **Artifacts:** `out/intelligibility_mel_s2.json` (+ `_demo` / `_oracle_dur` / `_train_free`), `out/mel_s2_demo_wavs/`, `out/mel_s2_oracle_dur_wavs/`, `out/mel_s2_train_free_wavs/`, `runs/mel_baseline_v01_s2/{align,mel}_coverage.json` (mirrored to Windows `out/` + metrics).
- **Verdict:** Data scale fixed the S1e OOD free gap (**0.69 → 0.079**). Train-free near-oracle (**0.021**). Oracle CER slightly worse than S1e (0.074 vs 0.031) — small-n ASR noise and/or less overfit; cover_ratio still 1.0. Next: optional neural vocoder for listening; learned duration head; then S3 (5 h) only if Gate-style CER stays ≪0.2 on a larger held-out set. Still **do not** restart S4/Flow/LoRA/VibeVoice train.

```bash
# S2 train (after CTC forced + mel cache for ~1h)
python scripts/27_train_mel.py \
  --config configs/mel_baseline_v01.yaml \
  --data configs/data_ruslan_local.yaml --stage S2 \
  --gap-mode expand_neighbor --run-dir runs/mel_baseline_v01_s2

# Free text-only demos
python scripts/28_infer_mel.py \
  --ckpt runs/mel_baseline_v01_s2/best.pt --demo \
  --duration-mode mean_stats --gap-mode expand_neighbor \
  --sil-policy none --no-forced-lookup \
  --data configs/data_ruslan_local.yaml --out out/mel_s2_demo_wavs
```

## Mel S3 — 5h scale, expand_neighbor (2026-07-30)

**Goal:** Talker-ladder S3 (`max_hours: 5.0`, 20000 steps) after S2 free CER≪0.2; warm-start from S2 best.

| Piece | Result |
|-------|--------|
| Stage | `max_hours: 5.0`, 20000 steps (`configs/mel_baseline_v01.yaml`; matches Talker S3) |
| Align | CTC merge `--skip-existing` → **6611** utts in `forced.json` (was 6431; +180 val for Gate 3) |
| S3 train cut | **5645** utts / **5.001 h** after `exclude_tags`; forced+mel cover **5645/5645** |
| Val | loader still caps 32; Gate 3 scored **100** contiguous forced+mel (279 available) |
| Mel cache | index **6445** (val extended for Gate 3) |
| Gap mode | `expand_neighbor` (no `sil`); vocab **39** |
| Init | `--init-ckpt runs/mel_baseline_v01_s2/best.pt` (54/54 weights) |
| Ckpt / run | `runs/mel_baseline_v01_s3/best.pt` (best @ step **7000**) |
| Val mel_l1 | **1.128** |
| Vocoder | griffin-lim |
| RAM | ~3 GiB host / ~1.8 GiB VRAM peak (lazy shards) |

| Metric | S1e (32) | S2 (≈1 h) | S3 (≈5 h) |
|--------|---------:|----------:|----------:|
| Val mel_l1 (best) | 2.069 | 1.381 | **1.128** |
| Oracle-dur CER (n=5 train) | 0.031 | 0.074 | **0.027** |
| Free train text-only CER (n=5) | 0.211 | 0.021 | **0.020** |
| Free OOD demo CER (n=5) | 0.690 | 0.079 | **0.041** |
| Free OOD demo WER | — | 0.086 | **0.057** |
| Free val100 CER (no lookup) | — | — | **0.057** (med 0.041; WER≈0.153) |

**Gate status vs [experiment_protocol.md](experiment_protocol.md) intelligibility gates (Track A mel proxies):**

| Gate | Criterion | S3 reading |
|------|-----------|------------|
| **1** | Single-utt CER ≤ 20% | **PASS** (oracle/train-free ≪0.20; packing fixed since S1d) |
| **2** | Train-set CER ≤ 10% | **PASS** (train-free n=5 CER≈0.020; not full 32-utt re-eval) |
| **3** | Val CER ≤ 15% on **100** held-out | **PASS** — free text-only val100 mean CER≈0.057 (median≈0.041), mean WER≈0.153, NEAR_SILENT=0 (`out/intelligibility_mel_s3_val100.json`) |

- **Artifacts:** `out/intelligibility_mel_s3.json` (+ `_demo` / `_oracle_dur` / `_train_free` / `_free_vs_oracle`), **`out/intelligibility_mel_s3_val100.json`**, `out/mel_s3_demo_wavs/`, `out/mel_s3_val100_wavs/` (+ `_summary.json`), `runs/mel_baseline_v01_s3/{best_metrics,s3_budget_post}.json` (JSON mirrored to Windows `out/`).
- **Verdict:** S3 improves OOD free CER **0.079 → 0.041** and val mel_l1 **1.381 → 1.128**. **Gate 3 PASS** on contiguous val100 free text-only (mean CER≈0.057≪0.15, median≈0.041, NEAR_SILENT=0). Still **do not** restart latent Talker S4/Flow/LoRA/VibeVoice train. Mel regression S4 (full clean) follows — see next section.

```bash
# S3 train (after CTC forced + mel cache for ~5h; warm-start S2)
python scripts/27_train_mel.py \
  --config configs/mel_baseline_v01.yaml \
  --data configs/data_ruslan_local.yaml --stage S3 \
  --gap-mode expand_neighbor --run-dir runs/mel_baseline_v01_s3 \
  --init-ckpt runs/mel_baseline_v01_s2/best.pt

python scripts/28_infer_mel.py \
  --ckpt runs/mel_baseline_v01_s3/best.pt --demo \
  --duration-mode mean_stats --gap-mode expand_neighbor \
  --sil-policy none --no-forced-lookup \
  --data configs/data_ruslan_local.yaml --out out/mel_s3_demo_wavs
```

## Mel S4 — full clean corpus, expand_neighbor (2026-07-30)

**Goal:** scale regression mel baseline past S3 5 h to full `exclude_tags`-filtered train; warm-start from S3. **CFM not trained** (parked after S2 A/B).

| Piece | Result |
|-------|--------|
| Stage | no `max_hours` (full clean), **40000** steps (`configs/mel_baseline_v01.yaml`) |
| Align | CTC merge `--skip-existing` → **13647** utts in `forced.json` (12490 train + 670 val; all `method=ctc`) |
| S4 train cut | **12490** utts / **15.409 h** after `exclude_tags`; forced+mel cover **12490/12490** |
| Val | loader cap 32; Gate-style scored **100** contiguous forced+mel (670 available) |
| Mel cache | index **13160** |
| Gap mode | `expand_neighbor` (no `sil`); vocab **39** |
| Init | `--init-ckpt runs/mel_baseline_v01_s3/best.pt` (54/54 weights) |
| Ckpt / run | `runs/mel_baseline_v01_s4/best.pt` (best @ step **7500**) |
| Val mel_l1 | **1.049** |
| Vocoder | griffin-lim (Gate metrics); HiFi-GAN listening at `out/mel_s4_demo_wavs_neural/` |
| RAM | ~3.6 GiB host / ~1.8 GiB VRAM peak (lazy shards; tmux `mel_s4`) |

| Metric | S2 (≈1 h) | S3 (≈5 h) | S4 (≈15.4 h) |
|--------|----------:|----------:|-------------:|
| Val mel_l1 (best) | 1.381 | 1.128 | **1.049** |
| Free OOD demo CER (n=5) | 0.079 | **0.041** | 0.048 |
| Free OOD demo WER | 0.086 | 0.057 | **0.057** |
| Free val100 CER (no lookup) | — | 0.057 (med 0.041) | **0.048** (med **0.026**; WER≈0.138) |

**Production recommendation:** `runs/mel_baseline_v01_s4/best.pt` — val100 free CER **0.048 < S3 0.057** (primary Gate-style metric). Demo5 slightly worse than S3 (0.048 vs 0.041) on n=5 ASR noise; keep S3 as backup if listening prefers the older demos.

- **Artifacts:** `out/intelligibility_mel_s4.json` (+ `_demo` / `_val100`), `out/mel_s4_demo_wavs/`, `out/mel_s4_val100_wavs/`, `runs/mel_baseline_v01_s4/{best_metrics,s4_budget_post,run_manifest}.json` (JSON + wavs mirrored to Windows `out/`).
- **Verdict:** Full-corpus scale improves held-out val100 CER **0.057 → 0.048** and val mel_l1 **1.128 → 1.049**. Gate 3 remains PASS. **CFM stays parked.** Still **do not** restart latent Talker S4/Flow/LoRA. Optional next: neural vocoder for listening; stress/OOD sets.

```bash
# S4 train (full clean CTC+mel; warm-start S3) — or: python scripts/_run_mel_s4.py
python scripts/27_train_mel.py \
  --config configs/mel_baseline_v01.yaml \
  --data configs/data_ruslan_local.yaml --stage S4 \
  --gap-mode expand_neighbor --run-dir runs/mel_baseline_v01_s4 \
  --init-ckpt runs/mel_baseline_v01_s3/best.pt

python scripts/28_infer_mel.py \
  --ckpt runs/mel_baseline_v01_s4/best.pt --demo \
  --duration-mode mean_stats --gap-mode expand_neighbor \
  --sil-policy none --no-forced-lookup \
  --data configs/data_ruslan_local.yaml --out out/mel_s4_demo_wavs

python scripts/_eval_mel_val100.py \
  --ckpt runs/mel_baseline_v01_s4/best.pt --n 100 \
  --out out/intelligibility_mel_s4_val100.json \
  --wav-dir out/mel_s4_val100_wavs --gap-mode expand_neighbor
```

## Env status (WSL Ubuntu, 2026-07-30)

| Piece | Status |
|-------|--------|
| **sudo apt** | Blocked (`sudo` password required). System `espeak-ng` package not installed via apt. |
| **espeak-ng** | User-local extract at `~/local/espeak-ng` + wrapper `~/local/bin/espeak-ng` (deps via `apt-get download` + `dpkg-deb -x`). `.venv/bin/activate` exports `LD_LIBRARY_PATH`, `ESPEAK_DATA_PATH`, `PHONEMIZER_ESPEAK_LIBRARY`. |
| **phonemizer** | Installed in venv (`uv pip install -e ".[g2p]"` → phonemizer 3.3.0). Sample: `phonemize('привет', language='ru', backend='espeak')` → `prʲivʲet`. `frontend/g2p.py` auto-`set_library` + ctypes preload for user-local lib. |
| **G2P smoke** | `26_build_forced_align.py --method uniform --max-utts 4` → `g2p_backend=phonemizer_espeak_ru`, IPA-like labels (not Cyrillic graphemes). Artifacts: `/tmp/tracka/forced_espeak_smoke.json`, `out/forced_espeak_smoke.json`. |
| **RU CTC** | `~/models/wav2vec2-large-xlsr-53-russian` + symlink `~/models/ru-ctc`. **`--method ctc` is real**: emissions → `forced_align` → `merge_tokens` → char spans in `forced.json` (S1c: 64/64 ok). |
| **Vocoder** | **HiFi-GAN V1** at `~/models/hifigan/generator.pt` (SpeechBrain LibriTTS 22050 Hz; `scripts/33_download_hifigan.py`). Auto-picked by `mel_to_wav` / `28_infer_mel.py`. griffin-lim remains fallback. |
| **Disk** | ~896 GB free on `$HOME` / `HF_HOME`. |

### Eval suite (plan c)

Frozen sets + unified runner (Whisper CER/WER; optional GigaAM / UTMOSv2 / NISQA / MCD-DTW / F0 RMSE skip if missing):

| Piece | Path |
|-------|------|
| Suite YAML | `configs/eval_mel_suite.yaml` |
| Sets | `configs/eval_sets/{demo5,val100,homograph,bank_entity}.json` |
| Loader / agg | `src/minispeech/eval/test_sets.py` |
| Optional metrics | `src/minispeech/eval/optional_metrics.py` |
| Runner | `scripts/32_eval_mel_suite.py` |
| Seeded S4 summary | `out/eval_mel_suite_s4.json` |
| Live S4 + HiFi-GAN | `out/eval_mel_suite_s4_live.json` (demo5/homograph/bank_entity; bank CER pre-fix) |
| Bank fix re-score | `out/eval_mel_suite_s4_bankfix.json` (demo5 + bank_entity) |
| Unit tests | `tests/test_eval_mel_suite.py`, bank TN tests in `tests/test_frontend_track_a.py` (CPU) |

```bash
# CPU: seed known S4 numbers into results JSON
python scripts/32_eval_mel_suite.py --seed-production --out out/eval_mel_suite_s4.json

# Live free-infer + Whisper on frozen phrase sets
python scripts/32_eval_mel_suite.py \
  --ckpt runs/mel_baseline_v01_s4/best.pt \
  --sets demo5,homograph,bank_entity \
  --prefer-vocoder neural \
  --out out/eval_mel_suite_s4_live.json

pytest -q tests/test_eval_mel_suite.py
```

### Neural vocoder (HiFi-GAN)

| Piece | Result |
|-------|--------|
| Source | `speechbrain/tts-hifigan-libritts-22050Hz` → `~/models/hifigan/` |
| Loader | `src/minispeech/infer/hifigan.py` + extended `infer/vocoder.py` (SpeechBrain `*.conv.*` state_dict remap) |
| Geometry | 22.05 kHz, hop=256, n_fft=1024, 80 mels — matches mel80; **fmax train=8 kHz** vs mel80 nyquist (mild mismatch; listening OK) |
| Demo5 CER (Whisper) | griffin-lim **≈0.048** → neural **≈0.042** (`out/intelligibility_mel_s4_demo_neural.json`) |
| Listening | `out/mel_s4_demo_wavs_neural/*.wav` |
| Compare JSON | `out/vocoder_demo5_compare.json` |

```bash
# One-time download (explicit; mel_to_wav never Hub-downloads)
python scripts/33_download_hifigan.py --smoke-forward

# Production demos with neural vocoder
python scripts/28_infer_mel.py \
  --ckpt runs/mel_baseline_v01_s4/best.pt --demo \
  --duration-mode mean_stats --gap-mode expand_neighbor \
  --sil-policy none --no-forced-lookup \
  --data configs/data_ruslan_local.yaml \
  --out out/mel_s4_demo_wavs_neural
```

### Live eval suite — S4 + HiFi-GAN (2026-07-30)

**Run:** WSL `~/projects/tts-self` @ `cd08a26`; `scripts/32_eval_mel_suite.py` with `--prefer-vocoder neural --vocoder-ckpt ~/models/hifigan/generator.pt`; ckpt `runs/mel_baseline_v01_s4/best.pt`. Sets: `demo5,homograph,bank_entity` (`val100` skipped — already scored offline ≈0.048). Artifact: `out/eval_mel_suite_s4_live.json` (mirrored to Windows `out/`). Wavs: `out/eval_suite_mel_baseline_v01_s4_wavs/`. All utterances used backend **neural**.

| Set | n | mean CER | median CER | mean WER | median WER |
|-----|---|----------|------------|----------|------------|
| demo5 | 5 | **0.0990** | 0.1163 | 0.1476 | 0.1667 |
| homograph | 8 | **0.0344** | 0.0119 | 0.1125 | 0.0625 |
| bank_entity (pre-fix) | 8 | **0.4584** | 0.4593 | 0.5036 | 0.5000 |

**Notes:** Homograph OOD is strong (mean CER≈0.034). Pre-fix bank CER≈0.458 was mostly an **ASR scoring artifact** (Whisper digit hyps vs spoken-word refs) plus digit prompts not exercised in the frozen set. Optional GigaAM/UTMOS/NISQA skipped (deps missing).

### Bank entity gap — diagnosed & fixed (2026-07-30)

**Root cause (not a Talker/mel regression):**
1. **Eval CER mismatch:** Whisper emits Arabic digits (`1234`, `92 рубля`) while refs were fully verbalized → `normalize_for_asr` alone inflated mean CER≈0.458. Fix: score with `normalize_verbalized_for_asr` on both ref and hyp in `scripts/32_eval_mel_suite.py`.
2. **Eval set did not stress TN:** phrases were pre-verbalized. Set now uses digit money/dates/times + account codes so infer TN (`normalize_for_tts` inside `transcript_for_ctc` / `text_to_ctc_char_labels`) matches train.
3. **TN hardening:** grouped thousands (`15 700`), money+kopecks, month-word dates, account digit-spell (`номер счёта 4082`), clock `16.30`, letterisms `ИНН`/`КПП`.

**Re-score (S4 + HiFi-GAN):** `out/eval_mel_suite_s4_bankfix.json`

| Set | n | mean CER | before → after | finetune? |
|-----|---|----------|----------------|-----------|
| demo5 | 5 | **0.0990** | unchanged smoke | no |
| bank_entity | 8 | **0.0931** | **0.458 → 0.093** | **no** (below 0.25) |

Residual failures: `bank_08` ИНН/КПП letterisms still weak under ASR (hyp drops acronyms; CER≈0.56); `bank_02` occasional семьсот↔семьдесят. **No bank-heavy finetune / S4 retrain** — gap was scoring + TN coverage.

### Next recommended steps

**Mel S4 done — production = S4** (`runs/mel_baseline_v01_s4/best.pt`): val100 free CER≈0.048 (med≈0.026) vs S3≈0.057; demo5≈0.048 (GL) / ≈0.042 (HiFi-GAN dedicated) / suite-live neural **≈0.099**; val mel_l1≈1.049; train **12490** utts / **15.4 h**. **Bank gap closed for Gate scoring** (mean CER **0.093**; artifact `out/eval_mel_suite_s4_bankfix.json`). Homograph ≈0.034. **CFM parked**. **Do not** restart latent Talker S4 / Flow / LoRA / VibeVoice train; **do not** train CFM further. Next: optional MOS (UTMOSv2); optional letterism/acronym data if bank_08 matters for product.

```bash
cd ~/projects/tts-self && source .venv/bin/activate
export HF_HOME=$HOME/hf_cache
python scripts/28_infer_mel.py --ckpt runs/mel_baseline_v01_s4/best.pt --demo \
  --duration-mode mean_stats --gap-mode expand_neighbor --sil-policy none \
  --no-forced-lookup --data configs/data_ruslan_local.yaml \
  --out out/mel_s4_demo_wavs_neural   # uses ~/models/hifigan/generator.pt
python scripts/32_eval_mel_suite.py \
  --ckpt runs/mel_baseline_v01_s4/best.pt \
  --sets demo5,homograph,bank_entity \
  --prefer-vocoder neural \
  --out out/eval_mel_suite_s4_live.json
```

Also: espeak phone separator fix (`Separator(phone=' ', word='|')`) for uniform phone packing when not using CTC.

## Mel CFM (a5) — after Gate 3 PASS (2026-07-30)

**Goal:** duration-conditioned rectified-flow / CFM mel decoder as generative alternative to L1 regression; same Track A frontend + expand_neighbor char-CTC packing. No Talker / codec restart.

| Piece | Location |
|-------|----------|
| Model | `src/minispeech/models/mel_cfm.py` (`MelCFM`: backbone encode/upsample + `FlowHead` over mel80) |
| Trainer | `src/minispeech/train/mel_cfm_trainer.py` |
| Config | `configs/mel_cfm_v01.yaml` |
| Train | `scripts/31_train_mel_cfm.py` |
| Infer | `scripts/28_infer_mel.py` (auto-detects CFM ckpt; `--flow-steps` / `--flow-method` / `--cfg-scale`) |
| Tests | `tests/test_mel_cfm.py` |

```bash
# Unit tests (CPU)
pytest -q tests/test_mel_cfm.py

# S1 CFM smoke (32 utts; optional warm-start encoder from regression S3)
python scripts/31_train_mel_cfm.py \
  --config configs/mel_cfm_v01.yaml \
  --data configs/data_ruslan_local.yaml --stage S1 \
  --gap-mode expand_neighbor --run-dir runs/mel_cfm_v01_s1 \
  --init-ckpt runs/mel_baseline_v01_s3/best.pt

# Infer (griffin-lim OK)
python scripts/28_infer_mel.py \
  --ckpt runs/mel_cfm_v01_s1/best.pt --demo --flow-steps 16 \
  --duration-mode mean_stats --gap-mode expand_neighbor \
  --sil-policy none --no-forced-lookup \
  --data configs/data_ruslan_local.yaml --out out/mel_cfm_s1_demo_wavs
```

### Mel CFM S1 smoke (2026-07-30)

| Piece | Result |
|-------|--------|
| Stage | S1, 32 utts, **1500** steps, batch 8, bf16 |
| Align / gap | char CTC + `expand_neighbor` (same as regression S3) |
| Vocab | **39** from `runs/mel_baseline_v01_s3/phone_vocab.json` |
| Init | regression S3 → backbone (48/48 keys) |
| Ckpt / run | `runs/mel_cfm_v01_s1/best.pt` |
| Train loss | early mean **4.645** → late mean **0.794** (`loss_decreased=true`) |
| Final train | `flow_mse≈0.705`, `duration≈0.069`, total≈0.712 |
| Val `flow_mse` | ≈5.54 (noisy single-pass CFM val; ranking proxy only) |
| Vocoder | griffin-lim demos in `out/mel_cfm_s1_demo_wavs/` |
| Artifacts | `runs/mel_cfm_v01_s1/smoke_metrics.json`, `out/mel_cfm_s1_smoke_metrics.json` |

**Verdict:** CFM train loop is healthy — loss decreases ~6× on S1 with S3 encoder warm-start. **Ready for S2 CFM A/B** vs regression (same packing / vocab; optional longer S1 if demos are unintelligible). Full-corpus CFM not yet. Still **do not** restart latent S4/Flow/LoRA. Demo CER vs regression S3 deferred (optional follow-up).

### Mel CFM S2 A/B (2026-07-30)

| Piece | Result |
|-------|--------|
| Stage | S2, ~1 h / 1236 utts, **12000** steps, batch 8, bf16 |
| Align / gap | char CTC + `expand_neighbor` (same as regression S2) |
| Init | CFM S1 `best.pt` (91/91 keys); vocab 39 |
| Ckpt / run | `runs/mel_cfm_v01_s2/best.pt` |
| Train loss | early mean **1.146** → late mean **0.918** (`loss_decreased=True`) |
| Final train | `flow_mse≈0.886`, `duration≈0.189`, total≈0.905 |
| Best val `flow_mse` | ≈1.166 @ step 7500 |
| Free demo CER (n=5, flow_steps=24, griffin-lim) | **0.6993** |
| Val32 free CER | mean≈0.8848 / median≈0.8682 |
| Regression S2 demo CER | ≈0.0794 |
| Regression S3 demo CER | ≈0.0407 (val100≈0.057) |

**A/B:** CFM S2 demo CER 0.6993 vs regression S2 0.0794 / S3 0.0407 / S4 0.048.

**Recommendation:** `stick_to_regression` — CFM S2 demo CER clearly worse than regression → **CFM parked** (no S3/S4 CFM, no further CFM train). Production mel = regression **S4** (`runs/mel_baseline_v01_s4/best.pt`). MOS listening study and full-corpus CFM **deferred**. Still **do not** restart latent Talker S4/Flow/LoRA.

**Artifacts:** `out/intelligibility_mel_cfm_s2.json` (+ `_demo` / `_val32`), `out/mel_cfm_s2_demo_wavs/`, `runs/mel_cfm_v01_s2/{best.pt,best_metrics,smoke_metrics}.json` (JSON mirrored to Windows `out/`).


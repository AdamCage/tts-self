# Results — `n2_first_execution`

**Plan:** `.cursor/plans/n2_first_execution.plan.md`  
**Date:** 2026-08-01  
**Windows HEAD (authoring):** `2c4c4bd` (+ uncommitted wave-0/1/2 WIP; not committed per instructions)  
**WSL runtime HEAD:** `2c4c4bd` (clean clone; code via `PYTHONPATH=/mnt/c/projects/tts-self/src`)  
**Status:** **partial / N2 FAIL** — Wave 0 validity repair complete; Wave 1 caches in progress/partial; Wave 2 S1 overfit + text-usage measured; Gate N2 **FAIL** (hard-stop before 1h/5h ladder); H1/H2 skipped; science stream mostly done.

Methodology was read (README, 08_evaluation, 09_roadmap, REVIEW) and **not edited**.

---

## 1. Task status matrix

| ID | Status | Notes / artifact |
|----|--------|------------------|
| R1 | **PASS** | Track N sets v2.0.0; train 51234 clean; val/test/review explicit; `out/r1_freeze_result.json` |
| R3 | **PASS** | Dates/time/decimals; `tests/test_verbalize_r3.py`; probe JSON |
| R9 | **PASS** | 15×3 exact equality + coverage test; 81 frontend tests |
| R4 | **PASS** | `bank_entity` 3.0.0 independent norms; match 297/315; `docs/bank_entity_v3_audit.md` |
| R5 | **PASS** | `homograph` 3.0.0; v2 not-agreed 175/740; v3 0; `docs/homograph_v3_audit.md` |
| R2 | **PASS** | Fair 3-condition baseline; Silero+FE bank DER **0.103** vs Track A **0.624** |
| R6 | **PASS** | 2×2 remainder **0.0**; `out/bank_delta_2x2.json` |
| R7 | **PASS** | WSL cleaned; `configs/eval_set_hashes.json`; `docs/reproducibility.md`; `git_dirty` |
| R8 | **PASS** | `out/_*` cleared on Windows; junk removed; telephony `n` → v1.0.1; `test_eval_mel_suite.py` already clean |
| C1 | **PASS*** | Encoded **46070** latents, `sample=False`, 244M; copy-synth mean CER **0.019** (n=50) vs anchor 0.073; post-encode n_frames match 1.0 on sample. *3269 pre-encode manifest n_frames mismatches were corrected on write.* |
| C2 | **PASS*** | Text-hidden n=46070, H=1024, 3 layers, frac_at_max=0, parity 20/20; disk 4.4G. *Manifest briefly truncated then restored to 46783; filter write fixed.* |
| C3 | **PASS*** | Coverage **1.0** on 46112 non-exclude rows; grapheme-uniform (espeak too slow). Level-A priors written for **46783** rows (`align_*.safetensors`). `out/c3_align_cv17.json` |
| C4 | **CODE PASS** | `MultiCorpusDataset` + tests (4 passed); full report after caches |
| A1–A3 | **PASS (unit)** | Backbone/diffusion/collator; 18 tests passed, 2 GPU skipped |
| A4 | **PARTIAL** | S1 TF overfit on RUSLAN-32; FR R² **−6.64**; ASR CER null |
| A5 | **PASS (plan §7)** | Shuffle/mask break (rises 0.43 / 1.63); prefix optional fail; `out/n2_text_usage.json` |
| A6 | **FAIL** | Hard-stop: S1 FR worse than trivial constant_mean DTW; no S2/S3; `out/n2_gate.json` |
| A7 | **PARTIAL** | Side-by-side folders exist (Silero raw/FE, Track A, N2 S1 demo); `listen_index.json` |
| H1/H2 | **SKIPPED** | Require A6 PASS |
| S1 | **READY / NOT RUN** | C3 `forced.json` now present (46112); SA re-measure left as follow-up |
| S2 | **PASS** | 673/673; return **77.44%** exclude-hours vs sample 71.92%; `out/e6_full_cv17.json` |
| S3 | **PASS** | Paired bootstrap; no difference at n=30 |
| S4 | **BLOCKED** | 8 figures from real artifacts; 0 listeners; MOS unmeasured |

---

## 2. Key metrics (with artifact paths)

### Wave 0 — validity

| Metric | Value | Artifact |
|--------|------:|----------|
| Track N train n / hours / speakers | 51234 / 72.98 h / 1088 | `out/r1_freeze_result.json` |
| Active `qc_tag=None` | 0 | `out/r1_qc_tag_table.json` |
| bank_entity v3 norm↔frontend match | 297/315 = 0.9429 | `out/bank_entity_v3_audit.json` |
| percent category match | 0.4762 | same |
| homograph v2 not-agreed | 175/740 = 0.2365 | `out/homograph_v3_audit.json` |
| homograph v3 not-agreed | 0/719 | same |
| Silero raw → +FE bank DER | 0.9734 → 0.1030 (Δ −0.8704) | `out/external_baseline_silero_v3.json` |
| Track A bank DER (v3) | 0.6238 | same |
| DER 2×2 remainder | 0.0 | `out/bank_delta_2x2.json` |
| Telephony paired ΔCER CI95 | both include 0 | `out/p1_telephony_channels.json` |

### Wave 1 — caches

| Metric | Value | Artifact |
|--------|------:|----------|
| CV17 latents encoded | 46070 | `~/datasets/.../latents/encode_meta.json`, `out/c1_latents_cv17.json` |
| Copy-synth mean CER (n=50) | **0.0193** (anchor 0.073) | `out/c1_copysynth_cer.json` |
| Text-hidden n / layers / H | 46070 / 3 / 1024 | `out/c2_text_hidden_cv17.json` |
| max_length hit fraction | 0.0 | same |
| Train/infer frontend parity | 20/20 | `out/c2_parity.json` |
| Forced align coverage | 1.0 (46112/46112) | `out/c3_align_cv17.json` |
| Align duration frames mean/max | 40.45 / 87; zero_or_neg=0 | same |

### Wave 2 — N2 signal

| Metric | Value | Artifact |
|--------|------:|----------|
| S1 final train loss | 0.056 | `out/n2_s1_ruslan.json` |
| S1 FR R² / var_ratio / DTW L1 | −6.64 / 6.72 / 2.076 | same |
| Trivial constant_mean DTW L1 | 0.757 | `out/trivial_baselines.json` |
| Text shuffle / mask relative rise | 0.428 / 1.631 | `out/n2_text_usage.json` |
| **N2 verdict** | **FAIL** | `out/n2_gate.json` |

### Science

| Metric | Value | Artifact |
|--------|------:|----------|
| CTC return of exclude-hours | 0.7744 (508/673 utts) | `out/e6_full_cv17.json` |
| Preference listeners | 0 (BLOCKED) | `out/paper1_figures/preference_protocol.json` |

---

## 3. Code / config changes (uncommitted)

Major paths (not exhaustive):

- Frontend: `src/minispeech/frontend/verbalize.py`, `tests/test_verbalize_r3.py`, `tests/test_frontend_track_a.py`
- Eval sets: `configs/eval_sets/{bank_entity,homograph,telephony,trackn_*}.json`, `configs/eval_set_hashes.json`
- Scripts: `39_external_baseline_eval.py`, `54_eval_sets.py`, `57/58` merge/freeze, `61`, `62`, `63`, `64`, `65_train_n2.py`, `66_infer_n2.py`, `67_eval_n2_text_usage.py`, `04/05` cache filters
- Models: `backbone_lm.py`, `latent_io.py`, `head_diffusion.py`, `collate_latent_seq.py`, `n2_trainer.py`
- Config: `configs/n2_qwen_latent_v01.yaml`, `configs/data_common_voice_ru17.yaml`, `configs/data_multicorpus.yaml`
- Docs: `reproducibility.md`, `bank_entity_v3_audit.md`, `homograph_v3_audit.md`, `frozen_numbers_mel80.md`, `corpus_inventory.md`, `paper1_*.md`

---

## 4. Listen paths (Windows)

Play these first:

```
C:\projects\tts-self\out\n2_first_execution\n2_s1_listen_demo\01_demo.wav
…\05_demo.wav

C:\projects\tts-self\out\n2_first_execution\silero_frontend\demo5\
C:\projects\tts-self\out\n2_first_execution\silero_raw\demo5\
C:\projects\tts-self\out\n2_first_execution\track_a\demo5\
```

Full suite wav trees (1256 each) also under Linux `~/projects/tts-self/out/n2_first_execution/{silero_raw,silero_frontend,track_a}/`.  
Index: `.cursor/results/n2_first_execution/listen_index.json`.

---

## 5. Gate exit (§7)

**Part 1 (validity):** essentially **met** (sets v3, fair Silero, 2×2, provenance, hygiene, category tests).

**Part 2 (N2 signal):** **FAIL recorded.** S1 free-running content collapses vs trivial baseline proxy; 1h/5h ladder not started. Text shuffle/mask pass on loss proxy. Wavs on disk.

**Fork recommendation (plan order):** (1) more data T2, (2) LoRA with measured justification, (3) other head, (4) own tokenizer P2.

---

## 6. Follow-ups / blockers

1. Finish C2 meta + C1 copy-synth CER (≥50) + C3 align; then S1 stress classifier.
2. Commit authoring tree so WSL can `git pull` without `PYTHONPATH=/mnt/c` / `MINISPEECH_ALLOW_DIRTY`.
3. Wire N2 ASR CER into text-usage / A6 (currently loss/DTW proxies).
4. Human preference panel for S4 (currently blocked at 0 listeners).
5. Do **not** scale N2 ladder until FR content beats trivial baseline.

---

## 7. Logs

Directory: `.cursor/results/n2_first_execution/logs/`  
Per-task reports: `R2_REPORT.md`, `R6_REPORT.md`, `A{1,2,3,4,5}_REPORT.md`, `S2_REPORT.md`, `S4_REPORT.md`, etc.

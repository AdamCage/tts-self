# Frozen numbers — mel80 (`fmax = sr/2`)

Tag: **`paper2-frozen-mel80`** @ commit `56c8182`.

These numbers were measured in the original Track A mel domain (`MEL80`, `fmax = sample_rate/2 = 11025 Hz`). After any new mel target (e.g. `mel80_hifigan`) absolute CER values are **not** comparable to this table without re-running under the frozen tag/domain.

Do **not** push the tag without an explicit user request.

## Production checkpoint

| Field | Value | Artifact |
|-------|-------|----------|
| Ckpt | `runs/mel_baseline_v01_s4/best.pt` | WSL run dir |
| Train size | 12490 utts / 15.41 h | `runs/mel_baseline_v01_s4/run_manifest.json`, `s4_budget_post.json` |
| Gap mode | `expand_neighbor` | same |

## Free synthesis (S4)

| Metric | Value | Artifact |
|--------|------:|----------|
| val100 Whisper CER (mean) | **0.048** | `out/intelligibility_mel_s4_val100.json` |
| val100 CER (median) | 0.026 | same |
| val100 WER (mean) | 0.153 | same (also cited as ≈0.138 in summary tables; use JSON) |
| val100 NEAR_SILENT | 0 | same |
| demo5 CER (griffin-lim) | **0.048** | `out/intelligibility_mel_s4_demo.json` / `out/intelligibility_mel_s4.json` |
| demo5 CER (HiFi-GAN) | **0.042** | `out/intelligibility_mel_s4_demo_neural.json` |

## Live eval suite (S4 + HiFi-GAN)

| Set | mean CER | Artifact |
|-----|---------:|----------|
| demo5 | 0.099 | `out/eval_mel_suite_s4_live.json` |
| homograph | 0.034 | same |
| bank_entity (after TN/ASR-score fix) | 0.093 | `out/eval_mel_suite_s4_bankfix.json` |

## Copy-synthesis / oracle toplines

| Condition | CER | Artifact |
|-----------|----:|----------|
| GT mel → Griffin-Lim | **0.027** | `out/intelligibility_gt_mel_griffin.json` |
| S1e oracle-dur | 0.031 | `out/intelligibility_mel_s1e_oracle_dur.json` |
| S1d oracle-dur | 0.052 | `out/intelligibility_mel_s1d_oracle_dur.json` |
| S1e free train (text-only) | 0.211 | `out/intelligibility_mel_s1e_train_free.json` |

## Timeline coverage (CTC blank gaps)

| Condition | cover_ratio | Artifact |
|-----------|------------:|----------|
| Raw CTC (pre-fix, S1c era) | **0.318** | `out/cover_ratio_s1d.json` (pre) / diagnosis in `docs/next_intelligibility.md` |
| After `expand_neighbor` (S1e) | **1.000** | `out/cover_ratio_s1e.json` |
| After `sil_fill` (S1d) | **1.000** | `out/cover_ratio_s1d.json` |

## CFM (parked)

| Condition | CER | Artifact |
|-----------|----:|----------|
| Mel CFM S2 free demo | **0.699** | docs / CFM run artifacts under `out/intelligibility_mel_cfm*` |
| Mel regression S2 free demo | **0.079** | `out/intelligibility_mel_s2.json` |

## Notes

- Primary narrative: `docs/next_intelligibility.md`.
- Naturalness (UTMOS / NISQA) was **not** measured at freeze time — see roadmap tasks T02/T04.
- JSON paths above live primarily under WSL `~/projects/tts-self/out/`; many were mirrored to Windows `C:\projects\tts-self\out\`.

## Track N P0 baseline (N10) — 2026-07-31

База для сравнения Track N с production Track A по расширенному RU-Critical и метрикам DER / EER / SA.
Чекпойнт: `runs/mel_stress_v01_s4/best.pt`. Вокодер: neural (HiFi-GAN). Артефакт:
`out/baseline_track_a_critical.json` (прогон 2026-07-31, WSL).

Набор `latin` не был в `configs/eval_mel_suite.yaml`; зарегистрирован через
`--suite out/eval_mel_suite_n10.yaml` (временный suite YAML под `out/`, без правки
общего конфига).

### Eval-set hashes (sha256)

| Set | n | sha256 |
|-----|--:|--------|
| `bank_entity.json` | 315 | `34671d57adddce5f6fbea30aedf51bfa23afb64961b6aacb84935464c44480ab` |
| `homograph.json` | 895 | `60067ae03bc288e9325ebe438554b3aa495958458f9ffa9077dc830adec9bbc6` |
| `latin.json` | 117 | `5cec49347f3f49551436e5a907f6d6c85cc89f96240081fe56a49782727cec21` |

Также: `out/n10_eval_set_hashes.txt`.

### Free synth (neural)

| Set | n | mean CER | DER | EER | SA | Notes |
|-----|--:|---------:|----:|----:|----|-------|
| demo5 | 5 | 0.0935 | — | — | — | нет critical_spans / stress_target |
| val100 | 100 | **0.0310** | — | — | — | production CER якорь |
| bank_entity | 315 | 0.1616 | **0.3647** | **0.4889** | — | n_spans=315, n_digits=1456 |
| homograph | 895 | 0.0916 | — | — | **skipped** | n_sa_skipped=895 (ASR не различает лексемы) |
| latin | 117 | 0.2408 | — | — | — | нет critical_spans / stress_target в наборе |

SA=`skipped` — допустимый исход N07/N10: текстовый ASR не различает пары омографов.

### Copy-synth topline (val100, utt_ids с корпусом)

| Condition | mean CER | n | Artifact field |
|-----------|---------:|--:|----------------|
| Free synth (model) | 0.0310 | 100 | `sets.val100.mean_cer` |
| Real corpus wav → ASR | 0.0601 | 100 | `utterances[].copy_synth_topline.real.cer` |
| GT mel → neural vocoder → ASR | 0.0640 | 100 | `utterances[].copy_synth_topline.gt_mel_vocoder.cer` |

Phrase-наборы (`demo5`, `bank_entity`, `homograph`, `latin`) не имеют corpus `utt_id` →
copy-synth topline там не считается (ожидаемо).

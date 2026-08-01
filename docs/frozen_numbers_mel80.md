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

Контракт P1 разделяет хеш содержимого и хеш файла. `items_sha256` считается по
каноническому JSON массива `items` (для `kind: utt_ids` — массива `utt_ids`) и
не зависит от метаданных. `file_sha256` изменился при добавлении
`version`; сами элементы неизменённых наборов не менялись.

| Set | version | n | items_sha256 | file_sha256 |
|-----|---------|--:|--------------|-------------|
| `bank_entity.json` | 2.0.0 | 315 | `02be6b39f2c37c00ec8ffe5d5f00677856133e3a64d5c4745f76e0ed4890b1b6` | `b91f49189aaa1d1df9bb07e64b782ebd0722a025bc910202ed98bcc98b8a7d57` |
| `demo5.json` | 1.0.0 | 5 | `72b43f9d2107133ce8192c0c96be27869e0c865edd1639b6d3a2aafd7a31b70e` | `8d7d53e243c4b08aefbabe73cd09a0e873feb4217e29fd4fb2d5919af19a3bee` |
| `homograph.json` | 2.0.0 | 740 | `ffafa4db14ff585a949939ed13638559934f2a2816c51e7731e130b2c4b936f6` | `319575e56ef873014fdcba69a4ec5b1012059a3994c174e31bdc435ce4e14a19` |
| `latin.json` | 1.0.0 | 117 | `3120970f696edb30d05e5d3ee21f04762a81f2c68d5d900aa1507b9a6d1d7ce0` | `3d1da42093072df688851d0953044ec2ba8dda4548246369af2910ab147268db` |
| `telephony.json` | 1.0.0 | 30 | `ee0ad46e1a6fea4b22e18e02d013d761af883aaa73cf9bb0201d342a02a732b4` | `510d92abbba9c717d0c25965f89bf858cfa621cc9e31c5c760ca5692482e596d` |
| `trackn_test.json` | 1.0.0 | 3458 | `fc00d40f81d7949b28ba4ccfac9c58646eeb69b5b7a2372de9a8bbff4094d85c` | `11efb5b01a2cb54a4d60e49fca4463e0e6af278bef14a1ebbcc3a586c77addc4` |
| `trackn_train.json` | 1.0.0 | 61866 | `bdb057ffee7cf8f50f06815cff991e84d0fa8a7aa61c82fc5230f3a33d1b2dc9` | `98e2e25835fe760ff78ff6b14c78d35b95013c14f03eb023021d78799cf00c7c` |
| `val100.json` | 1.0.0 | 100 | `49762248caa03e945fff343716d9e4a267e713263abf5f344511cfe8520ac556` | `35853c892e2ada16d9fbe56ed48ae3a4a674fa59c6defc2a152f6c59e5883047` |

Реестр: `out/eval_set_hashes.json`. V1-хеши до аудита:
`bank_entity items=80b21ec8… file=074ebb64…`,
`homograph items=c80eb2a6… file=d57fdf75…`. Старые `file_sha256` до добавления
метаданных для трёх исходных наборов сохранены в `out/n10_eval_set_hashes.txt`.

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

## Phase M baseline (M09) — 2026-07-31

Новая точка отсчёта после починки метрик (M04–M06) и акустического SA (M05).
Чекпойнт тот же: `runs/mel_stress_v01_s4/best.pt`, neural HiFi-GAN.
Артефакт: `out/baseline_m_critical.json` (прогон с `--max-utts 50` на
bank/homograph/latin; demo5 полный n=5).

**Несравнимо с P0 DER/EER/SA:** старые DER≈0.365 и EER≈0.489 были фактически
EER по unmatched digits (пустые `hyp_digits`); SA=`skipped` на всех 895 из‑за
стресс-stripped лексем. Новые DER/EER считают числительные словами; SA —
акустический (`sa_mode=acoustic`).

Eval-set hashes **не менялись** (`out/n10_eval_set_hashes.txt`).

### Free synth (neural), median + CI95

| Set | n | mean CER | median CER | CER CI95 | DER | EER | SA | Notes |
|-----|--:|---------:|-----------:|----------|----:|----:|----|-------|
| demo5 | 5 | 0.0935 | 0.1163 | [0.023, 0.166] | — | — | — | |
| bank_entity | 50* | 0.0852 | **0.000** | [0.050, 0.124] | **0.115** | **0.240** | — | DER≠EER |
| homograph | 50* | 0.0596 | 0.0389 | [0.039, 0.082] | — | — | **0.34** | n_sa_scored=50, skipped=0 |
| latin | 50* | 0.1325 | 0.1061 | [0.096, 0.171] | — | — | — | n_cer_gt_1=0 на этой подвыборке |

\* Подвыборка `--max-utts 50` для wall-clock; полный пересчёт n=315/895/117 —
следующий прогон без cap.

### M05 acoustic SA calibration

`out/m05_acoustic_sa_calibration.json`: word-local accuracy **0.358** on 779
words (110 val utts) — **below 0.90 gate**. Detector is diagnostic-only.
See `docs/m05_acoustic_sa.md`.

### M07 vocoder FT (short)

`runs/vocoder_ft_v01/` — 2000 steps, ~1 h RUSLAN, init SpeechBrain LibriTTS.
`out/m07_best_metrics.json`: val_mel ≈ 0.317. Human blind A/B vs stock
generator **pending listening** (ballot harness: M08).

### M08/M09 preference harness

Duration A/B ballot: `out/m09_pref_dur/` (mean_stats vs pred_dur from M02).
Trap sanity (auto): `trap_same_rate=1.0`. Human votes not yet collected —
`significant_vs_half` requires listener responses; coin-flip simulation stays
inside CI around 0.5 as expected.

## P1 full baselines: v1 and v2 — 2026-08-01

Оба прогона используют один и тот же checkpoint
`runs/mel_stress_v01_s4/best.pt`, neural HiFi-GAN, полный набор без
`--max-utts`, акустический SA и copy-synth topline. Артефакты:
`out/baseline_p1_v1_full.json` и `out/baseline_p1_v2_full.json`.

### V1, до аудита

| Set | n | mean CER | median CER | CER CI95 | DER | EER | SA |
|-----|--:|---------:|-----------:|----------|----:|----:|---:|
| demo5 | 5 | 0.0935 | 0.1163 | [0.0233, 0.1662] | — | — | — |
| val100 | 100 | 0.0310 | 0.0195 | [0.0241, 0.0381] | — | — | — |
| bank_entity v1 | 315 | 0.1616 | 0.0938 | [0.1412, 0.1839] | 0.4729 | 0.5270 | — |
| homograph v1 | 895 | 0.0916 | 0.0625 | [0.0848, 0.0986] | — | — | 0.4922 |
| latin | 117 | 0.2408 | 0.1304 | [0.1241, 0.4543] | — | — | — |

### V2, после аудита

| Set | n | mean CER | median CER | CER CI95 | DER | EER | SA |
|-----|--:|---------:|-----------:|----------|----:|----:|---:|
| demo5 | 5 | 0.0935 | 0.1163 | [0.0233, 0.1662] | — | — | — |
| val100 | 100 | 0.0319 | 0.0224 | [0.0249, 0.0390] | — | — | — |
| bank_entity v2 | 315 | 0.1907 | 0.0750 | [0.1321, 0.2700] | 0.6142 | 0.3651 | — |
| homograph v2 | 740 | 0.0703 | 0.0318 | [0.0631, 0.0769] | — | — | 0.4614 |
| latin | 117 | 0.2386 | 0.1304 | [0.1221, 0.4520] | — | — | — |

**Прямое межверсионное сравнение недопустимо.** В `bank_entity` исправлены
44 из 315 expected labels (13.97%), а текущий frontend меняет нормализованный
вход синтеза. В `homograph` исправлены формы и удалены 155 невалидных пар, так
что изменились и состав, и контексты набора.

Чтобы отделить эффект разметки, гипотезы неизменного v1-прогона были повторно
оценены по v2 labels (`out/bank_entity_v1_hyp_v2_labels.json`):

- DER: `585/1237 = 0.4729` → `512/1226 = 0.4176`,
  label-only delta **−0.0553** (73 меньше digit errors);
- EER: `166/315 = 0.5270` → `169/315 = 0.5365`,
  label-only delta **+0.0095** (3 больше span errors).

В свежем v2-прогоне относительно v1 DER изменился на `+0.1413`, EER на
`−0.1619`. Остаток нельзя приписывать label fixes: frontend и синтезируемый
вход уже другие, а в bank v2 есть четыре выброса CER > 1. SA здесь остаётся
диагностикой: калибровка M05 ниже порога валидности 0.90.

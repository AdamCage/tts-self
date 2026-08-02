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

**Оговорка `ref_norm` (v1→v2):** дельты CER на `val100` и `latin` несравнимы
как чистый эффект модели/разметки. Между `out/baseline_p1_v1_full.json` и
`out/baseline_p1_v2_full.json` изменился `ref_norm` у **2 / 100** строк
`val100` и **6 / 117** строк `latin` (нормализация ссылки ASR), поэтому
опубликованные Δ CER на этих наборах смешивают смену эталона с прочим.

### Track A против внешнего Silero v4_ru (v2, unequal — historical)

Таблица ниже — исторический прогон P1 на наборах v2. У **1 275 / 1 277** строк
Silero `model_input_text == text` (сырой текст с цифрами), тогда как Track A
получал TN+verbalize. Это **не** сравнение систем. Артефакт:
`out/external_baseline_silero_v2.json`.

| Set | System | n | mean CER | median CER | CER CI95 | DER | EER | SA |
|-----|--------|--:|---------:|-----------:|----------|----:|----:|---:|
| demo5 | Track A | 5 | 0.0935 | 0.1163 | [0.0233, 0.1662] | — | — | — |
| demo5 | Silero raw | 5 | 0.0305 | 0.0000 | [0.0000, 0.0915] | — | — | — |
| val100 | Track A | 100 | 0.0319 | 0.0224 | [0.0249, 0.0390] | — | — | — |
| val100 | Silero raw | 100 | 0.0078 | 0.0000 | [0.0040, 0.0131] | — | — | — |
| bank_entity v2 | Track A | 315 | 0.1907 | 0.0750 | [0.1321, 0.2700] | 0.6142 | 0.3651 | — |
| bank_entity v2 | Silero raw | 315 | 0.5654 | 0.5758 | [0.5428, 0.5875] | 0.9739 | 0.9937 | — |
| homograph v2 | Track A | 740 | 0.0703 | 0.0318 | [0.0631, 0.0769] | — | — | 0.4614 |
| homograph v2 | Silero raw | 740 | 0.0094 | 0.0000 | [0.0074, 0.0115] | — | — | 0.5210 |
| latin | Track A | 117 | 0.2386 | 0.1304 | [0.1221, 0.4520] | — | — | — |
| latin | Silero raw | 117 | 0.3338 | 0.3226 | [0.3154, 0.3533] | — | — | — |

### Fair three-condition Silero (v3 sets) — N2 / R2

Три условия на одних позициях v3 (`demo5`, `val100`, `bank_entity` 3.0.0,
`homograph` 3.0.0, `latin`), одном ASR `configs/asr.yaml`, одном ckpt Track A
`runs/mel_stress_v01_s4/best.pt`, одних функциях CER/DER/EER/SA.

| Condition | Role | JSON | wav |
|-----------|------|------|-----|
| Silero raw (`--input-mode raw`) | вклад фронтенда (legacy unequal) | `out/external_baseline_silero_v3_raw.json` | `out/n2_first_execution/silero_raw/` |
| Silero + frontend (`--input-mode frontend`) | **сравнение систем** (общий TN+verbalize) | `out/external_baseline_silero_v3_frontend.json` | `out/n2_first_execution/silero_frontend/` |
| Track A | **сравнение систем** | `out/external_baseline_track_a_v3.json` | `out/n2_first_execution/track_a/` |

Сводка: `out/external_baseline_silero_v3.json`. Статусы Silero: `ok`, `n_errors=0`.
Silero raw: 2 fallback-а на `normalize_for_tts` (`bank_156`, `bank_162`).
Silero+frontend: fallback-ов 0.

| Set | System | n | mean CER | median CER | CER CI95 | DER | EER | SA |
|-----|--------|--:|---------:|-----------:|----------|----:|----:|---:|
| demo5 | Track A | 5 | 0.0935 | 0.1163 | [0.0233, 0.1662] | — | — | — |
| demo5 | Silero raw | 5 | 0.0305 | 0.0000 | [0.0000, 0.0915] | — | — | — |
| demo5 | Silero+frontend | 5 | **0.0169** | **0.0000** | [0.0000, 0.0508] | — | — | — |
| val100 | Track A | 100 | 0.0319 | 0.0224 | [0.0249, 0.0390] | — | — | — |
| val100 | Silero raw | 100 | **0.0078** | **0.0000** | [0.0040, 0.0131] | — | — | — |
| val100 | Silero+frontend | 100 | **0.0078** | **0.0000** | [0.0040, 0.0131] | — | — | — |
| bank_entity v3 | Track A | 315 | 0.1818 | 0.0676 | [0.1233, 0.2600] | 0.6238 | 0.3397 | — |
| bank_entity v3 | Silero raw | 315 | 0.5730 | 0.5938 | [0.5500, 0.5951] | 0.9734 | 0.9937 | — |
| bank_entity v3 | Silero+frontend | 315 | **0.0488** | **0.0000** | [0.0381, 0.0608] | **0.1030** | **0.1556** | — |
| homograph v3 | Track A | 719 | 0.0655 | 0.0256 | [0.0587, 0.0727] | — | — | 0.4437 |
| homograph v3 | Silero raw | 719 | **0.0068** | **0.0000** | [0.0050, 0.0089] | — | — | 0.5146 |
| homograph v3 | Silero+frontend | 719 | **0.0068** | **0.0000** | [0.0050, 0.0089] | — | — | 0.5146 |
| latin | Track A | 117 | 0.2386 | 0.1304 | [0.1221, 0.4520] | — | — | — |
| latin | Silero raw | 117 | 0.3338 | 0.3226 | [0.3154, 0.3533] | — | — | — |
| latin | Silero+frontend | 117 | **0.0418** | **0.0000** | [0.0312, 0.0527] | — | — | — |

**Сравнение систем** = Silero+frontend vs Track A (общий фронтенд). На
`bank_entity` v3 Silero+frontend **лучше** Track A: DER 0.1030 против 0.6238,
mean CER 0.0488 против 0.1818. На `latin` то же: 0.0418 против 0.2386. Нельзя
утверждать победу Track A без указания условия.

**Вклад фронтенда** = Silero raw vs Silero+frontend на `bank_entity` v3:
DER 0.9734 → 0.1030, Δ DER = **−0.8704** (наш TN+verbalize).

Жирным — меньшая ошибка внутри тройки на строке набора. SA только диагностика
(порог валидности 0.90 не достигнут; см. B05 `out/stress_classifier_eval.json`).

Listen demo5 (Windows mirror):
`C:\projects\tts-self\out\n2_first_execution\{silero_raw,silero_frontend,track_a}\demo5\`.

### Bank DER/EER delta 2×2 (labels × frontend) — N2 / R6

Один ckpt `runs/mel_stress_v01_s4/best.pt`. Ячейки:

| Frontend | Labels | Hyps source | Role |
|----------|--------|-------------|------|
| after R3 (current verbalize) | v3 | R2 Track A `out/external_baseline_track_a_v3.json` | re-score / live |
| after R3 | v2 (`git show 2c4c4bd:…/bank_entity.json`) | same R2 hyps | label-only re-score |
| before R3 (P1-era synth) | v2 | `out/baseline_p1_v2_full.json` | label-only re-score |
| before R3 | v3 | same P1 hyps | label-only re-score |

Артефакт: `out/bank_delta_2x2.json`. Все четыре ячейки прогнаны **одним** текущим
`critical_spans` scorer (внутренне согласованы). Абсолютный DER для
P1 hyps+v2 labels при пересчёте (0.6566) отличается от агрегата, записанного в
`baseline_p1_v2_full.json` (0.6142); в разложении ниже используются только
числа одного scorer. CI95 — utterance-cluster bootstrap (2000), DER как
edit-distance ratio может иметь верхнюю границу > 1.

| Cell | n | DER [CI95] | EER [CI95] | mean CER |
|------|--:|------------|------------|---------:|
| frontend_before × labels_v2 | 315 | 0.6566 [0.2209, 1.4163] | 0.4222 [0.3683, 0.4762] | 0.1907 |
| frontend_before × labels_v3 | 315 | 0.6370 [0.1919, 1.4079] | 0.4730 [0.4159, 0.5270] | 0.1907 |
| frontend_after × labels_v2 | 315 | 0.6493 [0.2135, 1.4118] | 0.3937 [0.3397, 0.4476] | 0.1818 |
| frontend_after × labels_v3 | 315 | 0.6238 [0.1776, 1.3958] | 0.3397 [0.2857, 0.3905] | 0.1818 |

**Наблюдаемая дельта** (before×v2 → after×v3):

| Metric | Observed Δ | Label @ before | Label @ after | Frontend @ v2 | Frontend @ v3 | Path check | Unexplained remainder |
|--------|-----------:|---------------:|--------------:|--------------:|--------------:|-----------:|----------------------:|
| DER | −0.0329 | −0.0196 | −0.0255 | −0.0073 | −0.0133 | −0.0329 | **0.0** |
| EER | −0.0825 | +0.0508 | −0.0540 | −0.0286 | −0.1333 | −0.0825 | **0.0** |

Label avg (DER) −0.0225; frontend avg −0.0103; interaction −0.0059.
Сумма вкладов по пути `label_before + frontend_v3` совпадает с observed;
необъяснённый остаток = 0.

`val100` / `latin`: между P1 v2 и R2 Track A `ref_norm` не менялся (0/100 и
0/117). Несравнимость CER относится к историческому шагу **v1→v2** (см. оговорку
выше), не к этому 2×2.

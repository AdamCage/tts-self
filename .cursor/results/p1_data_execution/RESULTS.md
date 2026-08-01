# Результаты — Track N, фаза P1 «Данные»

Дата отчёта: 2026-08-01.  
План: [`.cursor/plans/p1_data_execution.plan.md`](../../plans/p1_data_execution.plan.md)  
Slug: `p1_data_execution`  
Author copy git SHA: `66445e5` (`feat(p1): build audited multi-corpus data pipeline`)  
Методология: читалась из `.cursor/methodology/` (в ходе реализации **не** редактировалась).

Числа ниже только из артефактов на диске.

> **Ревью:** результаты фазы независимо перепроверены — см. [`REVIEW.md`](REVIEW.md).
> Все числа этого отчёта подтвердились, но найдены три дефекта валидности
> (замороженные наборы Track N собраны до QC; labels `bank_entity` v2
> воспроизводят наш же вербализатор; Silero сравнивался на сыром тексте против
> вербализованного у Track A). Выводы ниже читать вместе с ревью.

---

## Статус гейта P1

**GO** — выходной гейт фазы данных выполнен:

- доступны объёмы для ступеней T1–T2 (пилот CV17: **67.5 clean hours**, порог ≥50);
- реестр происхождения / лицензий оформлен;
- дедуп и speaker-holdout работают на ≥2 корпусах;
- бенчмарк RU-Critical выпущен в v2; внешний baseline Silero измерен;
- N5 честно закрыт как **неизмеримый** (SA-детектор < 0.90).

**Новый TTS checkpoint в P1 не обучался** (Track A заморожен планом).  
Production ckpt для сравнения: `runs/mel_stress_v01_s4/best.pt`.

---

## Что сделано по задачам плана

| ID | Суть | Статус |
|----|------|--------|
| D00 | cleanup `out/_*`, roadmap wording | done |
| D01 | version + `items_sha256` eval sets, `scripts/54_eval_sets.py`, latin в suite | done |
| D02 | полный baseline Track A на v1 | done → `out/baseline_p1_v1_full.json` |
| D03 | инвентарь корпусов через Hub API | done → `docs/corpus_inventory.md`, `out/probe_corpus_inventory.json` |
| D04 | provenance registry + legal questions | done → `docs/corpus_registry.md`, `docs/legal_questions_p1.md` |
| D05–D08 | container_id, ingest, merge/dedup, frozen Track N sets | done |
| D09–D10 | CV17 download + ASR QC / clean hours | done → `out/pilot_qc.json` |
| D11 | E6 CTC recheck excluded hours | done → `out/e6_recheck.json` |
| D12 | telephony set + channel CER | done → `out/p1_telephony_channels.json` |
| F01–F02 | verbalize phones/codes/units; 15-category coverage | done |
| B01–B02 | bank_entity / homograph v2 audits | done |
| B03 | full baseline on v2 | done → `out/baseline_p1_v2_full.json` |
| B04 | Silero external baseline all critical sets | done → `out/external_baseline_silero_v2.json` |
| B05 | supervised stress classifier | done, **fail vs 0.90** → N5 unmeasurable |

Код: 57 files in `66445e5` (`scripts/54–64`, `src/minispeech/{data,frontend,eval}/`, configs, tests).  
`pytest -q`: 196 passed (WSL, per execution report).

---

## Данные

### Пилот Common Voice RU 17

Артефакт: `out/pilot_qc.json` (копия: `artifacts/pilot_qc.json`).

| Метрика | Значение |
|---------|---------:|
| Utterances | 46 783 |
| Hours (all) | 69.1003 |
| Speakers | 2 671 |
| Clean hours (CER≤0.15) | **67.5021** |
| Review hours | 0.8297 |
| Exclude hours | 0.7686 |
| Speaker overlap vs RUSLAN | 0 |
| `clean_hours_gate_50` | true |

### Merge / dedup

Артефакт: `out/merge_dedup_report.json`.

| Метрика | Значение |
|---------|---------:|
| Input rows | 68 425 |
| Active | 68 383 |
| Marked duplicate | 42 (text) |
| Priority | ruslan > common_voice_ru_17_hf_mirror |

### Telephony (Track A synth → channel)

Артефакт: `out/p1_telephony_channels.json`.

| Condition | mean CER |
|-----------|---------:|
| wideband | 0.08231 |
| G.711 µ-law | 0.08561 |
| Opus narrowband | 0.08332 |

---

## Бенчмарк и baselines

Источник таблиц: `docs/frozen_numbers_mel80.md` § «P1 full baselines»;  
JSON: `out/baseline_p1_{v1,v2}_full.json`, `out/external_baseline_silero_v2.json`.

Ckpt Track A: `runs/mel_stress_v01_s4/best.pt`, neural HiFi-GAN.  
**Прямое сравнение v1↔v2 недопустимо** (изменены labels и frontend).

### Track A — v2 (после аудита)

| Set | n | mean CER | DER | EER | SA (diag) |
|-----|--:|---------:|----:|----:|----------:|
| demo5 | 5 | 0.0935 | — | — | — |
| val100 | 100 | 0.0319 | — | — | — |
| bank_entity v2 | 315 | 0.1907 | 0.6142 | 0.3651 | — |
| homograph v2 | 740 | 0.0703 | — | — | 0.4614 |
| latin | 117 | 0.2386 | — | — | — |

Аудит: bank 44/315 labels; homograph 155 invalid pairs removed → 740 retained.

### Silero v4_ru (xenia) vs Track A на v2

| Set | Track A CER | Silero CER | Победитель (ниже CER/ошибка) |
|-----|------------:|-----------:|------------------------------|
| demo5 | 0.0935 | **0.0305** | Silero |
| val100 | 0.0319 | **0.0078** | Silero |
| bank_entity | **0.1907** (DER **0.614**) | 0.5654 (DER 0.974) | Track A — **условия неравны** |
| homograph | 0.0703 | **0.0094** | Silero |
| latin | **0.2386** | 0.3338 | Track A — **условия неравны** |

SA на homograph невалидна для гейта (см. B05).

**Оговорка по сравнению (добавлена после ревью).** Silero получал сырой текст
с арабскими цифрами (`model_input_text == text` у 1275 из 1277 строк), Track A —
текст после нашей TN и вербализации. Поэтому строки `bank_entity` и `latin`
не являются сравнением акустических систем; см. [`REVIEW.md`](REVIEW.md) § 4.3.

### B05 — супервизный SA-детектор

Артефакт: `out/stress_classifier_eval.json`.

| Split | n words | accuracy | CI95 |
|-------|--------:|---------:|------|
| train | 2142 | 0.8231 | [0.8063, 0.8386] |
| held-out | 459 | **0.8257** | [0.7883, 0.8577] |

Порог валидности: 0.90. `pass_ge_0_90=false` → **N5 неизмерим**.

---

## Listen / инференс (обязательные wav)

Каталог по slug плана: `out/p1_data_execution/`.  
Индекс: [`listen_index.json`](listen_index.json).  
Логи: [`logs/silero_listen_demo5_latin.log`](logs/silero_listen_demo5_latin.log), [`logs/track_a_listen_demo.log`](logs/track_a_listen_demo.log).

### Сторона A/B — demo5

| # | Silero | Track A |
|---|--------|---------|
| 1 | `out/p1_data_execution/silero/demo5/0001_demo_01.wav` | `out/p1_data_execution/track_a_listen_demo/01_demo_01.wav` |
| 2 | `out/p1_data_execution/silero/demo5/0002_demo_02.wav` | `out/p1_data_execution/track_a_listen_demo/02_demo_02.wav` |
| 3 | `out/p1_data_execution/silero/demo5/0003_demo_03.wav` | `out/p1_data_execution/track_a_listen_demo/03_demo_03.wav` |
| 4 | `out/p1_data_execution/silero/demo5/0004_demo_04.wav` | `out/p1_data_execution/track_a_listen_demo/04_demo_04.wav` |
| 5 | `out/p1_data_execution/silero/demo5/0005_demo_05.wav` | `out/p1_data_execution/track_a_listen_demo/05_demo_05.wav` |

Silero demo5 mean CER (этот listen-прогон): **0.0305**  
(`out/p1_data_execution/silero_listen_demo5_latin.json`).

Дополнительно: 117 latin wav — `out/p1_data_execution/silero/latin/`.

Полный метрический Silero на всех critical sets (без повторного listen в этом отчёте):  
`out/external_baseline_silero_v2.json`.

---

## Соответствие методологии / целям

Читались: `09_roadmap.md` (P1 gate), `08_evaluation.md` (версионирование, N5), `07_data.md`, `12_publications.md`.

| Цель | Оценка после P1 |
|------|-----------------|
| Научная новизна (бенчмарк/протокол) | **Продвинута**: v2 RU-Critical + Silero baseline + честный fail SA. Архитектурная новизна (Н1/Н2) — фазы P2–P4. |
| Качественная русская TTS-модель | **Не цель фазы**. Новый ckpt нет. На общем CER Silero сильнее; на bank/latin Track A лучше метрически, звучание по-прежнему baseline. |

---

## Провенанс прогонов (уточнено ревью)

- Канонический прогон B04 выполнен из **чистого worktree** `~/projects/tts-self-p1`,
  отведённого ровно на `66445e5` (рабочее дерево чистое, 196 тестов проходят).
- Остальные P1-прогоны шли из `~/projects/tts-self`, где HEAD = `cd08a26` и
  311 грязных файлов: P1-код попал туда копированием, а не `git pull`. Содержимое
  P1-модулей совпадает с коммитом с точностью до CRLF, но **`git_sha` в манифестах
  этих прогонов неверен**.
- Listen-прогон (`out/p1_data_execution/silero/`) тоже шёл из основного клона
  (устаревший `scripts/39_...`); для `demo5` и `latin` fallback не требуется, и
  агрегаты совпали с каноническими до последнего знака.

## Follow-ups

1. **P1.5 — ремонт валидности** (R1–R9 в [`REVIEW.md`](REVIEW.md) § 8.1): переморозить
   наборы Track N после QC, честное сравнение с Silero, `bank_entity` v3 с независимой
   разметкой, `homograph` v3 с валидатором согласования, починка дат/времени/дробей
   во фронтенде, восстановление провенанса.
2. Перемерить SA-детектор на CV17 с настоящим speaker-holdout — данные этой фазы
   снимают блокер B05 (2 671 диктор вместо одного).
3. P2 — токенизаторы; не размораживать Track A без явного плана. Вход в P2 требует R1.
4. Парные предпочтения / слушатели — для работы 1 из `12_publications.md` (P.808 по-прежнему отложен).
5. Agent scaffolding обновлён под wav/`results`/methodology rules (этот пакет результатов — первый по новой конвенции директорий).
6. Техдолг D00 снят только в WSL: на Windows остаются 65 файлов `out/_*`.

---

## Артефакты в этом каталоге

| Path | Содержание |
|------|------------|
| `RESULTS.md` | этот файл |
| `REVIEW.md` | независимое ревью фазы: соответствие плану и методологии, дефекты, оценка целей, дальнейшие шаги |
| `listen_index.json` | пути wav для A/B |
| `logs/` | console tee Silero + Track A listen |
| `artifacts/` | копии ключевых JSON из `out/` + `review_*.json` |

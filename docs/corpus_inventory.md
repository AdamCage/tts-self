# Инвентаризация русского аудио (P1 D03)

Дата проверки: 2026-08-01. Метод: API/карточки первоисточников; **аудио в
D03 не загружалось**. Машинный артефакт:
[`out/corpus_probe_v2.json`](../out/corpus_probe_v2.json).

Правило: точный размер записан в байтах только когда источник его отдаёт.
`—` означает отсутствие данных, а не оценку.

## Проверенные кандидаты

| Корпус / URL | Часы | Дикторы | Размер, байт | Gated | Лицензия источника | Допустимость обучения |
|---|---:|---:|---:|:---:|---|---|
| [RUSLAN](https://ruslan-corpus.github.io/) | 31.5486 | 1 | — (страница: 7 GB) | нет | CC BY-NC-SA 4.0 | research/NC; не pilot |
| [Golos](https://github.com/salute-developers/golos) | 1240 | — | 30,575,316,100 (HF mirror) | нет | custom; mirror: CC BY-NC 4.0 | **ЮРИСТ**: конфликт |
| [Common Voice 17 ru mirror](https://huggingface.co/datasets/fixie-ai/common_voice_17_0) | — для выбранных шардов; source validated 218 | 2858 (source index) | **1,734,268,016** | нет | CC0-1.0 | да; выбранный компромисс |
| [Common Voice 25 ru](https://mozilladatacollective.com/datasets/cmn2h1dg201gro107lpynbbd6) | 290.51 / 251.94 validated | 3695 | — | **да** (MDC terms/API key) | CC0-1.0 | да, но доступ не подтверждён |
| [OpenSTT](https://github.com/snakers4/open_stt) | 21852 | — | 10,606,970,778 (metadata mirror) | нет | CC BY-NC 4.0 | research/NC |
| [SOVA RuDevices](https://huggingface.co/datasets/bond005/sova_rudevices) | 101 | —; `speaker_id` отсутствует | 46,754,947,325 | нет | CC BY 4.0 | да, но speaker holdout невозможен |
| [M-AILABS ru](https://www.openslr.org/94/) | 46.78 | 3 | 2,719,813,273 (filtered mirror) | нет | M-AILABS; mirror: CC BY-NC-ND 4.0 | **ЮРИСТ**: конфликт |
| [YODAS ru](https://huggingface.co/datasets/espnet/yodas) | 39622.289 | — | 36,567,268,396,871 (весь multilingual repo) | нет | CC BY 3.0 | **ЮРИСТ**: YouTube provenance |
| [RuLS](https://www.openslr.org/96/) | 98 | — | — (страница: 9.1G) | нет | Public Domain in USA | да |
| [Balalaika](https://huggingface.co/datasets/lab260/Balalaika) | 5078 | — | 0 (в Hub только metadata) | нет, но внешние ключи | CC BY-NC-ND 4.0 + source licenses | **ЮРИСТ** |
| [ESpeech Podcasts](https://huggingface.co/datasets/ESpeech/ESpeech-podcasts) | — | — | 508,497,718,812 | **manual** | CC BY-NC 4.0 | доступ не подтверждён |
| [ESpeech Webinars 2](https://huggingface.co/datasets/ESpeech/ESpeech-webinars2) | 850 | — | 152,310,745,995 | нет | Apache-2.0 в metadata / MIT в body | **ЮРИСТ**: конфликт |

Числа **не аддитивны**: Balalaika агрегирует OpenSTT/Golos/ESpeech и другие
источники; YouTube-домены пересекаются.

## Рекомендованный pilot

Выбран `common_voice_ru_17_hf_mirror` как **лучший компромисс**, а не как
строго прошедший все критерии:

- download: 1,734,268,016 байт < 60 GB;
- ingest скачанных шардов: **46,783 utts, 69.1003 ч, 2,671 диктор** по
  `client_id`; speaker-disjoint split: train 62.1834 ч, val 3.4617 ч,
  test 3.4552 ч;
- полный ASR/DSP QC при зафиксированных порогах дал **67.5021 clean-ч**
  (45,476 utts), 0.8297 review-ч (634 utts) и 0.7686 exclude-ч
  (673 utts); ошибок QC: 0.

Гейт T1 ≥ 50 clean-ч пройден без изменения порогов.

## Проверка E6: CTC-переразметка exclude

Пороги Whisper/CTC QC **не менялись**: `clean_max_cer = 0.15`,
`review_max_cer = 0.25` (как в pilot QC).

### Выборка P1 (200 / 673)

Локальная `wav2vec2-large-xlsr-53-russian` повторно проверила 200 из 673
исключённых utterance (0.2357 ч). В `clean` вернулись 118, в `review` — 21:
итого **139 utts / 0.1695 ч**, или **71.92 %** проверенных exclude-часов.
Машинный отчёт выборки: [`out/e6_recheck.json`](../out/e6_recheck.json).

### Полный прогон N2 (673 / 673)

Все **673** exclude-utterance (0.7686 ч) прогнаны тем же CTC и теми же
порогами. Вернулись **508 utts / 0.5951 ч**: 434 → `clean`, 74 → `review`;
остались в `exclude` 165; ошибок CTC: 0. Доля возврата по часам —
**77.44 %** (`returned_fraction = 0.7743606386144409`), выше выборки P1
71.92 % на **+5.52 п.п.** Машинный отчёт:
[`out/e6_full_cv17.json`](../out/e6_full_cv17.json).

Распределение расхождений Whisper-exclude ↔ CTC (по числу utts):

| Бакет | n | часы |
|---|---:|---:|
| whisper_exclude → ctc_clean | 278 | 0.3777 |
| whisper_exclude → ctc_clean, короткая числовая | 156 | 0.1370 |
| whisper_exclude → ctc_review | 62 | 0.0702 |
| whisper_exclude → ctc_review, короткая числовая | 12 | 0.0103 |
| both exclude, CTC CER ниже | 56 | 0.0667 |
| both exclude, CTC CER выше | 74 | 0.0788 |
| both exclude, CER равны | 27 | 0.0215 |
| both exclude, короткая числовая | 8 | 0.0064 |

Короткие числовые фразы (≤4 слова, есть числительное) составляют
**168 / 508** возвращённых utts — основной класс, где Whisper и CTC
систематически расходятся; автоматическая подмена тега здесь запрещена.

### Правило масштабирования для T2 / T3

1. Пороги Whisper-QC (`0.15` / `0.25`) **не ослаблять** по итогам CTC.
2. Любой корпус с ≥ ~0.5 ч (или ≥ ~200 utts) в `exclude` обязан пройти
   **второй голос CTC** на полном множестве exclude до решения о списании
   часов; выборка 200 недостаточна как финальный вердикт (P1 71.92 % →
   полный 77.44 %).
3. Возврат в train/val допускается только через явную очередь `review`
   (или ручной audit): CTC → `clean` — кандидат на возврат; CTC → `review`
   и короткие числовые — только после ручной проверки; CTC → `exclude` —
   оставить исключёнными.
4. На T2 (~500 ч) и T3 (тысячи ч) второй голос **обязателен** как этап
   пайплайна QC, а не опциональный аудит: ожидаемая перебраковка Whisper
   порядка **¾ exclude-часов** делает отказ от CTC систематической потерей
   данных.

## Телефонный домен (8 кГц)

| Источник | Часы | Native 8 kHz? | Открытость | Комментарий |
|----------|-----:|:-------------:|------------|-------------|
| OpenSTT `asr_public_phone_calls_*` + val | 601+211+7.7 = **819.7** | нет (ресемпл в 16 kHz) | CC BY-NC | Единственный крупный *открытый* phone-domain на русском в этой инвентаризации |
| CALLFRIEND Russian LDC2023S08 | ~48 | **да** | LDC, платно/членство | [catalog](https://catalog.ldc.upenn.edu/LDC2023S08) |
| Nexdata 93 h telephony | 93 | **да** (8 kHz a-law/u-law) | Commercial License | [product](https://www.nexdata.ai/datasets/speechrecog/1312) |

На замороженных 30 IVR-фразах Track A: mean CER **0.08231** (wideband),
**0.08561** (G.711 µ-law), **0.08332** (Opus narrowband). Парные дельты
mean CER vs wideband: G.711 **+0.00330**, Opus **+0.00101**. Точные нули
попозиционно: **20/30** (G.711) и **19/30** (Opus). Парный bootstrap CI95
дельты (10 000 ресемплов, seed 42) включает ноль в обоих случаях
(G.711 **[−0.03582, +0.03348]**, p_boot **0.913**; Opus
**[−0.03339, +0.02799]**, p_boot **0.979**); знаковый тест двухсторонний
p **0.109** / **0.227**. **Различие не обнаружено при n = 30.** Opus
выполнен через доступный codec backend. Полный отчёт:
[`out/p1_telephony_channels.json`](../out/p1_telephony_channels.json)
(`paired_significance`).

**Итог по телефонии:** свободного открытого русского корпуса с **сохранёнными** 8 кГц и ясной free/commercial лицензией **не найдено**. Критический пробел из [`07_data.md`](../.cursor/methodology/07_data.md) подтверждён: нужен собственный банковский материал + симуляция (N12).

## Достижимость ступеней T1 / T2 / T3

Ориентиры объёма: T1 ~50 ч, T2 ~500 ч, T3 тысячи ч неразмеченного аудио ([`07_data.md`](../.cursor/methodology/07_data.md)).

| Ступень | Достижима? | Обоснование | Чего не хватает |
|---------|:----------:|-------------|-----------------|
| **T1 (50 ч)** | **да** | CV mirror дал 67.5021 clean-ч и 2671 диктора после полного QC | Для CV-пилота пробела нет; у SOVA по-прежнему нет speaker IDs |
| **T2 (500 ч)** | **да для research; право продукта не закрыто** | Одного Golos 1240 ч физически достаточно, но его source/mirror license конфликтует | Юридическое решение по Golos; разнообразие каналов, особенно телефония 8 кГц |
| **T3 (тысячи ч)** | **да для research; право продукта не закрыто** | YODAS RU 39622.289 ч; OpenSTT 21852 ч (NC) | Дедупликация YouTube-источников; юридическая проверка provenance; **native 8 kHz** для N-Lite; NC-корпуса не закрывают коммерческий путь без соглашений |

### Коммерчески более «чистые» открытые объёмы (по заявлениям источников)

Без скачивания и без вычитания пересечений (верхняя оценка по каталогу, не рабочий бюджет):

- Golos 1240 + CV 25 290.51 + SOVA RuDevices 101 + YODAS RU 39622.289 +
  RuLS 98 + M-AILABS 46.78 → **тысячи часов формально доступны**, но часть
  источников заблокирована доступом или юридической проверкой.
- Для продукта вычитать NC и проверять ShareAlike / provenance YouTube.

## Не установлено достоверно

- Число дикторов: Golos, OpenSTT, SOVA, YODAS, Balalaika, ESpeech.
- Sample rate Golos в первоисточнике (только вторичный HF).
- Sample rate RuLS на OpenSLR.
- Лицензия ESpeech-podcasts (gated).
- Точный статус commercial use для LDC CALLFRIEND без чтения соглашения LDC.

## Команды / воспроизведение

Инвентаризация выполнена через `hf datasets info`, datasets-server, GitHub raw,
OpenSLR, Mozilla Data Collective и карточки источников. Команды и поле evidence
сохранены в `out/corpus_probe_v2.json`; локальных загрузок аудио в D03 не было.

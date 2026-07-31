# Инвентаризация русского аудио (зонд З3 / N03)

Дата: 2026-07-31.  
Метод: только веб-инвентаризация по первоисточникам; **корпуса не загружались**.  
Машинный артефакт: [`out/probe_corpus_inventory.json`](../out/probe_corpus_inventory.json).

Правило: каждое число со ссылкой на источник. Если в первоисточнике нет — `unknown` / «не установлено», без оценок.

## Сводная таблица кандидатов

| Корпус | Часы (источник) | Дикторы | SR / канал | Лицензия | Коммерч. обучение | Транскрипты | Загрузка | 8 кГц тел. |
|--------|----------------:|--------:|------------|----------|-------------------|-------------|----------|------------|
| RUSLAN | >31 ([homepage](https://ruslan-corpus.github.io/)); у нас clean ~15.4 ([methodology](../.cursor/methodology/README.md)) | 1 | 22050 в локальном конфиге (`configs/data_ruslan_local.yaml`); на homepage SR не указан | [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/) | **нет** (NC) | да, TTS-аннотации | Google Drive, 7 Gb | нет |
| Golos | 1240; train 1227.4 + test 12.6 ([GitHub](https://github.com/sberdevices/golos/)) | unknown (анонимизированы, [arXiv](https://arxiv.org/abs/2106.10161)) | **не указан** в README/статье; вторично HF 16 kHz | custom ShareAlike; [портал](https://developers.sber.ru/portal/products/golos) допускает research+commercial; PDF: [en](https://github.com/sberdevices/golos/blob/master/license/en_us.pdf) | **да** (по порталу; условия ShareAlike — в PDF) | ручная | opus 20.5 GB или wav-шарды | нет |
| Common Voice ru v26 | 291.58 (validated 253.69) ([MDC](https://mozilladatacollective.com/datasets/cmqinj9g500vsnr07qf4hmr3j)) | 3744 | MP3; Hz на datasheet не указан | CC0-1.0 | **да** (CC0; запрет deanonymize / re-host) | scripted + votes | 6.58 GB MP3 | нет |
| OpenSTT | 20108 ([GitHub](https://github.com/snakers4/open_stt/)) | unknown | mono 16 kHz int16 (нормализация) | CC BY-NC 4.0; commercial по соглашению | **нет** по умолчанию | смешанные (align/ASR/subs/manual val) | 356G opus / 2.3 TB wav | домен phone есть (~819.7 ч), **но SR=16 kHz** |
| SOVA (RU части) | 298+101+17451 = **17850** ([GitHub](https://github.com/sovaai/sova-dataset)) | unknown | 16 kHz, mono, 16-bit | CC BY 4.0 | **да** | manual / ASR по частям | Yandex Disk; RuYoutube 1873 Gb | нет |
| M-AILABS ru | 46h 47m ([mirror](https://github.com/i-celeste-aurora/m-ailabs-dataset)); 46.78 ([MFA](https://mfa-models.readthedocs.io/en/latest/corpus/Russian/M-AILABS%20Russian.html)) | 3 (MFA index) | wav mono 16 kHz | custom BSD-like, commercial OK | **да** | metadata.csv / normalized | ~3.6 GiB DL / 5.1 GiB | нет |
| YODAS ru | **39622.289** (сумма шардов `ru000`…`ru106` в [HF README](https://huggingface.co/datasets/espnet/yodas)) | unknown | 16 kHz (пример в README); YODAS2 — 24 kHz | CC BY 3.0 | **да** (attribution) | YouTube captions (manual/auto) | HF streaming/shards | нет |
| RuLS (доп.) | ~98 / 98.25 ([OpenSLR 96](https://www.openslr.org/96/), [arXiv](https://arxiv.org/abs/2104.04896)) | 13 | unknown на OpenSLR | Public Domain (USA) | **да** | да | 9.1G tar | нет |
| Balalaika (доп.) | ~5078 ([arXiv](https://arxiv.org/abs/2507.13563)) | unknown | unknown | annotations CC BY-NC-ND 4.0 ([HF](https://huggingface.co/datasets/lab260/Balalaika)); аудио — лицензии источников | **нет** (NC) | rich ASR+stress+phones | HF + пайплайн | нет |
| ESpeech-podcasts (доп.) | 3200 ([HF](https://huggingface.co/datasets/ESpeech/ESpeech-podcasts)) | unknown | 44.1 kHz MP3 | **не установлено** (gated) | unknown | metadata claimed | HF gated | нет |
| ESpeech-webinars2 (доп.) | 850 ([HF](https://huggingface.co/datasets/ESpeech/ESpeech-webinars2)) | unknown | 44.1 kHz MP3 | card: Apache-2.0 / body: MIT — **конфликт** | вероятно да (permissive), проверить LICENSE-файл | metadata claimed | HF | нет |

Числа **не аддитивны**: Balalaika агрегирует OpenSTT/Golos/ESpeech и др.; YouTube-домены SOVA/YODAS/ESpeech пересекаются с высокой вероятностью.

## Телефонный домен (8 кГц)

| Источник | Часы | Native 8 kHz? | Открытость | Комментарий |
|----------|-----:|:-------------:|------------|-------------|
| OpenSTT `asr_public_phone_calls_*` + val | 601+211+7.7 = **819.7** | нет (ресемпл в 16 kHz) | CC BY-NC | Единственный крупный *открытый* phone-domain на русском в этой инвентаризации |
| CALLFRIEND Russian LDC2023S08 | ~48 | **да** | LDC, платно/членство | [catalog](https://catalog.ldc.upenn.edu/LDC2023S08) |
| Nexdata 93 h telephony | 93 | **да** (8 kHz a-law/u-law) | Commercial License | [product](https://www.nexdata.ai/datasets/speechrecog/1312) |

**Итог по телефонии:** свободного открытого русского корпуса с **сохранёнными** 8 кГц и ясной free/commercial лицензией **не найдено**. Критический пробел из [`07_data.md`](../.cursor/methodology/07_data.md) подтверждён: нужен собственный банковский материал + симуляция (N12).

## Достижимость ступеней T1 / T2 / T3

Ориентиры объёма: T1 ~50 ч, T2 ~500 ч, T3 тысячи ч неразмеченного аудио ([`07_data.md`](../.cursor/methodology/07_data.md)).

| Ступень | Достижима? | Обоснование | Чего не хватает |
|---------|:----------:|-------------|-----------------|
| **T1 (50 ч)** | **да** | Golos 1240, CV 291.58, RuLS 98.25, M-AILABS ~47, SOVA/YODAS многократно больше; многодикторность есть у CV (3744) и RuLS (13) | Реестр лицензий и отбор multi-speaker под манифест; у Golos число дикторов неизвестно |
| **T2 (500 ч)** | **да** | Одного Golos достаточно; коммерчески допустимые пути: Golos + CV + SOVA + YODAS + RuLS + M-AILABS | Разнообразие каналов, особенно телефония 8 кГц |
| **T3 (тысячи ч)** | **да** | SOVA RU ~17850; YODAS RU ~39622; OpenSTT 20108 (NC) | Дедупликация YouTube-источников; юридическая проверка Golos PDF / YouTube CC; **native 8 kHz** для N-Lite; NC-корпуса (OpenSTT, RUSLAN, Balalaika annotations) не закрывают коммерческий путь без соглашений |

### Коммерчески более «чистые» открытые объёмы (по заявлениям источников)

Без скачивания и без вычитания пересечений (верхняя оценка по каталогу, не рабочий бюджет):

- Golos 1240 + CV ~292 + SOVA RU 17850 + YODAS RU ~39622 + RuLS ~98 + M-AILABS ~47 → **тысячи часов формально доступны**.
- Для продукта вычитать NC и проверять ShareAlike / provenance YouTube.

## Не установлено достоверно

- Число дикторов: Golos, OpenSTT, SOVA, YODAS, Balalaika, ESpeech.
- Sample rate Golos в первоисточнике (только вторичный HF).
- Sample rate Common Voice ru v26 поклипово (datasheet молчит).
- Sample rate RuLS на OpenSLR.
- Лицензия ESpeech-podcasts (gated).
- Точный статус commercial use для LDC CALLFRIEND без чтения соглашения LDC.

## Команды / воспроизведение

Инвентаризация выполнена чтением публичных URL (GitHub raw, HF README, OpenSLR, Mozilla Data Collective, LDC, arXiv). Локальных загрузок аудио не было.

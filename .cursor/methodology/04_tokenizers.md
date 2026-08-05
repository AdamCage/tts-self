# 04 — Токенизаторы / кодеки

## Статус (2026-08)

| Роль | Компонент | Статус |
|------|-----------|--------|
| **Критический путь D1** | Frozen **Qwen3-TTS Tokenizer-12Hz** | Единственный акустический кодек train/infer D1 |
| Архив / baseline | VibeVoice continuous 7.5 Hz | E2 CER **0.073**; continuous N2 FAIL — не product |
| Будущая независимость | Свой кодек (continuous или discrete) | После D1 GO; блокер суверенного полного цикла |
| Исследовательский скелет | T0 `runs/tokenizer_t0_small/` | ~1 ч, copy-synth CER ≈ 0.84 — **не продукт** |

D1 **не** учит кодек. Encode TTS targets детерминированно; повторный encode
обязан быть bit-identical. Веса кодека регистрируются с версией, лицензией и
sha256.

## D1: frozen Qwen3-TTS Tokenizer-12Hz

### Почему дискретный чужой кодек сейчас

1. Continuous next-token diffusion на VibeVoice уже измерен и **провалил N2**
   (FR R² −6.64) — см. `.cursor/results/n2_first_execution/RESULTS.md`.
2. Собственный T0 до якоря E2 не дотягивает; ждать T2/T3 блокирует предъявляемый
   чекпойнт.
3. Открытый Apache-2.0 12Hz multi-codebook кодек даёт немедленный train path для
   **своего** Speech LM при честном national-component статусе
   ([02_legal.md](02_legal.md)).

Это **не** отменяет долгосрочную цель своего кодека. Это снимает блокеры с
критического пути content-модели.

### Спецификация (регистрировать в `TARGETS`)

Фактические числа — только после probe (`scripts/70_probe_qwen3_tts_tokenizer.py`);
ниже — контракт плана, помечать измеренное в манифесте.

| Параметр | Контракт D1 |
|----------|-------------|
| Тип | Discrete multi-codebook (RVQ-style), frozen |
| Sample rate | 24 kHz (подтвердить probe) |
| Frame rate | ~12 Hz (подтвердить probe) |
| Codebooks | 16 (CB0…CB15) |
| Trainable | **нет** — encode/decode only |
| Лицензия | Apache-2.0; attribution/NOTICE |
| Правовой статус в продукте | Открытый зарубежный компонент → D1 = national-component research, **не** sovereign |

Правило проекта: геометрия регистрируется в `src/minispeech/constants.py`
(`TARGETS`), как для `vibevoice64` и `mel80`. Параллельных констант не заводим.

### Hard gate D0 (до обучения LM)

Переиспользовать методику GO/NO-GO и те же позиции/ASR, что у существующих якорей.

| Решение | Критерий |
|---------|----------|
| GO | Copy-synth CER на E2-compatible наборе **не хуже 0.073**; `bank_entity` DER не деградирует относительно зафиксированного codec anchor |
| NO-GO | Хуже якоря / недетерминизм encode / telephony обвал — **обучение LM не начинается** |

Дополнительно измерять: codebook utilization, DER/EER, telephony degradation,
bit-identical re-encode. Любой decode пишет wav под
`out/d1_discrete_speech_lm/…`.

### Кэши кодов

- Детерминированное перекодирование текущего Track N train (~72.98 ч) в `[T,16]`
  integer shards.
- Старые VibeVoice latent / text-hidden caches **не** переиспользовать как
  train-path D1.
- Provenance: corpus/speaker/license, codec id+hash, text token IDs,
  offset-aligned stress/phoneme/span features, EOS.
- Train/infer packing — точное равенство на проверках.

## Архив: VibeVoice continuous (якорь и negative control)

| Факт | Значение | Артефакт |
|------|---------:|----------|
| Copy-synth CER (E2) | **0.073** | `out/latent_error_calibration.json` |
| Continuous N2 Gate | **FAIL**, FR R² −6.64 | `.cursor/results/n2_first_execution/RESULTS.md` |

VibeVoice остаётся:

- историческим якорем intelligibility кодека;
- отрицательным контролем для continuous diffusion;
- **не** продуктовым компонентом и **не** train-path D1.

## Будущее: свой кодек (независимость / суверенность)

Самый дорогой долгосрочный компонент. Обучается один раз на неразмеченном
аудио, затем замораживается. **Не блокирует D1.**

### Почему свои всё ещё нужны

- Суверенный полный цикл требует воспроизводимого обучения кодека
  ([02_legal.md](02_legal.md)).
- Гипотезы частоты кадров и узкой полосы (телефония) нельзя закрыть чужим
  замороженным 12Hz без своего трека.
- Риск лицензионной/поставочной зависимости от одного внешнего артефакта.

### Ориентир спецификации (план, не D1)

Исторический ориентир — σ-VAE / DAC из VibeVoice-линии; допустим и свой
discrete codec, если copy-synth и streaming гейты проходят.

| Параметр | Ориентир |
|----------|----------|
| Структура | Каузальный энкодер-декодер (streaming) |
| Вход | 24 кГц; узкополосный вариант отдельно |
| Частота кадров | Измерять (кандидаты 7.5 / 12.5 / 25 Гц) — после T0 |
| Целевая функция | DAC-подобные лоссы + дискриминаторы (`ОЦЕНКА`) |
| Приёмка | Phase 0 GO; copy-synth не хуже рабочих якорей на том же наборе/ASR; DER на `bank_entity` |

### Стадийность своего кодека (post-D1)

| Ступень | Данные | Цель |
|---------|--------|------|
| T0 | 1 ч | Пайплайн; сейчас CER ≈ 0.84 — FAIL как продукт |
| T1 | ~50 ч | Обобщение |
| T2 | ~500 ч | Устойчивость к каналу |
| T3 | тысячи часов | Продуктовый / суверенный кандидат |

Переход — только по критерию приёмки. Пока D1 идёт на Qwen3-TTS 12Hz, свой
кодек может готовиться **параллельно**, но не подменяет frozen component до GO.

## Семантический поток в D1 vs отдельный semantic tokenizer

В continuous-плане семантический токенизатор был отдельной ASR-прокси сетью.
В D1 роль «что сказано» несёт **CB0 (semantic codebook)** внутри multi-codebook
кодека + text prefix LM. Отдельный обучаемый semantic tokenizer **не** входит в
критический путь D1; может вернуться в research track своего кодека.

## Данные

Для D1 encode нужен размеченный/псевдоразмеченный Track N train (уже
speaker-disjoint, ~72.98 ч). Для будущего своего акустического кодека —
неразмеченное аудио с разнообразием каналов; телефонный домен обязателен для
N-Lite. Реестр лицензий — [02_legal.md](02_legal.md), [07_data.md](07_data.md).

500+ ч в D1 **не** добавлять (Golos/OpenSTT/YODAS заблокированы до юррешения).

## Чеклист реализации

### D1 (сейчас)

- [ ] Wrapper `qwen3_tts_wrapper.py` + регистрация в `TARGETS`
- [ ] Probe: bit-identical encode, copy-synth CER/DER, utilization, telephony
- [ ] Hard gate D0 PASS перед train LM
- [ ] Encode shards `[T,16]` с provenance; train/infer pack equality
- [ ] Манифест: codec version, license, sha256

### Свой кодек (будущее)

- [ ] Пайплайн неразмеченного аудио + реестр лицензий
- [ ] T0→T3 с Phase 0 на каждой ступени
- [ ] Сравнение с якорями на том же наборе/ASR
- [ ] Подмена frozen Qwen3-TTS только после GO — и явная смена правового статуса

# Методология MiniSpeech-RU — Track N

Рабочий свод по переходу от Track A (mel, from-scratch, ~10M) к **Track N** —
семейству собственной русской TTS на intelligibility-critical метриках.

Статус (2026-08-02): **активный критический путь — этап D1** (дискретный Speech LM).
Непрерывный next-token diffusion (LLM + DDPM-голова на VibeVoice-латентах) прошёл
измерительный пилот `n2_first_execution` и зафиксирован как **отрицательный
контроль**: Gate N2 FAIL (S1 FR R² **−6.64**, артефакт
`.cursor/results/n2_first_execution/RESULTS.md`). Продуктовым путём он не является.

Track A **заморожен как baseline** на `runs/mel_stress_v01_s4/best.pt`
(2026-07-31): дальнейшие улучшения не ведутся, планку натуральности задаёт
внешний baseline Silero. См. [09_roadmap.md](09_roadmap.md).

## Что такое Track N сейчас

| Слой | Контракт D1 | Долгосрочная независимость |
|------|-------------|----------------------------|
| Акустика | Замороженный Apache-2.0 **Qwen3-TTS Tokenizer-12Hz** (16 codebooks, ~12 Гц); веса кодека не обучаются | Собственный кодек (непрерывный или дискретный) — отдельный будущий этап; T0-скелет `runs/tokenizer_t0_small/` остаётся исследовательским артефактом |
| Модель | Собственный causal Speech LM ~250–400M (дефолт ~300M) + свои semantic/MTP heads | Масштабирование данных и бэкбона после гейтов D1 |
| Правовой статус этапа | **Исследовательский national-component path** — открытый зарубежный компонент (токенизатор) допустим; **суверенный полный цикл на D1 не заявляется** | Суверенность возможна только после замены чужого кодека на свой воспроизводимый |

Мы **не** дообучаем чужие веса Speech LM / TTS. Мы обучаем **свою** модель на
замороженном открытом кодеке и сравниваем с внешними верхними границами
(Qwen3-TTS 0.6B/1.7B, CosyVoice3, Silero).

## TL;DR стратегии

| Вопрос | Ответ |
|--------|-------|
| Зачем | Открытого качественного русского TTS нет; критичны числа/реквизиты/ударения, а не MOS |
| Что строим сейчас | **D1**: свой ~300M discrete Speech LM + frozen Qwen3-TTS 12Hz tokenizer |
| Что архивировано | Continuous N2/DDPM на VibeVoice — отрицательный результат (не масштабировать) |
| Чужие веса | Кодек — frozen component; полные чужие TTS — только baseline / верхняя граница |
| Научный тезис | Intelligibility-critical TTS: метрика — семантическая точность, а не MOS |
| Правовой статус D1 | National-component research; не sovereign full-cycle |
| Что переносим из Track A | Фронтенд, бенчмарк, гейты, ASR-обвязка, культура измерений |
| Что становится baseline | mel-модель `runs/mel_stress_v01_s4` + внешний Silero |

## Навигация

| Файл | Содержание |
|------|-----------|
| [01_strategy.md](01_strategy.md) | Мотивации, тезис, развилки, перенос активов |
| [02_legal.md](02_legal.md) | 243-ФЗ: суверенная vs национальная, влияние на архитектуру |
| [03_architecture.md](03_architecture.md) | Целевая архитектура D1 + архив continuous |
| [04_tokenizers.md](04_tokenizers.md) | Frozen Qwen3-TTS 12Hz сейчас; свой кодек — будущее |
| [05_backbone.md](05_backbone.md) | Собственный ~300M Speech LM; text-init из Qwen |
| [06_frontend.md](06_frontend.md) | Русский фронтенд и критические спаны |
| [07_data.md](07_data.md) | Корпуса, объёмы, псевдоразметка, лицензии |
| [08_evaluation.md](08_evaluation.md) | Бенчмарк RU-Critical и гейты N/D-серии |
| [09_roadmap.md](09_roadmap.md) | Фазы: D1 сейчас, свой кодек позже |
| [10_compute.md](10_compute.md) | Железо и вычислительные бюджеты |
| [11_probes.md](11_probes.md) | Ближайшие измерительные зонды |
| [12_publications.md](12_publications.md) | Цикл статей и диссертация |

Исполняемый план текущего этапа:
[.cursor/plans/d1-discrete-speech-lm_860c29d5.plan.md](../plans/d1-discrete-speech-lm_860c29d5.plan.md).

Архив отрицательного continuous-пилота:
[.cursor/results/n2_first_execution/RESULTS.md](../results/n2_first_execution/RESULTS.md).

## Правила работы с этими документами

1. **Разделяем измеренное и предполагаемое.** Числа из наших прогонов помечены
   ссылкой на артефакт (`out/*.json`, `docs/*.md`, `.cursor/results/…`). Внешние
   числа — ссылкой на источник. Оценки помечены `ОЦЕНКА`.
2. **Не переписываем историю.** Track A и Part 1 — не ошибка, а измерение.
   Continuous N2 FAIL — не «неудачный эксперимент для замалчивания», а контроль,
   запрещающий масштабирование того же тракта. См. [01_strategy.md](01_strategy.md).
3. **Ни один архитектурный выбор без гейта.** Каждая фаза в
   [09_roadmap.md](09_roadmap.md) имеет критерий перехода.
4. Обновляем при получении новых измерений; помечаем дату в разделе.
5. Во время **исполнения** плана эти файлы **read-only**; правки — только при
   планировании нового этапа.

## Ключевые внешние источники

| Источник | Что берём |
|----------|-----------|
| Qwen3-TTS / Tokenizer-12Hz (Apache-2.0) | Frozen discrete codec для D1; полные модели 0.6B/1.7B — внешний baseline |
| CosyVoice3 | Внешний multi-speaker baseline |
| VibeVoice Technical Report (arXiv 2508.19205) | Исторический референс continuous-тракта; не продуктовый путь |
| LatentLM / next-token diffusion (arXiv 2412.08635) | Концепция, от которой D1 сознательно отходит после N2 FAIL |
| DAC (Descript Audio Codec) | Целевая функция для **будущего** своего кодека |
| GigaChat Family (arXiv 2506.09440), веса `ai-sage` под MIT | Кандидат в бэкбон для позднего N-Pro (после D1) |
| ФЗ от 26.07.2026 № 243-ФЗ | Правовые требования к статусам моделей |

## Собственные измерения-якоря

| Факт | Значение | Артефакт |
|------|---------:|----------|
| Токенизатор VibeVoice на русском (copy-synth CER) | **0.073** | `out/latent_error_calibration.json` (E2) |
| Talker с нуля на 15.4 ч — Gate 0 | **FAIL**, R² ≈ −0.11 | `docs/part1_diagnosis.md` |
| Continuous N2 S1 (VibeVoice + DDPM) — Gate N2 | **FAIL**, FR R² **−6.64** | `.cursor/results/n2_first_execution/RESULTS.md`, `out/n2_s1_ruslan.json` |
| Track N train (speaker-disjoint) | **72.98 ч / 51234 utt / 1088 spk** | `out/r1_freeze_result.json` |
| Track A mel stress S4, val100 CER | **0.031** | `out/gate_v4_v5_summary.json` |
| Track A mel stress S4, val100 UTMOS | **1.277** | там же (метрика невалидна для RU, см. 08) |
| Реальная запись человека, val100 CER | **0.060** | `out/baseline_track_a_critical.json` — метрика насыщена (Ф4 в 08) |
| GT-мел через нейровокодер, val100 CER | **0.064** | там же |
| Истинный EER на `bank_entity` v1 после аудита | **0.264** [0.205, 0.332] | `docs/m01_eer_audit.md` — против отчётных 0.489 |
| Акустический детектор ударения, точность | **0.358** при пороге 0.90 | `out/m05_acoustic_sa_calibration.json` — измеритель SA невалиден |
| Пропускная способность 0.5B bf16 на 4080 Super | **≥ 14.8k tok/s** | `out/probe_lm_budget.json` — нижняя граница, GPU загружен на треть |

Первые три строки вместе обосновывают D1: кодек VibeVoice русский держит, но
обучение continuous seq-модели (с нуля в Part 1 и с diffusion-головой в N2) на
доступных данных **не дало content-сигнала**. Следующий измеряемый шаг —
дискретный Speech LM на замороженном 12Hz кодеке при тех же гейтах CER/DER и
запрете масштабирования при FAIL oracle ladder.

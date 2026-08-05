# 05 — Backbone / Speech LM

## Роль модели в архитектуре D1

В дискретном Speech LM «бэкбон» — это **собственная causal модель** над
объединённой последовательностью text prefix + speech tokens. Она:

- понимает русский текст (BPE prefix, stressed natural Russian);
- планирует просодию и содержание;
- авторегрессионно предсказывает semantic codebook CB0;
- через hidden state обусловливает MTP-головы residual codebooks.

**Нам не нужен универсальный ассистент.** Нужна модель с хорошим пониманием
русского текста и способностью удерживать реплику. Целевой размер D1 —
**250–400M параметров** (дефолтная конфигурация ~300M); фактический param count
и trainable/frozen lists пишутся в `run_manifest.json`.

Отличие от архивного continuous N2: нет DDPM-головы по непрерывным латентам;
основной loss — CE по codebooks. Урок N2: teacher-forced loss не является
критерием промоушена.

## Контракт инициализации

| Блок | Политика D1 |
|------|-------------|
| Text embeddings / текстовые блоки Transformer | Разрешена инициализация из **Qwen3-0.6B**; coverage mapping фиксируется |
| Speech embeddings, modality adapters | **С нуля** |
| MTP / output heads | **С нуля** |
| Speaker-ID embeddings | **С нуля**; не zero-shot cloning |
| Frozen codec | Qwen3-TTS 12Hz — вне trainable графа |

Абляция на 5h после PASS base: Qwen text-init vs random-init при одинаковых
speech heads; масштабировать только победителя.

## Кандидаты и семейство

### D1: свой Speech LM ~300M (критический путь)

| Аспект | Оценка |
|--------|--------|
| Правовой статус | **National-component research** из-за frozen foreign codec; LM свой |
| Порог 1B (243-ФЗ) | Не проходит |
| Латентность | Целиться в телефонию позже (N8); D1 ставит content gates раньше TTFB |
| Данные этапа | ~72.98 ч multi-speaker + research-only RUSLAN CPT |
| Риск | Мало данных vs индустриальное pretrain; смягчение — oracle ladder + hard-stop |

### N-Lite: свой текстовый/речевой ~0.5B + свой кодек (будущее)

Историческая цель суверенности. Не блокирует и не подменяет D1.
Предусловия из M10 сохраняются как ориентир compute:

| Предусловие | Артефакт | Состояние |
|-------------|----------|-----------|
| tokens/s ~0.5B bf16 | `out/probe_lm_budget.json` (~14.8k tok/s) | нижняя граница |
| Инвентарь текста | `docs/text_corpus_inventory.md` | есть |
| Перемер при реалистичном батче | — | **не сделано** |
| Свой кодек GO | — | после D1 / parallel track |

### N-Pro: GigaChat (после D1)

| Аспект | Оценка |
|--------|--------|
| Лицензия | MIT (`ai-sage`) |
| Правовой статус | Национальный; суверенность — вопрос `ЮРИСТ` |
| Реалтайм на 16 ГБ | Нет (10B MoE минимум среди открытых) |
| Роль | Не-realtime качество / длинный контекст |

### Внешние полные TTS — только baseline

Qwen3-TTS 0.6B/1.7B, CosyVoice3, Silero — **не** продуктовый бэкбон D1.
Сравнительная таблица обязательна ([08_evaluation.md](08_evaluation.md),
[12_publications.md](12_publications.md)).

Предостережение: многие «российские» LLM — производные Qwen/Llama; для
суверенности выигрыша не дают. Проверять происхождение.

## Сравнительная таблица

| Модель | Параметры | Кодек | Реалтайм цель | Порог 1B | Правовой путь |
|--------|----------:|-------|:-------------:|:--------:|---------------|
| **D1 Speech LM** | ~300M | frozen Qwen3-TTS 12Hz | позже (N8) | нет | national-component research |
| N-Lite + свой кодек | ~0.5B | свой | да | нет | кандидат в суверенный |
| GigaChat Lightning | 10B MoE | — | нет | да | национальный |
| Qwen3-TTS 0.6B/1.7B | 0.6B / 1.7B | свой (чужой полный стек) | да | зависит | **только baseline** |
| Continuous N2 | пилот | VibeVoice | — | нет | **архив FAIL** |

## Токенизация текста: кириллица и ударения

Токенизаторы, оптимизированные под английский, дробят кириллицу. Для D1
текстовый BPE совместим с выбранным text-init (Qwen-семейство), но:

1. В BPE подаётся **stressed natural Russian** (U+0301 из `frontend/stress.py`),
   не phoneme string.
2. Основной путь ударения — **отдельный канал признаков** параллельно токенам
   (после text-only base). Решение P0/M11 сохраняется: строковый U+0301 как
   единственный канал отклонён (З1: intact share у Qwen = 0.0).
3. Baseline «только U+0301 в строке» допустим как абляция.

## Что уже измерено (зонды P0 / M)

- [x] Фертильность токенизаторов → `out/probe_tokenizer_fertility.json` (N02)
- [x] U+0301 не intact у Qwen/GigaChat → тот же JSON
- [x] Профиль GigaChat / Qwen3.5-0.8B → `out/probe_backbone_profile.json` (N09)
- [x] Бюджет 0.5B (нижняя граница) → `out/probe_lm_budget.json` (M10)

## Решения, актуальные для D1 (ревизия 2026-08)

### 1. Критический путь модели

**Решение:** собственный discrete Speech LM ~300M + MTP heads; не continuous
diffusion; не дообучение полного Qwen3-TTS.

Историческое решение M11 («P3 = Qwen3.5-0.8B + diffusion») **снято** после
N2 FAIL continuous-пилота. Qwen остаётся источником **text-init** и внешним
полным TTS-baseline, не trainable product backbone.

### 2. Форма передачи ударения

Без изменений: parallel feature channel (Н1); U+0301 в строке — абляция.

### 3. Кодек

**Решение D1:** frozen Qwen3-TTS 12Hz после D0 GO.
Собственный кодек — future independence path ([04_tokenizers.md](04_tokenizers.md)).
Выбор 7.5 Гц для T0 своего кодека (З2) остаётся в codec-track, не в D1.

### 4. Trainer

bf16, 1–2 GPU DDP, grad accumulation, resume (optimizer/scheduler/RNG),
free-running checkpoint selection, clean `git_sha`. EMA — только если измерен
выигрыш.

## Открытые вопросы

| Вопрос | Кому | Блокирует |
|--------|------|-----------|
| Достаточно ли ~300M + ~73 ч для N3 на hard-set | измерение D1 | GO в D2 (500h) |
| Нужен ли Qwen text-init vs random | абляция 5h | выбор init для full ladder |
| Засчитывается ли GigaChat как компонент суверенной модели | `ЮРИСТ` | N-Pro статус |
| Когда стартовать свой кодек относительно D1 | план | суверенный нарратив |
| Throughput ~300M на 4080 / 6000 Pro | измерение | календарь (`ОЦЕНКА` до probe) |

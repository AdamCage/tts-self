Да. На домашнем ПК с **RTX 4080/4080 Super 16 ГБ** можно сделать небольшой, но полноценный исследовательский прототип современной русской TTS-модели. Предполагаю, что речь именно об одной из этих карт: официальной RTX 4080 Ti у NVIDIA нет, а RTX 4080 и 4080 Super имеют 16 ГБ памяти. ([NVIDIA][1])

Причём это не будет игрушечная Tacotron-подобная модель. Можно реализовать почти все **архитектурные идеи** современных систем, но не их масштаб.

## Оптимальный проект

### MiniSpeech-RU, около 200–500 млн параметров

Первая версия:

* русский язык;
* один диктор и zero-shot клонирование голоса;
* эмоциональная и управляемая речь;
* потоковая генерация;
* 24 кГц;
* фразы до 15–30 секунд;
* обучение на одной видеокарте;
* последующее масштабирование на серверные GPU без смены архитектуры.

Небольшой масштаб реалистичен: у VibeVoice уже существует потоковая версия на 0,5 млрд параметров с заявленной задержкой первого аудио около 200 мс; MOSS-TTS выпустил Nano-версию примерно на 100 млн параметров; MiniMind-O показывает полный speech-native контур на масштабе около 0,1 млрд параметров. ([GitHub][2])

## Какую архитектуру делать

```text
Текст / фонемы
      ↓
Небольшой языковой backbone
      ↓
Speech-specific Talker
      ↓
Next-token flow/diffusion head
      ↓
Низкочастотные акустические латенты
      ↓
Потоковый аудиодекодер
      ↓
Речь
```

### 1. Готовый замороженный аудиотокенизатор

Самостоятельно обучать одновременно codec, языковую модель и генератор на 16 ГБ неразумно.

Следует взять открытый низкочастотный токенизатор и заморозить его:

* continuous tokenizer на 7,5 Гц, как в VibeVoice;
* tokenizer на 12,5 Гц, как в Qwen3-TTS или MOSS-TTS;
* либо сделать сравнительный эксперимент с двумя вариантами.

Низкая частота резко сокращает последовательность: современные модели используют 7,5–25 аудиотокенов в секунду вместо сотен фреймов. VibeVoice работает с непрерывными латентами на 7,5 Гц, Qwen3-TTS имеет варианты на 12,5 и 25 Гц, MOSS использует causal tokenizer на 12,5 кадра в секунду. ([arxiv.org][3])

### 2. Небольшой языковой backbone

Два реалистичных варианта:

**Быстрый MVP:**

* Qwen3-0.6B;
* 4-bit загрузка;
* LoRA/QLoRA;
* замороженная большая часть слоёв;
* короткий контекст.

**Более исследовательский вариант:**

* собственный decoder-transformer на 150–300 млн параметров;
* 12–18 слоёв;
* hidden size порядка 768–1024;
* обучение с нуля или дистилляция.

Для первого результата я бы взял Qwen3-0.6B и обучал только LoRA, проекторы и речевые слои. Qwen3 официально включает модель 0,6B и распространяется по Apache 2.0. ([arxiv.org][4])

### 3. Разделение Thinker и Talker

Нижние слои отвечают за текст и русский язык, верхние — за речь, тембр и просодию.

Можно реализовать:

* замороженный текстовый backbone;
* 4–6 дополнительных speech-слоёв;
* передачу скрытого состояния из среднего слоя;
* отдельный Talker, генерирующий акустические представления.

Подобное разделение используется в VibeVoice, а MiniMind-O показывает, что связь через промежуточный слой особенно важна для маленьких speech-native моделей. ([VibeVoice][5])

### 4. Next-token diffusion или flow matching

Это главная исследовательская часть.

Вместо обычного предсказания дискретного аудиотокена модель на каждом шаге генерирует **непрерывный акустический латент** через маленький diffusion/flow-head.

Для домашней версии:

* 4–6 слоёв;
* примерно 20–60 млн параметров;
* 4–8 шагов генерации при обучении и inference;
* classifier-free guidance;
* быстрый ODE/DPM solver;
* возможность последующей дистилляции до 1–2 шагов.

VibeVoice использует next-token diffusion, CosyVoice — chunk-aware flow matching, а IndexTTS сочетает авторегрессионный семантический генератор с flow/diffusion-компонентом. ([arxiv.org][3])

### 5. Разделение содержания, голоса и стиля

Нужно отдельно кодировать:

* текстовое содержание;
* идентичность диктора;
* эмоцию;
* темп;
* высоту голоса;
* продолжительность;
* степень выразительности.

Практичная схема:

```text
Содержание: фонемы или семантические латенты
Голос: один глобальный speaker embedding
Стиль: отдельные embeddings эмоции и просодии
Акустика: continuous latent sequence
```

Spark-TTS показывает полезную идею разделения низкобитрейтных семантических токенов и глобальных признаков диктора; DSA-Tokenizer отдельно разделяет семантическое и акустическое содержание. ([arxiv.org][6])

### 6. Потоковая генерация

Сразу нужно проектировать модель как streaming:

* текст поступает частями;
* Talker работает оконно;
* аудио декодируется блоками;
* используется KV-cache;
* следующий текстовый блок обрабатывается параллельно с декодированием предыдущего аудио.

Это ближе к VibeVoice Realtime и dual-track подходу Qwen3-TTS. Последний использует отдельные параллельные тракты и заявляет первый аудиопакет примерно через 97 мс в своей крупномасштабной системе. ([GitHub][2])

Для домашнего прототипа разумная первая цель — **300–700 мс до первого аудио**, а не попытка сразу повторить 97–200 мс.

## Что поместится в 16 ГБ

Реалистичная конфигурация:

| Компонент                    |   Размер | Обучение                          |
| ---------------------------- | -------: | --------------------------------- |
| Аудиотокенизатор             | 100–400M | заморожен                         |
| Qwen3-0.6B или свой backbone | 200–600M | LoRA либо частичная заморозка     |
| Speech Talker                |  30–100M | полностью                         |
| Diffusion/flow head          |   20–60M | полностью                         |
| Speaker/style encoders       |    5–30M | заморожены или частично обучаются |

Одновременно обучаемых параметров лучше оставить **в пределах 20–100 млн**.

Настройки:

* BF16;
* batch size 1–2;
* gradient accumulation;
* gradient checkpointing;
* Flash Attention;
* 8-bit optimizer;
* длина аудио сначала 3–8 секунд;
* заранее рассчитанные акустические латенты;
* сохранение датасета уже в токенизированном виде.

Самое важное — **не прогонять аудиокодек внутри каждого training step**. Один раз преобразовать аудио в латенты и дальше обучать модель на готовых массивах.

## Минимальный объём данных

Инженерная оценка:

* 10–30 часов одного чистого диктора — проверка архитектуры;
* 50–100 часов — нормальная одноголосая модель;
* 200–500 часов и десятки дикторов — базовое клонирование голосов;
* 1–3 тысячи часов — уже серьёзная многоголосая русская модель.

Телефония подходит для интонаций и диалоговой речи, но первую модель лучше обучить на чистом широкополосном аудио. Телефонный корпус подключать позже для разговорного стиля и устойчивости.

## Что на одной 4080 сделать нельзя

Нереалистично одновременно получить:

* обучение всей системы с нуля;
* собственный качественный аудиокодек;
* модель 1,5–7B;
* контекст на 60–90 минут;
* четыре собеседника;
* универсальное клонирование любого голоса;
* качество лидирующих систем, обученных на миллионах часов.

Например, у VibeVoice-1.5B только encoder и decoder акустического токенизатора имеют примерно по 340 млн параметров, а diffusion head — около 123 млн. Полное обучение такого контура на 16 ГБ нецелесообразно. ([huggingface.co][7])

## Какие инновации объединять не стоит

Не все современные идеи совместимы. Это альтернативы:

* continuous latents **или** дискретные codec tokens;
* diffusion/flow head **или** прямое авторегрессионное предсказание кодов;
* отдельный semantic tokenizer **или** единый acoustic tokenizer;
* максимальное качество **или** минимальная streaming latency;
* длинный многосторонний диалог **или** компактная realtime-модель.

Поэтому правильный набор для первого прототипа:

1. Низкочастотный замороженный аудиотокенизатор.
2. Небольшой текстовый backbone.
3. Отдельный Talker.
4. Continuous next-token flow/diffusion.
5. Speaker и style conditioning.
6. Русский фонетический frontend с ударениями.
7. Потоковая оконная генерация.
8. LoRA и дистилляция.
9. Возможность дальнейшего масштабирования без смены формата данных.

## Практический план

**Версия 0.1:** один голос, 20–50 часов, фразы до 8 секунд, frozen codec, Qwen3-0.6B + LoRA, обычный acoustic head.

**Версия 0.2:** заменить head на next-token flow matching, добавить streaming и управление темпом.

**Версия 0.3:** 20–50 дикторов, speaker encoder, zero-shot cloning, эмоции и стиль.

**Версия 0.4:** перенести ту же архитектуру на RTX 6000 Pro, увеличить backbone, данные и контекст.

Итог: на домашней RTX 4080 можно создать **небольшую оригинальную модель с современной архитектурой**, пригодную для статьи и демонстрации. Но инновационность должна быть в удачной комбинации и адаптации методов под русский язык, а не в попытке воспроизвести масштаб VibeVoice целиком.

[1]: https://www.nvidia.com/nl-nl/geforce/graphics-cards/40-series/rtx-4080-family/?utm_source=chatgpt.com "GeForce RTX 4080 SUPER en RTX 4080 grafische kaarten | NVIDIA"
[2]: https://github.com/microsoft/VibeVoice/blob/main/docs/vibevoice-realtime-0.5b.md?utm_source=chatgpt.com "VibeVoice/docs/vibevoice-realtime-0.5b.md at main · microsoft/VibeVoice · GitHub"
[3]: https://arxiv.org/abs/2508.19205?utm_source=chatgpt.com "VibeVoice Technical Report"
[4]: https://arxiv.org/abs/2505.09388?utm_source=chatgpt.com "Qwen3 Technical Report"
[5]: https://egarciaf2-vibevoice.mintlify.app/architecture/overview?utm_source=chatgpt.com "Architecture Overview - VibeVoice"
[6]: https://arxiv.org/abs/2503.01710?utm_source=chatgpt.com "Spark-TTS: An Efficient LLM-Based Text-to-Speech Model with Single-Stream Decoupled Speech Tokens"
[7]: https://huggingface.co/vibevoice/VibeVoice-1.5B?utm_source=chatgpt.com "vibevoice/VibeVoice-1.5B · Hugging Face"

Заменить **Qwen3-0.6B на Qwen3.5-0.8B-Base — хорошая идея**. Для компактной современной речевой модели я бы сейчас выбрал именно его.

Qwen3.5-0.8B имеет 24 слоя, hidden size 1024 и гибридную архитектуру: три слоя Gated DeltaNet чередуются с одним слоем полного attention. Модель рассчитана на длинные последовательности, имеет нативный контекст 262 тыс. токенов и распространяется по Apache 2.0. ([Hugging Face][1])

### Почему он интереснее для TTS

**1. Меньше KV-cache.**
Большая часть слоёв использует линейный attention с рекуррентным состоянием. Это потенциально полезно для потоковой речи и длинных последовательностей аудиотокенов.

**2. Подходит по размеру.**
Checkpoint занимает около 1,75 ГБ в BF16. Замороженная модель вместе с небольшим речевым модулем нормально помещается в 16 ГБ VRAM. ([Hugging Face][2])

**3. Уже мультимодально предобучен.**
Хотя аудио в исходном обучении отсутствует, модель изначально обучалась объединять токены разных модальностей. Это может облегчить добавление нового типа speech-токенов по сравнению с чистой текстовой моделью. Это гипотеза, которую нужно проверять экспериментально. ([Hugging Face][3])

**4. Base-версия специально предназначена для дообучения.**
Следует брать `Qwen3.5-0.8B-Base`, а не постобученную conversational-версию. Официальная карточка прямо позиционирует Base для fine-tuning и PEFT. ([Hugging Face][1])

## Как его использовать правильно

Не нужно загружать всю мультимодальную модель с vision encoder. В Transformers предусмотрен отдельный `Qwen3_5TextModel`, поэтому можно извлечь только языковую часть и использовать её как речевой backbone. ([GitHub][4])

Рекомендуемая конструкция:

```text
Текст / фонемы
      ↓
Qwen3.5-0.8B-Base text tower
      ↓
Speech projector
      ↓
Небольшой Talker, 4–6 слоёв
      ↓
Flow-matching / diffusion head
      ↓
Низкочастотные speech-латенты
      ↓
Потоковый codec decoder
```

### Что обучать на RTX 4080 16 ГБ

| Компонент                  | Режим                               |
| -------------------------- | ----------------------------------- |
| Qwen3.5-0.8B               | сначала полностью заморожен         |
| Speech projector           | полностью обучается                 |
| Talker 40–80M              | полностью обучается                 |
| Flow/diffusion head 20–40M | полностью обучается                 |
| Аудиокодек                 | заморожен                           |
| Speaker/style encoder      | заморожен или небольшой fine-tuning |

После получения baseline можно добавить LoRA в последние 6–12 слоёв Qwen3.5. Полное обучение всех 0,8B параметров с AdamW на 16 ГБ практически нецелесообразно: веса помещаются, но градиенты, состояния оптимизатора и активации — уже нет.

## Важное ограничение

Я бы **не заставлял Qwen3.5 напрямую генерировать акустические латенты всеми 24 слоями**. Gated DeltaNet экономичен, но пока не доказано, что его рекуррентное состояние хорошо сохраняет мелкие фонетические и просодические детали.

Разумнее разделить задачи:

* **Qwen3.5** понимает текст, контекст, стиль и длинные зависимости;
* **Talker с обычным causal attention** формирует речевую последовательность;
* **flow/diffusion head** создаёт детальные акустические латенты;
* **codec** восстанавливает звук.

Это позволит использовать инновации Qwen3.5, VibeVoice-подобных систем и современных flow-based TTS, не связывая качество акустики с экспериментальной linear-attention архитектурой.

## Предлагаемая конфигурация первой модели

* Qwen3.5-0.8B-Base text tower;
* замороженный codec на 7,5–12,5 латентов в секунду;
* Talker около 60M;
* flow head около 30M;
* общий размер системы около 1 млрд параметров;
* реально обучаемых параметров — 90–120M;
* короткие сегменты 4–8 секунд;
* BF16, gradient checkpointing, batch 1–2;
* предварительно рассчитанные speech-латенты;
* сначала один диктор, затем несколько голосов.

**Итог:** Qwen3.5-0.8B-Base лучше подходит для такого эксперимента, чем Qwen3-0.6B. Он немного крупнее, но всё ещё помещается на домашней видеокарте, современнее архитектурно и потенциально лучше подходит для потоковой генерации. Основной технический риск — Gated DeltaNet, поэтому Qwen3.5 стоит использовать как языковое ядро, а акустическую генерацию вынести в отдельный небольшой Talker.

[1]: https://huggingface.co/Qwen/Qwen3.5-0.8B-Base?utm_source=chatgpt.com "Qwen/Qwen3.5-0.8B-Base · Hugging Face"
[2]: https://huggingface.co/Qwen/Qwen3.5-0.8B-Base/tree/main?utm_source=chatgpt.com "Qwen/Qwen3.5-0.8B-Base at main"
[3]: https://huggingface.co/Qwen/Qwen3.5-0.8B?utm_source=chatgpt.com "Qwen/Qwen3.5-0.8B · Hugging Face"
[4]: https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3_5/configuration_qwen3_5.py?utm_source=chatgpt.com "transformers/src/transformers/models/qwen3_5/configuration_qwen3_5.py at main · huggingface/transformers · GitHub"

Для первого эксперимента скачайте **RUSLAN Stressed** и **Dialogs**. Это около 15–20 ГБ суммарно с запасом под кеш.

### Windows PowerShell

```powershell
mkdir D:\datasets\tts

py -m venv .venv
Set-ExecutionPolicy -Scope Process Bypass
.\.venv\Scripts\Activate.ps1

python -m pip install -U huggingface_hub datasets pyarrow soundfile
```

Проверить объём перед загрузкой:

```powershell
hf download stilletto/ruslan-stressed `
  --repo-type dataset `
  --dry-run
```

Скачать RUSLAN с размеченными ударениями:

```powershell
hf download stilletto/ruslan-stressed `
  --repo-type dataset `
  --local-dir "D:\datasets\tts\ruslan"
```

Этот вариант содержит исходное аудио RUSLAN и тексты с расставленными ударениями. Лицензия указана как CC BY 4.0. ([Hugging Face][1])

Скачать эмоциональные диалоги:

```powershell
hf download langswap/dialogs-ru-emotional-conversations `
  --repo-type dataset `
  --local-dir "D:\datasets\tts\dialogs"
```

В корпусе около 20,6 часа студийной речи трёх актёров с 12 стилями и эмоциями. ([Hugging Face][2])

### Дополнительные корпуса

Russian LibriSpeech:

```powershell
hf download istupakov/russian_librispeech `
  --repo-type dataset `
  --local-dir "D:\datasets\tts\russian_librispeech"
```

Это около 98 часов и 57 тысяч фрагментов; размер репозитория — примерно 11,2 ГБ. ([Hugging Face][3])

Biggest-Ru-Book Balalaika:

```powershell
hf download lab260/biggest_ru_book_balalaika `
  --repo-type dataset `
  --local-dir "D:\datasets\tts\balalaika"
```

Корпус содержит около 528 часов, но для первого домашнего эксперимента он избыточен. Данные поставляются в Parquet и архивах, которые затем нужно распаковать. ([Hugging Face][4])

YodaLingua Russian:

1. Открыть страницу корпуса и запросить доступ.
2. Создать токен Hugging Face.
3. Авторизоваться:

```powershell
hf auth login
```

4. Скачать:

```powershell
hf download Thomcles/YodaLingua-Russian `
  --repo-type dataset `
  --local-dir "D:\datasets\tts\yodalingua"
```

Корпус gated: без принятия условий команда вернёт ошибку доступа. Его размер сейчас указан как 2,79 ГБ. ([Hugging Face][5])

Команда `hf download` официально поддерживает репозитории типа `dataset`, локальную директорию, фильтры файлов и режим `--dry-run`. ([Hugging Face][6])

Для MVP достаточно начать с:

```text
RUSLAN: 20–30 часов
Dialogs: 5–10 часов
```

Не следует сразу скачивать и обрабатывать сотни часов: сначала нужно добиться разборчивой генерации на одном дикторе.

[1]: https://huggingface.co/datasets/stilletto/ruslan-stressed "stilletto/ruslan-stressed · Datasets at Hugging Face"
[2]: https://huggingface.co/datasets/langswap/dialogs-ru-emotional-conversations?utm_source=chatgpt.com "langswap/dialogs-ru-emotional-conversations · Datasets at Hugging Face"
[3]: https://huggingface.co/datasets/istupakov/russian_librispeech "istupakov/russian_librispeech · Datasets at Hugging Face"
[4]: https://huggingface.co/datasets/MTUCI/biggest_ru_book_balalaika "lab260/biggest_ru_book_balalaika · Datasets at Hugging Face"
[5]: https://huggingface.co/datasets/Thomcles/YodaLingua-Russian?utm_source=chatgpt.com "Thomcles/YodaLingua-Russian · Datasets at Hugging Face"
[6]: https://huggingface.co/docs/huggingface_hub/main/en/package_reference/cli "hf · Hugging Face"

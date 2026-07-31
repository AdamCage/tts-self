# Инвентаризация русских текстовых корпусов (Phase M / M10)

Дата: 2026-07-31.  
Метод: desk-research по первоисточникам; корпуса не скачивались. Измерения
токенов локальным токенизатором не выполнялись.

Важно: «слова» и «токены» ниже — единицы, заявленные владельцем корпуса, и
**не** являются числом токенов конкретного LLM-токенизатора. Для обучения
нужны отдельные provenance-фильтры, дедупликация, PII/NSFW policy, проверка
лицензий исходных текстов и оценка перекрытий между web-corpora.

## Сводная таблица

| Корпус / источник | Объём из источника | Лицензия / права | Можно обучать? | Как получить |
|---|---:|---|---|---|
| [OSCAR 23.01, `ru`](https://huggingface.co/datasets/oscar-corpus/OSCAR-2301) | 78,032,029,344 токенов; 194,143,422 документов; 1.1 TB ([release table](https://oscar-project.github.io/documentation/versions/oscar-2301/)) | Упаковка, metadata и annotations: [CC0-1.0](https://creativecommons.org/publicdomain/zero/1.0/). Сам текст — Common Crawl; OSCAR не владеет его copyright, применимы [Terms of Use](https://commoncrawl.org/terms-of-use/). | **Только после legal/provenance review**; не считать CC0 разрешением на весь web-текст. Доступ на HF gated/временно ограничен. | Запросить доступ на HF; использовать версию и фильтры, указанные в карточке. |
| [Community-OSCAR, `ru`](https://huggingface.co/datasets/oscar-corpus/community-oscar) | 80.68B токенов (593.06B до дедупликации в таблице карточки); 982.11 GiB | Annotations: CC0-1.0; текст — Common Crawl и его Terms; проект прямо указывает research-oriented доступ и возможный отзыв доступа. | **Условно, после review**; высок риск overlap с OSCAR/mC4 и неоднородных web-прав. | Gated HF dataset; принять условия и зафиксировать snapshot. |
| [Taiga](https://tatianashavrina.github.io/taiga_site/) | ~5B слов | [CC BY-SA 3.0](https://creativecommons.org/licenses/by-sa/3.0/), как указано проектом; у составляющих источников могут быть дополнительные условия. | **Да, с attribution/share-alike review**; пригодность конкретных подкорпусов проверять по их metadata. | Сайт проекта / linked repositories and manifests. |
| [Russian Public Domain](https://huggingface.co/datasets/PleIAs/Russian-PD) | 8,525 titles; 995,163,165 слов | Dataset card заявляет public domain во всех регионах; текущий отбор — произведения до 1884, с проверкой срока прав. | **Да, наиболее чистый литературный источник**, но домен исторический и стилистически узкий. | Hugging Face dataset, parquet shards. |
| [Русская Википедия / Wikimedia dumps](https://dumps.wikimedia.org/ruwiki/) | Токены: **unknown** (rolling dump; объём меняется) | Текст обычно [CC BY-SA 4.0](https://dumps.wikimedia.org/legal.html) и GFDL; соблюдать attribution/share-alike и project-specific exceptions. | **Да, с license compliance**; полезна как чистый энциклопедический домен, но недостаточна сама по себе. | Официальные `pages-articles` dumps; извлечь plain text и пересчитать токены выбранным tokenizer. |
| [mC4 / AllenAI C4, `ru`](https://huggingface.co/datasets/allenai/c4) | Русский объём в токенах: **unknown**; мC4 целиком 9.7 TB / 108 языков | [ODC-BY](https://opendatacommons.org/licenses/by/1-0/) для dataset + Common Crawl Terms для содержимого. | **Условно, после provenance/legal review**; web overlap с OSCAR вероятен. | `datasets.load_dataset("allenai/c4", "ru")`; не использовать deprecated `legacy-datasets/mc4`. |
| [Russian National Corpus (RNC)](https://ruscorpora.ru/en/) | 13.5B word tokens по сайту, включая General Internet Corpus | Не свободная training-лицензия: только non-commercial research/education; тексты можно использовать как цитаты/примеры, offline copy требует соглашение ([terms](https://ruscorpora.ru/en/page/terms/), [offline conditions](https://ruscorpora.ru/en/page/corpora-datasets/)). | **Нет для LM pretraining** без отдельного письменного разрешения правообладателя. | Web query interface; для offline — подписанное соглашение. |
| [ruTenTen 2020](https://www.sketchengine.eu/rutenten-russian-corpus/) | 23+B tokens / 19B words; 40M+ web pages | Лицензия downloadable text: **unknown**; это hosted Sketch Engine corpus, не заявленный открытый training dump. | **Нет по умолчанию**; возможен только после отдельного license/access review. | Sketch Engine account/service; не считать доступ к query UI правом на выгрузку/обучение. |

## Рекомендуемый порядок отбора

1. Начать с Russian-PD и Wikimedia: яснее lineage, но зафиксировать
   attribution и domain skew.
2. Добавлять Taiga только после manifest-level проверки её подкорпусов и
   выполнения CC BY-SA obligations.
3. Рассматривать OSCAR/Community-OSCAR/mC4 как альтернативные web-pools, а не
   аддитивные объёмы: сначала near-dedup и URL/domain-level overlap audit.
4. Не включать RNC и ruTenTen в training mix без отдельного разрешения.

## Что нужно измерить перед запуском обучения

- Получить exact snapshot и manifest каждого принятого источника.
- Прогнать очистку и дедупликацию до смешивания источников.
- Подсчитать **после** фильтрации число токенов production-tokenizer’ом,
  документы, bytes и долю каждого домена.
- Сохранить license/provenance ledger по shard/source URL, а не только
  aggregate-лицензию корпуса.

Не складывать строки таблицы: OSCAR, Community-OSCAR и mC4 построены из
Common Crawl и могут существенно пересекаться.

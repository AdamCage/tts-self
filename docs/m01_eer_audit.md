# M01 — Аудит EER на 50 падающих спанах

Источник: `C:/projects/tts-self/out/baseline_track_a_critical.json` (bank_entity).
Всего failed spans: **154** / 315 (reported EER=0.4889).
Выборка: **50**, seed=42.

## Категории

| Категория | n | доля |
|-----------|--:|-----:|
| synthesis_error | 27 | 54.00% |
| case_artifact | 0 | 0.00% |
| metric_artifact | 23 | 46.00% |

## Оценка «истинного» EER

- Reported EER: **0.4889**
- Доля synthesis_error в выборке: **54.00%** (bootstrap CI95 доли: [42.00%, 68.00%])
- Точечная оценка true EER ≈ reported × frac_synth = **0.2640**
- CI95 true EER ≈ **[0.2053, 0.3324]**

Главный артефакт метрики: Whisper часто оставляет **арабские цифры**, а `expected_spoken` — пословная вербализация; substring-EER считает это отказом.
Файлы наборов **не правились** (заморозка хешей).

## Рекомендация для M04

- Основной путь: выравнивание + DER по **словам-числительным** (не Arabic digits).
- Допуск морфологии: общий префикс ≥ 4 (case в выборке редок; digit verbalization доминирует).

## Спорные expected_spoken / metric_artifact (для следующей версии набора)

Всего metric_artifact в выборке: **23** (список в JSON).

## Примеры по категориям

### synthesis_error

- `bank_258` (number): src=`100 г` exp=`сто граммов`
  hyp=`Нависка старбак.` — high CER=0.400; span content missing or heavily corrupted
- `bank_174` (code): src=`771234567890` exp=`семь семь один два три четыре пять шесть семь восемь девять `
  hyp=`12-ти значный и ин-ин-ин.` — high CER=0.474; span content missing or heavily corrupted
- `bank_152` (account): src=`42301810000001234567` exp=`четыре два три ноль один восемь один ноль ноль ноль ноль нол`
  hyp=`Депозитный счет.` — high CER=0.583; span content missing or heavily corrupted
- `bank_208` (entity): src=`МФО` exp=`эм эф о`
  hyp=`заим...` — high CER=0.700; span content missing or heavily corrupted
- `bank_213` (latin): src=`IT` exp=`ай-ти`
  hyp=`обратитесь в отдел поитис.` — latin/acronym not realized as expected spoken form
- `bank_176` (code): src=`502401001` exp=`пять ноль два четыре ноль один ноль ноль один`
  hyp=`Комплект филиала 52 401 000.` — no lexical/digit overlap between expected_spoken and hyp

### case_artifact

_нет в выборке_

### metric_artifact

- `bank_271` (number): src=`2 ч` exp=`два часа`
  hyp=`Страк обрабатки 2.` — hyp preserves ≥70% of source digits (overlap=1.00); expected_spoken is verbalized words — substring EER fails
- `bank_304` (number): src=`30–45` exp=`от тридцать до сорок пять`
  hyp=`Доставка карты 3045 дней.` — hyp preserves ≥70% of source digits (overlap=1.00); expected_spoken is verbalized words — substring EER fails
- `bank_141` (account): src=`+7 846 111-22-33` exp=`плюс семь восемь четыре шесть один один один два два три три`
  hyp=`Самара, 7 миллионов, 846 тысяч, 111, 22, 33.` — hyp preserves ≥70% of source digits (overlap=1.00); expected_spoken is verbalized words — substring EER fails
- `bank_290` (number): src=`20-й` exp=`двадцатый`
  hyp=`20 транш` — hyp preserves ≥70% of source digits (overlap=1.00); expected_spoken is verbalized words — substring EER fails
- `bank_310` (number): src=`7–14` exp=`от семь до четырнадцать`
  hyp=`Возврат за 714 дней.` — hyp preserves ≥70% of source digits (overlap=1.00); expected_spoken is verbalized words — substring EER fails
- `bank_263` (number): src=`25 °C` exp=`двадцать пять градусов Цельсия`
  hyp=`Температура в зале 25 с...` — hyp preserves ≥70% of source digits (overlap=1.00); expected_spoken is verbalized words — substring EER fails


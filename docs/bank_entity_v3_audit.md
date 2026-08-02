# Bank entity v3 audit — independent labels

## Independence

expected_spoken is written from Russian language norm tables in this auditor; verbalize_text is used only for frontend_spoken_at_v3 and match-rate measurement. No label was changed to match the verbalizer.

Matching the frontend is a **measurable quantity**, not the definition of truth. Field `frontend_spoken_at_v3` snapshots `verbalize_text` at release time for match-rate computation only.

## Summary

- Items / spans audited: **315** / **315**
- Overall norm↔frontend match: **297/315** (0.9429)
- Labels changed from v2: **70**
- Remaining mismatches (frontend bug candidates): **18**
- Target version: `3.0.0`

## Norm source per category

| category | n | match | match_rate | labels changed from v2 | norm source |
|---|---:|---:|---:|---:|---|
| `accounts` | 21 | 19 | 0.9048 | 0 | Банковские реквизиты: поцифровое чтение номеров счетов и карт |
| `codes` | 21 | 21 | 1.0000 | 0 | Реквизиты ИНН/КПП/БИК/СНИЛС: поцифровое чтение |
| `dates` | 21 | 21 | 1.0000 | 21 | Розенталь § о датах: день — порядковое ср. р. («первое»); после с/до/от/по — род. п. («первого»); год — всегда род. п. порядкового («две тысячи двадцать шестого года»); месяц — род. п. |
| `decimals` | 21 | 21 | 1.0000 | 21 | Розенталь / «Русская грамматика» АН: дробные числительные «N целых M десятых/сотых»; не «точка» для количественных дробей; при целой части 1 — «одна целая» |
| `integers` | 22 | 22 | 1.0000 | 0 | Розенталь, «Справочник по правописанию и литературной правке», разделы о количественных числительных; школьная норма именит. падежа |
| `lat_abbr` | 21 | 18 | 0.8571 | 0 | Устоявшиеся русские чтения латиницы в финтехе/IT; для спорных API/KPI — русские имена букв А→а, К→кэ (не копирование EN «эй/кей») |
| `loanwords` | 21 | 21 | 1.0000 | 1 | Орфоэпические адаптации заимствований (Академос / узус СМИ); WhatsApp → «ватсап» |
| `money` | 21 | 21 | 1.0000 | 0 | Банковская/кассовая норма озвучивания сумм (рубль/рубля/рублей, копейка/копейки/копеек); именительная словарная форма реквизита |
| `ordinals` | 20 | 19 | 0.9500 | 1 | Розенталь: порядковые; падеж по управлению («со 100-й» → «сотым»); формы на -го — родительный |
| `percent` | 21 | 10 | 0.4762 | 11 | Розенталь: согласование процент/процента/процентов; при дробном числительном существительное в род. п. ед. ч. («процента»); после предлогов с/до/от/по — родительный числительного |
| `phones` | 21 | 21 | 1.0000 | 0 | Телефонная IVR-конвенция РФ: «плюс семь» / «восемь» + группы по 3 (и 4 для междугородних кодов) словами, не поцифрово |
| `ranges` | 21 | 21 | 1.0000 | 1 | Розенталь: количественные диапазоны после от/до/с — родительный; временны́е диапазоны — digit-style с падежным управлением |
| `ru_abbr` | 21 | 21 | 1.0000 | 0 | Устная норма аббревиатур: побуквенно, кроме устоявшихся слов (БИК→бик); справочная практика деловой речи |
| `time` | 21 | 20 | 0.9524 | 14 | Канон digit-style для IVR/банкинга (зафиксирован как норма набора): «девять ноль пять», «четырнадцать ноль ноль»; после с/до/от/по — род. п.; после «к» — дат. п.; ведущий ноль минут сохраняется |
| `units` | 21 | 21 | 1.0000 | 0 | Розенталь: сочетание числительного с единицами СИ (килограмм/килограмма/килограммов и т.п.) |

## Remaining norm↔frontend mismatches (frontend bugs — do not fix in R4)

| utt_id | category | fragment | expected_spoken (norm) | frontend_spoken_at_v3 |
|---|---|---|---|---|
| bank_065 | percent | 0.5% | ноль целых пять десятых процента | ноль целых пять десятых процентов |
| bank_067 | percent | 1% | один процент | один процентов |
| bank_068 | percent | 3,5% | три целых пять десятых процента | три целых пять десятых процентов |
| bank_070 | percent | 0,1% | ноль целых одна десятая процента | ноль целых одна десятая процентов |
| bank_074 | percent | 50% | пятидесяти процентов | пятьдесят процентов |
| bank_077 | percent | 2.5% | два целых пять десятых процента | два целых пять десятых процентов |
| bank_078 | percent | 9,9% | девяти целых девяти десятых процента | девять целых девять десятых процентов |
| bank_081 | percent | 4% | четыре процента | четыре процентов |
| bank_082 | percent | 6.5% | шесть целых пять десятых процента | шесть целых пять десятых процентов |
| bank_084 | percent | 30% | тридцати процентов | тридцать процентов |
| bank_085 | percent | 0,01% | ноль целых одна сотая процента | ноль целых одна сотая процентов |
| bank_115 | time | 14:20 | четырнадцати двадцати | четырнадцать двадцать |
| bank_159 | accounts | 47422810 | четыре семь четыре два два восемь один ноль | сорок семь миллионов четыреста двадцать два тысяч восемьсот десять |
| bank_161 | accounts | 42106810 | четыре два один ноль шесть восемь один ноль | сорок два миллионов сто шесть тысяч восемьсот десять |
| bank_220 | lat_abbr | API | а-пи-ай | эй-пи-ай |
| bank_223 | lat_abbr | KPI | кэ-пи-ай | кей-пи-ай |
| bank_227 | lat_abbr | ATM | эй-ти-эм | эй-тэ-эм |
| bank_284 | ordinals | 100-й | сотым | сотый |

## v2 numbers that become incomparable

- **bank_entity DER/EER (Track A and Silero) on v2 labels** — v2 expected_spoken reproduced verbalize_text for 298/315 (94.6%), including grammatically wrong year forms and «точка» decimals; v3 is an independent norm reference
- **v2 per-category 'label == frontend' rates (13/15 at 100%)** — Those rates measured circularity, not correctness
- **Label-only re-score artifacts using v2 gold (e.g. out/bank_entity_v1_hyp_v2_labels.json Δ DER)** — Gold definition changed; deltas vs v2 gold are not comparable to v3 gold without the R6 2×2 design

## R6 first-type error audit plan

After R6 recomputes DER/EER on v3 labels, sample rejected critical spans (false DER/EER hits) and classify: (a) ASR error, (b) legal pronunciation variant, (c) remaining frontend bug, (d) metric alignment artifact. Report the share of (d) as first-type metric artifacts. Not executed in R4.

## Label changes v2 → v3 (norm rewrite)

Total: **70**. Full list is in the JSON report (`label_changes_v2_to_v3`).

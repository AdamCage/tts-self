# Homograph v3 audit

## Acceptance summary

| Criterion | Value | Artifact |
|-----------|------:|----------|
| Morph analyzer | pymorphy3 (+ `pymorphy3-dicts-ru`) | `scripts/64_audit_homograph.py` |
| v2 not-agreed fraction | **175 / 740 = 0.23648648648648649** (23.65%) | `out/homograph_v3_audit.json` |
| v2 disputed | **0** | same |
| v3 not-agreed (programmatic) | **0** / 719 | same; recheck on written set |
| Fixed | **154** | same (`fixes`) |
| Deleted | **21** | same (`deletions`) |
| Manual sample after fixes | **100** (pairs exist + agreement OK) | this doc + report `manual_sample` |
| Version | **3.0.0** | `configs/eval_sets/homograph.json` |
| `54_eval_sets.py verify` | OK | `out/eval_set_hashes.json` |

Deletion reasons: `inflected_forms_break_homograph` 8, `stress_transfer_fallback` 7, `adverb_homograph_only_in_instrumental` 5, `inflection_noop` 1. Stress was never “fixed” to destroy a homograph pair; such items were deleted.

- Morphological analyzer: **pymorphy3**
- Version: 2.0.0 → **3.0.0**
- Items before (v2): **740**
- Items after (v3): **719**
- v2 not-agreed (programmatic): **175** / 740 = **0.23648648648648649** (23.65%)
- v2 status counts: `{'agreed': 565, 'not_agreed': 175}`
- v3 not-agreed (programmatic): **0**
- v3 status counts: `{'agreed': 719}`
- Fixed (form repaired, stress contrast kept): **154**
- Deleted: **21**
- Manual sample size: **100**
- Manual sample all pairs exist: **True**
- Manual sample agreement OK: **True**
- Manual sample all programmatic agreed: **True**

## Incomparable v2 numbers

Homograph v2 → v3 changes item inventory (740 → 719) and many surface forms, so these v2 figures must not be compared to v3:

- any `homograph` mean CER / SA / DER from v2 evals (e.g. Track A / Silero tables)
- conclusions in `docs/homograph_v2_audit.md` that “all sampled contexts resolve stress”
- `out/baseline_p1_v2_full.json` and `out/external_baseline_silero_v2.json` homograph rows
- v2 listen-demo wavs tied to old `utt_id` texts when the text was rewritten

## Method

Programmatic check uses **pymorphy3** to require that the target lexeme has a NOUN parse whose case/number/gender jointly match modifiers and/or preposition government (plus measure-genitive and noun-noun genitive dependents). Verb/adverb pairs are exempt from nominal agreement. Repairs inflect both members of the stress pair to the required case and re-apply distinguishing stress by vowel index; pairs that would cease to be stress-only homographs are deleted, never re-stressed into a non-homograph.

## Manual sample (after fixes)

Sample is stratified across pairs and prioritises previously bad categories from the P1 review (полки/пропасть/хлопок/среды/замок/…).

| utt_id | text | lexeme | contrast | prog | status |
|---|---|---|---|---|---|
| homo_211 | Книги расставили по деревянным по́лкам. | по́лкам | полка́м | agreed | checked |
| homo_111 | Тропа обрывается у глубокой про́пасти. | про́пасти | пропа́сти | agreed | checked |
| homo_876 | Футболка сшита из натурального хло́пка. | хло́пка | хлопка́ | agreed | checked |
| homo_796 | Занятия перенесли на эту сре́ду. | сре́ду | среду́ | agreed | checked |
| homo_002 | Экскурсия ведёт к древнему за́мку у реки. | за́мку | замку́ | agreed | checked |
| homo_012 | В подземельях за́мка хранили сокровища. | за́мка | замка́ | agreed | checked |
| homo_044 | В а́тласе не хватает карты Антарктиды. | а́тласе | атла́се | agreed | checked |
| homo_086 | Педаль орга́на управляет басовыми трубами. | орга́на | о́ргана | agreed | checked |
| homo_097 | Натюрморт с и́рисом висел в галерее. | и́рисом | ири́сом | agreed | checked |
| homo_108 | Я купил килограмм ири́са к чаю. | ири́са | и́риса | agreed | checked |
| homo_231 | В яйце много полезного белка́. | белка́ | бе́лка | agreed | checked |
| homo_268 | В мешке осталось мало муки́. | муки́ | му́ки | agreed | checked |
| homo_279 | Хозяйка добавила муку́ в соус. | муку́ | му́ку | agreed | checked |
| homo_364 | В прибрежных горо́дах пахнет морем. | горо́дах | го́родах | agreed | checked |
| homo_017 | Ключ не подходит к этому замку́. | замку́ | за́мку | agreed | checked |
| homo_032 | Ученик нашёл страну в школьном а́тласе. | а́тласе | атла́се | agreed | checked |
| homo_046 | Платье сшили из гладкого атла́са. | атла́са | а́тласа | agreed | checked |
| homo_050 | Подушка обтянута мягким атла́сом. | атла́сом | а́тласом | agreed | checked |
| homo_078 | Звук орга́на заполнил весь зал. | орга́на | о́ргана | agreed | checked |
| homo_003 | В музее показали макет королевского за́мка. | за́мка | замка́ | agreed | checked |
| homo_005 | Мы поднялись к развалинам старого за́мка. | за́мка | замка́ | agreed | checked |
| homo_006 | Фотограф снял силуэт за́мка на закате. | за́мка | замка́ | agreed | checked |
| homo_008 | Историк описал осаду за́мка в пятнадцатом веке. | за́мка | замка́ | agreed | checked |
| homo_011 | Стража охраняла ворота за́мка круглые сутки. | за́мка | замка́ | agreed | checked |
| homo_013 | Туман скрыл очертания за́мка. | за́мка | замка́ | agreed | checked |
| homo_014 | Книга начинается с описания за́мка на скале. | за́мка | замка́ | agreed | checked |
| homo_028 | Я забыл комбинацию от замка́. | замка́ | за́мка | agreed | checked |
| homo_030 | В магазине большой выбор замка́ для дверей. | замка́ | за́мка | agreed | checked |
| homo_037 | Учитель показал океан на развороте а́тласа. | а́тласа | атла́са | agreed | checked |
| homo_040 | Сверьтесь с а́тласом, прежде чем ехать. | а́тласом | атла́сом | agreed | checked |
| homo_048 | Ткань атла́са блестит на свету. | атла́са | а́тласа | agreed | checked |
| homo_049 | В магазине продают метры атла́са. | атла́са | а́тласа | agreed | checked |
| homo_051 | Лента из атла́са украсила коробку. | атла́са | а́тласа | agreed | checked |
| homo_052 | Шторы из плотного атла́са затемняют комнату. | атла́са | а́тласа | agreed | checked |
| homo_055 | Образец атла́са прислали из фабрики. | атла́са | а́тласа | agreed | checked |
| homo_056 | Юбка из атла́са шуршит при ходьбе. | атла́са | а́тласа | agreed | checked |
| homo_059 | Галстук из атла́са выглядит официально. | атла́са | а́тласа | agreed | checked |
| homo_060 | Скатерть из атла́са легко пачкается. | атла́са | а́тласа | agreed | checked |
| homo_062 | Суд является независимым о́рганом власти. | о́рганом | орга́ном | agreed | checked |
| homo_080 | Концерт для орга́на с оркестром начался в семь. | орга́на | о́ргана | agreed | checked |
| homo_081 | Трубы орга́на требуют настройки. | орга́на | о́ргана | agreed | checked |
| homo_082 | Органист сел за клавиатуру орга́на. | орга́на | о́ргана | agreed | checked |
| homo_083 | Запись орга́на Баха звучала торжественно. | орга́на | о́ргана | agreed | checked |
| homo_085 | Учитель музыки объяснил устройство орга́на. | орга́на | о́ргана | agreed | checked |
| homo_093 | Букет из и́риса стоял на столе. | и́риса | ири́са | agreed | checked |
| homo_099 | В ботаническом саду коллекция и́риса. | и́риса | ири́са | agreed | checked |
| homo_103 | На прилавке стоят банки с ири́сом. | ири́сом | и́рисом | agreed | checked |
| homo_104 | Вкус ири́са напоминает детство. | ири́са | и́риса | agreed | checked |
| homo_105 | Гости угостились шоколадным ири́сом. | ири́сом | и́рисом | agreed | checked |
| homo_632 | Инструктор приказал: «Пали́ по мишени!» | Пали́ | Па́ли | agreed | checked |
| homo_401 | Проверьте правильность а́дреса доставки. | а́дреса | адреса́ | agreed | checked |
| homo_417 | Рассылка ушла по старым адреса́м. | адреса́м | а́дресам | agreed | checked |
| homo_031 | На уроке географии открыли а́тлас мира. | а́тлас | атла́с | agreed | checked |
| homo_661 | В парке прыгают рыжие бе́лки. | бе́лки | белки́ | agreed | checked |
| homo_244 | Фотограф поймал кадр с бе́лком. | бе́лком | белко́м | agreed | checked |
| homo_233 | Диетолог рекомендовал растительный бело́к. | бело́к | бе́лок | agreed | checked |
| homo_341 | История древнего го́рода насчитывает века. | го́рода | горо́да | agreed | checked |
| homo_360 | В этих горо́дах развит туризм. | горо́дах | го́родах | agreed | checked |
| homo_362 | Список малых горо́дов расширили. | горо́дов | го́родов | agreed | checked |
| homo_281 | Вечером все сидели до́ма у камина. | до́ма | дома́ | agreed | checked |
| homo_302 | В этих дома́х живут семьи педагогов. | дома́х | до́мах | agreed | checked |
| homo_145 | Фильм показывает встречу с ду́хами. | ду́хами | духа́ми | agreed | checked |
| homo_131 | Она нанесла дорогие духи́ перед выходом. | духи́ | ду́хи | agreed | checked |
| homo_132 | Аромат духо́в наполнил комнату. | духо́в | ду́хов | agreed | checked |
| homo_562 | На шее вздулась синяя жи́ла. | жи́ла | жила́ | agreed | checked |
| homo_563 | Хирург перевязал повреждённую жи́лу. | жи́лу | жилу́ | agreed | checked |
| homo_001 | Туристы осмотрели средневековый за́мок на холме. | за́мок | замо́к | agreed | checked |
| homo_091 | На клумбе расцвёл синий и́рис. | и́рис | ири́с | agreed | checked |
| homo_151 | На полке стоят керамические кру́жки. | кру́жки | кружки́ | agreed | checked |
| homo_311 | Грибники ушли в густые леса́. | леса́ | ле́са | agreed | checked |
| homo_316 | Охотник знает тропы этих лесо́в. | лесо́в | ле́сов | agreed | checked |
| homo_431 | Ученик перенял приём у опытного ма́стера. | ма́стера | мастера́ | agreed | checked |
| homo_433 | К ма́стерам записались на ремонт. | ма́стерам | мастера́м | agreed | checked |
| homo_447 | Цех нанял новых мастеро́в. | мастеро́в | ма́стеров | agreed | checked |
| homo_443 | Награда досталась молодому ма́стеру. | ма́стеру | мастеру́ | agreed | checked |
| homo_251 | Его му́ка длилась долгие годы. | му́ка | мука́ | agreed | checked |
| homo_260 | Никто не знал о его внутренней му́ке. | му́ке | муке́ | agreed | checked |
| homo_252 | Повесть описывает му́ку пленных. | му́ку | муку́ | agreed | checked |
| homo_602 | С на́чала урока прошло десять минут. | на́чала | начала́ | agreed | checked |
| homo_061 | Парламент — высший законодательный о́рган страны. | о́рган | орга́н | agreed | checked |
| homo_371 | Экспедиция достигла берега о́строва. | о́строва | острова́ | agreed | checked |
| homo_394 | Шторм отрезал связь с дальними острова́ми. | острова́ми | о́стровами | agreed | checked |
| homo_392 | На этих острова́х нет постоянного населения. | острова́х | о́стровах | agreed | checked |
| homo_397 | Список необитаемых острово́в длинный. | острово́в | о́стровов | agreed | checked |
| homo_622 | Солдаты па́ли в неравном бою. | па́ли | пали́ | agreed | checked |
| homo_521 | Гости пи́ли чай до полуночи. | пи́ли | пили́ | agreed | checked |
| homo_212 | На кухонных по́лках стоят банки со специями. | по́лках | полка́х | agreed | checked |
| homo_213 | Пыль покрыла верхние по́лки. | по́лки | полки́ | agreed | checked |
| homo_851 | Рубашка промокла по́том после бега. | по́том | пото́м | agreed | checked |
| homo_112 | Альпинист смотрит вниз в про́пасть. | про́пасть | пропа́сть | agreed | checked |
| homo_582 | Птица се́ла на ветку у окна. | се́ла | села́ | agreed | checked |
| homo_599 | На карте отмечено старое село́. | село́ | се́ло | agreed | checked |
| homo_592 | Дорога ведёт к соседнему селу́. | селу́ | се́лу | agreed | checked |
| homo_812 | Рыбы приспособлены к пресной среде́. | среде́ | сре́де | agreed | checked |
| homo_799 | Отчёт сдаём до сре́ды. | сре́ды | среды́ | agreed | checked |
| homo_491 | Билет сто́ит тысячу рублей. | сто́ит | стои́т | agreed | checked |
| homo_171 | Часовые стре́лки показывают полдень. | стре́лки | стрелки́ | agreed | checked |
| homo_826 | Эта улица у́же, чем соседняя. | у́же | уже́ | agreed | checked |
| homo_462 | Память о добром учи́теле жива. | учи́теле | учителе́ | agreed | checked |
| homo_488 | В архиве списки бывших учителе́й. | учителе́й | учи́телей | agreed | checked |

## Fix summary (first 80)

```json
[
  {
    "utt_id": "homo_002",
    "action": "fix",
    "reason": "inflect_to_datv",
    "lexeme_before": "за́мок",
    "lexeme_after": "за́мку",
    "contrast_before": "замо́к",
    "contrast_after": "замку́",
    "text_before": "Экскурсия ведёт к древнему за́мок у реки.",
    "text_after": "Экскурсия ведёт к древнему за́мку у реки.",
    "char_index_after": 28,
    "cases": [
      "datv"
    ],
    "recheck": "agreed"
  },
  {
    "utt_id": "homo_003",
    "action": "fix",
    "reason": "inflect_to_gent",
    "lexeme_before": "за́мок",
    "lexeme_after": "за́мка",
    "contrast_before": "замо́к",
    "contrast_after": "замка́",
    "text_before": "В музее показали макет королевского за́мок.",
    "text_after": "В музее показали макет королевского за́мка.",
    "char_index_after": 37,
    "cases": [
      "gent"
    ],
    "recheck": "agreed"
  },
  {
    "utt_id": "homo_005",
    "action": "fix",
    "reason": "inflect_to_gent",
    "lexeme_before": "за́мок",
    "lexeme_after": "за́мка",
    "contrast_before": "замо́к",
    "contrast_after": "замка́",
    "text_before": "Мы поднялись к развалинам старого за́мок.",
    "text_after": "Мы поднялись к развалинам старого за́мка.",
    "char_index_after": 35,
    "cases": [
      "gent"
    ],
    "recheck": "agreed"
  },
  {
    "utt_id": "homo_006",
    "action": "fix",
    "reason": "inflect_to_gent",
    "lexeme_before": "за́мок",
    "lexeme_after": "за́мка",
    "contrast_before": "замо́к",
    "contrast_after": "замка́",
    "text_before": "Фотограф снял силуэт за́мок на закате.",
    "text_after": "Фотограф снял силуэт за́мка на закате.",
    "char_index_after": 22,
    "cases": [
      "gent"
    ],
    "recheck": "agreed"
  },
  {
    "utt_id": "homo_008",
    "action": "fix",
    "reason": "inflect_to_gent",
    "lexeme_before": "за́мок",
    "lexeme_after": "за́мка",
    "contrast_before": "замо́к",
    "contrast_after": "замка́",
    "text_before": "Историк описал осаду за́мок в пятнадцатом веке.",
    "text_after": "Историк описал осаду за́мка в пятнадцатом веке.",
    "char_index_after": 22,
    "cases": [
      "gent"
    ],
    "recheck": "agreed"
  },
  {
    "utt_id": "homo_011",
    "action": "fix",
    "reason": "inflect_to_gent",
    "lexeme_before": "за́мок",
    "lexeme_after": "за́мка",
    "contrast_before": "замо́к",
    "contrast_after": "замка́",
    "text_before": "Стража охраняла ворота за́мок круглые сутки.",
    "text_after": "Стража охраняла ворота за́мка круглые сутки.",
    "char_index_after": 24,
    "cases": [
      "gent"
    ],
    "recheck": "agreed"
  },
  {
    "utt_id": "homo_012",
    "action": "fix",
    "reason": "inflect_to_gent",
    "lexeme_before": "за́мок",
    "lexeme_after": "за́мка",
    "contrast_before": "замо́к",
    "contrast_after": "замка́",
    "text_before": "В подземельях за́мок хранили сокровища.",
    "text_after": "В подземельях за́мка хранили сокровища.",
    "char_index_after": 15,
    "cases": [
      "gent"
    ],
    "recheck": "agreed"
  },
  {
    "utt_id": "homo_013",
    "action": "fix",
    "reason": "inflect_to_gent",
    "lexeme_before": "за́мок",
    "lexeme_after": "за́мка",
    "contrast_before": "замо́к",
    "contrast_after": "замка́",
    "text_before": "Туман скрыл очертания за́мок.",
    "text_after": "Туман скрыл очертания за́мка.",
    "char_index_after": 23,
    "cases": [
      "gent"
    ],
    "recheck": "agreed"
  },
  {
    "utt_id": "homo_014",
    "action": "fix",
    "reason": "inflect_to_gent",
    "lexeme_before": "за́мок",
    "lexeme_after": "за́мка",
    "contrast_before": "замо́к",
    "contrast_after": "замка́",
    "text_before": "Книга начинается с описания за́мок на скале.",
    "text_after": "Книга начинается с описания за́мка на скале.",
    "char_index_after": 29,
    "cases": [
      "gent"
    ],
    "recheck": "agreed"
  },
  {
    "utt_id": "homo_017",
    "action": "fix",
    "reason": "inflect_to_datv",
    "lexeme_before": "замо́к",
    "lexeme_after": "замку́",
    "contrast_before": "за́мок",
    "contrast_after": "за́мку",
    "text_before": "Ключ не подходит к этому замо́к.",
    "text_after": "Ключ не подходит к этому замку́.",
    "char_index_after": 29,
    "cases": [
      "datv"
    ],
    "recheck": "agreed"
  },
  {
    "utt_id": "homo_028",
    "action": "fix",
    "reason": "inflect_to_gent",
    "lexeme_before": "замо́к",
    "lexeme_after": "замка́",
    "contrast_before": "за́мок",
    "contrast_after": "за́мка",
    "text_before": "Я забыл комбинацию от замо́к.",
    "text_after": "Я забыл комбинацию от замка́.",
    "char_index_after": 26,
    "cases": [
      "gent"
    ],
    "recheck": "agreed"
  },
  {
    "utt_id": "homo_030",
    "action": "fix",
    "reason": "inflect_to_gent",
    "lexeme_before": "замо́к",
    "lexeme_after": "замка́",
    "contrast_before": "за́мок",
    "contrast_after": "за́мка",
    "text_before": "В магазине большой выбор замо́к для дверей.",
    "text_after": "В магазине большой выбор замка́ для дверей.",
    "char_index_after": 29,
    "cases": [
      "gent"
    ],
    "recheck": "agreed"
  },
  {
    "utt_id": "homo_032",
    "action": "fix",
    "reason": "inflect_to_loct",
    "lexeme_before": "а́тлас",
    "lexeme_after": "а́тласе",
    "contrast_before": "атла́с",
    "contrast_after": "атла́се",
    "text_before": "Ученик нашёл страну в школьном а́тлас.",
    "text_after": "Ученик нашёл страну в школьном а́тласе.",
    "char_index_after": 31,
    "cases": [
      "loct"
    ],
    "recheck": "agreed"
  },
  {
    "utt_id": "homo_037",
    "action": "fix",
    "reason": "inflect_to_gent",
    "lexeme_before": "а́тлас",
    "lexeme_after": "а́тласа",
    "contrast_before": "атла́с",
    "contrast_after": "атла́са",
    "text_before": "Учитель показал океан на развороте а́тлас.",
    "text_after": "Учитель показал океан на развороте а́тласа.",
    "char_index_after": 35,
    "cases": [
      "gent"
    ],
    "recheck": "agreed"
  },
  {
    "utt_id": "homo_040",
    "action": "fix",
    "reason": "inflect_to_ablt",
    "lexeme_before": "а́тлас",
    "lexeme_after": "а́тласом",
    "contrast_before": "атла́с",
    "contrast_after": "атла́сом",
    "text_before": "Сверьтесь с а́тлас, прежде чем ехать.",
    "text_after": "Сверьтесь с а́тласом, прежде чем ехать.",
    "char_index_after": 12,
    "cases": [
      "ablt"
    ],
    "recheck": "agreed"
  },
  {
    "utt_id": "homo_044",
    "action": "fix",
    "reason": "inflect_to_loct",
    "lexeme_before": "а́тлас",
    "lexeme_after": "а́тласе",
    "contrast_before": "атла́с",
    "contrast_after": "атла́се",
    "text_before": "В а́тлас не хватает карты Антарктиды.",
    "text_after": "В а́тласе не хватает карты Антарктиды.",
    "char_index_after": 2,
    "cases": [
      "loct"
    ],
    "recheck": "agreed"
  },
  {
    "utt_id": "homo_046",
    "action": "fix",
    "reason": "inflect_to_gent",
    "lexeme_before": "атла́с",
    "lexeme_after": "атла́са",
    "contrast_before": "а́тлас",
    "contrast_after": "а́тласа",
    "text_before": "Платье сшили из гладкого атла́с.",
    "text_after": "Платье сшили из гладкого атла́са.",
    "char_index_after": 28,
    "cases": [
      "gent"
    ],
    "recheck": "agreed"
  },
  {
    "utt_id": "homo_048",
    "action": "fix",
    "reason": "inflect_to_gent",
    "lexeme_before": "атла́с",
    "lexeme_after": "атла́са",
    "contrast_before": "а́тлас",
    "contrast_after": "а́тласа",
    "text_before": "Ткань атла́с блестит на свету.",
    "text_after": "Ткань атла́са блестит на свету.",
    "char_index_after": 9,
    "cases": [
      "gent"
    ],
    "recheck": "agreed"
  },
  {
    "utt_id": "homo_049",
    "action": "fix",
    "reason": "inflect_to_gent",
    "lexeme_before": "атла́с",
    "lexeme_after": "атла́са",
    "contrast_before": "а́тлас",
    "contrast_after": "а́тласа",
    "text_before": "В магазине продают метры атла́с.",
    "text_after": "В магазине продают метры атла́са.",
    "char_index_after": 28,
    "cases": [
      "gent"
    ],
    "recheck": "agreed"
  },
  {
    "utt_id": "homo_050",
    "action": "fix",
    "reason": "inflect_to_ablt",
    "lexeme_before": "атла́с",
    "lexeme_after": "атла́сом",
    "contrast_before": "а́тлас",
    "contrast_after": "а́тласом",
    "text_before": "Подушка обтянута мягким атла́с.",
    "text_after": "Подушка обтянута мягким атла́сом.",
    "char_index_after": 27,
    "cases": [
      "ablt"
    ],
    "recheck": "agreed"
  },
  {
    "utt_id": "homo_051",
    "action": "fix",
    "reason": "inflect_to_gent",
    "lexeme_before": "атла́с",
    "lexeme_after": "атла́са",
    "contrast_before": "а́тлас",
    "contrast_after": "а́тласа",
    "text_before": "Лента из атла́с украсила коробку.",
    "text_after": "Лента из атла́са украсила коробку.",
    "char_index_after": 12,
    "cases": [
      "gent"
    ],
    "recheck": "agreed"
  },
  {
    "utt_id": "homo_052",
    "action": "fix",
    "reason": "inflect_to_gent",
    "lexeme_before": "атла́с",
    "lexeme_after": "атла́са",
    "contrast_before": "а́тлас",
    "contrast_after": "а́тласа",
    "text_before": "Шторы из плотного атла́с затемняют комнату.",
    "text_after": "Шторы из плотного атла́са затемняют комнату.",
    "char_index_after": 21,
    "cases": [
      "gent"
    ],
    "recheck": "agreed"
  },
  {
    "utt_id": "homo_055",
    "action": "fix",
    "reason": "inflect_to_gent",
    "lexeme_before": "атла́с",
    "lexeme_after": "атла́са",
    "contrast_before": "а́тлас",
    "contrast_after": "а́тласа",
    "text_before": "Образец атла́с прислали из фабрики.",
    "text_after": "Образец атла́са прислали из фабрики.",
    "char_index_after": 11,
    "cases": [
      "gent"
    ],
    "recheck": "agreed"
  },
  {
    "utt_id": "homo_056",
    "action": "fix",
    "reason": "inflect_to_gent",
    "lexeme_before": "атла́с",
    "lexeme_after": "атла́са",
    "contrast_before": "а́тлас",
    "contrast_after": "а́тласа",
    "text_before": "Юбка из атла́с шуршит при ходьбе.",
    "text_after": "Юбка из атла́са шуршит при ходьбе.",
    "char_index_after": 11,
    "cases": [
      "gent"
    ],
    "recheck": "agreed"
  },
  {
    "utt_id": "homo_059",
    "action": "fix",
    "reason": "inflect_to_gent",
    "lexeme_before": "атла́с",
    "lexeme_after": "атла́са",
    "contrast_before": "а́тлас",
    "contrast_after": "а́тласа",
    "text_before": "Галстук из атла́с выглядит официально.",
    "text_after": "Галстук из атла́са выглядит официально.",
    "char_index_after": 14,
    "cases": [
      "gent"
    ],
    "recheck": "agreed"
  },
  {
    "utt_id": "homo_060",
    "action": "fix",
    "reason": "inflect_to_gent",
    "lexeme_before": "атла́с",
    "lexeme_after": "атла́са",
    "contrast_before": "а́тлас",
    "contrast_after": "а́тласа",
    "text_before": "Скатерть из атла́с легко пачкается.",
    "text_after": "Скатерть из атла́са легко пачкается.",
    "char_index_after": 15,
    "cases": [
      "gent"
    ],
    "recheck": "agreed"
  },
  {
    "utt_id": "homo_062",
    "action": "fix",
    "reason": "inflect_to_ablt",
    "lexeme_before": "о́рган",
    "lexeme_after": "о́рганом",
    "contrast_before": "орга́н",
    "contrast_after": "орга́ном",
    "text_before": "Суд является независимым о́рган власти.",
    "text_after": "Суд является независимым о́рганом власти.",
    "char_index_after": 25,
    "cases": [
      "ablt"
    ],
    "recheck": "agreed"
  },
  {
    "utt_id": "homo_078",
    "action": "fix",
    "reason": "inflect_to_gent",
    "lexeme_before": "орга́н",
    "lexeme_after": "орга́на",
    "contrast_before": "о́рган",
    "contrast_after": "о́ргана",
    "text_before": "Звук орга́н заполнил весь зал.",
    "text_after": "Звук орга́на заполнил весь зал.",
    "char_index_after": 8,
    "cases": [
      "gent"
    ],
    "recheck": "agreed"
  },
  {
    "utt_id": "homo_080",
    "action": "fix",
    "reason": "inflect_to_gent",
    "lexeme_before": "орга́н",
    "lexeme_after": "орга́на",
    "contrast_before": "о́рган",
    "contrast_after": "о́ргана",
    "text_before": "Концерт для орга́н с оркестром начался в семь.",
    "text_after": "Концерт для орга́на с оркестром начался в семь.",
    "char_index_after": 15,
    "cases": [
      "gent"
    ],
    "recheck": "agreed"
  },
  {
    "utt_id": "homo_081",
    "action": "fix",
    "reason": "inflect_to_gent",
    "lexeme_before": "орга́н",
    "lexeme_after": "орга́на",
    "contrast_before": "о́рган",
    "contrast_after": "о́ргана",
    "text_before": "Трубы орга́н требуют настройки.",
    "text_after": "Трубы орга́на требуют настройки.",
    "char_index_after": 9,
    "cases": [
      "gent"
    ],
    "recheck": "agreed"
  },
  {
    "utt_id": "homo_082",
    "action": "fix",
    "reason": "inflect_to_gent",
    "lexeme_before": "орга́н",
    "lexeme_after": "орга́на",
    "contrast_before": "о́рган",
    "contrast_after": "о́ргана",
    "text_before": "Органист сел за клавиатуру орга́н.",
    "text_after": "Органист сел за клавиатуру орга́на.",
    "char_index_after": 30,
    "cases": [
      "gent"
    ],
    "recheck": "agreed"
  },
  {
    "utt_id": "homo_083",
    "action": "fix",
    "reason": "inflect_to_gent",
    "lexeme_before": "орга́н",
    "lexeme_after": "орга́на",
    "contrast_before": "о́рган",
    "contrast_after": "о́ргана",
    "text_before": "Запись орга́н Баха звучала торжественно.",
    "text_after": "Запись орга́на Баха звучала торжественно.",
    "char_index_after": 10,
    "cases": [
      "gent"
    ],
    "recheck": "agreed"
  },
  {
    "utt_id": "homo_085",
    "action": "fix",
    "reason": "inflect_to_gent",
    "lexeme_before": "орга́н",
    "lexeme_after": "орга́на",
    "contrast_before": "о́рган",
    "contrast_after": "о́ргана",
    "text_before": "Учитель музыки объяснил устройство орга́н.",
    "text_after": "Учитель музыки объяснил устройство орга́на.",
    "char_index_after": 38,
    "cases": [
      "gent"
    ],
    "recheck": "agreed"
  },
  {
    "utt_id": "homo_086",
    "action": "fix",
    "reason": "inflect_to_gent",
    "lexeme_before": "орга́н",
    "lexeme_after": "орга́на",
    "contrast_before": "о́рган",
    "contrast_after": "о́ргана",
    "text_before": "Педаль орга́н управляет басовыми трубами.",
    "text_after": "Педаль орга́на управляет басовыми трубами.",
    "char_index_after": 10,
    "cases": [
      "gent"
    ],
    "recheck": "agreed"
  },
  {
    "utt_id": "homo_093",
    "action": "fix",
    "reason": "inflect_to_gent",
    "lexeme_before": "и́рис",
    "lexeme_after": "и́риса",
    "contrast_before": "ири́с",
    "contrast_after": "ири́са",
    "text_before": "Букет из и́рис стоял на столе.",
    "text_after": "Букет из и́риса стоял на столе.",
    "char_index_after": 9,
    "cases": [
      "gent"
    ],
    "recheck": "agreed"
  },
  {
    "utt_id": "homo_097",
    "action": "fix",
    "reason": "inflect_to_ablt",
    "lexeme_before": "и́рис",
    "lexeme_after": "и́рисом",
    "contrast_before": "ири́с",
    "contrast_after": "ири́сом",
    "text_before": "Натюрморт с и́рис висел в галерее.",
    "text_after": "Натюрморт с и́рисом висел в галерее.",
    "char_index_after": 12,
    "cases": [
      "ablt"
    ],
    "recheck": "agreed"
  },
  {
    "utt_id": "homo_099",
    "action": "fix",
    "reason": "inflect_to_gent",
    "lexeme_before": "и́рис",
    "lexeme_after": "и́риса",
    "contrast_before": "ири́с",
    "contrast_after": "ири́са",
    "text_before": "В ботаническом саду коллекция и́рис.",
    "text_after": "В ботаническом саду коллекция и́риса.",
    "char_index_after": 30,
    "cases": [
      "gent"
    ],
    "recheck": "agreed"
  },
  {
    "utt_id": "homo_103",
    "action": "fix",
    "reason": "inflect_to_ablt",
    "lexeme_before": "ири́с",
    "lexeme_after": "ири́сом",
    "contrast_before": "и́рис",
    "contrast_after": "и́рисом",
    "text_before": "На прилавке стоят банки с ири́с.",
    "text_after": "На прилавке стоят банки с ири́сом.",
    "char_index_after": 28,
    "cases": [
      "ablt"
    ],
    "recheck": "agreed"
  },
  {
    "utt_id": "homo_104",
    "action": "fix",
    "reason": "inflect_to_gent",
    "lexeme_before": "ири́с",
    "lexeme_after": "ири́са",
    "contrast_before": "и́рис",
    "contrast_after": "и́риса",
    "text_before": "Вкус ири́с напоминает детство.",
    "text_after": "Вкус ири́са напоминает детство.",
    "char_index_after": 7,
    "cases": [
      "gent"
    ],
    "recheck": "agreed"
  },
  {
    "utt_id": "homo_105",
    "action": "fix",
    "reason": "inflect_to_ablt",
    "lexeme_before": "ири́с",
    "lexeme_after": "ири́сом",
    "contrast_before": "и́рис",
    "contrast_after": "и́рисом",
    "text_before": "Гости угостились шоколадным ири́с.",
    "text_after": "Гости угостились шоколадным ири́сом.",
    "char_index_after": 30,
    "cases": [
      "ablt"
    ],
    "recheck": "agreed"
  },
  {
    "utt_id": "homo_107",
    "action": "fix",
    "reason": "inflect_to_gent",
    "lexeme_before": "ири́с",
    "lexeme_after": "ири́са",
    "contrast_before": "и́рис",
    "contrast_after": "и́риса",
    "text_before": "Упаковка ири́с ярко раскрашена.",
    "text_after": "Упаковка ири́са ярко раскрашена.",
    "char_index_after": 11,
    "cases": [
      "gent"
    ],
    "recheck": "agreed"
  },
  {
    "utt_id": "homo_108",
    "action": "fix",
    "reason": "inflect_to_gent",
    "lexeme_before": "ири́с",
    "lexeme_after": "ири́са",
    "contrast_before": "и́рис",
    "contrast_after": "и́риса",
    "text_before": "Я купил килограмм ири́с к чаю.",
    "text_after": "Я купил килограмм ири́са к чаю.",
    "char_index_after": 20,
    "cases": [
      "gent"
    ],
    "recheck": "agreed"
  },
  {
    "utt_id": "homo_109",
    "action": "fix",
    "reason": "inflect_to_gent",
    "lexeme_before": "ири́с",
    "lexeme_after": "ири́са",
    "contrast_before": "и́рис",
    "contrast_after": "и́риса",
    "text_before": "Рецепт домашнего ири́с передали бабушке.",
    "text_after": "Рецепт домашнего ири́са передали бабушке.",
    "char_index_after": 19,
    "cases": [
      "gent"
    ],
    "recheck": "agreed"
  },
  {
    "utt_id": "homo_111",
    "action": "fix",
    "reason": "inflect_to_gent",
    "lexeme_before": "про́пасть",
    "lexeme_after": "про́пасти",
    "contrast_before": "пропа́сть",
    "contrast_after": "пропа́сти",
    "text_before": "Тропа обрывается у глубокой про́пасть.",
    "text_after": "Тропа обрывается у глубокой про́пасти.",
    "char_index_after": 30,
    "cases": [
      "gent"
    ],
    "recheck": "agreed"
  },
  {
    "utt_id": "homo_113",
    "action": "fix",
    "reason": "inflect_to_gent",
    "lexeme_before": "про́пасть",
    "lexeme_after": "про́пасти",
    "contrast_before": "пропа́сть",
    "contrast_after": "пропа́сти",
    "text_before": "Туман поднимался со дна про́пасть.",
    "text_after": "Туман поднимался со дна про́пасти.",
    "char_index_after": 26,
    "cases": [
      "gent"
    ],
    "recheck": "agreed"
  },
  {
    "utt_id": "homo_116",
    "action": "fix",
    "reason": "inflect_to_gent",
    "lexeme_before": "про́пасть",
    "lexeme_after": "про́пасти",
    "contrast_before": "пропа́сть",
    "contrast_after": "пропа́сти",
    "text_before": "Предупреждение: не подходите к краю про́пасть.",
    "text_after": "Предупреждение: не подходите к краю про́пасти.",
    "char_index_after": 38,
    "cases": [
      "gent"
    ],
    "recheck": "agreed"
  },
  {
    "utt_id": "homo_117",
    "action": "fix",
    "reason": "inflect_to_gent",
    "lexeme_before": "про́пасть",
    "lexeme_after": "про́пасти",
    "contrast_before": "пропа́сть",
    "contrast_after": "пропа́сти",
    "text_before": "Эхо отозвалось из про́пасть.",
    "text_after": "Эхо отозвалось из про́пасти.",
    "char_index_after": 20,
    "cases": [
      "gent"
    ],
    "recheck": "agreed"
  },
  {
    "utt_id": "homo_119",
    "action": "fix",
    "reason": "inflect_to_gent",
    "lexeme_before": "про́пасть",
    "lexeme_after": "про́пасти",
    "contrast_before": "пропа́сть",
    "contrast_after": "пропа́сти",
    "text_before": "Легенда говорит о сокровищах на дне про́пасть.",
    "text_after": "Легенда говорит о сокровищах на дне про́пасти.",
    "char_index_after": 38,
    "cases": [
      "gent"
    ],
    "recheck": "agreed"
  },
  {
    "utt_id": "homo_124",
    "action": "fix",
    "reason": "inflect_to_gent",
    "lexeme_before": "пропа́сть",
    "lexeme_after": "пропа́сти",
    "contrast_before": "про́пасть",
    "contrast_after": "про́пасти",
    "text_before": "Не дайте шансу пропа́сть зря.",
    "text_after": "Не дайте шансу пропа́сти зря.",
    "char_index_after": 19,
    "cases": [
      "gent"
    ],
    "recheck": "agreed"
  },
  {
    "utt_id": "homo_130",
    "action": "fix",
    "reason": "inflect_to_gent",
    "lexeme_before": "пропа́сть",
    "lexeme_after": "пропа́сти",
    "contrast_before": "про́пасть",
    "contrast_after": "про́пасти",
    "text_before": "Не дайте идее пропа́сть без проверки.",
    "text_after": "Не дайте идее пропа́сти без проверки.",
    "char_index_after": 18,
    "cases": [
      "gent"
    ],
    "recheck": "agreed"
  },
  {
    "utt_id": "homo_132",
    "action": "fix",
    "reason": "inflect_to_gent",
    "lexeme_before": "духи́",
    "lexeme_after": "духо́в",
    "contrast_before": "ду́хи",
    "contrast_after": "ду́хов",
    "text_before": "Аромат духи́ наполнил комнату.",
    "text_after": "Аромат духо́в наполнил комнату.",
    "char_index_after": 10,
    "cases": [
      "gent"
    ],
    "recheck": "agreed"
  },
  {
    "utt_id": "homo_134",
    "action": "fix",
    "reason": "inflect_to_gent",
    "lexeme_before": "духи́",
    "lexeme_after": "духо́в",
    "contrast_before": "ду́хи",
    "contrast_after": "ду́хов",
    "text_before": "Флакон духи́ стоит на туалетном столике.",
    "text_after": "Флакон духо́в стоит на туалетном столике.",
    "char_index_after": 10,
    "cases": [
      "gent"
    ],
    "recheck": "agreed"
  },
  {
    "utt_id": "homo_135",
    "action": "fix",
    "reason": "inflect_to_gent",
    "lexeme_before": "духи́",
    "lexeme_after": "духо́в",
    "contrast_before": "ду́хи",
    "contrast_after": "ду́хов",
    "text_before": "Продавец дал попробовать пробник духи́.",
    "text_after": "Продавец дал попробовать пробник духо́в.",
    "char_index_after": 36,
    "cases": [
      "gent"
    ],
    "recheck": "agreed"
  },
  {
    "utt_id": "homo_137",
    "action": "fix",
    "reason": "inflect_to_gent",
    "lexeme_before": "духи́",
    "lexeme_after": "духо́в",
    "contrast_before": "ду́хи",
    "contrast_after": "ду́хов",
    "text_before": "В магазине открыли новый отдел духи́.",
    "text_after": "В магазине открыли новый отдел духо́в.",
    "char_index_after": 34,
    "cases": [
      "gent"
    ],
    "recheck": "agreed"
  },
  {
    "utt_id": "homo_138",
    "action": "fix",
    "reason": "inflect_to_gent",
    "lexeme_before": "духи́",
    "lexeme_after": "духо́в",
    "contrast_before": "ду́хи",
    "contrast_after": "ду́хов",
    "text_before": "Бутылочка духи́ уместилась в сумочке.",
    "text_after": "Бутылочка духо́в уместилась в сумочке.",
    "char_index_after": 13,
    "cases": [
      "gent"
    ],
    "recheck": "agreed"
  },
  {
    "utt_id": "homo_139",
    "action": "fix",
    "reason": "inflect_to_gent",
    "lexeme_before": "духи́",
    "lexeme_after": "духо́в",
    "contrast_before": "ду́хи",
    "contrast_after": "ду́хов",
    "text_before": "Критик описал ноты этих духи́.",
    "text_after": "Критик описал ноты этих духо́в.",
    "char_index_after": 27,
    "cases": [
      "gent"
    ],
    "recheck": "agreed"
  },
  {
    "utt_id": "homo_142",
    "action": "fix",
    "reason": "inflect_to_gent",
    "lexeme_before": "ду́хи",
    "lexeme_after": "ду́хов",
    "contrast_before": "духи́",
    "contrast_after": "духо́в",
    "text_before": "Шаман призывал добрых ду́хи.",
    "text_after": "Шаман призывал добрых ду́хов.",
    "char_index_after": 23,
    "cases": [
      "gent"
    ],
    "recheck": "agreed"
  },
  {
    "utt_id": "homo_143",
    "action": "fix",
    "reason": "inflect_to_gent",
    "lexeme_before": "ду́хи",
    "lexeme_after": "ду́хов",
    "contrast_before": "духи́",
    "contrast_after": "духо́в",
    "text_before": "Дети боятся ночных ду́хи.",
    "text_after": "Дети боятся ночных ду́хов.",
    "char_index_after": 20,
    "cases": [
      "gent"
    ],
    "recheck": "agreed"
  },
  {
    "utt_id": "homo_145",
    "action": "fix",
    "reason": "inflect_to_ablt",
    "lexeme_before": "ду́хи",
    "lexeme_after": "ду́хами",
    "contrast_before": "духи́",
    "contrast_after": "духа́ми",
    "text_before": "Фильм показывает встречу с ду́хи.",
    "text_after": "Фильм показывает встречу с ду́хами.",
    "char_index_after": 28,
    "cases": [
      "ablt"
    ],
    "recheck": "agreed"
  },
  {
    "utt_id": "homo_146",
    "action": "fix",
    "reason": "inflect_to_gent",
    "lexeme_before": "ду́хи",
    "lexeme_after": "ду́хов",
    "contrast_before": "духи́",
    "contrast_after": "духо́в",
    "text_before": "По преданию ду́хи охраняют храм.",
    "text_after": "По преданию ду́хов охраняют храм.",
    "char_index_after": 13,
    "cases": [
      "gent"
    ],
    "recheck": "agreed"
  },
  {
    "utt_id": "homo_147",
    "action": "fix",
    "reason": "inflect_to_gent",
    "lexeme_before": "ду́хи",
    "lexeme_after": "ду́хов",
    "contrast_before": "духи́",
    "contrast_after": "духо́в",
    "text_before": "Писатель описал невидимых ду́хи.",
    "text_after": "Писатель описал невидимых ду́хов.",
    "char_index_after": 27,
    "cases": [
      "gent"
    ],
    "recheck": "agreed"
  },
  {
    "utt_id": "homo_148",
    "action": "fix",
    "reason": "inflect_to_gent",
    "lexeme_before": "ду́хи",
    "lexeme_after": "ду́хов",
    "contrast_before": "духи́",
    "contrast_after": "духо́в",
    "text_before": "Обряд изгнания ду́хи длился час.",
    "text_after": "Обряд изгнания ду́хов длился час.",
    "char_index_after": 16,
    "cases": [
      "gent"
    ],
    "recheck": "agreed"
  },
  {
    "utt_id": "homo_149",
    "action": "fix",
    "reason": "inflect_to_gent",
    "lexeme_before": "ду́хи",
    "lexeme_after": "ду́хов",
    "contrast_before": "духи́",
    "contrast_after": "духо́в",
    "text_before": "Старуха верила в домашних ду́хи.",
    "text_after": "Старуха верила в домашних ду́хов.",
    "char_index_after": 27,
    "cases": [
      "gent"
    ],
    "recheck": "agreed"
  },
  {
    "utt_id": "homo_211",
    "action": "fix",
    "reason": "inflect_to_datv",
    "lexeme_before": "по́лки",
    "lexeme_after": "по́лкам",
    "contrast_before": "полки́",
    "contrast_after": "полка́м",
    "text_before": "Книги расставили по деревянным по́лки.",
    "text_after": "Книги расставили по деревянным по́лкам.",
    "char_index_after": 32,
    "cases": [
      "datv"
    ],
    "recheck": "agreed"
  },
  {
    "utt_id": "homo_212",
    "action": "fix",
    "reason": "inflect_to_loct",
    "lexeme_before": "по́лки",
    "lexeme_after": "по́лках",
    "contrast_before": "полки́",
    "contrast_after": "полка́х",
    "text_before": "На кухонных по́лки стоят банки со специями.",
    "text_after": "На кухонных по́лках стоят банки со специями.",
    "char_index_after": 13,
    "cases": [
      "loct"
    ],
    "recheck": "agreed"
  },
  {
    "utt_id": "homo_215",
    "action": "fix",
    "reason": "inflect_to_loct",
    "lexeme_before": "по́лки",
    "lexeme_after": "по́лках",
    "contrast_before": "полки́",
    "contrast_after": "полка́х",
    "text_before": "Вазы стоят на стеклянных по́лки.",
    "text_after": "Вазы стоят на стеклянных по́лках.",
    "char_index_after": 26,
    "cases": [
      "loct"
    ],
    "recheck": "agreed"
  },
  {
    "utt_id": "homo_231",
    "action": "fix",
    "reason": "inflect_to_gent",
    "lexeme_before": "бело́к",
    "lexeme_after": "белка́",
    "contrast_before": "бе́лок",
    "contrast_after": "бе́лка",
    "text_before": "В яйце много полезного бело́к.",
    "text_after": "В яйце много полезного белка́.",
    "char_index_after": 27,
    "cases": [
      "gent"
    ],
    "recheck": "agreed"
  },
  {
    "utt_id": "homo_232",
    "action": "fix",
    "reason": "inflect_to_gent",
    "lexeme_before": "бело́к",
    "lexeme_after": "белка́",
    "contrast_before": "бе́лок",
    "contrast_after": "бе́лка",
    "text_before": "Спортсмен считает граммы бело́к в рационе.",
    "text_after": "Спортсмен считает граммы белка́ в рационе.",
    "char_index_after": 29,
    "cases": [
      "gent"
    ],
    "recheck": "agreed"
  },
  {
    "utt_id": "homo_234",
    "action": "fix",
    "reason": "inflect_to_gent",
    "lexeme_before": "бело́к",
    "lexeme_after": "белка́",
    "contrast_before": "бе́лок",
    "contrast_after": "бе́лка",
    "text_before": "Анализ показал уровень бело́к в крови.",
    "text_after": "Анализ показал уровень белка́ в крови.",
    "char_index_after": 27,
    "cases": [
      "gent"
    ],
    "recheck": "agreed"
  },
  {
    "utt_id": "homo_237",
    "action": "fix",
    "reason": "inflect_to_gent",
    "lexeme_before": "бело́к",
    "lexeme_after": "белка́",
    "contrast_before": "бе́лок",
    "contrast_after": "бе́лка",
    "text_before": "Недостаток бело́к влияет на мышцы.",
    "text_after": "Недостаток белка́ влияет на мышцы.",
    "char_index_after": 15,
    "cases": [
      "gent"
    ],
    "recheck": "agreed"
  },
  {
    "utt_id": "homo_238",
    "action": "fix",
    "reason": "inflect_to_gent",
    "lexeme_before": "бело́к",
    "lexeme_after": "белка́",
    "contrast_before": "бе́лок",
    "contrast_after": "бе́лка",
    "text_before": "Этикетка указывает процент бело́к.",
    "text_after": "Этикетка указывает процент белка́.",
    "char_index_after": 31,
    "cases": [
      "gent"
    ],
    "recheck": "agreed"
  },
  {
    "utt_id": "homo_244",
    "action": "fix",
    "reason": "inflect_to_ablt",
    "lexeme_before": "бе́лок",
    "lexeme_after": "бе́лком",
    "contrast_before": "бело́к",
    "contrast_after": "белко́м",
    "text_before": "Фотограф поймал кадр с бе́лок.",
    "text_after": "Фотограф поймал кадр с бе́лком.",
    "char_index_after": 24,
    "cases": [
      "ablt"
    ],
    "recheck": "agreed"
  },
  {
    "utt_id": "homo_246",
    "action": "fix",
    "reason": "inflect_to_gent",
    "lexeme_before": "бе́лок",
    "lexeme_after": "бе́лка",
    "contrast_before": "бело́к",
    "contrast_after": "белка́",
    "text_before": "Гнездо бе́лок спрятано в дупле.",
    "text_after": "Гнездо бе́лка спрятано в дупле.",
    "char_index_after": 8,
    "cases": [
      "gent"
    ],
    "recheck": "agreed"
  },
  {
    "utt_id": "homo_250",
    "action": "fix",
    "reason": "inflect_to_gent",
    "lexeme_before": "бе́лок",
    "lexeme_after": "бе́лка",
    "contrast_before": "бело́к",
    "contrast_after": "белка́",
    "text_before": "Следы бе́лок видны на снегу.",
    "text_after": "Следы бе́лка видны на снегу.",
    "char_index_after": 7,
    "cases": [
      "gent"
    ],
    "recheck": "agreed"
  },
  {
    "utt_id": "homo_252",
    "action": "fix",
    "reason": "inflect_to_accs",
    "lexeme_before": "му́ка",
    "lexeme_after": "му́ку",
    "contrast_before": "мука́",
    "contrast_after": "муку́",
    "text_before": "Повесть описывает му́ка пленных.",
    "text_after": "Повесть описывает му́ку пленных.",
    "char_index_after": 19,
    "cases": [
      "accs"
    ],
    "recheck": "agreed"
  },
  {
    "utt_id": "homo_253",
    "action": "fix",
    "reason": "inflect_to_accs",
    "lexeme_before": "му́ка",
    "lexeme_after": "му́ку",
    "contrast_before": "мука́",
    "contrast_after": "муку́",
    "text_before": "Художник передал му́ка героя на холсте.",
    "text_after": "Художник передал му́ку героя на холсте.",
    "char_index_after": 18,
    "cases": [
      "accs"
    ],
    "recheck": "agreed"
  },
  {
    "utt_id": "homo_255",
    "action": "fix",
    "reason": "inflect_to_accs",
    "lexeme_before": "му́ка",
    "lexeme_after": "му́ку",
    "contrast_before": "мука́",
    "contrast_after": "муку́",
    "text_before": "Она скрывала свою му́ка от близких.",
    "text_after": "Она скрывала свою му́ку от близких.",
    "char_index_after": 19,
    "cases": [
      "accs"
    ],
    "recheck": "agreed"
  },
  {
    "utt_id": "homo_256",
    "action": "fix",
    "reason": "inflect_to_accs",
    "lexeme_before": "му́ка",
    "lexeme_after": "му́ку",
    "contrast_before": "мука́",
    "contrast_after": "муку́",
    "text_before": "Фильм показывает му́ка войны без прикрас.",
    "text_after": "Фильм показывает му́ку войны без прикрас.",
    "char_index_after": 18,
    "cases": [
      "accs"
    ],
    "recheck": "agreed"
  },
  {
    "utt_id": "homo_257",
    "action": "fix",
    "reason": "inflect_to_accs",
    "lexeme_before": "му́ка",
    "lexeme_after": "му́ку",
    "contrast_before": "мука́",
    "contrast_after": "муку́",
    "text_before": "Поэт воспел му́ка разлуки.",
    "text_after": "Поэт воспел му́ку разлуки.",
    "char_index_after": 13,
    "cases": [
      "accs"
    ],
    "recheck": "agreed"
  },
  {
    "utt_id": "homo_258",
    "action": "fix",
    "reason": "inflect_to_accs",
    "lexeme_before": "му́ка",
    "lexeme_after": "му́ку",
    "contrast_before": "мука́",
    "contrast_after": "муку́",
    "text_before": "Свидетели помнят му́ка осаждённых.",
    "text_after": "Свидетели помнят му́ку осаждённых.",
    "char_index_after": 18,
    "cases": [
      "accs"
    ],
    "recheck": "agreed"
  }
]
```

## Deletion summary (first 80)

```json
[
  {
    "utt_id": "homo_156",
    "text": "Магазин продаёт набор цветных кру́жки.",
    "lexeme": "кру́жки",
    "action": "delete",
    "reason": "inflected_forms_break_homograph",
    "plain_lexeme": "кружек",
    "plain_contrast": "кружков",
    "cases": [
      "gent"
    ]
  },
  {
    "utt_id": "homo_158",
    "text": "Сушилка полна мокрых кру́жки.",
    "lexeme": "кру́жки",
    "action": "delete",
    "reason": "inflected_forms_break_homograph",
    "plain_lexeme": "кружек",
    "plain_contrast": "кружков",
    "cases": [
      "gent"
    ]
  },
  {
    "utt_id": "homo_159",
    "text": "В автомате нет пластиковых кру́жки.",
    "lexeme": "кру́жки",
    "action": "delete",
    "reason": "inflected_forms_break_homograph",
    "plain_lexeme": "кружек",
    "plain_contrast": "кружков",
    "cases": [
      "gent"
    ]
  },
  {
    "utt_id": "homo_166",
    "text": "В доме культуры много бесплатных кружки́.",
    "lexeme": "кружки́",
    "action": "delete",
    "reason": "inflected_forms_break_homograph",
    "plain_lexeme": "кружков",
    "plain_contrast": "кружек",
    "cases": [
      "gent"
    ]
  },
  {
    "utt_id": "homo_217",
    "text": "Ребёнок достал игрушку с нижних по́лки.",
    "lexeme": "по́лки",
    "action": "delete",
    "reason": "inflected_forms_break_homograph",
    "plain_lexeme": "полок",
    "plain_contrast": "полков",
    "cases": [
      "gent"
    ]
  },
  {
    "utt_id": "homo_229",
    "text": "Историк изучил состав царских полки́.",
    "lexeme": "полки́",
    "action": "delete",
    "reason": "inflected_forms_break_homograph",
    "plain_lexeme": "полков",
    "plain_contrast": "полок",
    "cases": [
      "gent"
    ]
  },
  {
    "utt_id": "homo_248",
    "text": "Сказка про хитрого бе́лок смешит малышей.",
    "lexeme": "бе́лок",
    "action": "delete",
    "reason": "inflection_noop",
    "cases": [
      "accs"
    ]
  },
  {
    "utt_id": "homo_402",
    "text": "В письме указан домашний а́дреса.",
    "lexeme": "а́дреса",
    "action": "delete",
    "reason": "stress_transfer_fallback",
    "new_lexeme": "а́дрес",
    "new_contrast": "адре́с",
    "cases": [
      "nomn"
    ]
  },
  {
    "utt_id": "homo_403",
    "text": "Курьер спросил точный а́дреса.",
    "lexeme": "а́дреса",
    "action": "delete",
    "reason": "stress_transfer_fallback",
    "new_lexeme": "а́дрес",
    "new_contrast": "адре́с",
    "cases": [
      "accs"
    ]
  },
  {
    "utt_id": "homo_404",
    "text": "Форма требует почтовый а́дреса.",
    "lexeme": "а́дреса",
    "action": "delete",
    "reason": "stress_transfer_fallback",
    "new_lexeme": "а́дрес",
    "new_contrast": "адре́с",
    "cases": [
      "accs"
    ]
  },
  {
    "utt_id": "homo_407",
    "text": "На визитке напечатан рабочий а́дреса.",
    "lexeme": "а́дреса",
    "action": "delete",
    "reason": "stress_transfer_fallback",
    "new_lexeme": "а́дрес",
    "new_contrast": "адре́с",
    "cases": [
      "nomn"
    ]
  },
  {
    "utt_id": "homo_411",
    "text": "Он назвал временный а́дреса.",
    "lexeme": "а́дреса",
    "action": "delete",
    "reason": "stress_transfer_fallback",
    "new_lexeme": "а́дрес",
    "new_contrast": "адре́с",
    "cases": [
      "accs"
    ]
  },
  {
    "utt_id": "homo_413",
    "text": "Система сохранила прежний а́дреса.",
    "lexeme": "а́дреса",
    "action": "delete",
    "reason": "stress_transfer_fallback",
    "new_lexeme": "а́дрес",
    "new_contrast": "адре́с",
    "cases": [
      "accs"
    ]
  },
  {
    "utt_id": "homo_415",
    "text": "В договоре прописан юридический а́дреса.",
    "lexeme": "а́дреса",
    "action": "delete",
    "reason": "stress_transfer_fallback",
    "new_lexeme": "а́дрес",
    "new_contrast": "адре́с",
    "cases": [
      "nomn"
    ]
  },
  {
    "utt_id": "homo_673",
    "text": "Учёные кольцуют местных бе́лки.",
    "lexeme": "бе́лки",
    "action": "delete",
    "reason": "inflected_forms_break_homograph",
    "plain_lexeme": "белок",
    "plain_contrast": "белков",
    "cases": [
      "gent"
    ]
  },
  {
    "utt_id": "homo_675",
    "text": "Туристы фотографируют ручных бе́лки.",
    "lexeme": "бе́лки",
    "action": "delete",
    "reason": "inflected_forms_break_homograph",
    "plain_lexeme": "белок",
    "plain_contrast": "белков",
    "cases": [
      "gent"
    ]
  },
  {
    "utt_id": "homo_855",
    "text": "Капли по́том стекали по вискам.",
    "lexeme": "по́том",
    "action": "delete",
    "reason": "adverb_homograph_only_in_instrumental",
    "cases": [
      "gent"
    ]
  },
  {
    "utt_id": "homo_856",
    "text": "После зала всё было в по́том.",
    "lexeme": "по́том",
    "action": "delete",
    "reason": "adverb_homograph_only_in_instrumental",
    "cases": [
      "accs",
      "loct"
    ]
  },
  {
    "utt_id": "homo_858",
    "text": "Он пах по́том и пылью.",
    "lexeme": "по́том",
    "action": "delete",
    "reason": "adverb_homograph_only_in_instrumental",
    "cases": [
      "gent"
    ]
  },
  {
    "utt_id": "homo_859",
    "text": "Полотенце впитало по́том.",
    "lexeme": "по́том",
    "action": "delete",
    "reason": "adverb_homograph_only_in_instrumental",
    "cases": [
      "accs",
      "gent"
    ]
  },
  {
    "utt_id": "homo_860",
    "text": "Тренировка оставила одежду мокрой по́том.",
    "lexeme": "по́том",
    "action": "delete",
    "reason": "adverb_homograph_only_in_instrumental",
    "cases": [
      "gent"
    ]
  }
]
```

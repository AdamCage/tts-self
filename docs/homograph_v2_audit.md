# Homograph v2 audit

- Items checked programmatically: **895**
- Valid items retained in v2: **740**
- Distinct stress pairs: **30**
- Broken `char_index`: **250** (27.933%)
- Broken `char_index` after corrections: **0**
- Canonical stress/form corrections: **380**
- Items removed for invalid stress-only pairs: **155**
- Other structural errors: **0**
- Manually reviewed contexts: **100**
- All sampled pairs exist: **True**
- All sampled contexts resolve stress: **True**

The manual sample is deterministic and stratified to include every pair, then filled evenly across the remaining inventory.

## Manual sample

| utt_id | text | lexeme | contrast | status |
|---|---|---|---|---|
| homo_632 | Инструктор приказал: «Пали́ по мишени!» | Пали́ | Па́ли | checked |
| homo_401 | Проверьте правильность а́дреса доставки. | а́дреса | адреса́ | checked |
| homo_031 | На уроке географии открыли а́тлас мира. | а́тлас | атла́с | checked |
| homo_661 | В парке прыгают рыжие бе́лки. | бе́лки | белки́ | checked |
| homo_231 | В яйце много полезного бело́к. | бело́к | бе́лок | checked |
| homo_341 | История древнего го́рода насчитывает века. | го́рода | горо́да | checked |
| homo_281 | Вечером все сидели до́ма у камина. | до́ма | дома́ | checked |
| homo_131 | Она нанесла дорогие духи́ перед выходом. | духи́ | ду́хи | checked |
| homo_562 | На шее вздулась синяя жи́ла. | жи́ла | жила́ | checked |
| homo_001 | Туристы осмотрели средневековый за́мок на холме. | за́мок | замо́к | checked |
| homo_091 | На клумбе расцвёл синий и́рис. | и́рис | ири́с | checked |
| homo_151 | На полке стоят керамические кру́жки. | кру́жки | кружки́ | checked |
| homo_311 | Грибники ушли в густые леса́. | леса́ | ле́са | checked |
| homo_431 | Ученик перенял приём у опытного ма́стера. | ма́стера | мастера́ | checked |
| homo_251 | Его му́ка длилась долгие годы. | му́ка | мука́ | checked |
| homo_602 | С на́чала урока прошло десять минут. | на́чала | начала́ | checked |
| homo_061 | Парламент — высший законодательный о́рган страны. | о́рган | орга́н | checked |
| homo_371 | Экспедиция достигла берега о́строва. | о́строва | острова́ | checked |
| homo_622 | Солдаты па́ли в неравном бою. | па́ли | пали́ | checked |
| homo_521 | Гости пи́ли чай до полуночи. | пи́ли | пили́ | checked |
| homo_211 | Книги расставили по деревянным по́лки. | по́лки | полки́ | checked |
| homo_851 | Рубашка промокла по́том после бега. | по́том | пото́м | checked |
| homo_111 | Тропа обрывается у глубокой про́пасть. | про́пасть | пропа́сть | checked |
| homo_582 | Птица се́ла на ветку у окна. | се́ла | села́ | checked |
| homo_796 | Занятия перенесли на эту сре́ды. | сре́ды | среды́ | checked |
| homo_491 | Билет сто́ит тысячу рублей. | сто́ит | стои́т | checked |
| homo_171 | Часовые стре́лки показывают полдень. | стре́лки | стрелки́ | checked |
| homo_826 | Эта улица у́же, чем соседняя. | у́же | уже́ | checked |
| homo_461 | Совет учи́теля помог выбрать профессию. | учи́теля | учителя́ | checked |
| homo_876 | Футболка сшита из натурального хло́пок. | хло́пок | хлопо́к | checked |
| homo_002 | Экскурсия ведёт к древнему за́мок у реки. | за́мок | замо́к | checked |
| homo_012 | В подземельях за́мок хранили сокровища. | за́мок | замо́к | checked |
| homo_023 | Без ключа открыть замо́к невозможно. | замо́к | за́мок | checked |
| homo_034 | Картограф составил новый а́тлас регионов. | а́тлас | атла́с | checked |
| homo_044 | В а́тлас не хватает карты Антарктиды. | а́тлас | атла́с | checked |
| homo_054 | Модельер предпочитает чёрный атла́с. | атла́с | а́тлас | checked |
| homo_066 | Контрольный о́рган проверил отчётность. | о́рган | орга́н | checked |
| homo_076 | В соборе зазвучал огромный орга́н. | орга́н | о́рган | checked |
| homo_086 | Педаль орга́н управляет басовыми трубами. | орга́н | о́рган | checked |
| homo_097 | Натюрморт с и́рис висел в галерее. | и́рис | ири́с | checked |
| homo_108 | Я купил килограмм ири́с к чаю. | ири́с | и́рис | checked |
| homo_119 | Легенда говорит о сокровищах на дне про́пасть. | про́пасть | пропа́сть | checked |
| homo_129 | Шанс победы может пропа́сть из-за ошибки. | пропа́сть | про́пасть | checked |
| homo_141 | В сказке злые ду́хи живут в лесу. | ду́хи | духи́ | checked |
| homo_152 | Налей чай в чистые кру́жки. | кру́жки | кружки́ | checked |
| homo_162 | В школе открыли новые творческие кружки́. | кружки́ | кру́жки | checked |
| homo_173 | На циферблате золотые стре́лки. | стре́лки | стрелки́ | checked |
| homo_184 | Меткие стрелки́ прикрывали отряд. | стрелки́ | стре́лки | checked |
| homo_215 | Вазы стоят на стеклянных по́лки. | по́лки | полки́ | checked |
| homo_225 | Командир обошёл полки́ на рассвете. | полки́ | по́лки | checked |
| homo_237 | Недостаток бело́к влияет на мышцы. | бело́к | бе́лок | checked |
| homo_247 | Охотник увидел бе́лок на опушке. | бе́лок | бело́к | checked |
| homo_258 | Свидетели помнят му́ка осаждённых. | му́ка | мука́ | checked |
| homo_268 | В мешке осталось мало мука́. | мука́ | му́ка | checked |
| homo_279 | Хозяйка добавила мука́ в соус. | мука́ | му́ка | checked |
| homo_290 | В выходные остаёмся до́ма. | до́ма | дома́ | checked |
| homo_300 | Туристы фотографировали старые дома́. | дома́ | до́ма | checked |
| homo_310 | Вид на дома́ открывается с холма. | дома́ | до́ма | checked |
| homo_322 | Ветер шумит в верхушках леса́. | леса́ | ле́са | checked |
| homo_332 | Животные прячутся в глубине ле́са. | ле́са | леса́ | checked |
| homo_343 | Жители го́рода голосуют завтра. | го́рода | горо́да | checked |
| homo_354 | Новости го́рода публикуют ежедневно. | го́рода | горо́да | checked |
| homo_364 | В прибрежных горо́да пахнет морем. | горо́да | го́рода | checked |
| homo_375 | Порт о́строва принимает паромы. | о́строва | острова́ | checked |
| homo_385 | Маяк о́строва виден издалека. | о́строва | острова́ | checked |
| homo_396 | Коралловые острова́ привлекают дайверов. | острова́ | о́строва | checked |
| homo_407 | На визитке напечатан рабочий а́дреса. | а́дреса | адреса́ | checked |
| homo_417 | Рассылка ушла по старым адреса́. | адреса́ | а́дреса | checked |
| homo_427 | Сверка адреса́ заняла весь день. | адреса́ | а́дреса | checked |
| homo_439 | Урок ма́стера запомнился надолго. | ма́стера | мастера́ | checked |
| homo_449 | Конкурс выиграли юные мастера́. | мастера́ | ма́стера | checked |
| homo_459 | Каталог представляет современных мастера́. | мастера́ | ма́стера | checked |
| homo_471 | Опыт учи́теля передают молодым. | учи́теля | учителя́ | checked |
| homo_481 | Летние курсы для учителя́ открыты. | учителя́ | учи́теля | checked |
| homo_492 | Этот телефон того сто́ит. | сто́ит | стои́т | checked |
| homo_502 | Момент сто́ит того, чтобы подождать. | сто́ит | стои́т | checked |
| homo_513 | Машина стои́т под запрещающим знаком. | стои́т | сто́ит | checked |
| homo_524 | Дети пи́ли компот после обеда. | пи́ли | пили́ | checked |
| homo_565 | На виске билась тонкая жи́ла. | жи́ла | жила́ | checked |
| homo_575 | Кошка жила́ у соседей всё лето. | жила́ | жи́ла | checked |
| homo_587 | Усталость се́ла на плечи. | се́ла | села́ | checked |
| homo_597 | В музее история этого села́. | села́ | се́ла | checked |
| homo_608 | С на́чала сезона команда лидирует. | на́чала | начала́ | checked |
| homo_619 | Она начала́ читать вслух детям. | начала́ | на́чала | checked |
| homo_630 | Занавеси па́ли, скрыв сцену. | па́ли | пали́ | checked |
| homo_641 | Часовой услышал приказ: «Пали́ при тревоге!» | Пали́ | Па́ли | checked |
| homo_651 | Мастера пили́ толстый брус. | пили́ | пи́ли | checked |
| homo_663 | Зимой бе́лки ищут спрятанные орехи. | бе́лки | белки́ | checked |
| homo_673 | Учёные кольцуют местных бе́лки. | бе́лки | белки́ | checked |
| homo_683 | Окулист проверил состояние белки́. | белки́ | бе́лки | checked |
| homo_800 | В прошлую сре́ды шёл дождь. | сре́ды | среды́ | checked |
| homo_810 | Тренировка проходит по сре́ды. | сре́ды | среды́ | checked |
| homo_820 | Раствор охладили в ледяной среды́. | среды́ | сре́ды | checked |
| homo_831 | Коридор здесь заметно у́же. | у́же | уже́ | checked |
| homo_842 | Дети уже́ спят. | уже́ | у́же | checked |
| homo_853 | Он вытер лоб, мокрый по́том. | по́том | пото́м | checked |
| homo_863 | Позвоните мне пото́м, после обеда. | пото́м | по́том | checked |
| homo_873 | Сначала теория, пото́м задачи. | пото́м | по́том | checked |
| homo_885 | Импортёр закупил партию хло́пок. | хло́пок | хлопо́к | checked |
| homo_895 | Молния ударила с оглушительным хлопо́к. | хлопо́к | хло́пок | checked |

## Removed invalid pairs

```json
[
  {
    "utt_id": "homo_191",
    "base": "языки",
    "lexeme": "яз́ыки",
    "contrast": "языќи",
    "reason": "the proposed meanings do not form a normative stress contrast"
  },
  {
    "utt_id": "homo_192",
    "base": "языки",
    "lexeme": "яз́ыки",
    "contrast": "языќи",
    "reason": "the proposed meanings do not form a normative stress contrast"
  },
  {
    "utt_id": "homo_193",
    "base": "языки",
    "lexeme": "яз́ыки",
    "contrast": "языќи",
    "reason": "the proposed meanings do not form a normative stress contrast"
  },
  {
    "utt_id": "homo_194",
    "base": "языки",
    "lexeme": "яз́ыки",
    "contrast": "языќи",
    "reason": "the proposed meanings do not form a normative stress contrast"
  },
  {
    "utt_id": "homo_195",
    "base": "языки",
    "lexeme": "яз́ыки",
    "contrast": "языќи",
    "reason": "the proposed meanings do not form a normative stress contrast"
  },
  {
    "utt_id": "homo_196",
    "base": "языки",
    "lexeme": "яз́ыки",
    "contrast": "языќи",
    "reason": "the proposed meanings do not form a normative stress contrast"
  },
  {
    "utt_id": "homo_197",
    "base": "языки",
    "lexeme": "яз́ыки",
    "contrast": "языќи",
    "reason": "the proposed meanings do not form a normative stress contrast"
  },
  {
    "utt_id": "homo_198",
    "base": "языки",
    "lexeme": "яз́ыки",
    "contrast": "языќи",
    "reason": "the proposed meanings do not form a normative stress contrast"
  },
  {
    "utt_id": "homo_199",
    "base": "языки",
    "lexeme": "яз́ыки",
    "contrast": "языќи",
    "reason": "the proposed meanings do not form a normative stress contrast"
  },
  {
    "utt_id": "homo_200",
    "base": "языки",
    "lexeme": "яз́ыки",
    "contrast": "языќи",
    "reason": "the proposed meanings do not form a normative stress contrast"
  },
  {
    "utt_id": "homo_201",
    "base": "языки",
    "lexeme": "языќи",
    "contrast": "яз́ыки",
    "reason": "the proposed meanings do not form a normative stress contrast"
  },
  {
    "utt_id": "homo_202",
    "base": "языки",
    "lexeme": "языќи",
    "contrast": "яз́ыки",
    "reason": "the proposed meanings do not form a normative stress contrast"
  },
  {
    "utt_id": "homo_203",
    "base": "языки",
    "lexeme": "языќи",
    "contrast": "яз́ыки",
    "reason": "the proposed meanings do not form a normative stress contrast"
  },
  {
    "utt_id": "homo_204",
    "base": "языки",
    "lexeme": "языќи",
    "contrast": "яз́ыки",
    "reason": "the proposed meanings do not form a normative stress contrast"
  },
  {
    "utt_id": "homo_205",
    "base": "языки",
    "lexeme": "языќи",
    "contrast": "яз́ыки",
    "reason": "the proposed meanings do not form a normative stress contrast"
  },
  {
    "utt_id": "homo_206",
    "base": "языки",
    "lexeme": "языќи",
    "contrast": "яз́ыки",
    "reason": "the proposed meanings do not form a normative stress contrast"
  },
  {
    "utt_id": "homo_207",
    "base": "языки",
    "lexeme": "языќи",
    "contrast": "яз́ыки",
    "reason": "the proposed meanings do not form a normative stress contrast"
  },
  {
    "utt_id": "homo_208",
    "base": "языки",
    "lexeme": "языќи",
    "contrast": "яз́ыки",
    "reason": "the proposed meanings do not form a normative stress contrast"
  },
  {
    "utt_id": "homo_209",
    "base": "языки",
    "lexeme": "языќи",
    "contrast": "яз́ыки",
    "reason": "the proposed meanings do not form a normative stress contrast"
  },
  {
    "utt_id": "homo_210",
    "base": "языки",
    "lexeme": "языќи",
    "contrast": "яз́ыки",
    "reason": "the proposed meanings do not form a normative stress contrast"
  },
  {
    "utt_id": "homo_532",
    "base": "берег",
    "lexeme": "бе́рег",
    "contrast": "бере́г",
    "reason": "the past tense requires ё (берёг), so it is not a stress-only contrast under explicit Russian orthography"
  },
  {
    "utt_id": "homo_533",
    "base": "берег",
    "lexeme": "бе́рег",
    "contrast": "бере́г",
    "reason": "the past tense requires ё (берёг), so it is not a stress-only contrast under explicit Russian orthography"
  },
  {
    "utt_id": "homo_534",
    "base": "берег",
    "lexeme": "бе́рег",
    "contrast": "бере́г",
    "reason": "the past tense requires ё (берёг), so it is not a stress-only contrast under explicit Russian orthography"
  },
  {
    "utt_id": "homo_535",
    "base": "берег",
    "lexeme": "бе́рег",
    "contrast": "бере́г",
    "reason": "the past tense requires ё (берёг), so it is not a stress-only contrast under explicit Russian orthography"
  },
  {
    "utt_id": "homo_536",
    "base": "берег",
    "lexeme": "бе́рег",
    "contrast": "бере́г",
    "reason": "the past tense requires ё (берёг), so it is not a stress-only contrast under explicit Russian orthography"
  },
  {
    "utt_id": "homo_537",
    "base": "берег",
    "lexeme": "бе́рег",
    "contrast": "бере́г",
    "reason": "the past tense requires ё (берёг), so it is not a stress-only contrast under explicit Russian orthography"
  },
  {
    "utt_id": "homo_538",
    "base": "берег",
    "lexeme": "бе́рег",
    "contrast": "бере́г",
    "reason": "the past tense requires ё (берёг), so it is not a stress-only contrast under explicit Russian orthography"
  },
  {
    "utt_id": "homo_539",
    "base": "берег",
    "lexeme": "бе́рег",
    "contrast": "бере́г",
    "reason": "the past tense requires ё (берёг), so it is not a stress-only contrast under explicit Russian orthography"
  },
  {
    "utt_id": "homo_540",
    "base": "берег",
    "lexeme": "бе́рег",
    "contrast": "бере́г",
    "reason": "the past tense requires ё (берёг), so it is not a stress-only contrast under explicit Russian orthography"
  },
  {
    "utt_id": "homo_541",
    "base": "берег",
    "lexeme": "бе́рег",
    "contrast": "бере́г",
    "reason": "the past tense requires ё (берёг), so it is not a stress-only contrast under explicit Russian orthography"
  },
  {
    "utt_id": "homo_542",
    "base": "берег",
    "lexeme": "бе́рег",
    "contrast": "бере́г",
    "reason": "the past tense requires ё (берёг), so it is not a stress-only contrast under explicit Russian orthography"
  },
  {
    "utt_id": "homo_543",
    "base": "берег",
    "lexeme": "бе́рег",
    "contrast": "бере́г",
    "reason": "the past tense requires ё (берёг), so it is not a stress-only contrast under explicit Russian orthography"
  },
  {
    "utt_id": "homo_544",
    "base": "берег",
    "lexeme": "бе́рег",
    "contrast": "бере́г",
    "reason": "the past tense requires ё (берёг), so it is not a stress-only contrast under explicit Russian orthography"
  },
  {
    "utt_id": "homo_545",
    "base": "берег",
    "lexeme": "бе́рег",
    "contrast": "бере́г",
    "reason": "the past tense requires ё (берёг), so it is not a stress-only contrast under explicit Russian orthography"
  },
  {
    "utt_id": "homo_546",
    "base": "берег",
    "lexeme": "бе́рег",
    "contrast": "бере́г",
    "reason": "the past tense requires ё (берёг), so it is not a stress-only contrast under explicit Russian orthography"
  },
  {
    "utt_id": "homo_547",
    "base": "берег",
    "lexeme": "бере́г",
    "contrast": "бе́рег",
    "reason": "the past tense requires ё (берёг), so it is not a stress-only contrast under explicit Russian orthography"
  },
  {
    "utt_id": "homo_548",
    "base": "берег",
    "lexeme": "бере́г",
    "contrast": "бе́рег",
    "reason": "the past tense requires ё (берёг), so it is not a stress-only contrast under explicit Russian orthography"
  },
  {
    "utt_id": "homo_549",
    "base": "берег",
    "lexeme": "бере́г",
    "contrast": "бе́рег",
    "reason": "the past tense requires ё (берёг), so it is not a stress-only contrast under explicit Russian orthography"
  },
  {
    "utt_id": "homo_550",
    "base": "берег",
    "lexeme": "бере́г",
    "contrast": "бе́рег",
    "reason": "the past tense requires ё (берёг), so it is not a stress-only contrast under explicit Russian orthography"
  },
  {
    "utt_id": "homo_551",
    "base": "берег",
    "lexeme": "бере́г",
    "contrast": "бе́рег",
    "reason": "the past tense requires ё (берёг), so it is not a stress-only contrast under explicit Russian orthography"
  },
  {
    "utt_id": "homo_552",
    "base": "берег",
    "lexeme": "бере́г",
    "contrast": "бе́рег",
    "reason": "the past tense requires ё (берёг), so it is not a stress-only contrast under explicit Russian orthography"
  },
  {
    "utt_id": "homo_553",
    "base": "берег",
    "lexeme": "бере́г",
    "contrast": "бе́рег",
    "reason": "the past tense requires ё (берёг), so it is not a stress-only contrast under explicit Russian orthography"
  },
  {
    "utt_id": "homo_554",
    "base": "берег",
    "lexeme": "бере́г",
    "contrast": "бе́рег",
    "reason": "the past tense requires ё (берёг), so it is not a stress-only contrast under explicit Russian orthography"
  },
  {
    "utt_id": "homo_555",
    "base": "берег",
    "lexeme": "бере́г",
    "contrast": "бе́рег",
    "reason": "the past tense requires ё (берёг), so it is not a stress-only contrast under explicit Russian orthography"
  },
  {
    "utt_id": "homo_556",
    "base": "берег",
    "lexeme": "бере́г",
    "contrast": "бе́рег",
    "reason": "the past tense requires ё (берёг), so it is not a stress-only contrast under explicit Russian orthography"
  },
  {
    "utt_id": "homo_557",
    "base": "берег",
    "lexeme": "бере́г",
    "contrast": "бе́рег",
    "reason": "the past tense requires ё (берёг), so it is not a stress-only contrast under explicit Russian orthography"
  },
  {
    "utt_id": "homo_558",
    "base": "берег",
    "lexeme": "бере́г",
    "contrast": "бе́рег",
    "reason": "the past tense requires ё (берёг), so it is not a stress-only contrast under explicit Russian orthography"
  },
  {
    "utt_id": "homo_559",
    "base": "берег",
    "lexeme": "бере́г",
    "contrast": "бе́рег",
    "reason": "the past tense requires ё (берёг), so it is not a stress-only contrast under explicit Russian orthography"
  },
  {
    "utt_id": "homo_560",
    "base": "берег",
    "lexeme": "бере́г",
    "contrast": "бе́рег",
    "reason": "the past tense requires ё (берёг), so it is not a stress-only contrast under explicit Russian orthography"
  },
  {
    "utt_id": "homo_561",
    "base": "берег",
    "lexeme": "бере́г",
    "contrast": "бе́рег",
    "reason": "the past tense requires ё (берёг), so it is not a stress-only contrast under explicit Russian orthography"
  },
  {
    "utt_id": "homo_691",
    "base": "волосы",
    "lexeme": "во́лосы",
    "contrast": "воло́сы",
    "reason": "no normative stress contrast for the same written form"
  },
  {
    "utt_id": "homo_692",
    "base": "волосы",
    "lexeme": "во́лосы",
    "contrast": "воло́сы",
    "reason": "no normative stress contrast for the same written form"
  },
  {
    "utt_id": "homo_693",
    "base": "волосы",
    "lexeme": "во́лосы",
    "contrast": "воло́сы",
    "reason": "no normative stress contrast for the same written form"
  },
  {
    "utt_id": "homo_694",
    "base": "волосы",
    "lexeme": "во́лосы",
    "contrast": "воло́сы",
    "reason": "no normative stress contrast for the same written form"
  },
  {
    "utt_id": "homo_695",
    "base": "волосы",
    "lexeme": "во́лосы",
    "contrast": "воло́сы",
    "reason": "no normative stress contrast for the same written form"
  },
  {
    "utt_id": "homo_696",
    "base": "волосы",
    "lexeme": "во́лосы",
    "contrast": "воло́сы",
    "reason": "no normative stress contrast for the same written form"
  },
  {
    "utt_id": "homo_697",
    "base": "волосы",
    "lexeme": "во́лосы",
    "contrast": "воло́сы",
    "reason": "no normative stress contrast for the same written form"
  },
  {
    "utt_id": "homo_698",
    "base": "волосы",
    "lexeme": "во́лосы",
    "contrast": "воло́сы",
    "reason": "no normative stress contrast for the same written form"
  },
  {
    "utt_id": "homo_699",
    "base": "волосы",
    "lexeme": "во́лосы",
    "contrast": "воло́сы",
    "reason": "no normative stress contrast for the same written form"
  },
  {
    "utt_id": "homo_700",
    "base": "волосы",
    "lexeme": "во́лосы",
    "contrast": "воло́сы",
    "reason": "no normative stress contrast for the same written form"
  },
  {
    "utt_id": "homo_701",
    "base": "волосы",
    "lexeme": "во́лосы",
    "contrast": "воло́сы",
    "reason": "no normative stress contrast for the same written form"
  },
  {
    "utt_id": "homo_702",
    "base": "волосы",
    "lexeme": "во́лосы",
    "contrast": "воло́сы",
    "reason": "no normative stress contrast for the same written form"
  },
  {
    "utt_id": "homo_703",
    "base": "волосы",
    "lexeme": "во́лосы",
    "contrast": "воло́сы",
    "reason": "no normative stress contrast for the same written form"
  },
  {
    "utt_id": "homo_704",
    "base": "волосы",
    "lexeme": "во́лосы",
    "contrast": "воло́сы",
    "reason": "no normative stress contrast for the same written form"
  },
  {
    "utt_id": "homo_705",
    "base": "волосы",
    "lexeme": "во́лосы",
    "contrast": "воло́сы",
    "reason": "no normative stress contrast for the same written form"
  },
  {
    "utt_id": "homo_706",
    "base": "зубы",
    "lexeme": "зу́бы",
    "contrast": "зубы́",
    "reason": "technical plural requires зубья, not a stress-only contrast"
  },
  {
    "utt_id": "homo_707",
    "base": "зубы",
    "lexeme": "зу́бы",
    "contrast": "зубы́",
    "reason": "technical plural requires зубья, not a stress-only contrast"
  },
  {
    "utt_id": "homo_708",
    "base": "зубы",
    "lexeme": "зу́бы",
    "contrast": "зубы́",
    "reason": "technical plural requires зубья, not a stress-only contrast"
  },
  {
    "utt_id": "homo_709",
    "base": "зубы",
    "lexeme": "зу́бы",
    "contrast": "зубы́",
    "reason": "technical plural requires зубья, not a stress-only contrast"
  },
  {
    "utt_id": "homo_710",
    "base": "зубы",
    "lexeme": "зу́бы",
    "contrast": "зубы́",
    "reason": "technical plural requires зубья, not a stress-only contrast"
  },
  {
    "utt_id": "homo_711",
    "base": "зубы",
    "lexeme": "зу́бы",
    "contrast": "зубы́",
    "reason": "technical plural requires зубья, not a stress-only contrast"
  },
  {
    "utt_id": "homo_712",
    "base": "зубы",
    "lexeme": "зу́бы",
    "contrast": "зубы́",
    "reason": "technical plural requires зубья, not a stress-only contrast"
  },
  {
    "utt_id": "homo_713",
    "base": "зубы",
    "lexeme": "зу́бы",
    "contrast": "зубы́",
    "reason": "technical plural requires зубья, not a stress-only contrast"
  },
  {
    "utt_id": "homo_714",
    "base": "зубы",
    "lexeme": "зу́бы",
    "contrast": "зубы́",
    "reason": "technical plural requires зубья, not a stress-only contrast"
  },
  {
    "utt_id": "homo_715",
    "base": "зубы",
    "lexeme": "зу́бы",
    "contrast": "зубы́",
    "reason": "technical plural requires зубья, not a stress-only contrast"
  },
  {
    "utt_id": "homo_716",
    "base": "зубы",
    "lexeme": "зу́бы",
    "contrast": "зубы́",
    "reason": "technical plural requires зубья, not a stress-only contrast"
  },
  {
    "utt_id": "homo_717",
    "base": "зубы",
    "lexeme": "зу́бы",
    "contrast": "зубы́",
    "reason": "technical plural requires зубья, not a stress-only contrast"
  },
  {
    "utt_id": "homo_718",
    "base": "зубы",
    "lexeme": "зу́бы",
    "contrast": "зубы́",
    "reason": "technical plural requires зубья, not a stress-only contrast"
  },
  {
    "utt_id": "homo_719",
    "base": "зубы",
    "lexeme": "зу́бы",
    "contrast": "зубы́",
    "reason": "technical plural requires зубья, not a stress-only contrast"
  },
  {
    "utt_id": "homo_720",
    "base": "зубы",
    "lexeme": "зу́бы",
    "contrast": "зубы́",
    "reason": "technical plural requires зубья, not a stress-only contrast"
  },
  {
    "utt_id": "homo_721",
    "base": "зубы",
    "lexeme": "зубы́",
    "contrast": "зу́бы",
    "reason": "technical plural requires зубья, not a stress-only contrast"
  },
  {
    "utt_id": "homo_722",
    "base": "зубы",
    "lexeme": "зубы́",
    "contrast": "зу́бы",
    "reason": "technical plural requires зубья, not a stress-only contrast"
  },
  {
    "utt_id": "homo_723",
    "base": "зубы",
    "lexeme": "зубы́",
    "contrast": "зу́бы",
    "reason": "technical plural requires зубья, not a stress-only contrast"
  },
  {
    "utt_id": "homo_724",
    "base": "зубы",
    "lexeme": "зубы́",
    "contrast": "зу́бы",
    "reason": "technical plural requires зубья, not a stress-only contrast"
  },
  {
    "utt_id": "homo_725",
    "base": "зубы",
    "lexeme": "зубы́",
    "contrast": "зу́бы",
    "reason": "technical plural requires зубья, not a stress-only contrast"
  },
  {
    "utt_id": "homo_726",
    "base": "зубы",
    "lexeme": "зубы́",
    "contrast": "зу́бы",
    "reason": "technical plural requires зубья, not a stress-only contrast"
  },
  {
    "utt_id": "homo_727",
    "base": "зубы",
    "lexeme": "зубы́",
    "contrast": "зу́бы",
    "reason": "technical plural requires зубья, not a stress-only contrast"
  },
  {
    "utt_id": "homo_728",
    "base": "зубы",
    "lexeme": "зубы́",
    "contrast": "зу́бы",
    "reason": "technical plural requires зубья, not a stress-only contrast"
  },
  {
    "utt_id": "homo_729",
    "base": "зубы",
    "lexeme": "зубы́",
    "contrast": "зу́бы",
    "reason": "technical plural requires зубья, not a stress-only contrast"
  },
  {
    "utt_id": "homo_730",
    "base": "зубы",
    "lexeme": "зубы́",
    "contrast": "зу́бы",
    "reason": "technical plural requires зубья, not a stress-only contrast"
  },
  {
    "utt_id": "homo_731",
    "base": "зубы",
    "lexeme": "зубы́",
    "contrast": "зу́бы",
    "reason": "technical plural requires зубья, not a stress-only contrast"
  },
  {
    "utt_id": "homo_732",
    "base": "зубы",
    "lexeme": "зубы́",
    "contrast": "зу́бы",
    "reason": "technical plural requires зубья, not a stress-only contrast"
  },
  {
    "utt_id": "homo_733",
    "base": "зубы",
    "lexeme": "зубы́",
    "contrast": "зу́бы",
    "reason": "technical plural requires зубья, not a stress-only contrast"
  },
  {
    "utt_id": "homo_734",
    "base": "зубы",
    "lexeme": "зубы́",
    "contrast": "зу́бы",
    "reason": "technical plural requires зубья, not a stress-only contrast"
  },
  {
    "utt_id": "homo_735",
    "base": "зубы",
    "lexeme": "зубы́",
    "contrast": "зу́бы",
    "reason": "technical plural requires зубья, not a stress-only contrast"
  },
  {
    "utt_id": "homo_736",
    "base": "листья",
    "lexeme": "ли́стья",
    "contrast": "листь́я",
    "reason": "paper sheets are листы, not a stress-only contrast"
  },
  {
    "utt_id": "homo_737",
    "base": "листья",
    "lexeme": "ли́стья",
    "contrast": "листь́я",
    "reason": "paper sheets are листы, not a stress-only contrast"
  },
  {
    "utt_id": "homo_738",
    "base": "листья",
    "lexeme": "ли́стья",
    "contrast": "листь́я",
    "reason": "paper sheets are листы, not a stress-only contrast"
  },
  {
    "utt_id": "homo_739",
    "base": "листья",
    "lexeme": "ли́стья",
    "contrast": "листь́я",
    "reason": "paper sheets are листы, not a stress-only contrast"
  },
  {
    "utt_id": "homo_740",
    "base": "листья",
    "lexeme": "ли́стья",
    "contrast": "листь́я",
    "reason": "paper sheets are листы, not a stress-only contrast"
  },
  {
    "utt_id": "homo_741",
    "base": "листья",
    "lexeme": "ли́стья",
    "contrast": "листь́я",
    "reason": "paper sheets are листы, not a stress-only contrast"
  },
  {
    "utt_id": "homo_742",
    "base": "листья",
    "lexeme": "ли́стья",
    "contrast": "листь́я",
    "reason": "paper sheets are листы, not a stress-only contrast"
  },
  {
    "utt_id": "homo_743",
    "base": "листья",
    "lexeme": "ли́стья",
    "contrast": "листь́я",
    "reason": "paper sheets are листы, not a stress-only contrast"
  },
  {
    "utt_id": "homo_744",
    "base": "листья",
    "lexeme": "ли́стья",
    "contrast": "листь́я",
    "reason": "paper sheets are листы, not a stress-only contrast"
  },
  {
    "utt_id": "homo_745",
    "base": "листья",
    "lexeme": "ли́стья",
    "contrast": "листь́я",
    "reason": "paper sheets are листы, not a stress-only contrast"
  },
  {
    "utt_id": "homo_746",
    "base": "листья",
    "lexeme": "ли́стья",
    "contrast": "листь́я",
    "reason": "paper sheets are листы, not a stress-only contrast"
  },
  {
    "utt_id": "homo_747",
    "base": "листья",
    "lexeme": "ли́стья",
    "contrast": "листь́я",
    "reason": "paper sheets are листы, not a stress-only contrast"
  },
  {
    "utt_id": "homo_748",
    "base": "листья",
    "lexeme": "ли́стья",
    "contrast": "листь́я",
    "reason": "paper sheets are листы, not a stress-only contrast"
  },
  {
    "utt_id": "homo_749",
    "base": "листья",
    "lexeme": "ли́стья",
    "contrast": "листь́я",
    "reason": "paper sheets are листы, not a stress-only contrast"
  },
  {
    "utt_id": "homo_750",
    "base": "листья",
    "lexeme": "ли́стья",
    "contrast": "листь́я",
    "reason": "paper sheets are листы, not a stress-only contrast"
  },
  {
    "utt_id": "homo_751",
    "base": "листья",
    "lexeme": "листь́я",
    "contrast": "ли́стья",
    "reason": "paper sheets are листы, not a stress-only contrast"
  },
  {
    "utt_id": "homo_752",
    "base": "листья",
    "lexeme": "листь́я",
    "contrast": "ли́стья",
    "reason": "paper sheets are листы, not a stress-only contrast"
  },
  {
    "utt_id": "homo_753",
    "base": "листья",
    "lexeme": "листь́я",
    "contrast": "ли́стья",
    "reason": "paper sheets are листы, not a stress-only contrast"
  },
  {
    "utt_id": "homo_754",
    "base": "листья",
    "lexeme": "листь́я",
    "contrast": "ли́стья",
    "reason": "paper sheets are листы, not a stress-only contrast"
  },
  {
    "utt_id": "homo_755",
    "base": "листья",
    "lexeme": "листь́я",
    "contrast": "ли́стья",
    "reason": "paper sheets are листы, not a stress-only contrast"
  },
  {
    "utt_id": "homo_756",
    "base": "листья",
    "lexeme": "листь́я",
    "contrast": "ли́стья",
    "reason": "paper sheets are листы, not a stress-only contrast"
  },
  {
    "utt_id": "homo_757",
    "base": "листья",
    "lexeme": "листь́я",
    "contrast": "ли́стья",
    "reason": "paper sheets are листы, not a stress-only contrast"
  },
  {
    "utt_id": "homo_758",
    "base": "листья",
    "lexeme": "листь́я",
    "contrast": "ли́стья",
    "reason": "paper sheets are листы, not a stress-only contrast"
  },
  {
    "utt_id": "homo_759",
    "base": "листья",
    "lexeme": "листь́я",
    "contrast": "ли́стья",
    "reason": "paper sheets are листы, not a stress-only contrast"
  },
  {
    "utt_id": "homo_760",
    "base": "листья",
    "lexeme": "листь́я",
    "contrast": "ли́стья",
    "reason": "paper sheets are листы, not a stress-only contrast"
  },
  {
    "utt_id": "homo_761",
    "base": "листья",
    "lexeme": "листь́я",
    "contrast": "ли́стья",
    "reason": "paper sheets are листы, not a stress-only contrast"
  },
  {
    "utt_id": "homo_762",
    "base": "листья",
    "lexeme": "листь́я",
    "contrast": "ли́стья",
    "reason": "paper sheets are листы, not a stress-only contrast"
  },
  {
    "utt_id": "homo_763",
    "base": "листья",
    "lexeme": "листь́я",
    "contrast": "ли́стья",
    "reason": "paper sheets are листы, not a stress-only contrast"
  },
  {
    "utt_id": "homo_764",
    "base": "листья",
    "lexeme": "листь́я",
    "contrast": "ли́стья",
    "reason": "paper sheets are листы, not a stress-only contrast"
  },
  {
    "utt_id": "homo_765",
    "base": "листья",
    "lexeme": "листь́я",
    "contrast": "ли́стья",
    "reason": "paper sheets are листы, not a stress-only contrast"
  },
  {
    "utt_id": "homo_766",
    "base": "цветы",
    "lexeme": "цв́еты",
    "contrast": "цвет́ы",
    "reason": "colours are цвета; flowers and the proposed contrast are not homographs"
  },
  {
    "utt_id": "homo_767",
    "base": "цветы",
    "lexeme": "цв́еты",
    "contrast": "цвет́ы",
    "reason": "colours are цвета; flowers and the proposed contrast are not homographs"
  },
  {
    "utt_id": "homo_768",
    "base": "цветы",
    "lexeme": "цв́еты",
    "contrast": "цвет́ы",
    "reason": "colours are цвета; flowers and the proposed contrast are not homographs"
  },
  {
    "utt_id": "homo_769",
    "base": "цветы",
    "lexeme": "цв́еты",
    "contrast": "цвет́ы",
    "reason": "colours are цвета; flowers and the proposed contrast are not homographs"
  },
  {
    "utt_id": "homo_770",
    "base": "цветы",
    "lexeme": "цв́еты",
    "contrast": "цвет́ы",
    "reason": "colours are цвета; flowers and the proposed contrast are not homographs"
  },
  {
    "utt_id": "homo_771",
    "base": "цветы",
    "lexeme": "цв́еты",
    "contrast": "цвет́ы",
    "reason": "colours are цвета; flowers and the proposed contrast are not homographs"
  },
  {
    "utt_id": "homo_772",
    "base": "цветы",
    "lexeme": "цв́еты",
    "contrast": "цвет́ы",
    "reason": "colours are цвета; flowers and the proposed contrast are not homographs"
  },
  {
    "utt_id": "homo_773",
    "base": "цветы",
    "lexeme": "цв́еты",
    "contrast": "цвет́ы",
    "reason": "colours are цвета; flowers and the proposed contrast are not homographs"
  },
  {
    "utt_id": "homo_774",
    "base": "цветы",
    "lexeme": "цв́еты",
    "contrast": "цвет́ы",
    "reason": "colours are цвета; flowers and the proposed contrast are not homographs"
  },
  {
    "utt_id": "homo_775",
    "base": "цветы",
    "lexeme": "цв́еты",
    "contrast": "цвет́ы",
    "reason": "colours are цвета; flowers and the proposed contrast are not homographs"
  },
  {
    "utt_id": "homo_776",
    "base": "цветы",
    "lexeme": "цв́еты",
    "contrast": "цвет́ы",
    "reason": "colours are цвета; flowers and the proposed contrast are not homographs"
  },
  {
    "utt_id": "homo_777",
    "base": "цветы",
    "lexeme": "цв́еты",
    "contrast": "цвет́ы",
    "reason": "colours are цвета; flowers and the proposed contrast are not homographs"
  },
  {
    "utt_id": "homo_778",
    "base": "цветы",
    "lexeme": "цв́еты",
    "contrast": "цвет́ы",
    "reason": "colours are цвета; flowers and the proposed contrast are not homographs"
  },
  {
    "utt_id": "homo_779",
    "base": "цветы",
    "lexeme": "цв́еты",
    "contrast": "цвет́ы",
    "reason": "colours are цвета; flowers and the proposed contrast are not homographs"
  },
  {
    "utt_id": "homo_780",
    "base": "цветы",
    "lexeme": "цв́еты",
    "contrast": "цвет́ы",
    "reason": "colours are цвета; flowers and the proposed contrast are not homographs"
  },
  {
    "utt_id": "homo_781",
    "base": "цветы",
    "lexeme": "цвет́ы",
    "contrast": "цв́еты",
    "reason": "colours are цвета; flowers and the proposed contrast are not homographs"
  },
  {
    "utt_id": "homo_782",
    "base": "цветы",
    "lexeme": "цвет́ы",
    "contrast": "цв́еты",
    "reason": "colours are цвета; flowers and the proposed contrast are not homographs"
  },
  {
    "utt_id": "homo_783",
    "base": "цветы",
    "lexeme": "цвет́ы",
    "contrast": "цв́еты",
    "reason": "colours are цвета; flowers and the proposed contrast are not homographs"
  },
  {
    "utt_id": "homo_784",
    "base": "цветы",
    "lexeme": "цвет́ы",
    "contrast": "цв́еты",
    "reason": "colours are цвета; flowers and the proposed contrast are not homographs"
  },
  {
    "utt_id": "homo_785",
    "base": "цветы",
    "lexeme": "цвет́ы",
    "contrast": "цв́еты",
    "reason": "colours are цвета; flowers and the proposed contrast are not homographs"
  },
  {
    "utt_id": "homo_786",
    "base": "цветы",
    "lexeme": "цвет́ы",
    "contrast": "цв́еты",
    "reason": "colours are цвета; flowers and the proposed contrast are not homographs"
  },
  {
    "utt_id": "homo_787",
    "base": "цветы",
    "lexeme": "цвет́ы",
    "contrast": "цв́еты",
    "reason": "colours are цвета; flowers and the proposed contrast are not homographs"
  },
  {
    "utt_id": "homo_788",
    "base": "цветы",
    "lexeme": "цвет́ы",
    "contrast": "цв́еты",
    "reason": "colours are цвета; flowers and the proposed contrast are not homographs"
  },
  {
    "utt_id": "homo_789",
    "base": "цветы",
    "lexeme": "цвет́ы",
    "contrast": "цв́еты",
    "reason": "colours are цвета; flowers and the proposed contrast are not homographs"
  },
  {
    "utt_id": "homo_790",
    "base": "цветы",
    "lexeme": "цвет́ы",
    "contrast": "цв́еты",
    "reason": "colours are цвета; flowers and the proposed contrast are not homographs"
  },
  {
    "utt_id": "homo_791",
    "base": "цветы",
    "lexeme": "цвет́ы",
    "contrast": "цв́еты",
    "reason": "colours are цвета; flowers and the proposed contrast are not homographs"
  },
  {
    "utt_id": "homo_792",
    "base": "цветы",
    "lexeme": "цвет́ы",
    "contrast": "цв́еты",
    "reason": "colours are цвета; flowers and the proposed contrast are not homographs"
  },
  {
    "utt_id": "homo_793",
    "base": "цветы",
    "lexeme": "цвет́ы",
    "contrast": "цв́еты",
    "reason": "colours are цвета; flowers and the proposed contrast are not homographs"
  },
  {
    "utt_id": "homo_794",
    "base": "цветы",
    "lexeme": "цвет́ы",
    "contrast": "цв́еты",
    "reason": "colours are цвета; flowers and the proposed contrast are not homographs"
  },
  {
    "utt_id": "homo_795",
    "base": "цветы",
    "lexeme": "цвет́ы",
    "contrast": "цв́еты",
    "reason": "colours are цвета; flowers and the proposed contrast are not homographs"
  }
]
```

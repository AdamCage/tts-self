# Bank entity v2 audit

Every critical span was re-derived from the corrected deterministic frontend. Changes are benchmark-label corrections; item texts are frozen.

## Summary

- Items audited: **315**
- Critical spans audited: **315**
- Findings classified: **48**
- Labels changed: **44**
- Version: `1.0.0` → `2.0.0`
- Classification counts: `{"annotation_error": 44, "legal_variant_or_orthography": 4, "verifier_error": 21}`

## Verifier fixes found during audit

- dotted decimals were interpreted as clock times without context: `bank_031, bank_033, bank_038, bank_042`
- ungrouped four-digit ruble amounts missed the money matcher: `bank_044, bank_060`
- genitive ordinal suffix -го was emitted as nominative: `bank_285, bank_286, bank_287`
- four-digit Russian area codes lost their phone prefix: `bank_146, bank_147`
- established financial Latin acronyms lacked dictionary readings: `bank_216, bank_218, bank_219, bank_221, bank_225, bank_226`
- established loanwords lacked or had incorrect dictionary readings: `bank_234, bank_235, bank_239, bank_244`

## Changed labels

| utt_id | category | fragment | before | after | classification |
|---|---|---|---|---|---|
| bank_058 | money | 500 руб. 00 коп. | пятьсот рублей | пятьсот рублей ноль копеек | annotation_error |
| bank_062 | money | 42 руб. 42 коп. | сорок два рубля сорок две копеек | сорок два рубля сорок две копейки | annotation_error |
| bank_128 | phones | +7 900 123-45-67 | плюс семь девять ноль ноль один два три четыре пять шесть семь | плюс семь девятьсот сто двадцать три сорок пять шестьдесят семь | annotation_error |
| bank_129 | phones | +7 903 555-01-02 | плюс семь девять ноль три пять пять пять ноль один ноль два | плюс семь девятьсот три пятьсот пятьдесят пять ноль один ноль два | annotation_error |
| bank_130 | phones | +7 495 123-45-67 | плюс семь четыре девять пять один два три четыре пять шесть семь | плюс семь четыреста девяносто пять сто двадцать три сорок пять шестьдесят семь | annotation_error |
| bank_131 | phones | +7 812 200-30-40 | плюс семь восемь один два два ноль ноль три ноль четыре ноль | плюс семь восемьсот двенадцать двести тридцать сорок | annotation_error |
| bank_132 | phones | +7 916 777-88-99 | плюс семь девять один шесть семь семь семь восемь восемь девять девять | плюс семь девятьсот шестнадцать семьсот семьдесят семь восемьдесят восемь девяносто девять | annotation_error |
| bank_133 | phones | +7 999 111-22-33 | плюс семь девять девять девять один один один два два три три | плюс семь девятьсот девяносто девять сто одиннадцать двадцать два тридцать три | annotation_error |
| bank_134 | phones | +7 901 234-56-78 | плюс семь девять ноль один два три четыре пять шесть семь восемь | плюс семь девятьсот один двести тридцать четыре пятьдесят шесть семьдесят восемь | annotation_error |
| bank_135 | phones | +7 926 000-11-22 | плюс семь девять два шесть ноль ноль ноль один один два два | плюс семь девятьсот двадцать шесть ноль ноль ноль одиннадцать двадцать два | annotation_error |
| bank_136 | phones | +7 983 444-55-66 | плюс семь девять восемь три четыре четыре четыре пять пять шесть шесть | плюс семь девятьсот восемьдесят три четыреста сорок четыре пятьдесят пять шестьдесят шесть | annotation_error |
| bank_137 | phones | +7 800 555-35-35 | плюс семь восемь ноль ноль пять пять пять три пять три пять | плюс семь восемьсот пятьсот пятьдесят пять тридцать пять тридцать пять | annotation_error |
| bank_138 | phones | +7 495 789-00-11 | плюс семь четыре девять пять семь восемь девять ноль ноль один один | плюс семь четыреста девяносто пять семьсот восемьдесят девять ноль ноль одиннадцать | annotation_error |
| bank_139 | phones | +7 343 200-10-20 | плюс семь три четыре три два ноль ноль один ноль два ноль | плюс семь триста сорок три двести десять двадцать | annotation_error |
| bank_140 | phones | +7 383 300-40-50 | плюс семь три восемь три три ноль ноль четыре ноль пять ноль | плюс семь триста восемьдесят три триста сорок пятьдесят | annotation_error |
| bank_141 | phones | +7 846 111-22-33 | плюс семь восемь четыре шесть один один один два два три три | плюс семь восемьсот сорок шесть сто одиннадцать двадцать два тридцать три | annotation_error |
| bank_142 | phones | +7 861 555-66-77 | плюс семь восемь шесть один пять пять пять шесть шесть семь семь | плюс семь восемьсот шестьдесят один пятьсот пятьдесят пять шестьдесят шесть семьдесят семь | annotation_error |
| bank_143 | phones | +7 351 222-33-44 | плюс семь три пять один два два два три три четыре четыре | плюс семь триста пятьдесят один двести двадцать два тридцать три сорок четыре | annotation_error |
| bank_144 | phones | +7 831 100-20-30 | плюс семь восемь три один один ноль ноль два ноль три ноль | плюс семь восемьсот тридцать один сто двадцать тридцать | annotation_error |
| bank_145 | phones | +7 843 700-80-90 | плюс семь восемь четыре три семь ноль ноль восемь ноль девять ноль | плюс семь восемьсот сорок три семьсот восемьдесят девяносто | annotation_error |
| bank_146 | phones | +7 4212 12-34-56 | плюс семь четыре два один два один два три четыре пять шесть | плюс семь четыре два один два двенадцать тридцать четыре пятьдесят шесть | annotation_error |
| bank_147 | phones | +7 3952 60-70-80 | плюс семь три девять пять два шесть ноль семь ноль восемь ноль | плюс семь три девять пять два шестьдесят семьдесят восемьдесят | annotation_error |
| bank_148 | phones | 8 900 123-45-67 | восемь девять ноль ноль один два три четыре пять шесть семь | восемь девятьсот сто двадцать три сорок пять шестьдесят семь | annotation_error |
| bank_284 | ordinals | 100-й | сто | сотый | annotation_error |
| bank_295 | ranges | 10–15 | от десять до пятнадцать | от десяти до пятнадцати | annotation_error |
| bank_296 | ranges | 5-7 | от пять до семь | от пяти до семи | annotation_error |
| bank_297 | ranges | с 9 до 18 | с девять до восемнадцать | с девяти до восемнадцати | annotation_error |
| bank_298 | ranges | с 10 до 19 | с десять до девятнадцать | с десяти до девятнадцати | annotation_error |
| bank_299 | ranges | 1–3 | от один до три | от одного до трёх | annotation_error |
| bank_300 | ranges | 100–200 | от сто до двести | от ста до двухсот | annotation_error |
| bank_301 | ranges | 2–5 | от два до пять | от двух до пяти | annotation_error |
| bank_302 | ranges | с 8 до 20 | с восемь до двадцать | с восьми до двадцати | annotation_error |
| bank_303 | ranges | 15–20 | от пятнадцать до двадцать | от пятнадцати до двадцати | annotation_error |
| bank_304 | ranges | 30–45 | от тридцать до сорок пять | от тридцати до сорока пяти | annotation_error |
| bank_305 | ranges | с 1 до 10 | с один до десять | с одного до десяти | annotation_error |
| bank_306 | ranges | 50–70 | от пятьдесят до семьдесят | от пятидесяти до семидесяти | annotation_error |
| bank_307 | ranges | 3–6 | от три до шесть | от трёх до шести | annotation_error |
| bank_308 | ranges | 12–24 | от двенадцать до двадцать четыре | от двенадцати до двадцати четырёх | annotation_error |
| bank_309 | ranges | с 14 до 16 | с четырнадцать до шестнадцать | с четырнадцати до шестнадцати | annotation_error |
| bank_310 | ranges | 7–14 | от семь до четырнадцать | от семи до четырнадцати | annotation_error |
| bank_311 | ranges | 20–25 | от двадцать до двадцать пять | от двадцати до двадцати пяти | annotation_error |
| bank_312 | ranges | с 0 до 100 | с ноль до сто | с нуля до ста | annotation_error |
| bank_313 | ranges | 4–8 | от четыре до восемь | от четырёх до восьми | annotation_error |
| bank_314 | ranges | 90–180 | от девяносто до сто восемьдесят | от девяноста до ста восьмидесяти | annotation_error |

## Legal variants retained

| utt_id | category | retained | frontend variant |
|---|---|---|---|
| bank_220 | lat_abbr | а-пи-ай | эй-пи-ай |
| bank_223 | lat_abbr | кэ-пи-ай | кей-пи-ай |
| bank_227 | lat_abbr | эй-ти-эм | эй-тэ-эм |
| bank_236 | loanwords | вотсап | ватсап |

## Manual spot-check

| utt_id | category | fragment | expected_after | status |
|---|---|---|---|---|
| bank_031 | decimals | 0.01 | ноль точка ноль один | checked |
| bank_044 | money | 1234 руб. 50 коп. | одна тысяча двести тридцать четыре рубля пятьдесят копеек | checked |
| bank_128 | phones | +7 900 123-45-67 | плюс семь девятьсот сто двадцать три сорок пять шестьдесят семь | checked |
| bank_166 | accounts | 4260 9876 5432 1098 | четыре два шесть ноль девять восемь семь шесть пять четыре три два один ноль девять восемь | checked |
| bank_190 | codes | 046015207 | ноль четыре шесть ноль один пять два ноль семь | checked |
| bank_191 | ru_abbr | ИНН | и эн эн | checked |
| bank_234 | loanwords | Apple Pay | эпл пэй | checked |
| bank_263 | units | 25 °C | двадцать пять градусов Цельсия | checked |
| bank_284 | ordinals | 100-й | сотый | checked |
| bank_295 | ranges | 10–15 | от десяти до пятнадцати | checked |

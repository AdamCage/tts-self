# §6 отчёт — S2 CTC second voice on all 673 CV17 exclude

## 1. Идентификатор задачи

**S2** — прогнан CTC-второй голос QC на всех 673 exclude CV17; доля
возврата на полном множестве названа и сопоставлена с выборкой P1 71.92 %;
правило масштабирования для T2/T3 записано в `docs/corpus_inventory.md`.

## 2. Критерий приёмки

Цитата плана:

> - доля возврата на полном множестве названа числом и сопоставлена с 71.92 % на
>   выборке
> - правило для масштабирования записано в док
> - пороги QC не менялись

| Критерий | Результат |
|---|---|
| Доля возврата на полном множестве | **достигнут**: `returned_fraction = 0.7743606386144409` → **77.44 %** exclude-часов (508 utts / 0.5951 ч из 673 / 0.7686 ч). Выборка P1: **71.92 %** (`out/e6_recheck.json`). Дельта: **+5.52 п.п.** |
| Правило для T2/T3 в доке | **достигнут**: раздел «Правило масштабирования для T2 / T3» в `docs/corpus_inventory.md` |
| Пороги QC не менялись | **достигнут**: `clean_max_cer = 0.15`, `review_max_cer = 0.25` в артефакте и в CLI; те же значения, что pilot QC / P1 E6 |

Дополнительно (шаг 2 плана): распределение расхождений Whisper↔CTC —
в `out/e6_full_cv17.json` ключи `disagreement_n` / `disagreement_hours` и
таблица в E6-разделе дока. Ошибок CTC: **0**.

## 3. Изменённые и созданные файлы

- `scripts/61_recheck_excluded_ctc.py` — сводка: `still_exclude_n`,
  `ctc_tag_counts`, бакеты `disagreement_*`, якорь `sample_p1_returned_fraction`
- `docs/corpus_inventory.md` — раздел E6: полный прогон + правило T2/T3
- `out/e6_full_cv17.json` — артефакт полного прогона
- `out/_n2_s2_run.sh` — launcher WSL
- `.cursor/results/n2_first_execution/logs/s2_ctc_full_exclude.log`
- `.cursor/results/n2_first_execution/S2_REPORT.md` (этот файл)

## 4. Артефакты

| Путь | Содержание |
|---|---|
| `out/e6_full_cv17.json` | полный отчёт 673 utts |
| `out/e6_recheck.json` | выборка P1 200 utts (не перезаписывался) |
| `docs/corpus_inventory.md` § E6 | числа + правило T2/T3 |
| `.cursor/results/n2_first_execution/logs/s2_ctc_full_exclude.log` | консольный лог |

Ключевые числа из `out/e6_full_cv17.json`:

- `n_checked = 673`, `n_exclude_available = 673`, `n_errors = 0`
- `returned_n = 508` (clean 434, review 74), `still_exclude_n = 165`
- `returned_hours = 0.5951441898148158`
- `original_exclude_hours_checked = 0.7685620370370373`
- `returned_fraction = 0.7743606386144409`

## 5. Команды для воспроизведения

```bash
wsl -d Ubuntu -- bash -lc '
cd ~/projects/tts-self
source .venv/bin/activate
export HF_HOME=$HOME/hf_cache
export PYTHONPATH=/mnt/c/projects/tts-self/src
export MINISPEECH_ALLOW_DIRTY=1
git rev-parse --short HEAD
git status --porcelain | head
sed -i "s/\r$//" /mnt/c/projects/tts-self/out/_n2_s2_run.sh
bash /mnt/c/projects/tts-self/out/_n2_s2_run.sh
'
```

Эквивалентный прямой вызов:

```bash
python /mnt/c/projects/tts-self/scripts/61_recheck_excluded_ctc.py \
  --config /mnt/c/projects/tts-self/configs/data_common_voice_ru17.yaml \
  --model ~/models/wav2vec2-large-xlsr-53-russian \
  --out /mnt/c/projects/tts-self/out/e6_full_cv17.json \
  --max-utts 0 \
  --clean-max 0.15 \
  --review-max 0.25
```

Провенанс прогона (из лога):

- `HEAD=2c4c4bd`
- `DIRTY=0` (`git status --porcelain` пуст в WSL-клоне)
- GPU: ждал завершения C1 (`04_encode_latents.py`, util ~63 %), затем
  запустил S2; C1 не убивался
- wall-clock S2 ≈ 18 с после загрузки модели

## 6. Найдено попутно

- C1 финализатор записал `"n_index": 0` при `disk: 244M` latents — возможный
  баг индексации шардов C1; S2 не чинил.
- Среди возвращённых **168 / 508** — короткие числовые; это подтверждает
  качественный вывод P1 о расхождении моделей на числовых фразах, но на
  полном множестве доля возврата ещё выше выборки.
- Скрипт по умолчанию `--max-utts 200`; полный прогон требует `--max-utts 0`.

## 7. Блокеры

Нет. Критерий приёмки достигнут.

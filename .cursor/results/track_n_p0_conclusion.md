# Заключение — Track N, фаза P0

Дата: 2026-07-31.  
План: `.cursor/plans/track_n_phase0_execution.plan.md` (N00–N15).  
Числа ниже только из артефактов на диске; выдуманных значений нет.

---

## Статус плана

**Фаза P0 закрыта по исполнительным задачам N00–N15** (зонды З1–З4, бенчмарк, инфра, гейт N14, скелет T0 / N15).

Гейт выхода P0 (решения по бэкбону / ударению / частоте кадров) зафиксирован в
[`.cursor/methodology/11_probes.md`](../methodology/11_probes.md) и
[`.cursor/methodology/05_backbone.md`](../methodology/05_backbone.md).

Оговорки (не блокеры гейта N14, но пробелы фазы):

- технический долг «свернуть ~60 `scripts/_*.py`» в плане всё ещё отмечен как незакрытый checklist;
- код/docs P0 **не закоммичены** (по правилу плана — без явного указания пользователя);
- N15 T0 — skeleton/overfit на ~1 ч, **не** продуктовый токенизатор (copy-synth CER ≫ якоря E2).

---

## Три решения гейта P0 (N14)

| # | Вопрос | Решение | Опора |
|---|--------|---------|-------|
| 1 | Бэкбон | **Свой ~0.5B (N-Lite) на критическом пути.** GigaChat 10B — кандидат N-Pro, но на RTX 4080 Super 16 ГБ **не** брать в realtime-петлю | З1 + З4 |
| 2 | Форма ударения | **Отдельный канал признаков** параллельно токенам; U+0301 строкой ненадёжен | З1 |
| 3 | Частота кадров T0 | **7.5 Гц** (геометрия референса) | З2 |

Источник формулировок: `.cursor/methodology/11_probes.md` § «Решения гейта P0 (N14)».

---

## Ключевые числа и артефакты

### З1 — фертильность токенизаторов (`out/probe_tokenizer_fertility.json`)

Проверено на Linux-клоне: `/home/adam/projects/tts-self/out/probe_tokenizer_fertility.json`
(зеркало Windows: `C:\projects\tts-self\out\probe_tokenizer_fertility.json`).

| Модель | Доля кириллицы в словаре | ruslan_200 ток/слово | U+0301 single-token | in-word intact (ruslan_200) |
|--------|-------------------------:|---------------------:|---------------------|----------------------------:|
| Qwen3.5-0.8B-Base | 0.075 | 3.41 | **no** | 0.0 |
| GigaChat3.1-10B | 0.295 | 3.32 | **no** | 0.195 |

### З2 — референсный токенизатор (`out/probe_ref_tokenizer_domains.json`)

| Условие | mean CER | Примечание |
|---------|---------:|------------|
| E2 anchor (gt latent decode, n=50) | **0.07253** | `anchor_reproduced=true`, Δ vs 0.073 = 0.00047 |
| E2 + telephony µ-law sim | **0.08720** | ΔCER vs clean = **+0.01467** |
| bank_entity source (Track A synth wav, n=40) | 0.200 / DER 0.479 / EER 0.525 | не GT corpus wav |
| bank_entity VibeVoice roundtrip | 0.496 / DER **0.532** / EER 0.675 | осторожно интерпретировать |
| multi-speaker | — | **skipped** (локально только RUSLAN) |

### З3 — инвентаризация (`out/probe_corpus_inventory.json`, `docs/corpus_inventory.md`)

- T1/T2/T3 по объёму **достижимы** (Golos 1240 ч; SOVA RU ~17850; YODAS RU ~39622).
- Свободного **native 8 кГц** telephony с clear free/commercial лицензией **не найдено**.
- OpenSTT phone ~820 ч, но 16 кГц + NC.

### З4 — профиль бэкбона (`out/probe_backbone_profile.json`)

GPU: RTX 4080 SUPER ~16 ГБ.

| Вариант | Peak alloc (после load / forward) | Forward latency |
|---------|-----------------------------------|-----------------|
| Qwen3.5-0.8B bf16 | ~1.44 / ~1.45 ГБ | ~**105 мс** (seq≈12) |
| GigaChat3.1-10B bf16 | ~**20.9** / ~20.4 ГБ | ~136 мс (seq≈11; shared/oversubscribe) |
| GigaChat bf16_auto_offload | ~13.5 / ~**13.7** ГБ | ~**1.08 с** |
| GigaChat int8/int4 | load **fail** | — |
| GigaChat FP8 native | load ok, forward **fail** (`kernels`) | — |

### N10 — baseline Track A critical (`out/baseline_track_a_critical.json`)

Ckpt: `runs/mel_stress_v01_s4/best.pt`, vocoder neural. Сводка также в
`docs/frozen_numbers_mel80.md` § «Track N P0 baseline (N10)».

| Set | n | mean CER | DER | EER | SA |
|-----|--:|---------:|----:|----:|----|
| demo5 | 5 | 0.0935 | — | — | — |
| val100 | 100 | **0.0310** | — | — | — |
| bank_entity | 315 | 0.1616 | **0.3647** | **0.4889** | — |
| homograph | 895 | 0.0916 | — | — | **skipped** (n_sa_skipped=895) |
| latin | 117 | 0.2408 | — | — | — |

Copy-synth topline val100: real wav CER 0.0601; GT mel→neural 0.0640
(поля в JSON / таблица в `docs/frozen_numbers_mel80.md`).

Eval-set hashes: `out/n10_eval_set_hashes.txt`; suite YAML:
`out/eval_mel_suite_n10.yaml` (latin **не** в основном `configs/eval_mel_suite.yaml`).

### N15 — T0 σ-VAE (`runs/tokenizer_t0_small/`, WSL Linux FS)

| Поле | Значение | Артефакт |
|------|----------|----------|
| Данные | 1236 utts / ~0.999 ч | `train_summary.json`, `run_manifest.json` |
| Params | ~26.1M (enc+dec) | same |
| Loss | early 3.27 → late 2.54 → best **2.25**; `monotone_drop=true` | `train_summary.json` |
| Copy-synth mean CER | **0.844** (n=32 train subset) | `copy_synth_train1h.json`, `n15_acceptance.json` |
| Lean overfit CER | **0.0** (1 фраза) | `n15_acceptance.json` |
| Target / rate | `sigma_vae64_t0`, 7.5 Гц, reconstruction_only | `run_manifest.json` |
| git_sha run | `cd08a26e…` | `run_manifest.json` |

Это **roundtrip токенизатора**, не TTS. Качество copy-synth на 1 ч ожидаемо далеко от якоря E2 0.073.

---

## Что сделано по волнам (кратко)

| Волна | Задачи | Суть |
|-------|--------|------|
| 0 | N00–N03 | Расчистка/реестры (частично), З1 fertility, З3 inventory |
| 1 | N04–N07 | Latin translit, bank/homograph/latin sets, DER/EER/SA |
| 2 | N08–N10 | З2 ref tokenizer domains, З4 backbone profile, Track A baseline N10 |
| 3 | N11–N13 | Multi-corpus manifest, telephony sim (N12), DDP/resume smoke |
| 4 | N14–N15 | Гейт P0 записан; T0 σ-VAE skeleton `tokenizer_t0_small` |

Детали зондов и решений: `.cursor/methodology/11_probes.md`.

---

## Пробелы и попутные находки

1. **SA = skipped** на homograph (N10): ASR не различает целевые лексемы омографов — допустимый исход N07/N10, не «провал метрики».
2. **Native telephony 8 кГц** в открытом free/commercial виде не найден (З3); работаем через симуляцию N12 + будущий собственный/платный материал.
3. **latin** живёт в `configs/eval_sets/latin.json` и в временном suite `out/eval_mel_suite_n10.yaml`, но **не** в основном `configs/eval_mel_suite.yaml`.
4. **Коммитов по P0 нет** — артефакты и код на диске Windows/WSL, git history не отражает фазу.
5. З2 bank_entity roundtrip опирается на **Track A synth** источники, не GT wav — DER/CER там осторожно.
6. Multi-speaker в З2 **skipped**.
7. N15 T0 copy-synth CER 0.84 — скелет/overfit, не претензия на N0.
8. Checklist «свернуть `scripts/_*.py`» в `11_probes.md` остаётся открытым.

---

## Что дальше (P1–P5)

По [`09_roadmap.md`](../methodology/09_roadmap.md):

| Фаза | Фокус | Гейт выхода |
|------|-------|-------------|
| **P1** Данные | Многокорпусный пайплайн, реестр, псевдоразметка, telephony data | объёмы T1–T2 доступны и размечены |
| **P2** Токенизаторы | T0→T3, semantic tokenizer, кривая 7.5/12.5/25 Гц, узкополосный | **N0**, **N1** (copy-synth ≤ E2 0.073) |
| **P3** Вторая стадия | LLM + diffusion, N-Lite ~0.5B, text-usage | **N2**, **N3** |
| **P4** Русская специфика | Канал Н1, критические спаны Н2 | **N4**, **N5** |
| **P5** Продукт | Telephony product, N-Pro, публикация | **N6**, **N7**, **N8** |

Параллельно: довести latin в основной suite YAML; коммит/синк Windows↔WSL по решению пользователя; продолжить расчистку одноразовых скриптов.

---

## Где слушать

Все production/N15 wav лежат в **Linux FS** WSL (`/home/adam/projects/tts-self/...`), кроме N12 sample на Windows.  
В Explorer: префикс `\\wsl.localhost\Ubuntu\home\adam\projects\tts-self\...`.

### 1. Production Track A (intelligibility) — слушать в первую очередь

**Модель:** `runs/mel_stress_v01_s4/best.pt` + HiFi-GAN neural.  
**Не** latent Talker / CFM.

Готовые demo5 (N10 suite, 2026-07-31):

| # | WSL path |
|---|----------|
| dir | `/home/adam/projects/tts-self/out/eval_suite_mel_stress_v01_s4_wavs/demo5/` |
| 01 | `.../01_привет_я_полностью_самостоятельно_обученная_tts-.wav` |
| 02 | `.../02_как_у_вас_дела_чем_вам_помочь.wav` |
| 03 | `.../03_сегодня_хорошая_погода_можно_пройтись_пешком.wav` |
| 04 | `.../04_пожалуйста_повторите_ещё_раз_чуть_медленнее.wav` |
| 05 | `.../05_спасибо_за_внимание_до_скорой_встречи.wav` |

Дубликат listen-папки (те же 5 фраз):  
`/home/adam/projects/tts-self/out/mel_stress_s4_listen_demo/`

Windows UNC:

```text
\\wsl.localhost\Ubuntu\home\adam\projects\tts-self\out\eval_suite_mel_stress_v01_s4_wavs\demo5
\\wsl.localhost\Ubuntu\home\adam\projects\tts-self\out\mel_stress_s4_listen_demo
```

Открыть:

```powershell
explorer.exe "\\wsl.localhost\Ubuntu\home\adam\projects\tts-self\out\mel_stress_s4_listen_demo"
```

Или в WSL: `ffplay <path>.wav` / двойной клик из Explorer.

Пересинтез (если нужно заново), из WSL `~/projects/tts-self`:

```bash
cd ~/projects/tts-self && source .venv/bin/activate
export HF_HOME=$HOME/hf_cache
python scripts/28_infer_mel.py \
  --ckpt runs/mel_stress_v01_s4/best.pt \
  --demo \
  --prefer-vocoder neural \
  --gap-mode expand_neighbor \
  --out out/mel_stress_s4_listen_demo_fresh
```

Полный suite (как N10):

```bash
python scripts/32_eval_mel_suite.py \
  --ckpt runs/mel_stress_v01_s4/best.pt \
  --suite out/eval_mel_suite_n10.yaml \
  --sets demo5 \
  --prefer-vocoder neural \
  --out out/baseline_track_a_critical_demo5_rerun.json
```

### 2. N15 T0 σ-VAE — copy-synth (не TTS)

**Модель:** `runs/tokenizer_t0_small/best.pt` (acoustic σ-VAE roundtrip).

```text
/home/adam/projects/tts-self/runs/tokenizer_t0_small/copy_synth_wavs/
```

32 wav (`ruslan_000000_RUSLAN.wav` …). UNC:

```text
\\wsl.localhost\Ubuntu\home\adam\projects\tts-self\runs\tokenizer_t0_small\copy_synth_wavs
```

Метаданные CER по utt: `runs/tokenizer_t0_small/copy_synth_train1h.json`.

### 3. Прочие свежие demo от P0

| Что | Путь | Примечание |
|-----|------|------------|
| N12 telephony sample (µ-law sim) | `C:\projects\tts-self\out\n12_telephony_000000_RUSLAN.wav` | есть на Windows; JSON: `out/n12_telephony_probe.json` |
| N10 bank source wavs (для З2) | `/home/adam/projects/tts-self/out/n08_bank_src_wavs/*.wav` | Track A synth, не production demo |
| Метрики N10 | `C:\projects\tts-self\out\baseline_track_a_critical.json` | пути wav внутри JSON указывают на Linux |

N10 **не** писал отдельную папку «n10_demo» сверх suite wavs выше — слушать
`eval_suite_mel_stress_v01_s4_wavs/demo5` / `mel_stress_s4_listen_demo`.

---

## Итог одной строкой

P0 закрыт: три решения гейта записаны; Track A production остаётся тем, что
**слушать** для intelligibility; N15 — слышимый, но слабый roundtrip скелета
токенизатора на 1 ч; дальше — P1 данные и P2 доведение T0→N0.

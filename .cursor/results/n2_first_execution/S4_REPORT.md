# §6 отчёт — S4 Paper 1 figures and preference tests

## 1. Идентификатор задачи

**S4** — собраны фигуры Работы 1 из существующих измеренных артефактов;
подготовлен runnable протокол парных предпочтений с ловушками self-vs-self;
человеческое исследование закрыто как **BLOCKED** (слушателей нет); MOS помечен
как **unmeasured**.

## 2. Критерий приёмки

Цитата плана:

> - каждая фигура имеет путь к породившему её артефакту
> - число слушателей и число сравнений названы; биномиальный CI95 приведён
> - MOS в тексте помечен как не измеренный
> - если слушателей собрать не удалось — задача закрывается как блокированная, без
>   подстановки чисел

| Критерий | Результат |
|---|---|
| Каждая фигура → source artifact | **достигнут**: 8/8 `status=ok` в `out/paper1_figures/manifest.json`; таблица в `docs/paper1_figures.md` |
| Слушатели + сравнения названы **или** blocked | **достигнут (BLOCKED)**: `listener_count = 0`; сравнение названо `silero_frontend_vs_track_a` (5 content + 3 traps). Human CI95 **не применим** (нет ответов). Dry-run scoring path показывает Wilson CI95 только на синтетических ответах (`self_check.json`, явно помечено «not human») |
| MOS не измерен | **достигнут**: `mos_status: unmeasured` в docs + `preference_protocol.json` + status.json |
| Без подстановки чисел слушателей | **достигнут**: нет fake MOS / preference rates |

N2 исключён из панели: `out/n2_gate.json` → `verdict: FAIL`.

## 3. Изменённые и созданные файлы

- `scripts/68_paper1_figures.py` — **новый** билдер фигур из JSON
- `scripts/51_preference_test.py` — match-mode order/name, staging, `protocol`, status
- `docs/paper1_figures.md` — **новый** индекс фигур
- `docs/paper1_preference_status.md` — **новый** статус предпочтений
- `docs/paper1_proxy_metrics_trap.md` — §6 обновлён путями; MOS/BLOCKED в шапке
- `docs/paper1_t16_remaining.md` — чеклист S4
- `out/paper1_figures/**` — PNG + JSON + preference dry-run
- `.cursor/results/n2_first_execution/logs/s4_*.log`
- `.cursor/results/n2_first_execution/S4_REPORT.md` (этот файл)

## 4. Артефакты

| Путь | Содержание |
|---|---|
| `out/paper1_figures/fig01_cer_sigma_alpha.{png,json}` | source: `out/latent_error_calibration.json` |
| `out/paper1_figures/fig02_r2_three_systems.{png,json}` | source: `out/external_baseline_silero_v3.json` |
| `out/paper1_figures/fig03_bank_delta_2x2.{png,json}` | source: `out/bank_delta_2x2.json` |
| `out/paper1_figures/fig04_gate0_panel.{png,json}` | sources: head_diagnostics + trivial_baselines |
| `out/paper1_figures/fig05_e2_vs_e4_gap.{png,json}` | sources: calibration + intelligibility_S3 |
| `out/paper1_figures/fig06_diagonality.{png,json}` | source: `out/align_audit.json` |
| `out/paper1_figures/fig07_telephony_channels.{png,json}` | source: `out/p1_telephony_channels.json` |
| `out/paper1_figures/fig08_track_a_ladder.{png,json}` | mel suite + gate_v4_v5 |
| `out/paper1_figures/manifest.json` | сводка, `mos_status=unmeasured` |
| `out/paper1_figures/preference_protocol.json` | `study_status=BLOCKED`, listener_count=0 |
| `out/paper1_figures/preference_silero_fe_vs_track_a/` | ballot/key/html/self_check |

Ключевые измеренные числа (из источников, не пересчитывались заново):

- Gate 0: `r2_overall = -0.110254…`, `gate0_r2_ge_0.28 = false`
- E2 CER ≈ 0.0725 vs E4 S3 CER ≈ 0.889
- bank DER: Silero raw 0.973 / Silero+FE 0.103 / Track A 0.624; Δ frontend−raw = −0.870
- telephony: `difference_detected_any = false` (n=30)

## 5. Команды для воспроизведения

```powershell
cd C:\projects\tts-self
$env:PYTHONPATH = "C:\projects\tts-self\src"
$env:PYTHONIOENCODING = "utf-8"

git rev-parse --short HEAD   # this run: 2c4c4bd
git status --porcelain       # dirty (план R7; S4 без GPU, не чистили дерево)

python scripts/68_paper1_figures.py --out-dir out/paper1_figures

python scripts/51_preference_test.py protocol `
  --out out/paper1_figures/preference_protocol.json

python scripts/51_preference_test.py `
  --dir-a out/n2_first_execution/silero_frontend/demo5 `
  --dir-b out/n2_first_execution/track_a/demo5 `
  --name-a silero_frontend --name-b track_a `
  --match-mode order --n-traps 3 --seed 42 `
  --status blocked_no_listeners `
  --out out/paper1_figures/preference_silero_fe_vs_track_a

python scripts/51_preference_test.py self-check `
  --key out/paper1_figures/preference_silero_fe_vs_track_a/key.json `
  --out out/paper1_figures/preference_silero_fe_vs_track_a/self_check.json

python -m pytest tests/test_preference.py -q   # 2 passed
```

Логи: `.cursor/results/n2_first_execution/logs/s4_figures.log`,
`s4_preference_protocol.log`, `s4_preference_ballot.log`,
`s4_preference_selfcheck.log`, `s4_pytest_preference.log`.

## 6. Найдено попутно

- Demo wav filenames различаются между Silero (`0001_demo_01.wav`) и Track A
  (кириллические имена) — для панели нужен `--match-mode order` (staging).
- Windows-зеркало Track A demo5 имеет повреждённые кириллические имена в FS listing;
  файлы читаются, pairing по sorted order.
- Полные R2 wav-деревья (1256 utts) на Linux; на Windows для S4 использованы demo5.
- `git status` грязный (много файлов других задач этапа) — R7 не выполнялся здесь.
- Stub-генератор `scripts/38_paper1_figure_stubs.py` не удалялся; S4 пишет в
  `out/paper1_figures/`, не в `docs/figures/`.

## 7. Блокеры

- **Человеческие слушатели недоступны** → preference study **BLOCKED**;
  human binomial CI95 и trap_same_rate панели не измерены.
- Это ожидаемое закрытие по плану («если слушателей собрать не удалось — …
  блокированная, без подстановки чисел»), не провал фигур/протокола.

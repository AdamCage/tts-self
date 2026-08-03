---
name: d1b hardstop closure
overview: Закрыть D1b как воспроизводимый HARD-STOP, устранить оставшиеся fail-closed и тестовые дефекты и подготовить чистую границу для отдельного D1c termination-repair. План не пытается переиграть confirm_v2 и не разрешает 1h/5h/full.
todos:
  - id: freeze-hardstop
    content: Зафиксировать seed43 clean FAIL как основание HARD-STOP и корректно классифицировать seed42 dirty eval.
    status: completed
  - id: harden-fail-closed
    content: Закрыть canonical provenance, max_frames, select8 hash и eval-TF-CE escape paths с тестами.
    status: completed
  - id: restore-test-gate
    content: Восстановить latent_io и покрыть atomic checkpoint save, чтобы полный test gate снова воспроизводился.
    status: completed
  - id: reconcile-closure
    content: Согласовать RESULTS, PROTOCOL, formal gate, listen index и статусы плана без изменения frozen artifacts.
    status: completed
  - id: verify-and-package
    content: Пройти WSL test/hash gates и подготовить изолированный D1b source/results пакет к разрешённому commit/push.
    status: completed
isProject: false
---

# План закрытия D1b HARD-STOP

## Граница этапа
- Зафиксировать итог D1b как **HARD-STOP**: сильный content-сигнал подтверждён, termination/stability не пройдены.
- Не запускать новые training/eval/inference jobs и не пересчитывать burned CONFIRM v1/v2.
- Не менять `.cursor/methodology/**`; существующие несвязанные правки и временные `configs/_tmp_*` исключить из D1b-пакета.
- Следующий допустимый этап — отдельный D1c termination-repair, не 1h.

## Работы
1. **Зафиксировать доказательную базу без ретроспективного переписывания.** Сохранить contract 1.2.0 и первичные confirm_v2 artifacts неизменными; в [`.cursor/results/d1b_n2_reliability/d1b_reliability_gate.json`](.cursor/results/d1b_n2_reliability/d1b_reliability_gate.json) явно указать, что clean seed43 самостоятельно устанавливает HARD-STOP, а seed42 eval остаётся diagnostic из-за `git_dirty=true`.
2. **Закрыть оставшиеся fail-closed дыры перед будущим reconfirm.** В [`scripts/74_eval_discrete_lm.py`](scripts/74_eval_discrete_lm.py) отклонять score-only bundle при отсутствующем или несовпадающем `max_frames`/provenance; в [`scripts/73_infer_discrete_lm.py`](scripts/73_infer_discrete_lm.py) запретить canonical CLI override frozen infer settings; в [`scripts/76_select_discrete_checkpoint.py`](scripts/76_select_discrete_checkpoint.py) не штамповать отсутствующий `items_sha256`, требовать проверенный select8 и eval-computed finite TF CE для canonical D1b. Добавить регрессии в [`tests/test_discrete_n2_protocol.py`](tests/test_discrete_n2_protocol.py) и [`tests/test_d1b_oracles.py`](tests/test_d1b_oracles.py).
3. **Восстановить воспроизводимый test gate.** Вернуть [`src/minispeech/models/latent_io.py`](src/minispeech/models/latent_io.py) из последнего tracked варианта commit `b048254` вместо отключения сохранившегося [`tests/test_n2_collate_latent_seq.py`](tests/test_n2_collate_latent_seq.py); добавить тест успешной и аварийной atomic-save семантики в [`tests/test_discrete_lm_trainer_steps.py`](tests/test_discrete_lm_trainer_steps.py).
4. **Согласовать closure-документы.** Обновить [`.cursor/results/d1b_n2_reliability/PROTOCOL.md`](.cursor/results/d1b_n2_reliability/PROTOCOL.md), убрав stale v1 PASS; уточнить clean/dirty SHA в [`RESULTS.md`](.cursor/results/d1b_n2_reliability/RESULTS.md); дополнить [`listen_index.json`](.cursor/results/d1b_n2_reliability/listen_index.json) checkpoint/contract/subset hashes и condition; отметить последние два todo в [`.cursor/plans/d1b_n2_reliability_7bd71c65.plan.md`](.cursor/plans/d1b_n2_reliability_7bd71c65.plan.md) как завершённые с terminal result HARD-STOP. Старые v1 artifacts не удалять, а явно маркировать superseded/invalid.
5. **Пройти финальную верификацию на одном pushed-source candidate SHA.** В нормативном WSL venv выполнить focused D1b suite, полный `pytest -q`, WORLD_SIZE=1 DDP smoke и проверки frozen hashes/provenance. Сохранить новые логи в [`.cursor/results/d1b_n2_reliability/logs/`](.cursor/results/d1b_n2_reliability/logs/) без GPU training или synthesis.
6. **Подготовить воспроизводимую передачу.** Отделить D1b source/results от несвязанных methodology/tmp changes, проверить отсутствие checkpoints/audio/HF cache в индексе и сформировать небольшие reviewable commits. Commit/push выполнять только после отдельного явного разрешения; после публикации Windows и WSL должны указывать на один remote SHA.

## Критерии выхода
- Ни одного актуального документа, заявляющего D1b PASS или разрешающего 1h.
- Seed43 clean FAIL и seed42 diagnostic status согласованы во всех summaries.
- Canonical selector/evaluator fail-closed по TF CE, subset hash, contract hash и frozen infer settings.
- Focused D1b tests, полный pytest и DDP smoke воспроизводимо зелёные на одном SHA.
- Results/logs/listen index готовы к публикации; тяжёлые artifacts остаются только в Linux `runs/`/`out/`.
- После этого разрешено планировать D1c termination-only; 1h остаётся запрещён до нового N2 PASS.
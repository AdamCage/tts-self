# REPAIR_ROUND5 — D1c INFRA repair (plan §8.4 distinct checkpoint identity)

**Plan:** `d1c_semantic_eos_repair_b81ecc3e`  
**Worktree:** `C:\projects\tts-self-d1c` (`d1c/semantic-eos`)  
**Input:** `REAUDIT_ROUND4.md` (verdict **INFRA_READY_FOR_HUMAN_AUDIT**; MEDIUM gap: duplicate-checkpoint candidate set + inventable `--checkpoint-step`)  
**Date:** 2026-08-04  
**Constraints:** no commit, no push, no GPU (`CUDA_VISIBLE_DEVICES=`), no seed invented, no methodology edit, no Wave 0–2 PASS claim.

---

## Verdict

> **INFRA_REPAIR_ROUND5_COMPLETE_AWAITING_HUMAN_REAUDIT**
>
> The Round-4 MEDIUM §8.4 gap is closed fail-closed: the formal selector now
> requires five **distinct** `checkpoint.sha256` values, `candidate_step_<step>.pt`
> basename agreement, and on-disk hash verification. `--checkpoint-step` may only
> **confirm** a payload/filename-derived step on the D1c formal path (cannot invent
> a foreign step for `best.pt`). Live probe of the round-4 attack path now refuses
> at invent + selector + schema layers. Prior FAIL #2/#4/#7 and step stamping are
> not regressively weakened.
>
> Waves 0–2 remain **not PASS**. Wave 3 remains **NO-GO**.

---

## Contract decisions (documented)

### Distinct-hash / path identity (selector)

`select_d1c_formal_reports` (`_require_distinct_checkpoint_identities`) refuses when:

1. the five `checkpoint.sha256` values are not pairwise distinct;
2. any `checkpoint.path` basename ≠ `candidate_step_<step:04d>.pt` for that report's step;
3. `checkpoint.path` is not readable, **or** live file sha256 ≠ claimed `checkpoint.sha256`;
4. resolved absolute paths collide (same underlying file published twice).

This implements plan §8.4 "один checkpoint hash/step match на report" and the
required "checkpoint hash/step mismatch rejected" case.

### CLI `--checkpoint-step` (evaluator)

**Chosen contract: confirm-only on formal path (`require=True`).**

| Situation | Behavior |
|---|---|
| Payload and/or `candidate_step_XXXX.pt` derive step S; CLI omitted | OK → S |
| Derived S; CLI = S | OK (confirm) |
| Derived S; CLI ≠ S | refuse (`conflicts`) |
| No derived step; CLI set; `require=True` (D1c formal) | refuse (`cannot invent`) |
| No derived step; CLI set; `require=False` | still allowed (non-formal / diagnostic only) |

Schema additionally rejects `checkpoint.sha256` ≠ on-disk bytes when the path is
readable, and rejects path basename step ≠ `checkpoint.step` when the name matches
`candidate_step_*.pt`.

---

## Fixes

### 1. Selector fail-closed on duplicate identities — **FIXED**

`scripts/76_select_discrete_checkpoint.py`: `_require_distinct_checkpoint_identities`
before provenance/eligibility ranking.

### 2. Hash/step consistency — **FIXED**

- Selector: on-disk verify required for every formal candidate.
- Schema (`validate_formal_report`): when path is readable, claimed sha must match;
  basename step must agree with `checkpoint.step`.

### 3. `--checkpoint-step` invent closed — **FIXED**

`resolve_checkpoint_step(..., require=True)` refuses CLI-only invent. Help text
updated. Emitter message no longer advertises invent-via-CLI.

### 4. Negative tests — **ADDED**

- `test_selector_refuses_duplicate_checkpoint_hashes` (round-4 attack shape)
- `test_selector_refuses_checkpoint_hash_mismatch`
- `test_validate_formal_report_rejects_checkpoint_hash_mismatch`
- `test_resolve_checkpoint_step_payload_filename_cli` now asserts invent refuse + confirm OK
- `test_main_stamps_checkpoint_step_and_selector_e2e` writes distinct on-disk ckpts and
  asserts attack set is refused

---

## Live probe

`logs/repair_round5_probe_dupckpt.py` → `logs/repair_round5_probe_dupckpt.log`:

```text
N1  PASS five best.pt + --checkpoint-step invents refused
N1b PASS resolve invent refuse (cannot invent)
N2  PASS selector refused duplicate checkpoint.sha256 (distinct=1, n=5)
N3  PASS schema refused hash mismatch
C1  PASS CLI confirm of derived step still works
SUMMARY pass=5 fail=0
```

---

## Sealed digests (post-edit reseal)

| Field | Value |
|---|---|
| `contract_sha256` / `recipe_sha256` | `cff9e9f5dd11ae66d66e44eddc8768f4989daa8c0339822d77252289a155baad` |
| `recipe_source_manifest_sha256` | `e505acf99640a3c801f9a9451bc60a19c74d2460413564cb0d68b2ab6fc2b596` |
| Locked sources | 19 (resealed after `74` + `76` + `discrete_report_schema`) |
| Manifest mismatches | `[]` |

Artifacts: `artifacts/repair_round5_hashes.json`, updated W1/W2 YAML binding hashes,
`configs/eval_contracts/d1c_semantic_eos_v2_0_0.json` (+ v2_0_1 parent pointer).
Round-4 digests (`9922ce4d…` / `62eeeae2…`) are superseded.

---

## Pytest

| Check | Result |
|---|---|
| Full suite | **404 passed, 3 skipped, 2 warnings in 55.52s** |
| Collected | **407** (= round-4 404 + 3 new tests) |
| Log | `logs/repair_round5_full_pytest.log` |
| Environment | WSL Ubuntu, `CUDA_VISIBLE_DEVICES=`, `GIT_DIR`/`GIT_WORK_TREE` worktree override |
| Focused | `28 passed` (`logs/repair_round5_focused_pytest.log`) |

---

## Remaining issues

1. **[LOW]** Cover real `_load_model` returns payload with `step` (round-4 X4 note).
2. **[LOW]** Weight EOS accuracy aggregate / publish min alongside mean.
3. **[LOW]** Pin `verify_recipe_source_manifest` to repo root.
4. **[LOW]** Assert rather than overwrite `recipe_hash` in scripts/72.
5. **[LOW]** Non-D1c resume fail-open.
6. **[LOW]** CER margin tolerance at scripts/76.
7. Standing non-code: no clean committed D1c SHA; no external confirm-seed; Wave 3/4 NO-GO pending human re-audit.

---

## Key files changed

- `scripts/74_eval_discrete_lm.py` — confirm-only `--checkpoint-step` on formal path
- `scripts/76_select_discrete_checkpoint.py` — distinct hash/path/on-disk verify
- `src/minispeech/eval/discrete_report_schema.py` — readable-path hash + basename step check
- `tests/test_d1c_infra_repair.py` — attack + mismatch negatives; e2e distinct ckpts
- `configs/eval_contracts/d1c_semantic_eos_v2_0_0.json` (+ v2_0_1) — resealed
- `configs/discrete_speech_lm_d1c_semantic_eos_w1.yaml` / `w2.yaml` — rebound hashes

## Confirmations

- No commit / no push
- No GPU work / no training / no inference seed invent
- No methodology edit
- No Waves 0–2 PASS claim
- Wave 3 still **NO-GO**

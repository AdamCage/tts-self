# REAUDIT — independent adversarial re-verification of the D1c INFRA repair

**Plan:** `.cursor/plans/d1c_semantic_eos_repair_b81ecc3e.plan.md`
**Slug:** `d1c_semantic_eos_repair_b81ecc3e`
**Worktree audited:** `C:\projects\tts-self-d1c`, branch `d1c/semantic-eos`
**Branch tip:** `0661d10eb616362e74dc1750e03cf25c33a7943e` (baseline `origin/master`; the branch
carries **no commits of its own** — all D1c work is uncommitted working-tree state, 495 dirty
`git status --porcelain` lines)
**Auditor:** independent re-audit agent. The repair agent's `REPAIR_AUDIT.md` claims were treated
as unverified assertions.
**Date:** 2026-08-04
**Constraints honoured:** no commit, no push, no GPU, no seed invented, no methodology edits.

## Overall verdict

> ## STILL_INVALID_RUN
>
> 3 of the 9 blockers do not hold on live code. Waves 0–2 remain **not PASS**.
> Wave 3 remains **NO-GO** (independently: no GPU, no clean committed SHA, no external
> confirmation-seed commitment).

The repair is substantially real — 6 of 9 blockers are genuinely fixed and I reproduced the
pytest attestation exactly — but the three failures are load-bearing: the canonical W1/W2
recipes are bound to a **stale, non-existent eval-contract hash**, the D1c v2 report emitter
**raises on real evaluator rows** (so the evaluator→selector formal interface has never
executed end-to-end), and three of the four newly added resume hash checks are **inert** for
the exact configs they were added for.

## Method

Everything below was executed, not read. Probe scripts and logs are committed alongside this
report:

| Artifact | What it does |
|---|---|
| `logs/reaudit_full_pytest.log` | full suite, default env, from the worktree |
| `logs/reaudit_full_pytest_gitenv.log` | full suite with `GIT_DIR`/`GIT_WORK_TREE` worktree override |
| `logs/reaudit_probe.py` / `.log` | B1–B8 live probes (grad flow, contract tamper, formal load, CLI, empty EOS, selector hash) |
| `logs/reaudit_probe2.py` / `.log` | B5 batched EOS, B7 dynamic resume enforcement, B8 manifest recompute |
| `logs/reaudit_probe3.py` / `.log` | canonical W1/W2 YAML ↔ contract ↔ `resolve_repro_hashes` binding |
| `logs/reaudit_probe4.py` / `.log` | v2 report emitter fed evaluator-shaped rows |
| `logs/reaudit_probe5.py` / `.log` | fail-open surface of the emitted v2 report |
| `logs/reaudit_probe6.py` / `.log` | selector eligibility with consistently re-hashed reports |
| `logs/reaudit_postcheck.py` / `.log` | proves the probes mutated no attested source |

Post-audit integrity (`logs/reaudit_postcheck.log`) — the probes left every attested artifact
untouched:

```text
post_probe_manifest_mismatches=[]
post_probe_contract_selfcheck_ok=True
post_probe_formal_load dev32/select8/gate24/full32: ok=True
```

Test environment: WSL2 Ubuntu, venv `/home/adam/projects/tts-self/.venv`, rootdir
`/mnt/c/projects/tts-self-d1c`. `pyproject.toml` sets `pythonpath = ["src"]`, so pytest and the
probes import the **worktree** `src/minispeech`, not the WSL clone (which is still at `0661d10`
without any D1c code — verified).

## Per-blocker verdicts

| # | Blocker | Verdict |
|---|---------|---------|
| 1 | Residual loss detached in semantic-EOS mode | **PASS** |
| 2 | Contract 2.0.0 self-check / no legacy stop fields | **FAIL** |
| 3 | DEV oracle formal load, `file_sha256`, select8 2+6 | **PASS** (with caveat) |
| 4 | CLI semantic wiring / `emit_d1c_v2_report` reachable | **FAIL** |
| 5 | Immediate EOS phantom frame | **PASS** |
| 6 | Selector `_sha256_file` + eligibility from rows | **PASS** (with material gap) |
| 7 | Checkpoint resume hash checks | **FAIL** |
| 8 | Frozen source manifest mismatches | **PASS** (with coverage gap) |
| 9 | Pytest attestation freshness | **PASS** |

---

### 1. Residual loss detached — **PASS**

`MultiCodebookHead.loss()` now returns live tensors
(`src/minispeech/models/multi_codebook_head.py:172-182`), and the semantic-EOS path composes
`residual_ce` into the backward graph
(`src/minispeech/models/discrete_speech_lm.py:680-694`, where `total = cb0_weight *
cb0_frame_ce + residual_ce + eos_loss_weight * semantic_eos_ce` and only the logging fields at
lines 697-705 are detached).

Live evidence (`logs/reaudit_probe.log`):

```text
REAUDIT B1 PASS residual params without grad: 0/45 (default cb0_w=1 eos_w=1)
REAUDIT B1 PASS residual params without grad with cb0_w=0,eos_w=0: 0/45
REAUDIT B1 PASS loss['total'].requires_grad=True
REAUDIT B1 PASS MultiCodebookHead.loss ce_residual.requires_grad=True ce_cb0.requires_grad=True
```

The second check is the decisive one: with `cb0_weight=0` **and** `eos_loss_weight=0`, all 45
residual-head parameters still receive non-zero gradients, which is only possible if the
residual CE term itself is in the graph. The repair agent's own test
(`tests/test_d1c_infra_repair.py:61-79`) also exercises the real forward/backward path.

### 2. Contract 2.0.0 self-check — **FAIL**

The narrow self-check the repair claimed **does** hold:

```text
REAUDIT B2 PASS declared=c636562ee0f158f722dc02d2161ea3dac6424c58f5404e7f346cc10f9cc76f71
                computed=c636562ee0f158f722dc02d2161ea3dac6424c58f5404e7f346cc10f9cc76f71
REAUDIT B2 PASS raw-file recompute=c636562ee0f158f722dc02d2161ea3dac6424c58f5404e7f346cc10f9cc76f71
REAUDIT B2 PASS tamper(max_frames=999) still matches declared sha: False
REAUDIT B2 PASS infer keys=['max_cb0_repeat','max_frames','min_frames','repetition_penalty',
                            'temperature','top_k'] legacy_present=[]
REAUDIT B2 PASS contract.inference keys=[] legacy_declared=[]
```

Recorded hashes:

| Field | Value |
|---|---|
| `sha256` (= `recipe_sha256`) | `c636562ee0f158f722dc02d2161ea3dac6424c58f5404e7f346cc10f9cc76f71` |
| `recipe_source_manifest_sha256` | `15869c58ecd1f01dc33a3c8b5f8432c8c303c47d34243cbb0bc2fbc3e76ec261` |

**But the canonical recipes were never re-bound to the repaired contract.** Both W1 and W2 still
declare the **pre-repair** contract hash — the very `a59ada9…` value `REPAIR_AUDIT.md:47` names
as the broken "before" state:

- `configs/discrete_speech_lm_d1c_semantic_eos_w1.yaml:39`
  `eval_contract_hash: a59ada9c1e0a64f067b27dd2b4d93ca270702bd41881746d1bb9a437d2f33772`
- `configs/discrete_speech_lm_d1c_semantic_eos_w2.yaml:39` — identical stale value
- `configs/eval_contracts/d1c_semantic_eos_v2_0_1.json:8-9` — `parent_sha256` and
  `parent_recipe_sha256` both point at `a59ada9…`, so the confirmation-bound contract's parent
  binding is broken and plan §7's "2.0.1 must preserve `recipe_sha256` from 2.0.0" cannot hold
- `WAVE1_REPORT.md:41-42` still publishes `a59ada9…` as the contract/recipe SHA, directly
  contradicting `REPAIR_AUDIT.md:26` inside the same results pack

```text
REAUDIT B2 FAIL w1: train.eval_contract_hash=a59ada9c… vs live contract sha256=c636562e… match=False
REAUDIT B2 FAIL w2: train.eval_contract_hash=a59ada9c… vs live contract sha256=c636562e… match=False
```

Aggravating factor: nothing fails closed on this. `scripts/72_train_discrete_lm.py:455-457`
**silently overwrites** the declared value instead of asserting equality:

```python
tcfg["eval_contract_hash"] = contract.get("sha256") or tcfg.get("eval_contract_hash")
```

So a canonical D1c train run would neither fail nor warn; it would discard the recipe's declared
provenance. And because the W1/W2 YAMLs are themselves hash-locked in
`recipe_source_manifest` (blocker 8 reports "0 mismatches"), the frozen manifest has **frozen in
the stale hash**. Fixing it invalidates `recipe_source_manifest_sha256` and therefore the
contract SHA — the contract, recipe files, and manifest must be regenerated together.

What passes: W1 vs W2 differ in exactly `loss.eos_loss_weight` (1.0 → 2.0) plus `train.run_name`,
and both `train.subset_hash` values match the contract's `full32` `items_sha256`.

### 3. DEV oracle formal load — **PASS** (with caveat)

All four registry entries carry `file_sha256`, all four formal loads succeed, and the select8
composition lock is real (`src/minispeech/data/discrete_oracles.py:155-162`):

```text
REAUDIT B3 PASS registry entries lacking file_sha256=[]
REAUDIT B3 PASS formal load dev32:   n=32 split=full32  formal_eligible=True
REAUDIT B3 PASS formal load select8: n=8  split=select8 formal_eligible=True
REAUDIT B3 PASS formal load gate24:  n=24 split=gate24  formal_eligible=True
REAUDIT B3 PASS formal load full32:  n=32 split=full32  formal_eligible=True
REAUDIT B3 PASS select8 composition = 2 RUSLAN + 6 CV
REAUDIT B3 PASS select8|gate24 == full32 and disjoint (|s|=8 |g|=24 |f|=32)
REAUDIT B3 PASS select8 stratification: buckets=['long','medium','short'] n_low_ratio=2 n_high_ratio=3
```

I additionally verified select8 satisfies plan §9.3 stratification (all three duration buckets
present, both ratio extremes represented) — the repair agent never checked this, and the
sorted-looking CV IDs made it worth testing.

**Caveat (semantics changed, not just fixed):** the repair redefined `file_sha256` from
"SHA256 of the file's bytes" to "SHA256 of a canonical re-serialisation of the parsed body
without the `file_sha256` field" (`discrete_oracles.py:96-104`). The real on-disk byte digest of
`configs/train_subsets/d1c_dev32.json` is `88518b178f39185c…`; the declared/verified value is
`f9f82b708b23ab61…`. No formal code path checks the actual file bytes — `file_sha256()` (real
bytes) is used only in `load_diagnostic_subset` (`discrete_oracles.py:185`). This is a
defensible way to resolve a self-referential hash, but it is weaker than plan §8.2's "file SHA"
and should be stated explicitly rather than left implicit.

### 4. Semantic generation CLI wiring — **FAIL**

Sub-items that pass:

- `scripts/73_infer_discrete_lm.py:313-318` unwraps `DiscreteGenerationOutput` before any
  `.shape`/`.cpu` use, and refuses codec decode of empty generations (lines 337-342).
- `scripts/74_eval_discrete_lm.py:245-252` `_unwrap_generation` normalizes correctly and raises
  `TypeError` on non-tensor input.
- `scripts/74:227-234` `teacher_forced_ce` refuses a batch without `semantic_targets`/
  `semantic_mask` in semantic mode.
- `estimate_stop_reason` has exactly one call site (`scripts/74:378`), inside the
  `else` of `if semantic_mode:` (lines 374-386) — unreachable on the D1c v2 path.
- `emit_d1c_v2_report` is exported and wired at three sites: `scripts/74:735` (inside
  `_emit_d1c_v2_from_gate`), reached from `scripts/74:907` (`--score-only`) and
  `scripts/74:1426` (main eval). Not dead code.

**The blocking defect: the emitter raises on real evaluator rows.** `validate_bound_rows`
requires every row to carry `ordinal == index`
(`src/minispeech/eval/discrete_report_schema.py:84-85`), but neither
`free_running_asr_eval`'s `row_out` (`scripts/74:421-431`) nor `_enrich_v2_condition_rows`
(`scripts/74:624-681`) ever stamps `ordinal`. Fed rows shaped exactly as the evaluator builds
them (`logs/reaudit_probe4.log`):

```text
ValueError: formal D1c report refused: conditions.model.rows[0].ordinal must equal 0
REAUDIT B4 FAIL _emit_d1c_v2_from_gate RAISED on evaluator-shaped rows
```

Neither call site wraps this in `try/except`, so the first real D1c eval run crashes *after*
writing `n2_gate.json` and **never produces a v2 report**. The evaluator→selector direct
interface (plan §8.4) has therefore never executed end-to-end. The repair agent's test
(`tests/test_d1c_infra_repair.py:256-271`) only asserts the function is callable and rejects an
empty condition set, so it could not catch this.

Stamping `ordinal` manually lets the emitter run, which exposes four further fail-open defects
(`logs/reaudit_probe5.log`):

```text
REAUDIT B4 FAIL fabricated TF provenance: teacher_forced_ce_source='eval_computed'
                tf_eos_final_top1_accuracy=1.0 tf_eos_early_top1_false_positive_count=0
                (never measured by the evaluator)
REAUDIT B4 FAIL formal_eligible hardcoded True by emitter
REAUDIT B4 FAIL pcm_sha256 is a copy of file_sha256 (no real PCM hash)
REAUDIT B4 FAIL row.catastrophic never stamped -> selector n_catastrophic always 0
```

1. `scripts/74:100` sets `teacher_forced_metrics["teacher_forced_ce_source"] = "eval_computed"`
   unconditionally — including in `--score-only`, where the TF CE is read out of the decode
   bundle (`scripts/74:913`) rather than computed by the scorer.
2. `scripts/74:725-726` defaults `tf_eos_final_top1_accuracy` to `1.0` and
   `tf_eos_early_top1_false_positive_count` to `0` — both *passing* values. `teacher_forced_ce`
   (`scripts/74:238-242`) returns only `loss`/`ce_cb0`/`ce_residual`, so these EOS metrics are
   **always** fabricated. Plan §12.3 requires measured 100% / 0.
3. `scripts/74:102` hardcodes `formal_eligible=True`.
4. `scripts/74:641-646` sets `pcm_digest = file_digest`, and `scripts/74:641,650-653` substitute
   `"0"*64` when a wav is missing (10 such placeholder literals in the file) — a missing artifact
   yields a syntactically valid hash instead of a refusal.

### 5. Immediate EOS phantom frame — **PASS**

`generate_semantic_eos` trims to `max(lengths)` and emits `(B, 0, n_codebooks)` when empty
(`src/minispeech/models/discrete_speech_lm.py:863-870`), and `generate()` dispatches to it in
semantic mode (`discrete_speech_lm.py:905-912`). I tested through the **public** `generate()`
entry point, not the private helper the repair agent's test uses:

```text
REAUDIT B5 PASS generate() dispatch: codes.shape=(1, 0, 16) lengths=[0] mask=(1, 0) reason=learned_eos
REAUDIT B5 PASS mixed batch: lengths=[0, 3] codes.shape=(2, 3, 16) row0_mask_any=False
                reasons=['learned_eos', 'safety_max_frames']
REAUDIT B5 PASS min_frames=4: early_eos_suppressed_count=4 len=4 reason=learned_eos
```

The mixed-batch case matters and holds: row 0 stops immediately with `length=0` and an all-False
`frame_mask`, while row 1 keeps decoding — no phantom frame is attributed to row 0 inside the
padded tensor. Min-frame suppression correctly records `early_eos_suppressed_count=4`.

### 6. Selector `_sha256_file` + eligibility — **PASS** (with material gap)

`_sha256_file` (`scripts/76_select_discrete_checkpoint.py:47-52`) is correct — verified on a
3 MiB blob so the chunk loop actually iterates, matching `hashlib.sha256` exactly. Eligibility is
recomputed from bound rows (`scripts/76:89-120`) and does not trust `report["gates"]["eligible"]`;
`rows_sha256`/`report_sha256` tamper detection fires; and `scripts/76:141-143` refuses least-bad
selection when nothing is eligible.

```text
REAUDIT B6 PASS _sha256_file 3MiB: match=True
REAUDIT B6 PASS all-safety candidates refused: found no eligible candidate; refusing least-bad
REAUDIT B6 PASS catastrophic rows refused (explicit catastrophic=True)
```

**Material gap against plan §11.** The recomputed predicate (`scripts/76:108-120`) is only
`bool(rows) and all(reason == "learned_eos") and n_cat == 0 and all(cer < 0.5)`, plus three terms
that the emitter makes unconditionally true (`formal_eligible`, `teacher_forced_ce_source`, and
the two fabricated EOS TF metrics — see blocker 4). With consistently re-hashed reports so
refusals cannot be attributed to tamper detection (`logs/reaudit_probe6.log`):

```text
REAUDIT B6 FAIL selector SELECTED a candidate with early_eos_suppressed_count=7 and max_cb0_run=99
REAUDIT B6 FAIL selector SELECTED a candidate with zero-frame generations
REAUDIT B6 FAIL selector SELECTED a candidate with per-row duration ratio ~9.5x
REAUDIT B6 FAIL selector declared ELIGIBLE a report with zero measured EOS-TF metrics
```

Plan §11 additionally requires `early_eos_suppressed_count == 0`, CB0 max run ≤ 12, no empty
generation, per-row duration ratio ∈ [0.5, 2.0], mean duration ratio ∈ [0.90, 1.10], model macro
CER ≤ copy_synth macro CER + 0.03 (not a hardcoded `< 0.5`), text-usage PASS, and no phrase
loop/runaway. None of these are enforced. I grade the *stated* blocker PASS because the
`NameError` is gone and eligibility is genuinely row-derived, but the gate is not yet the §11
gate.

### 7. Checkpoint resume hash checks — **FAIL**

The machinery is real. All eight hashes are stored (`discrete_lm_trainer.py:617`, populated by
`resolve_repro_hashes` at lines 246-259) and enforced in `_validate_resume`
(`discrete_lm_trainer.py:735-758`). Dynamic probes confirm enforcement
(`logs/reaudit_probe2.log`):

```text
REAUDIT B7 PASS checkpoint stores all 8 hash keys: True (run_manifest=rm_a)
REAUDIT B7 PASS run_manifest mismatch refused: resume fail-closed: run_manifest mismatch ckpt=rm_a current=rm_b
REAUDIT B7 PASS config mismatch refused: resume fail-closed: config mismatch ckpt=ec15e… current=77f46…
REAUDIT B7 PASS checkpoint lacking run_manifest refused: current run_manifest=rm_a but checkpoint hashes lack this key
```

**But for the canonical D1c configs, three of the four newly added checks are inert.**
`discrete_lm_trainer.py:745-748` skips any key whose *current* value is `None`:

```python
for key in required_keys:
    cur_v = current.get(key)
    if cur_v is None:
        continue
```

Running `resolve_repro_hashes` on the real W1/W2 YAMLs (`logs/reaudit_probe3.log`):

```text
REAUDIT B7 INFO w1: resolve_repro_hashes -> {"config": "8b3597866ed41a6c", "model_config": "0fb389c327c5301a",
  "speaker_map": "64e7cdc5c3447a43…", "subset": "501330481c07d928…", "eval_contract": "a59ada9c1e0a64f0…",
  "recipe": null, "recipe_source_manifest": null, "run_manifest": null}
REAUDIT B7 FAIL w1: resume hash checks that are INERT (None): ['recipe', 'recipe_source_manifest', 'run_manifest']
REAUDIT B7 FAIL w1: stamped eval_contract hash == live contract sha: False
REAUDIT B7 FAIL w2: (identical)
```

Root cause: `resolve_repro_hashes` only back-fills `recipe` / `recipe_source_manifest` from the
contract inside `if out["eval_contract"] is None:` (`discrete_lm_trainer.py:260-278`). Because
W1/W2 *do* declare `eval_contract_hash` (with the stale value), that whole fallback is skipped,
and `scripts/72_train_discrete_lm.py` never stamps `recipe_hash`,
`recipe_source_manifest_hash`, or `run_manifest_hash` into `tcfg`. So the one active check is
bound to a stale hash and the three new ones are no-ops.

There is also a general fail-open: dropping a hash from the config both disables its check and
leaves the aggregate `config` digest unchanged, because `_stable_config_for_hash`
(`discrete_lm_trainer.py:159-175`) deliberately pops those same fields:

```text
REAUDIT B7 FAIL fail-OPEN: resume accepted after dropping run_manifest_hash from config
                (current=None -> check skipped)
```

`tests/test_discrete_lm_resume.py` covers grad_accum, speaker_map, subset, model_config, config,
recipe, and recipe_source_manifest mismatches, but has no `run_manifest` case and no
unstamped-key case.

### 8. Frozen source manifest — **PASS** (with coverage gap)

Recomputed all declared SHAs against the current files (`logs/reaudit_probe2.log`):

```text
REAUDIT B8 PASS n_locked=14 missing=[] mismatches=[]
```

14/14 declared digests match on-disk content, so the repair agent's "0 mismatches" claim is
accurate. Note this is what freezes the stale `eval_contract_hash` into the manifest (blocker 2).

**Coverage gap:** eight D1c-critical files are not hash-locked at all:

```text
REAUDIT B8 FAIL d1c-critical files NOT in frozen manifest:
  src/minispeech/data/discrete_oracles.py      <- the formal subset loader itself
  src/minispeech/data/d1b_oracles.py           <- provides load_eval_contract + the contract self-check
  src/minispeech/eval/asr.py
  src/minispeech/data/codec_seq_protocol.py
  src/minispeech/constants.py
  configs/eval_contracts/schemas/discrete_eval_report_v2.schema.json
  configs/train_subsets/d1c_dev32.json
  configs/asr.yaml
```

`d1c_dev32.json` and `configs/asr.yaml` are bound elsewhere in the contract (`subsets[*].file_sha256`
and `asr.file_sha256`), so those two are acceptable. The first five are executable sources on the
formal path — including the two modules that implement the contract self-check and the formal
oracle loader — and are silently mutable without invalidating any declared hash.

### 9. Pytest attestation — **PASS**

Freshness: `logs/repair_full_pytest.log` has mtime `2026-08-04 09:14:30`, newer than every
repaired source it attests (latest source edit `09:12:11`, `configs/eval_contracts/
d1c_semantic_eos_v2_0_0.json` and `configs/train_subsets/d1c_dev32.json`). No source is newer
than the log.

Content match: the worktree collects **389 tests**, exactly `387 passed + 2 skipped`. I
reproduced the claimed result byte-for-byte (`logs/reaudit_full_pytest_gitenv.log`):

```text
387 passed, 2 skipped, 2 warnings in 52.78s
```

Both skips are explained and pre-existing opt-ins (`MINISPEECH_DDP_GLOO`,
`MINISPEECH_DDP_SMOKE`).

Two caveats worth recording rather than failing on:

1. The result requires `GIT_DIR`/`GIT_WORK_TREE` pointing at
   `/mnt/c/projects/tts-self/.git/worktrees/tts-self-d1c`. Without it, `tests/test_mel_baseline.py::
   test_git_sha_and_run_manifest` fails (`git_sha()` → `"unknown"`) and the run is
   **1 failed, 386 passed, 2 skipped** (`logs/reaudit_full_pytest.log`). This is an artifact of
   running a Windows worktree through `/mnt/c`; it would not occur in the canonical WSL clone.
   `REPAIR_AUDIT.md:14` mentions the override but the log itself records neither the env nor the
   command.
2. The log was written with bare `-q`, so it carries no header: no rootdir, no git SHA, no
   timestamp, no plugin list. It is only attestable because I re-ran it. Future attestations
   should tee the pytest header plus `git rev-parse HEAD` and `date -u`.

---

## Critical remaining defects, in fix order

1. **Re-bind the canonical recipes to contract 2.0.0.** Update
   `eval_contract_hash` in both `discrete_speech_lm_d1c_semantic_eos_w1.yaml:39` and `w2:39` to
   `c636562e…`, then regenerate `recipe_source_manifest` → `recipe_source_manifest_sha256` →
   contract `sha256`/`recipe_sha256` (they are mutually dependent), and repoint
   `d1c_semantic_eos_v2_0_1.json:8-9`. Correct `WAVE1_REPORT.md:41-42`.
2. **Make the contract binding fail closed.** Replace the silent overwrite at
   `scripts/72_train_discrete_lm.py:455-457` with an equality assertion.
3. **Stamp `ordinal` on every row** in `scripts/74:_enrich_v2_condition_rows` (and preferably in
   `free_running_asr_eval`'s `row_out`) so the v2 emitter can run at all. Add a regression test
   that drives `_emit_d1c_v2_from_gate` with evaluator-shaped rows — the existing test cannot
   catch this class of defect.
4. **Stop fabricating EOS teacher-forced metrics.** Have `teacher_forced_ce` return
   `ce_semantic_eos`, `tf_eos_final_top1_accuracy`, and
   `tf_eos_early_top1_false_positive_count` (the model already computes all three at
   `discrete_speech_lm.py:701-705`), and remove the passing defaults at `scripts/74:725-726`.
   Derive `teacher_forced_ce_source` and `formal_eligible` from provenance instead of hardcoding
   them at `scripts/74:100-102`.
5. **Remove the `"0"*64` placeholder hashes and the `pcm_sha256 = file_sha256` shortcut**
   (`scripts/74:641-653`); a missing wav must refuse, and PCM must be hashed from decoded samples.
6. **Stamp `catastrophic` per row** so the selector's `n_catastrophic == 0` term is not vacuous.
7. **Complete the §11 eligibility predicate** in `scripts/76:108-120`
   (`early_eos_suppressed_count`, CB0 run ≤ 12, non-empty generation, per-row and mean duration
   ratio, copy-synth-relative CER, text-usage, phrase-loop).
8. **Populate `recipe` / `recipe_source_manifest` / `run_manifest` for the canonical configs**
   (either in the YAMLs or stamped by `scripts/72`), and make `_validate_resume` refuse a
   `None` current value for D1c formal runs instead of skipping it
   (`discrete_lm_trainer.py:745-748`). Add `run_manifest` and unstamped-key resume tests.
9. **Extend the frozen manifest** to `discrete_oracles.py`, `d1b_oracles.py`, `eval/asr.py`,
   `codec_seq_protocol.py`, `constants.py`.
10. **Add D1c awareness to `scripts/72_train_discrete_lm.py`.** `is_d1b_cfg`
    (line 435) matches only `d1b_443m`/`d1b_n2`, so for a D1c config the "silent first-N is
    forbidden" guard (line 436) never fires, eval-contract enforcement (line 447) is skipped when
    `--subset` is absent, `--skip-eval-contract` stays available, and the `--allow-dirty` warning
    (line 509) never prints. `require_clean_sha=False` is hardcoded at line 453. Plan §8.3 and
    §18 forbid all of these on the formal D1c path.

## Standing non-code blockers (unchanged, correctly disclosed by the repair agent)

1. No clean committed D1c source SHA. Branch `d1c/semantic-eos` is still at baseline `0661d10`
   with 495 dirty working-tree lines; the WSL clone has no D1c code. Requires explicit user
   commit/push authorization.
2. External confirmation-seed commitment still absent (correctly **not** invented).
3. Wave 3 GPU smoke and Wave 4 DEV remain NO-GO.

## Main-repository status

`C:\projects\tts-self` (branch `master`, `0661d10`) contains **only** the untracked plan file
`.cursor/plans/d1c_semantic_eos_repair_b81ecc3e.plan.md`. It has **no copy** of this REAUDIT, the
results pack, the D1c configs, contracts, subsets, or sources. This report exists only in the
worktree at
`C:\projects\tts-self-d1c\.cursor\results\d1c_semantic_eos_repair_b81ecc3e\REAUDIT.md`.

## Scope confirmations

- No commit created, no push, no branch mutation.
- No GPU work; no training, inference, decode, or synthesis. No new `best.pt`/`last.pt` was
  produced outside pytest/probe temporary directories, so no listen-demo obligation arises.
- No seed invented; no methodology file touched.
- Waves 0–2 remain **not PASS**. Wave 3 remains **NO-GO**.

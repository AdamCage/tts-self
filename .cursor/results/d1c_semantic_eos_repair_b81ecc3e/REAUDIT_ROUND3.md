# REAUDIT_ROUND3 — independent adversarial re-verification of the D1c INFRA repair (round 3)

**Plan:** `.cursor/plans/d1c_semantic_eos_repair_b81ecc3e.plan.md`
**Slug:** `d1c_semantic_eos_repair_b81ecc3e`
**Worktree audited:** `C:\projects\tts-self-d1c`, branch `d1c/semantic-eos`
**Branch tip:** `0661d10eb616362e74dc1750e03cf25c33a7943e` (baseline `origin/master`; the branch still
carries **no commits of its own** — all D1c work remains uncommitted working-tree state)
**Input under audit:** `REPAIR_ROUND3.md` (claims FAIL #4 FIXED on the production path, HIGH docs
fixed, four MEDIUM gaps closed, fresh pytest `398 passed, 3 skipped`)
**Prior audits:** `REAUDIT.md` (round 1), `REAUDIT_ROUND2.md` (round 2 — verdict STILL_INVALID_RUN,
FAIL #4 still live)
**Auditor:** independent re-audit agent. Every claim in `REPAIR_ROUND3.md` was treated as an
unverified assertion and re-tested with live probes that drive the **production entry points**.
**Date:** 2026-08-04
**Constraints honoured:** no commit, no push, no GPU work (`CUDA_VISIBLE_DEVICES=` on every run),
no seed invented, no methodology edit, no training / inference / synthesis. No new `best.pt` or
`last.pt` was produced, so no listen-demo obligation arises.

---

## Overall verdict

> ## STILL_INVALID_RUN
>
> **FAIL #4 is genuinely fixed.** I drove `scripts/74_eval_discrete_lm.py::main()` end to end and
> the measured teacher-forced EOS metrics now survive aggregation, reach the emitter, and produce a
> schema-valid formal v2 report. FAIL #2 and FAIL #7 still hold, the HIGH doc defect is closed, all
> four MEDIUM gaps are closed, and the pytest attestation reproduces exactly.
>
> **But a critical defect on the formal eval path is still live, and it is the same *class* of
> defect for the third round running:** the evaluator never stamps the candidate checkpoint step.
> `checkpoint.step` is hard-zero in every production-emitted report, so
> `select_d1c_formal_reports` refuses all five candidates and the **evaluator→selector formal
> interface (plan §8.4) has still never executed end to end**.
>
> Waves 0–2 remain **not PASS**. Wave 3 remains **NO-GO**.

Round 1 shipped an emitter that crashed. Round 2 fixed the emitter but the evaluator threw the EOS
measurements away before the emitter saw them. Round 3 fixes that link — and the very next link
drops a different value that is already sitting in the checkpoint being loaded. The repair is real
and substantial; it is simply one link short of the interface the plan requires.

## Per-item verdicts

| Item | Round-3 claim | This audit | Evidence |
|---|---|---|---|
| **FAIL #4** — evaluator→emitter EOS plumbing | FIXED | **PASS** | `logs/reaudit3_probe_mainpath.log` G0a–G1n, G2–G8 |
| **FAIL #2** — contract / recipe / manifest binding | still PASS | **PASS** | `logs/reaudit3_probe_rest.log` R2a–R2i |
| **FAIL #7** — resume hashes | still PASS | **PASS** | `logs/reaudit3_probe_rest.log` R7a–R7d |
| `--score-only` contract ((a)/(b) hybrid) | decided + implemented | **PASS** | G5a–G8b |
| **HIGH** — stale DEV32 provenance docs | FIXED | **PASS** | H0–H9 |
| MEDIUM — §11 fields in emitter | Fixed | **PASS** | G1k, G1l |
| MEDIUM — selector refuses absent `text_usage` | Fixed | **PASS** | M4a |
| MEDIUM — runtime `recipe_source_manifest` verify | Fixed | **PASS** | M5a–M5c |
| MEDIUM — D1c subset aliases | Fixed | **PASS** | M6a–M6b |
| New regression test catches the round-2 bug | implied | **PASS** (helpers) / **gap** (main) | `logs/reaudit3_probe_mutation.log` X1–X4 |
| Pytest attestation `398 passed, 3 skipped` | claimed | **VALID** — exact reproduction | `logs/reaudit3_full_pytest.log` |
| Regression spot checks (grads / select8 / empty EOS) | — | **all hold** | RXa–RXd |
| **NEW BLOCKER — `checkpoint.step` never stamped** | not disclosed | **FAIL** | G1o, M3b–M3g |

## Method

Everything below was executed, not read. Probe scripts and logs are stored alongside this report.

| Artifact | What it does |
|---|---|
| `logs/reaudit3_probe_mainpath.py` / `.log` | **decisive FAIL #4 probe** — runs `scripts/74::main()` itself through argparse (real TF loop, real aggregation, real gate, real emitter), plus AST proof of the wiring, plus bare-float / legacy-aggregation / score-only negatives |
| `logs/reaudit3_probe_rest.py` / `.log` | multi-batch aggregation arithmetic, aggregation fail-closed cases, evaluator→selector §8.4 interface, selector `text_usage`, runtime manifest verify, D1c aliases, published-hash audit, FAIL #2 / #7 regression, prior-PASS spot checks |
| `logs/reaudit3_probe_mutation.py` / `.log` | mutation testing: splices the round-2 code back into `scripts/74` in memory and re-runs the new regression test to see whether it actually fails |
| `logs/reaudit3_run_full_pytest.sh` → `logs/reaudit3_full_pytest.log` | independent full-suite reproduction with the same `GIT_DIR` override and GPU masked |
| `logs/reaudit3_testdiff.sh` | which test ids are new in round 3, and whether my collected set matches the claim |
| `logs/reaudit3_integrity_win.log` | post-probe source integrity (authoritative Windows-side git) |
| `logs/reaudit3_integrity.log` | manifest re-verify after all probes (**see caveat below**) |

Test environment: WSL2 Ubuntu, venv `/home/adam/projects/tts-self/.venv`, Python 3.11.15, rootdir
`/mnt/c/projects/tts-self-d1c`, `PYTHONPATH` pinned to the **worktree** `src`, `CUDA_VISIBLE_DEVICES=`
on every run.

**Caveat on `logs/reaudit3_integrity.log`:** its `git status` section ran under WSL with the
`GIT_DIR` override and reports every tracked file as modified. That is a pure CRLF/`core.autocrlf`
artifact of viewing a Windows worktree through `/mnt/c` — it is **not** evidence of mutation. The
authoritative check is `logs/reaudit3_integrity_win.log`, which is byte-identical to the pre-audit
snapshot (13 modified files, the same untracked set). My probes mutated no attested source. The
frozen manifest still verifies after every probe:

```text
n_locked = 19
mismatches = []
contract sha256 = 25c370a94f589127913623cc04db61d1fd593d4b4158f692ded9e72408006f82
```

---

## FAIL #4 — evaluator→emitter EOS plumbing — **PASS**

### Why I trust it this time: the probe drives `main()`, not the helpers

The round-3 repair probe (`logs/repair_round3_probe_realpath.log`) is a genuine improvement over
round 2 — it feeds real per-batch measurements through `aggregate_teacher_forced_batches` and
`teacher_forced_metrics_for_emitter` instead of a hand-written dict. But it still labels a direct
`_emit_d1c_v2_from_gate(...)` call "H4 MAIN measured path", and it never proves that `main()` uses
those helpers. That is exactly the gap that let round 2 self-certify.

So I rebuilt the probe to run `main()` itself. Only the GPU/codec/ASR-bound outer dependencies are
stubbed (`_load_model`, `load_codec_items`, `enforce_eval_contract`, `_resolve_speaker_registry`,
`_ref_texts_from_manifest`, `free_running_asr_eval`, `load_asr_engine`). Everything on the metric
path is the real code: `teacher_forced_ce` on a real semantic-EOS model, the real `DataLoader`
aggregation loop, `aggregate_teacher_forced_batches`, `teacher_forced_metrics_for_emitter`,
`evaluate_n2_gate_d1b`, `text_usage_acceptance`, `stability_acceptance`,
`_emit_d1c_v2_from_gate`, `emit_d1c_v2_report`, `build_formal_report`, `validate_formal_report`.

**AST proof of the wiring** (not a text heuristic):

```text
G0a PASS main() calls aggregate_teacher_forced_batches
G0b PASS main() calls teacher_forced_metrics_for_emitter
G0c PASS tf_ce := teacher_forced_metrics_for_emitter()
G0d PASS main() emit call kwarg teacher_forced_ce=["Name(id='tf_ce', ctx=Load())"]
G0e PASS decode bundle carries tf_ce dict
G0f PASS old bare-float reduction 'tf_ce = summary.get("loss")' absent
```

**Live `main()` run, measured path:**

```text
G1a PASS main(--n2-gate) rc=0 traceback=False
G1b PASS production main() wrote discrete_eval_report_v2.json
G1c PASS formal_eligible=True
G1d PASS teacher_forced_ce_source='eval_computed'
G1e PASS tf_eos_final_top1_accuracy=0.0
G1f PASS tf_eos_early_fp=0
G1g PASS ce_semantic_eos=3.6959688663482666
G1h PASS total_ce=10.86877727508545
G1i PASS untrained model does not report the 1.0/0 default (acc=0.0, fp=0)
G1j PASS unpatched validate_formal_report accepts the report
G1k PASS §11 fields present: gates.text_usage / summaries.phrase_loop_repetition / runaway_or_repetition
G1l PASS gates.text_usage.pass mirrors the live gate acceptance block (report=True, gate=True)
G1m PASS eval_summary teacher_forced_ce_source='eval_computed' n_batches=1
G1n PASS emitted EOS metrics reproduce an independent recompute (acc 0.0 vs 0.0; fp 0 vs 0)
```

`G1i` and `G1n` are the decisive controls. An untrained tiny model reports EOS accuracy `0.0`,
which no hardcoded 1.0 default could produce, and the value in the report reproduces an
independent recomputation I performed outside the evaluator.

### Both round-2 defect modes are gone, and the new code is load-bearing

I reinstated each round-2 defect inside a live `main()` run:

```text
G2a PASS bare-float injection: rc=0 unhandled_traceback=False
G2b PASS gate JSON still written before the refusal
G2c PASS no formal v2 report emitted from a bare float (fail-closed)
G2d PASS refusal reason printed: D1c v2 formal report refused (non-fatal after gate write):
         formal D1c report refused: formal_eligible=False
         (teacher_forced_ce_source='unavailable', eos_measured=False)
G3a PASS legacy discarding aggregation => NO formal report
         (the new aggregation is load-bearing, not decorative)
G3b PASS legacy aggregation still fails soft: rc=0 traceback=False
```

The round-2 traceback-after-gate-write is closed: `_emit_d1c_v2_from_gate` wraps the impl and
converts `ValueError` into a printed refusal plus `None`.

### Aggregation arithmetic and fail-closed behaviour

```text
M1a PASS n_batches=3
M1b PASS multi-batch EOS accuracy is the batch mean (0.11111111442248027)
M1c PASS early-EOS false positives are summed (1)
M1d PASS source=eval_computed for a fully-measured multi-batch run
M2a PASS one batch missing EOS accuracy => whole run unavailable (no partial credit)
M2b PASS mixed per-batch sources => unavailable
M2c PASS empty batch list => unavailable (not a fabricated 0.0 pass)
M2d PASS projection preserves unavailable
```

### `--score-only` contract — the (a)/(b) hybrid works as documented

Verified through the production `--decode-only` → `--score-only` chain (bundle produced by a real
`main()` run; `_score_n2_bundle` is the function `main()` delegates to):

```text
G4a PASS decode-only bundle written
G4b PASS bundle.teacher_forced_ce is a measured dict: dict source='eval_computed'
G5a PASS --score-only measured rc=0 traceback=False
G5b PASS --score-only + measured TF => formal v2 report
G5c PASS score-only report is stamped eval_computed with score_only provenance
G5d PASS score-only report passes unpatched validation
G6a PASS --score-only bare float rc=0 traceback=False
G6b PASS --score-only bare float: no formal report, gate JSON intact
G6c PASS skip message: D1c v2 formal report skipped: --score-only without measured
         eval_computed teacher-forced EOS metrics in decode bundle (source='unavailable')
G7b PASS decode_bundle provenance never yields a formal report
G8b PASS eval_computed source without measured EOS is still refused (no fabricated 1.0/0)
```

`G8b` is the important negative: claiming `eval_computed` is not sufficient — the EOS numbers must
actually be present, so a caller cannot launder provenance.

**Disclosure:** for the `--score-only` runs I stubbed `assert_score_only_d1b_bundle` and
`assert_score_only_wavs_present`. Those are provenance/`max_frames` gates orthogonal to FAIL #4 and
were verified live against `scripts/74` in round 2 (`logs/reaudit2_probe_72cli*.log`); everything
downstream of them — contract load, gate evaluation, emitter, schema validation — ran unpatched.

---

## NEW BLOCKER — the evaluator never stamps the candidate checkpoint step

### The defect

`_emit_d1c_v2_from_gate_impl` derives the step like this:

```1039:1039:scripts/74_eval_discrete_lm.py
            "step": int(provenance.get("checkpoint_step") or gate.get("step") or 0),
```

Neither source is ever populated on the production path:

- `main()` builds `provenance` at `scripts/74:1592-1614` with no `checkpoint_step` key.
- `evaluate_n2_gate_d1b` never sets `gate["step"]`, and `main()` adds only `artifacts`,
  `teacher_forced_ce`, `provenance`, `text_usage_acceptance`, `stability_acceptance`.
- `scripts/74` exposes no CLI flag to supply it.
- The only occurrences of `checkpoint_step` anywhere in the repository are in the round-1/2/3
  **probe and test files**, which hand-write `"checkpoint_step": 750` into a synthetic provenance
  dict. That is precisely why three rounds of probes never saw this.

Measured on the real `main()` run and on five production-shaped emissions:

```text
G1o FAIL checkpoint.step stamped by the REAL main() run: 0
         (0 => the selector's five-step requirement can never be met)
M3b FAIL checkpoint.step stamped by the PRODUCTION provenance shape -> step=0
M3c INFO steps stamped on five production-shaped reports: [0, 0, 0, 0, 0]
M3d FAIL evaluator -> selector (plan §8.4) STILL cannot run end-to-end:
         ValueError: D1c candidate steps must be exactly [250, 375, 500, 625, 750]
M3f PASS scripts/74 exposes NO CLI flag to supply the candidate step (found []);
         nothing in production sets provenance['checkpoint_step'] or gate['step']
M3g INFO other scripts referencing checkpoint_step: []
```

The control isolates it to exactly one field — same reports, same contract, same rows:

```text
M3e PASS CONTROL: identical reports with step hand-patched + EOS==1.0 select -> step=250
```

### Why this is a blocker, not a nit

Plan §8.4 requires `scripts/74` to "itself emit a selector-consumable v2 report" and requires
`scripts/76` in D1c formal mode to accept "exact candidate steps `[250,375,500,625,750]`, each
exactly once" with "one checkpoint hash/step match per report". With `step = 0` on all five,
`select_d1c_formal_reports` raises before it evaluates a single candidate. There is no operator
workaround: the only way to fix a report is to hand-edit it and recompute `report_sha256`, which
defeats the tamper-evidence the whole v2 schema exists to provide. **The evaluator→selector formal
interface has still never executed end to end.**

### The information is already in the file being loaded

The trainer writes candidates as `candidate_step_XXXX.pt` with `"step": int(step)` inside the
payload (`discrete_lm_trainer.py:646,690`), and `scripts/74::_load_model` already loads that exact
payload — then discards everything except `cfg` and `model`:

```156:160:scripts/74_eval_discrete_lm.py
def _load_model(ckpt: Path, device: torch.device):
    payload = torch.load(ckpt, map_location=device, weights_only=False)
    cfg = payload.get("cfg") or load_yaml("configs/discrete_speech_lm_d1b_443m.yaml")
    model = build_discrete_speech_lm(cfg)
    sd = payload.get("model") or payload
```

This is the identical failure mode as FAIL #4: the value is measured/available, and then dropped
before the emitter sees it. The fix is to return the payload step from `_load_model` (or parse the
immutable `candidate_step_XXXX.pt` filename), put it into `provenance["checkpoint_step"]`, and
cross-check the two against each other. Add a `--checkpoint-step` override plus a fail-closed
assertion that the value is in `contract.train.candidate_steps`.

### Related fail-open in the schema (LOW)

`validate_formal_report` checks `checkpoint.get("step") in (None, "")`, so `0` passes validation.
The schema should require a positive integer drawn from the contract's `candidate_steps`; had it
done so, the emitter would have failed closed instead of writing five indistinguishable reports.

---

## FAIL #2 — contract / recipe / manifest binding — **PASS** (no regression after the reseal)

The round-3 reseal is internally consistent and I recomputed it independently of the repair
agent's helpers:

```text
R2a PASS independent content recompute 25c370a94f589127 == declared 25c370a94f589127
R2b PASS library recompute agrees
R2c PASS tampering infer.max_frames changes the content sha
R2d PASS recipe_source_manifest n_locked=19 mismatches=[]
R2e PASS manifest dict sha 2d31bb9b0b37d548 == declared 2d31bb9b0b37d548
R2f-w1 PASS w1 binds contract=25c370a94f589127 recipe=25c370a94f589127 manifest=2d31bb9b0b37d548
R2f-w2 PASS w2 binds contract=25c370a94f589127 recipe=25c370a94f589127 manifest=2d31bb9b0b37d548
R2g PASS v2.0.1 parent pointers repointed at the live 2.0.0 digests (25c370a94f589127)
R2h PASS scripts/72 no longer silently overwrites eval_contract_hash
R2i PASS scripts/72 still asserts the declared contract hash before stamping
```

The round-2 digests (`32f0ea2b…` / `cb697d4f…`) are correctly superseded by the round-3 reseal
(`25c370a9…` / `2d31bb9b…`) after the `74` / `76` / `d1b_oracles` edits, and W1/W2 were rebound as
a set. `scripts/72:461-482` still asserts `declared == live` before stamping, so the round-1 silent
overwrite stays closed.

**Residual (low, unchanged from round 2):** `scripts/72:483-488` still *overwrites* `recipe_hash`
and `recipe_source_manifest_hash` from the contract rather than asserting equality with the
declared YAML values — the exact pattern that was fixed for `eval_contract_hash`. Contained in
practice (`recipe_source_manifest_sha256` is inside the content hash), but a recipe can publish a
wrong `recipe_hash` that no gate reports.

## FAIL #7 — resume hash checks — **PASS** (no regression)

```text
R7a-w1 PASS w1 resolve_repro_hashes inert keys=[] (n=8)
R7a-w2 PASS w2 resolve_repro_hashes inert keys=[] (n=8)
R7b PASS stale eval_contract_hash raises: eval_contract_hash mismatch: declared=aaaa… live=25…
R7c-run_manifest_hash PASS dropping train.run_manifest_hash still resolves all keys (None=[])
R7c-recipe_hash PASS
R7c-recipe_source_manifest_hash PASS
R7c-eval_contract_hash PASS
R7d PASS D1c resume fail-closed message still present in the trainer
```

All eight repro keys resolve for both recipes, backfill still works when a hash is dropped from the
YAML (so removing a field cannot disable its check), and a stale declared hash raises. The round-1
fail-open remains closed.

**Residual (low, unchanged):** the non-D1c branch of `discrete_lm_trainer.py:784-792` still
`continue`s past a `None` current value when `termination.mode != "semantic_eos_v1"`.

---

## HIGH — stale DEV32 provenance docs — **PASS**

Every live digest of `configs/train_subsets/d1c_dev32.json` now appears in both published
artifacts, and all three round-2 stale values are gone:

```text
H0 INFO live d1c_dev32: {"file_sha256": "f9f82b708b23ab61", "items_sha256": "501330481c07d928",
                         "select8": "882691c9fa7e799b", "gate24": "5485d37be12342d1"}
H1 PASS WAVE1_REPORT.md contains every live DEV32 digest: all true
H2 PASS round-2 stale digests absent from WAVE1_REPORT.md: []
H3 PASS oracle_overlap_report.json matches live: all true
H4 PASS stale digests absent from oracle_overlap_report.json: []
H5 PASS WAVE1_REPORT.md states the file_sha256 redefinition caveat
H6 PASS WAVE1_REPORT.md carries live contract sha 25c370a94f589127
H7 PASS repair_round3_hashes.json carries live contract sha 25c370a94f589127
H8 PASS REPAIR_ROUND3.md quotes the live contract sha
H9 PASS REPAIR_ROUND3.md quotes live recipe_source_manifest_sha256 2d31bb9b0b37d548
```

The `file_sha256` redefinition (payload-content digest of the canonically re-serialised body with
the self-referential field removed, not the raw file bytes) is now stated explicitly where it is
published, which was the round-2 request.

## MEDIUM claims — all four **PASS**

**§11 fields populated (G1k, G1l).** A real `main()` run emits `gates.text_usage`,
`summaries.phrase_loop_repetition`, `summaries.runaway_or_repetition` and
`gates.runaway_or_repetition`, and `gates.text_usage.pass` mirrors the live
`text_usage_acceptance` block in `n2_gate.json` rather than a constant. The three structurally
vacuous §11 terms from round 2 are now real.

**Selector refuses an absent `text_usage` block (M4a).** `scripts/76:141-145` treats a missing
block as not eligible. Live: five otherwise-eligible reports with `gates.text_usage` deleted →
`D1c formal selection found no eligible candidate; refusing least-bad selection`.

**Runtime `recipe_source_manifest` verification (M5a–M5c).** `enforce_eval_contract` now verifies
the manifest for `semantic_eos_v1` contracts. I proved it is genuinely executed rather than dead
code by running it from a foreign cwd, where it fails closed with
`recipe_source_manifest verify failed`.

*Caveat (low):* the call passes `root=cwd`, so `enforce_eval_contract` hard-fails whenever the job
is launched outside the repo root. Fail-closed, therefore safe, but an operator foot-gun that
should be pinned to the repo root explicitly.

**D1c subset aliases (M6).** All four aliases resolve to the D1c oracle with digests that match the
contract, and the D1b aliases are untouched:

```text
M6-d1c_dev32   PASS n=32 items_sha256=501330481c07d928 contract[dev32]=501330481c07d928  match=True
M6-d1c_full32  PASS n=32 items_sha256=501330481c07d928 contract[full32]=501330481c07d928 match=True
M6-d1c_select8 PASS n=8  items_sha256=882691c9fa7e799b contract[select8]=882691c9fa7e799b match=True
M6-d1c_gate24  PASS n=24 items_sha256=5485d37be12342d1 contract[gate24]=5485d37be12342d1  match=True
M6b-dev32/select8/gate24/full32 PASS D1b aliases still point at the D1b oracles
```

---

## Does the new regression test actually catch the round-2 bug?

Mutation testing (`logs/reaudit3_probe_mutation.log`): I spliced the round-2 code back into
`scripts/74` **in memory** and re-ran `test_production_tf_aggregation_to_emitter_survives` through
its own `_load_script` seam. No file on disk was modified.

```text
X0 PASS unmutated source: test PASSED
X1 PASS round-2 discarding aggregation spliced back in -> the new test FAILS as required:
        KeyError: 'teacher_forced_ce_source'
X2 PASS round-2 bare-float projection spliced back in -> the new test FAILS as required:
        TypeError: 'float' object is not subscriptable
X3 INFO main() rewired back to `tf_ce = summary.get("loss")` (helpers untouched)
        -> the new test STILL PASSES
X4 FAIL COVERAGE: the new test only covers the helper chain; a main() regression would ship green
```

So the test is real and non-tautological against the round-2 defect — but it stops one level short.
It never calls `main()`, so it cannot protect the production wiring, and it cannot see the
`checkpoint.step` defect at all. **[MEDIUM] Promote it to a `main()`-level integration test** with
the outer GPU/codec/ASR dependencies stubbed; `logs/reaudit3_probe_mainpath.py` is a working
template.

## Other findings

**[MEDIUM] `evaluate_n2_gate_d1c` is dead code.** `src/minispeech/eval/discrete_n2.py:902` defines
the D1c gate that enforces `learned_eos_count == len(rows)`, zero safety terminations and
`tf_eos_final_top1_accuracy == 1.0`. Nothing calls it — `main()` and `_score_n2_bundle` both call
`evaluate_n2_gate_d1b`. Consequence: `n2_gate.json` carries a D1b verdict, so
`gates.eligible = true` can appear in a v2 report for a run whose terminations were all
`safety_max_frames`. Not a blocker, because `scripts/76` independently recomputes those predicates
from the bound rows and refuses (round-2 E9/E10/E11, and my M4a), but a reader of the gate artifact
would be misled, and the D1c termination gate is not exercised anywhere.

**[LOW] Unweighted batch mean for `tf_eos_final_top1_accuracy`.** The model returns a per-batch mean
over utterances; `aggregate_teacher_forced_batches` then takes an unweighted mean across batches, so
a partial final batch skews the published number (demonstrated with batches of 2/3/1 in M1b). The
strict `== 1.0` selector predicate is unaffected, but the number published in a formal report is not
the corpus mean. Weight by utterance count, or report min as well.

**[LOW] The emitter's soft-refuse only catches `ValueError`.** Every current refusal path in
`_emit_d1c_v2_from_gate_impl` and `discrete_report_schema._fail` raises `ValueError`, so the
guarantee holds today (verified in G2/G3/G6/G8). An unexpected `TypeError` / `AttributeError` /
`OSError` from row enrichment would still traceback after `n2_gate.json` is written. Consider
catching `Exception` and re-raising only on a debug flag.

---

## Pytest attestation — **VALID**

| Check | Result |
|---|---|
| Claimed result | `398 passed, 3 skipped, 2 warnings in 50.96s`, `401 collected` |
| Independent reproduction | `398 passed, 3 skipped, 2 warnings in 51.92s`, `401 collected` — **exact match** |
| Same environment | rootdir `/mnt/c/projects/tts-self-d1c`, `GIT_DIR`/`GIT_WORK_TREE` override, `CUDA_VISIBLE_DEVICES=`, `git rev-parse HEAD = 0661d10e…` in both |
| Collected test-id sets | **identical** (`logs/reaudit3_testdiff.sh`: `IDENTICAL test id sets`) |
| New tests vs round 2 | exactly two, as claimed: `test_d1c_subset_aliases_resolve_to_d1c_oracle`, `test_production_tf_aggregation_to_emitter_survives` |
| Arithmetic | round-2 GPU-masked `396 + 3 = 399` → round-3 `398 + 3 = 401`; +2 matches the two new tests |
| Log mtime | `2026-08-04 10:21:13 +0300` (run started `07:20:21 UTC`) |
| Newest attested source | `configs/eval_contracts/d1c_semantic_eos_v2_0_1.json` at `10:19:05` — **no source post-dates the log** |
| GPU skip caveat | correctly disclosed; the three skips are `test_bf16_forward_cuda` (CUDA masked) plus the two pre-existing DDP opt-ins |

Standing caveat, correctly disclosed in the log header: the result requires `GIT_DIR`/`GIT_WORK_TREE`
pointing at `/mnt/c/projects/tts-self/.git/worktrees/tts-self-d1c`, otherwise
`tests/test_mel_baseline.py::test_git_sha_and_run_manifest` fails. That is an artifact of running a
Windows worktree through `/mnt/c` and would not occur in the canonical WSL clone.

**Attestation caveat (unchanged in spirit):** a green suite is not evidence that the D1c formal path
works. The suite passes at 398/401 while the evaluator→selector interface cannot run, because no
test drives `main()`, and every selector test supplies `checkpoint.step` itself via a hand-built
report — none asserts that the *evaluator* stamps it.

## Prior-PASS spot checks — all still hold

```text
RXa PASS residual params with grads (cb0_w=0, eos_w=0): 45/45
RXb PASS immediate EOS -> codes.shape=(1, 0, 16) lengths=[0] (no phantom frame)
RXc PASS select8 composition: 2 RUSLAN + 6 CV
RXd PASS select8 | gate24 == full32 and disjoint (|s|=8 |g|=24 |f|=32)
```

The `cb0_weight=0, eos_loss_weight=0` gradient variant is the decisive one: all 45 residual
parameters still receive non-zero gradients, which is only possible if the residual CE term is
genuinely in the backward graph.

---

## Critical remaining defects, in fix order

1. **[BLOCKER] Stamp the candidate checkpoint step in `scripts/74`.** Return the payload `step`
   from `_load_model` (and/or parse the immutable `candidate_step_XXXX.pt` filename), cross-check
   the two, put it into `provenance["checkpoint_step"]`, add a `--checkpoint-step` override, and
   fail closed when the value is not in `contract.train.candidate_steps`. Then add a `main()`-level
   test that produces five reports and drives `select_d1c_formal_reports` to a real selection.
2. **[LOW, same defect] Tighten `validate_formal_report`** to require `checkpoint.step` to be a
   positive integer in the contract's `candidate_steps`, so this class of miss fails closed.
3. **[MEDIUM] Promote the new regression test to `main()` level** — the current one cannot see a
   production-wiring regression (proved by X3/X4).
4. **[MEDIUM] Wire `evaluate_n2_gate_d1c` in, or delete it.** The D1c termination gate is currently
   unreachable and `n2_gate.json` publishes a D1b verdict on a D1c run.
5. **[LOW] Weight the EOS accuracy aggregate** by utterance count (or publish min alongside mean).
6. **[LOW] Pin `verify_recipe_source_manifest` to the repo root** in `enforce_eval_contract` instead
   of relying on `cwd`.
7. **[LOW] Assert rather than overwrite** `recipe_hash` / `recipe_source_manifest_hash` at
   `scripts/72:483-488`; consider including `recipe_sha256` in the contract content hash.
8. **[LOW] Close the non-D1c resume fail-open** at `discrete_lm_trainer.py:784-792`.
9. **[LOW] Broaden the emitter soft-refuse** beyond `ValueError`.
10. **[LOW] Use a tolerance** for `macro_cer <= copy_macro + 0.03` at `scripts/76:133-135` if the
    bound is meant to be inclusive (carried over from round 2, unchanged).

## Standing non-code blockers (unchanged, correctly disclosed by the repair agent)

1. No clean committed D1c source SHA. Branch `d1c/semantic-eos` is still at baseline `0661d10` with
   all D1c work uncommitted; `contract.repro` has no `clean_sha_available`, so `require_clean_sha`
   refuses — correctly. Requires explicit user commit/push authorization.
2. External confirmation-seed commitment still absent and correctly **not** invented.
3. Wave 3 GPU smoke and Wave 4 DEV remain NO-GO.

## Scope confirmations

- No commit created, no push, no branch mutation.
- No GPU work; `CUDA_VISIBLE_DEVICES=` on every run. No training, decode, or synthesis; no new
  `best.pt` / `last.pt`, so no listen-demo obligation arises.
- No seed invented; no methodology file touched.
- Probe scratch files were written only under `/tmp` and
  `.cursor/results/d1c_semantic_eos_repair_b81ecc3e/logs/`. Post-probe source integrity is
  byte-identical to the pre-audit snapshot and the frozen manifest verifies with 0 mismatches.
- Waves 0–2 remain **not PASS**. Wave 3 remains **NO-GO**.

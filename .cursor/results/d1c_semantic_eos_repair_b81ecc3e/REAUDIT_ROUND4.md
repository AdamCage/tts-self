# REAUDIT_ROUND4 — independent adversarial re-verification of the D1c INFRA repair (round 4)

**Plan:** `.cursor/plans/d1c_semantic_eos_repair_b81ecc3e.plan.md`
**Slug:** `d1c_semantic_eos_repair_b81ecc3e`
**Worktree audited:** `C:\projects\tts-self-d1c`, branch `d1c/semantic-eos`
**Branch tip:** `0661d10eb616362e74dc1750e03cf25c33a7943e` (baseline `origin/master`; the branch still
carries **no commits of its own** — all D1c work remains uncommitted working-tree state)
**Input under audit:** `REPAIR_ROUND4.md` (claims the round-3 `checkpoint.step` blocker is FIXED on
the production path, `evaluate_n2_gate_d1c` wired, main()-level regression added, pytest
`401 passed, 3 skipped`)
**Prior audits:** `REAUDIT.md` (r1), `REAUDIT_ROUND2.md` (r2), `REAUDIT_ROUND3.md` (r3 — verdict
STILL_INVALID_RUN, NEW BLOCKER `checkpoint.step=0`)
**Auditor:** independent re-audit agent. Every claim in `REPAIR_ROUND4.md` was treated as an
unverified assertion and re-tested with live probes that drive the **production entry points**.
**Date:** 2026-08-04
**Constraints honoured:** no commit, no push, no GPU work (`CUDA_VISIBLE_DEVICES=` on every run), no
seed invented, no methodology edit, no training / inference / synthesis. No new `best.pt` / `last.pt`
was produced, so no listen-demo obligation arises.

---

## Overall verdict

> ## INFRA_READY_FOR_HUMAN_AUDIT
>
> **The round-3 blocker is genuinely fixed on the production path, and I verified it harder than
> the repair agent did.** Driving `scripts/74::main()` with the **real, unstubbed `_load_model`**
> and five **real** `candidate_step_XXXX.pt` files, the evaluator stamps `checkpoint.step` from the
> checkpoint payload cross-checked against the immutable filename, and the five **untouched**
> production reports are accepted by `select_d1c_formal_reports`, which returns a real selection.
> **The evaluator→selector formal interface (plan §8.4) has now executed end to end for the first
> time.** `evaluate_n2_gate_d1c` is genuinely wired and load-bearing, FAIL #4 / #2 / #7 all still
> hold, every prior-PASS spot check holds, and the pytest attestation reproduces exactly
> (`401 passed, 3 skipped`, 404 collected).
>
> **NEW finding (MEDIUM, not verdict-blocking under the stated rules):** the `--checkpoint-step`
> override introduced by round 4 makes an unimplemented plan §8.4 requirement exploitable. The
> selector never requires the five reports to reference *distinct* checkpoints, so **one physical
> checkpoint file can be published five times as the complete candidate set** and the selector
> accepts it and selects a winner. Plan §8.4 explicitly requires "один checkpoint hash/step match на
> report" and lists "checkpoint hash/step mismatch rejected" as a required test case. This must be
> closed before Wave 3/4, but it does not break any of the four blocker conditions.
>
> Waves 0–2 remain **not PASS**. Wave 3 remains **NO-GO**.

Round 1 shipped an emitter that crashed. Round 2 fixed the emitter but the evaluator threw the EOS
measurements away. Round 3 fixed that link and dropped the checkpoint step. Round 4 is the first
round where the whole chain runs, and the residual issue is a *different* one: not a dropped value,
but an under-constrained one.

## Per-item verdicts

| # | Check | Round-4 claim | This audit | Evidence |
|---|---|---|---|---|
| 1 | `checkpoint.step` stamped via **production `main()`** | FIXED | **PASS** | `reaudit4_probe_prod.log` A1–A4, B1–B5, C1a–C1i, C2a |
| 1b | Verified with the **real, unstubbed `_load_model`** | not tested by repair | **PASS** | B1–B3 |
| 2 | Mismatch / missing step → fail closed | FIXED | **PASS** | E1–E4 |
| 2b | Schema rejects `step ≤ 0` / non-int / bool / str / float / missing | FIXED | **PASS** | F1–F7 |
| 3 | Evaluator→selector §8.4 E2E, steps {250,375,500,625,750} | live probe | **PASS** (stronger: five real `main()` runs, untouched reports) | D1–D4 |
| 3b | Control: `step=0` report still refused | claimed | **PASS** | D5 |
| 4 | `evaluate_n2_gate_d1c` used on the D1c path (not D1b published as D1c) | WIRED | **PASS** | C1f, C1g, G1, G2, G3 |
| 5 | main()-level regression test catches missing-step / discarded-metrics | FIXED | **PASS** (mutation-proved) | `reaudit4_probe_mutation.log` X0–X6 |
| 6 | FAIL #4 — evaluator→emitter EOS plumbing | still PASS | **PASS** | C1h, C1i, X2 |
| 6 | FAIL #2 — contract / recipe / manifest binding after reseal | resealed | **PASS** | `reaudit4_probe_rest.log` R2a–R2k |
| 6 | FAIL #7 — resume hashes | still PASS | **PASS** | R7a–R7d |
| 6 | Residual grads / select8 2+6 / immediate EOS | — | **all hold** | RXa–RXe |
| 7 | Pytest attestation `401 passed, 3 skipped` (404 collected) | claimed | **VALID — exact reproduction** | `reaudit4_full_pytest.log`, `reaudit4_attest.log` P1–P9 |
| — | **NEW: duplicate-checkpoint candidate set accepted by selector** | not disclosed | **FAIL (MEDIUM)** | `reaudit4_probe_dupckpt.log` N1–N3 |

Totals across my probes: **41/41** production-path checks, **7/7** mutation checks, **28/28**
regression/seal checks, **10/10** attestation checks, and **2 FAILs** in the new-surface probe that
I wrote specifically to attack round 4's new flag.

## Method — what I did differently from the repair agent

Everything below was executed, not read. Probe scripts and logs sit next to this report.

| Artifact | What it does |
|---|---|
| `logs/reaudit4_probe_prod.py` / `.log` | **decisive probe.** Real `torch.save` → **real `_load_model`** → real `resolve_checkpoint_step` → real gate → real emitter → real schema → real selector. Five real `main()` runs on five real checkpoints; the emitted JSONs are fed to the selector **untouched**. Plus main()-level negatives, schema negatives, D1c-gate load-bearing checks, decode-bundle step recovery |
| `logs/reaudit4_probe_mutation.py` / `.log` | mutation testing of `test_main_stamps_checkpoint_step_and_selector_e2e`: five defects spliced back into `scripts/74` / the schema **in memory**, then the test is re-run through its own `_load_script` seam |
| `logs/reaudit4_probe_rest.py` / `.log` | round-4 reseal recompute, W1/W2 rebinding, v2.0.1 parent pointers, resume-hash regression, published-digest audit, prior-PASS spot checks |
| `logs/reaudit4_probe_dupckpt.py` / `.log` | new-surface attack on the round-4 `--checkpoint-step` flag |
| `logs/reaudit4_full_pytest.log` | independent full-suite reproduction (same `GIT_DIR` override, GPU masked) |
| `logs/reaudit4_attest.py` / `.log` | pytest attestation audit (counts, collected ids, new-test presence, log-vs-source mtimes) + post-probe source digests |
| `logs/reaudit4_post_probe_hashes.json` | post-probe digests of every source my probes could have touched |

Test environment: WSL2 Ubuntu, venv `/home/adam/projects/tts-self/.venv`, Python 3.11.15, rootdir
`/mnt/c/projects/tts-self-d1c`, `PYTHONPATH` pinned to the **worktree** `src`,
`CUDA_VISIBLE_DEVICES=` on every run.

**Why the repair agent's own probe was not sufficient.** `logs/repair_round4_probe_step.py:248` does

```python
mod._load_model = lambda p, d: (model, copy.deepcopy(cfg), dict(payload))
```

That is, it replaces `_load_model` with a lambda that *hands the payload back*. A `_load_model` that
discards the payload would still make that probe green — which is the same self-certification
pattern that let rounds 2 and 3 pass their own probes. Its selector §8.4 evidence is also
one `main()` run plus four `copy.deepcopy` clones with `cloned["checkpoint"]["step"] = s`
hand-written in, so the *evaluator* only ever produced one of the five steps under test. I removed
both shortcuts.

---

## 1–2. `checkpoint.step` on the production path — **PASS**

### The real `_load_model` really returns the payload

```text
A4 PASS _load_model returns 3-tuple on every return path (n_returns=1)
B1 PASS REAL _load_model returned tuple len=3
B2 PASS REAL _load_model 3rd element carries payload step=750
       (NOT stubbed - this is the check the repair probe could not make)
B3 PASS REAL _load_model rebuilt a live model (DiscreteSpeechLM)
```

`scripts/74:227-236` now returns `(model, cfg, payload)`, and I confirmed it against a real
`torch.save`d checkpoint rather than an AST assertion alone.

### A real `main()` run stamps the step, with the metric path fully unpatched

Run C1 keeps `teacher_forced_ce` completely real (untrained tiny semantic model):

```text
C1a PASS main() rc=0 tb=False exit=''
C1b PASS production main() emitted formal v2 report
C1c PASS REAL _load_model + REAL tf metrics -> report checkpoint.step=750
C1d PASS gate.step=750
C1e PASS provenance.checkpoint_step=750
C1f PASS gate id=N2_D1c_semantic_eos_v1 (evaluate_n2_gate_d1c, not D1b)
C1g PASS criteria.termination_d1c present
C1h PASS tf source=eval_computed acc=0.0 fp=0
C1i PASS unpatched validate_formal_report accepts the report
C2a PASS CONTROL different real ckpt -> step=250 (not a hardcoded 750)
```

`C1h` is the FAIL #4 control carried forward: an untrained model reports EOS accuracy `0.0`, which
no fabricated `1.0` default could produce. `C2a` is the anti-constant control — the same code path
on a different real checkpoint yields `250`, so `750` is not baked in anywhere.

### Fail-closed behaviour on every degenerate step source

```text
E1 PASS payload/filename mismatch fails closed:
       'checkpoint step mismatch: payload step=500 != filename step=750 (…/candidate_step_0750.pt)'
E2 PASS missing payload step + non-candidate filename fails closed:
       "checkpoint step unresolved: pass --checkpoint-step or evaluate candidate_step_XXXX.pt …"
E3 PASS --checkpoint-step conflict fails closed:
       '--checkpoint-step=375 conflicts with inferred step=750'
E4 PASS --checkpoint-step outside candidate_steps fails closed:
       'checkpoint step 999 not in contract.train.candidate_steps [250, 375, 500, 625, 750]'
E5 PASS --checkpoint-step override on best.pt -> step=625
E6 PASS payload optimizer_step fallback -> step=500
```

E1/E2 also assert that **no formal report is written** in the refusal cases, not merely that an
exception was raised.

### Schema now fails closed on the whole `step` type family

```text
F1 step=0     refused   F2 step=-5      refused   F3 step=True refused
F4 step='750' refused   F5 step=750.0   refused   F6 missing   refused
F7 PASS allowed_candidate_steps membership enforced
```

`F3` (bool is an `int` subclass) and `F5` (float) are the two the round-3 recommendation did not
explicitly ask for; both are correctly rejected by `discrete_report_schema.py:140-148`.

---

## 3. Evaluator→selector §8.4 — **PASS, end to end, for the first time**

Five separate `main()` invocations on five separate real `candidate_step_XXXX.pt` files, five
production-written `discrete_eval_report_v2.json` files handed to the selector **byte-untouched**:

```text
D1 PASS five REAL main() runs stamped steps=[250, 375, 500, 625, 750]
D2 PASS five untouched production reports on disk: 5
D3 PASS selector accepted UNTOUCHED production reports steps=[250, 375, 500, 625, 750]
D4 PASS selector selected step=250
D5 PASS CONTROL step=0 refused: ValueError: formal D1c report refused:
       checkpoint.step must be a positive integer (not 0/missing)
```

This is the interface that round 3 recorded as "has still never executed end to end". It now does.

**Disclosure.** For the five eligibility runs I wrapped the real `teacher_forced_ce` and overrode
only the two EOS scalars to their perfect values (`accuracy=1.0`, `false_positives=0`), because an
untrained tiny model cannot reach EOS accuracy `1.0` and the selector requires strict `== 1.0`.
Everything else in those runs is real: the model forward, the CE terms, the batch aggregation, the
emitter projection, the D1c gate, the step resolution, the schema validation, and the selector.
Run C1 above uses the metric path with **no** override at all and still stamps the step correctly,
so the two concerns are separated. Outer GPU/codec/ASR/dataset dependencies (`load_codec_items`,
`enforce_eval_contract`, `_resolve_speaker_registry`, `_ref_texts_from_manifest`,
`free_running_asr_eval`, `load_yaml`) are stubbed on the same seam round 3 used.

---

## 4. `evaluate_n2_gate_d1c` — **PASS, wired and load-bearing**

```text
C1f PASS gate id=N2_D1c_semantic_eos_v1 on a D1c contract
C1g PASS criteria.termination_d1c present
G1  PASS D1c gate flags an all-safety_max_frames run (term_pass=False)
G2  PASS D1c termination passes on a clean learned_eos run
G3  PASS D1b contract keeps the D1b gate: gate='N2_D1b', formal_v2_emitted=False
X3  PASS MUTATION main() uses evaluate_n2_gate_d1b on a D1c contract -> the new test FAILS
```

`G3` is the negative that matters: a D1b contract still gets `N2_D1b` and produces no D1c formal
report, so the D1c gate is selected by contract content (`_is_d1c_semantic_contract`) rather than
applied blanket. `X3` proves the wiring is protected by a test. Round 3's "`evaluate_n2_gate_d1c` is
dead code" finding is closed.

---

## 5. Does the new regression test actually catch anything? — **PASS (mutation-proved)**

Round 3's `X3/X4` recorded that the then-new test "only covers the helper chain; a main() regression
would ship green". I re-ran that exact experiment against round 4's `main()`-level test, splicing
defects into `scripts/74` **in memory** (no file on disk modified):

```text
X0 PASS unmutated source: test_main_stamps_checkpoint_step_and_selector_e2e -> PASSED
X1 PASS MUTATION drop provenance['checkpoint_step'] + gate['step'] in main()
       -> AssertionError: main() must emit formal v2 report
X2 PASS MUTATION round-2 main() bare-float rewire (round-3 X3/X4 blind spot)
       -> AssertionError: main() must emit formal v2 report
X3 PASS MUTATION main() uses evaluate_n2_gate_d1b on a D1c contract -> AssertionError
X4 INFO MUTATION _load_model discards payload -> PASSED
       (test stubs _load_model, and candidate_step_XXXX.pt is a second independent
        source, so this is expected not to be caught here)
X5 PASS MUTATION round-3 schema (step=0 allowed) -> step-zero test DID NOT RAISE (caught)
X6 PASS round-3 helper test still green on unmutated source
```

**The round-3 `X3/X4` coverage hole is closed:** the exact `main()` rewire that shipped green in
round 3 now turns the suite red (`X2`).

`X4` is a residual coverage note, not a defect. The test stubs `_load_model`, so a `_load_model`
that discarded the payload would not be caught by the suite — but it also would not be a live bug,
because the immutable `candidate_step_XXXX.pt` filename is an independent second source and my `B2`
check confirms the real function does return the payload. **[LOW] Add one assertion that the real
`_load_model` returns a payload carrying `step`**, so the suite covers what `B2` covers.

---

## NEW finding — one checkpoint can be published as all five candidates — **MEDIUM**

### What I found

Plan §8.4 requires the D1c formal selector to enforce **"требует один checkpoint hash/step match на
report"**, and §"Report/selector cases" lists **"checkpoint hash/step mismatch rejected"** as a
required test case. Neither is implemented. `select_d1c_formal_reports`
(`scripts/76:55-83`) checks that the five *steps* are exactly `[250,375,500,625,750]` and distinct,
and that provenance/infer/row-binding are common — but it never looks at
`checkpoint.sha256` beyond copying it into the candidate row and using it as a tie-break key.

Round 4's new `--checkpoint-step` flag turns that omission into an exploitable path, because the
step no longer has to be derived from the artifact at all (`E5` above: `best.pt` +
`--checkpoint-step 625` produces a fully valid formal report). I drove production `main()` five
times against **one single `best.pt`**, varying only the flag:

```text
N1 INFO five reports produced from ONE file: steps=[250, 375, 500, 625, 750]
        distinct_ckpt_sha256=1
N2 FAIL SELECTOR ACCEPTED five reports that all point at the SAME checkpoint
        sha256=578374ee87e8a69a (selected step=250)
        — plan §8.4 'one checkpoint hash/step match per report' is NOT enforced
N3 FAIL schema accepted a checkpoint.sha256 that does not match the file at
        checkpoint.path (no on-disk hash verification)
```

### Severity and why it is not a blocker under the stated rules

It is not one of the four blocker conditions: step stamping works on `main()`, the selector E2E
passes, the D1b gate is not published on D1c, no regression fired, and the pytest attestation is
valid. Exploiting it requires deliberate operator misuse — the real Wave 3/4 protocol evaluates
five distinct `candidate_step_XXXX.pt` files, where the step comes from the payload cross-checked
against the immutable filename and `--checkpoint-step` is unnecessary. `N3` is likewise bounded in
practice: the emitter always computes `checkpoint.sha256` from the real file
(`scripts/74:1070`), so only a hand-forged report can carry a false hash, and forging one requires
recomputing `report_sha256` (which is a plain digest, not a signature).

But it is a plan requirement that is written down, untested, and now reachable, and it directly
undermines the tamper-evidence the v2 schema exists to provide.

### Recommended fix (before Wave 3/4)

1. In `select_d1c_formal_reports`, require the five `checkpoint.sha256` values to be **distinct**,
   and require `checkpoint.path` basenames to match `candidate_step_<step>.pt` for the report's own
   step. Add the plan's "checkpoint hash/step mismatch rejected" test case.
2. Restrict `--checkpoint-step` on the D1c formal path: allow it only as a *cross-check* of a value
   that already resolves from payload/filename, or gate it behind an explicit
   `--allow-unbound-checkpoint-step` diagnostic flag that also forces `formal_eligible=false`.
3. Optionally have the selector re-verify `checkpoint.sha256` against `checkpoint.path` when the
   file is reachable.

---

## 6. Prior FAILs and spot checks — all still hold

### FAIL #2 — contract / recipe / manifest binding after the round-4 reseal — **PASS**

```text
R2a PASS declared sha256 == independent recompute 9922ce4d75932dca
R2b PASS REPAIR_ROUND4 claimed contract_sha256 matches live
R2c PASS recipe_sha256 == contract sha
R2d PASS tampering infer.max_frames changes the content sha (seal is live)
R2e PASS recipe_source_manifest n_locked=19 mismatches=[]
R2f PASS recipe_source_manifest_sha256 live=62eeeae2c6360e88 == claim
R2g PASS round-4 edited sources are inside the locked manifest (missing=[])
R2h-w1/w2 PASS both recipes bind contract/recipe/manifest to the live digests
R2i PASS v2.0.1 parent pointer repointed at the live 2.0.0 digest
R2j PASS v2.0.1 no longer references the superseded round-3 contract digest
R2k PASS scripts/72 still asserts declared eval_contract_hash before stamping
```

`R2g` is the check that matters after an edit-then-reseal round: the three files round 4 changed
(`scripts/74`, `discrete_report_schema.py`, `scripts/76`) are all inside the 19 locked sources, so
the reseal covers the edits rather than routing around them. The round-3 digests
(`25c370a9…` / `2d31bb9b…`) are correctly superseded everywhere they were published.

### FAIL #7 — resume hash checks — **PASS**

```text
R7a-w1/w2 PASS resolve_repro_hashes inert keys=[] (n=8)
R7b PASS stale eval_contract_hash raises (declared=aaaa… live=99…)
R7c-{run_manifest,recipe,recipe_source_manifest,eval_contract}_hash PASS backfill still works
R7d PASS D1c resume fail-closed message still present in the trainer
```

### Published-digest audit — **PASS**

```text
H-REPAIR_ROUND4.md          PASS live_digest_present=True  round3_stale_digest_present=False
H-d1c_reliability_gate.json PASS live_digest_present=True  round3_stale_digest_present=False
H-WAVE1_REPORT.md           PASS live_digest_present=True  round3_stale_digest_present=False
```

### Standing spot checks — **all hold**

```text
RXa PASS residual params with non-zero grads (cb0_w=0, eos_w=0): 45/45
RXb PASS immediate EOS -> codes.shape=(1, 0, 16) lengths=[0] reason=learned_eos
RXc PASS select8 composition: 2 RUSLAN + 6 CV
RXd PASS select8 | gate24 == full32 and disjoint (|s|=8 |g|=24 |f|=32)
RXe PASS D1c subset aliases still registered
```

The `cb0_weight=0, eos_loss_weight=0` gradient variant remains the decisive one: all 45 residual
parameters still receive non-zero gradients, which is only possible if the residual CE term is
genuinely in the backward graph.

---

## 7. Pytest attestation — **VALID**

| Check | Result |
|---|---|
| Claimed | `401 passed, 3 skipped, 2 warnings in 52.51s`, 404 collected |
| Independent reproduction | `401 passed, 3 skipped, 2 warnings in 53.24s`, 404 collected — **exact match** |
| Same environment | rootdir `/mnt/c/projects/tts-self-d1c`, `GIT_DIR`/`GIT_WORK_TREE` override, `CUDA_VISIBLE_DEVICES=`, `git rev-parse HEAD = 0661d10e…` |
| Collected ids | 404, all unique |
| Three claimed new tests present | `test_validate_formal_report_rejects_checkpoint_step_zero`, `test_resolve_checkpoint_step_payload_filename_cli`, `test_main_stamps_checkpoint_step_and_selector_e2e` — all collected |
| Arithmetic | round-3 401 collected → round-4 404; +3 matches the three new tests |
| Claimed log mtime | `2026-08-04 07:51:18 UTC` |
| Newest attested source | `configs/eval_contracts/d1c_semantic_eos_v2_0_1.json` @ `2026-08-04 07:48:34 UTC` — **no attested source post-dates the log** |
| GPU skip caveat | correctly disclosed; three skips are `test_bf16_forward_cuda` (CUDA masked) plus two pre-existing DDP opt-ins |

Standing caveat, correctly disclosed by the repair agent: the green result requires
`GIT_DIR`/`GIT_WORK_TREE` pointing at `/mnt/c/projects/tts-self/.git/worktrees/tts-self-d1c`,
otherwise `tests/test_mel_baseline.py::test_git_sha_and_run_manifest` fails. That is an artifact of
running a Windows worktree through `/mnt/c` and would not occur in the canonical WSL clone.

**Attestation caveat.** A green suite still is not evidence that the D1c formal path works — but for
the first time the suite would actually go red on a production-wiring regression (`X1`, `X2`, `X3`).
The suite would *not* catch the duplicate-checkpoint hole above, because no test asserts distinct
checkpoint hashes across the candidate set.

---

## LOW-item claims in `REPAIR_ROUND4.md`

| Claim | This audit |
|---|---|
| Soft-refuse broadened beyond `ValueError` | **Confirmed** — `scripts/74:984` catches `Exception`; refusals observed non-fatally in the mutation runs |
| Unweighted EOS batch mean — deferred | **Agreed, still open** (LOW; selector uses strict `== 1.0`, so only the published corpus number is skewed) |
| `cwd`-rooted manifest verify — deferred | **Agreed, still open** (fail-closed, therefore safe; operator foot-gun outside repo root) |
| `recipe_hash` overwrite in `scripts/72` — deferred | **Agreed, still open** (contained; `recipe_source_manifest_sha256` is inside the content hash) |
| Non-D1c resume fail-open — deferred | **Agreed, still open** (D1c path is fail-closed) |
| CER margin tolerance at `scripts/76` — deferred | **Agreed, still open** |

---

## Remaining defects, in fix order

1. **[MEDIUM — NEW] Enforce one checkpoint per candidate report.** Require distinct
   `checkpoint.sha256` across the five §8.4 reports, require the `candidate_step_<step>.pt`
   basename to agree with the report's step, and add the plan's "checkpoint hash/step mismatch
   rejected" test. Restrict or flag `--checkpoint-step` on the formal path. (`N1`–`N3`)
2. **[LOW] Cover the real `_load_model`** with one assertion that it returns a payload carrying
   `step`, closing the `X4` coverage note.
3. **[LOW] Consider verifying `checkpoint.sha256` against `checkpoint.path`** in the selector when
   the file is reachable.
4. **[LOW] Weight the EOS accuracy aggregate** by utterance count (or publish min alongside mean).
5. **[LOW] Pin `verify_recipe_source_manifest` to the repo root** in `enforce_eval_contract`.
6. **[LOW] Assert rather than overwrite** `recipe_hash` / `recipe_source_manifest_hash` at
   `scripts/72:483-488`.
7. **[LOW] Close the non-D1c resume fail-open** at `discrete_lm_trainer.py:784-792`.
8. **[LOW] Use a tolerance** for `macro_cer <= copy_macro + 0.03` at `scripts/76:133-135`.

## Standing non-code blockers (unchanged, correctly disclosed by the repair agent)

1. No clean committed D1c source SHA. Branch `d1c/semantic-eos` is still at baseline `0661d10` with
   all D1c work uncommitted, so `require_clean_sha` refuses — correctly. Requires explicit user
   commit/push authorization.
2. External confirmation-seed commitment still absent and correctly **not** invented.
3. Wave 3 GPU smoke and Wave 4 DEV remain NO-GO pending human sign-off.

## Scope confirmations and post-audit integrity

- No commit created, no push, no branch mutation.
- No GPU work; `CUDA_VISIBLE_DEVICES=` on every run. No training, decode, or synthesis; no new
  `best.pt` / `last.pt`, so no listen-demo obligation arises.
- No seed invented; no methodology file touched.
- Probe scratch files were written only under `/tmp` and
  `.cursor/results/d1c_semantic_eos_repair_b81ecc3e/logs/`.
- Post-audit source integrity is byte-identical to the pre-audit snapshot (authoritative Windows-side
  `git status`: 31 entries before and after; `logs/reaudit4_post_probe_hashes.json`):

```text
f6c59fbf64f1b1d6  scripts/74_eval_discrete_lm.py
a5824c6f8218d43f  scripts/76_select_discrete_checkpoint.py
62ac323301c66bda  src/minispeech/eval/discrete_report_schema.py
7d85475e6ab3db19  src/minispeech/eval/discrete_n2.py
b54f7ebe6bc3102e  tests/test_d1c_infra_repair.py
e39461d5e7097997  configs/eval_contracts/d1c_semantic_eos_v2_0_0.json
```

- The frozen manifest still verifies after every probe: `n_locked = 19`, `mismatches = []`,
  `contract sha256 = 9922ce4d75932dcaff255dd20601f731f48c369647677c68249b9ff652d938af`.
- Waves 0–2 remain **not PASS**. Wave 3 remains **NO-GO**.

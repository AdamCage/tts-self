# REAUDIT_ROUND5 — independent adversarial re-verification of the D1c INFRA repair (round 5)

**Plan:** `.cursor/plans/d1c_semantic_eos_repair_b81ecc3e.plan.md`
**Slug:** `d1c_semantic_eos_repair_b81ecc3e`
**Worktree audited:** `C:\projects\tts-self-d1c`, branch `d1c/semantic-eos`
**Branch tip:** `0661d10eb616362e74dc1750e03cf25c33a7943e` (baseline `origin/master`; the branch still
carries **no commits of its own** — all D1c work remains uncommitted working-tree state)
**Input under audit:** `REPAIR_ROUND5.md` (claims the round-4 MEDIUM §8.4 gap is closed fail-closed:
distinct `checkpoint.sha256`, `candidate_step_<step:04d>.pt` basename, on-disk hash verification,
confirm-only `--checkpoint-step`; live probe 5/5; pytest `404 passed, 3 skipped`, 407 collected)
**Prior audits:** `REAUDIT.md` (r1), `REAUDIT_ROUND2.md` (r2), `REAUDIT_ROUND3.md` (r3),
`REAUDIT_ROUND4.md` (r4 — verdict INFRA_READY_FOR_HUMAN_AUDIT, NEW MEDIUM: duplicate-checkpoint
candidate set accepted)
**Auditor:** independent re-audit agent. Every claim in `REPAIR_ROUND5.md` was treated as an
unverified assertion and re-tested with live probes that drive the **production entry points**,
including the `scripts/76 --d1c-formal` **command line**.
**Date:** 2026-08-04
**Constraints honoured:** no commit, no push, no GPU (`CUDA_VISIBLE_DEVICES=` on every run), no seed
invented, no methodology edit, no training / inference / synthesis, no Wave 0–2 PASS claim.

---

## Overall verdict

> ## INFRA_READY_FOR_HUMAN_AUDIT
>
> **The round-4 MEDIUM is genuinely closed, and I attacked it harder than the repair agent did.**
> The round-4 attack script re-run **verbatim** now produces **zero** formal reports — all five
> `main()` invocations refuse at `resolve_checkpoint_step`. Five reports carrying one identical
> `checkpoint.sha256` are hard-refused by the selector **after passing the real, unpatched schema**
> (the repair agent's own probe replaced `validate_formal_report` with a pass-through, so it never
> demonstrated this). A forged `checkpoint.sha256` is refused at both layers, an unreachable
> `checkpoint.path` is refused by the selector even though the schema skips it, and all of this
> holds at the **production CLI**: `scripts/76 --d1c-formal` exits non-zero and writes **no**
> `checkpoint_selection.json`. Critically, the fix is **not a blanket refusal** — five real
> `main()` runs on five real distinct `candidate_step_XXXX.pt` files still produce reports the
> selector accepts and selects from. Mutation testing shows four of the five round-5 guards go red
> in the suite when removed. The pytest attestation reproduces **exactly**
> (`404 passed, 3 skipped`, 407 collected), and no prior blocker regressed.
>
> **NEW finding (MEDIUM, not verdict-blocking under the stated rules):** closing the CLI closed one
> door, not the room. `resolve_checkpoint_step(require=True)` still accepts a step derived from the
> **filename alone**, so an operator can assert any candidate step by *renaming* — I re-saved one
> model state five times as `candidate_step_0250.pt … candidate_step_0750.pt` with **no payload
> step**, and the full chain (evaluator → schema → selector) accepted the set and selected a
> winner. Round-5's rules prove the five *files* are distinct; they do not prove the five
> *training steps* are. The trainer already stamps `step` **and** `hashes` (run/recipe/contract) into
> every checkpoint payload, and the evaluator reads neither as a requirement.
>
> **NEW finding (MEDIUM–LOW):** the round-5 reseal did not update `WAVE1_REPORT.md`, which still
> publishes the **superseded round-4** digests as the live contract hashes. Rounds 3 and 4 both kept
> this artifact in sync; a human auditor following it would verify against the wrong digest.
>
> Waves 0–2 remain **not PASS**. Wave 3 remains **NO-GO**.

Round 1 shipped an emitter that crashed. Round 2 fixed the emitter but the evaluator threw the EOS
measurements away. Round 3 fixed that link and dropped the checkpoint step. Round 4 made the whole
chain run but left the candidate set under-constrained. Round 5 constrains the candidate set
correctly — the residual is one layer further out: nothing binds a candidate *file* to the training
*step* it claims, except a filename an operator controls.

**Scope note on the verdict rule.** The stated rule flips to STILL_INVALID_RUN if "the duplicate-ckpt
attack still works". The attack as defined and demonstrated in round 4 — *one physical checkpoint
file* published five times, enabled by CLI step invention — does **not** work; I reproduced it
verbatim and it is refused at three layers. The M2 finding below is a *different* attack that
requires the operator to deliberately re-save checkpoint bytes. If the parent reads
"duplicate-ckpt attack" as covering semantic duplicates (same weights, different bytes), the verdict
flips to STILL_INVALID_RUN; on the mechanism round 5 was asked to close, it is closed.

## Per-item verdicts

| # | Check | Round-5 claim | This audit | Evidence |
|---|---|---|---|---|
| 1 | Round-4 attack reproduced (one file, five `--checkpoint-step`) → refuse on the formal path | FIXED | **PASS** — 0/5 reports emitted, 0/5 gate JSONs leaked | `reaudit5_probe_attack.log` A1–A3; `reaudit5_rerun_r4_dupckpt.log` (round-4 script re-run verbatim) |
| 2 | Five reports with identical `checkpoint.sha256` → selector hard refuse | FIXED | **PASS**, and stronger: the five reports pass the **real** schema first | `reaudit5_probe_attack.log` D1–D3; CLI `reaudit5_probe_cli.log` K2–K3 |
| 3 | Claimed sha ≠ file at path → refuse | FIXED | **PASS** at schema, selector helper, end-to-end and CLI | A-probe H1–H3; CLI K4–K5 |
| 3b | Unreadable `checkpoint.path` → refuse | implied | **PASS** (schema skips, selector refuses — layering is real) | H4; CLI K6 |
| 3c | Basename must be `candidate_step_<step:04d>.pt` | FIXED | **PASS** live | H5, H6 |
| 4 | Legitimate set of five distinct candidates still accepted | asserted | **PASS** — five real `main()` runs, untouched reports, selector selects | L1–L4; CLI K1 |
| 5 | `--checkpoint-step` confirm-only | FIXED | **PASS** — confirm OK, foreign refuse, invent refuse, membership intact | C1–C7 |
| 6 | Step stamping via production `main()` | not regressed | **PASS** (`step=750`, `250` control, `optimizer_step` fallback) | A4; r4-prod re-run B1–B5, C1a–C2a, E6 |
| 6 | FAIL #4 — evaluator→emitter EOS plumbing | not regressed | **PASS** (untrained model reports `acc=0.0`, not a fabricated `1.0`) | A5; r4-prod C1h |
| 6 | `evaluate_n2_gate_d1c` wired, D1b contract keeps D1b gate | not regressed | **PASS** | A6, G1; r4-prod C1f/C1g/G1–G3 |
| 6 | FAIL #2 — contract / recipe / manifest binding after the round-5 reseal | resealed | **PASS** (19 locked, `mismatches=[]`, edits inside the manifest) | `reaudit5_probe_rest.log` R2a–R2k |
| 6 | FAIL #7 — resume hashes | not regressed | **PASS** (+ a config pinned to the round-4 digest is now refused) | R7a–R7d, R7b2 |
| 6 | Residual grads / select8 2+6 / immediate EOS / aliases | not regressed | **all hold** | RXa–RXe |
| 6 | main()-level regression test still load-bearing | not claimed | **PASS** (round-4 mutation probe re-run: 0 fails) | `reaudit5_rerun_r4_mutation.log` X0–X6 |
| 7 | Pytest attestation `404 passed, 3 skipped` (407 collected) | claimed | **VALID — exact reproduction** | `reaudit5_full_pytest_tail.log`, `reaudit5_attest.log` P1–P12 |
| 8 | Round-5 guards are protected by tests | implied by "negative tests ADDED" | **4 of 5 protected**; the basename rule is **not** | `reaudit5_probe_mutation.log` Y1–Y6, **Y3 FAIL** |
| — | **NEW: rename / re-save publishes one model state as five candidates** | not disclosed | **FAIL (MEDIUM)** | A-probe **M1–M3** |
| — | **NEW: `WAVE1_REPORT.md` publishes superseded round-4 digests** | not disclosed | **FAIL (MEDIUM–LOW)** | rest-probe **H-WAVE1_REPORT.md** |

Totals across my probes: **28/32** attack-surface checks PASS (1 FAIL = M2, 3 INFO), **30/33**
reseal/regression checks PASS (1 FAIL = stale WAVE1 digest, 2 INFO), **7/7** production-CLI checks,
**14/16** mutation checks PASS (1 FAIL = Y3 coverage hole, 1 INFO), **11/14** attestation checks PASS
(0 FAIL, 3 INFO). Round-4 probes re-run verbatim: dupckpt attack fully closed, prod **39/41** (the
two "failures" are the intentional contract change, see §5), mutation **7/7**.

## Method — what I did differently from the repair agent

Everything below was executed, not read. Probe scripts and logs sit next to this report.

| Artifact | What it does |
|---|---|
| `logs/reaudit5_probe_attack.py` / `.log` | **decisive probe.** Real `torch.save` → **real `_load_model`** → real `resolve_checkpoint_step` → real gate → real emitter → **real unpatched schema** → real selector. Round-4 attack reproduction, duplicate-identity set that survives the schema, forged/unreadable/cross-wired/symlinked paths, legitimate-set acceptance, the full `--checkpoint-step` matrix, and the new rename attack |
| `logs/reaudit5_probe_cli.py` / `.log` | the same refusals driven through `python scripts/76_select_discrete_checkpoint.py --d1c-formal` as a **subprocess**: exit code, message, and whether a selection file was written |
| `logs/reaudit5_probe_mutation.py` / `.log` | mutation testing of the **round-5** guards: each guard spliced out of `scripts/76` / `scripts/74` / the schema **in memory**, then the round-5 negative tests re-run through their own `_load_script` seam |
| `logs/reaudit5_probe_rest.py` / `.log` | round-5 reseal recompute, W1/W2 rebinding, v2.0.1 parent pointer, round-4 supersession, resume-hash regression, published-digest audit, prior-PASS spot checks |
| `logs/reaudit5_rerun_round4_probes.sh` + `reaudit5_rerun_r4_{dupckpt,prod,mutation}.log` | the **round-4 probes re-run verbatim** as a regression harness |
| `logs/reaudit5_full_pytest_tail.log` | independent full-suite reproduction (same `GIT_DIR` override, GPU masked) |
| `logs/reaudit5_attest.py` / `.log` | attestation audit (counts, collected ids, skip identities, new/changed tests, log-vs-source mtimes) + post-probe digests |
| `logs/reaudit5_post_probe_hashes.json` | post-probe digests of every source my probes could have touched |

Test environment: WSL2 Ubuntu, venv `/home/adam/projects/tts-self/.venv`, Python 3.11.15, rootdir
`/mnt/c/projects/tts-self-d1c`, `PYTHONPATH` pinned to the **worktree** `src`,
`CUDA_VISIBLE_DEVICES=` on every run.

**Why the repair agent's own probe was not sufficient.** `logs/repair_round5_probe_dupckpt.py:344`
does

```python
schema.validate_formal_report = lambda data, **kwargs: data
```

before exercising the selector, so its headline `N2 PASS` only shows that the selector refuses
**schema-bypassed forgeries**. It never showed that a duplicate-identity set which the *real* schema
accepts is refused — and it has to be shown, because identical files make claimed sha equal live
sha, so the schema is happy (my `D1`: 5/5 pass the real schema). Its `N1` also stubs `_load_model`
with `lambda p, d: (model, dict(cfg), {"cfg": cfg})`, a payload with no step, which is the exact
condition the test needs — a self-fulfilling seam. And it never checked that the *legitimate* set is
still accepted, which is the first thing a fail-closed rule can break.

---

## 1. Round-4 attack, reproduced on the production path — **PASS**

The round-4 attack script was re-run **byte-unmodified**:

```text
SystemExit: --checkpoint-step=250 cannot invent a step for /tmp/.../best.pt: formal D1c
            requires payload['step'] and/or candidate_step_XXXX.pt; CLI may only confirm
            a derived step
   … identically for 375, 500, 625, 750 …
N1 INFO five reports produced from ONE file: steps=[] distinct_ckpt_sha256=0
N2 INFO only 0 reports emitted; cannot test
```

My own reproduction adds the checks the round-4 script did not make:

```text
A1 PASS five main() runs on ONE best.pt emitted 0 formal reports; steps_emitted=[]
A2 PASS all five refusals are 'cannot invent'
A3 PASS no n2_gate.json leaked from refused runs: []
```

`A3` matters: the refusal happens before any eval work, so a refused run leaves **no** partial gate
artifact that a later step could pick up.

## 2. Duplicate `checkpoint.sha256` — **PASS, through the real schema**

```text
D1 PASS 5/5 duplicate-identity reports pass the REAL schema (claimed sha == live sha of
        identical files) — so the selector is the only layer that can refuse this shape
D2 PASS selector hard-refuses duplicate sha256: D1c formal selection refuses duplicate
        checkpoint.sha256 across candidates (distinct=1, n=5); plan §8.4 requires one
        checkpoint hash/step match per report
D3 PASS partial duplicate (2 of 5) refused (distinct=4, n=5)
```

`D3` is the case the repair agent did not test: an attacker does not need all five to collide, and
two colliding candidates are refused just as hard.

At the production CLI:

```text
K2 PASS CLI refuses the duplicate-checkpoint set: rc=1
K3 PASS no checkpoint_selection.json written on refusal (exists=False)
```

## 3. Hash / path verification — **PASS at every layer**

```text
H1 PASS schema refuses forged sha: checkpoint.sha256 does not match file at checkpoint.path
H2 PASS selector refuses a forged-sha report end to end
H3 PASS selector's OWN helper verifies live bytes (claimed=bbbb… live=dbcd8434…)
H4 PASS unreadable path: schema_accepts=True, selector_refuses ("not readable")
H5 PASS basename rule: 'best.pt' does not match expected 'candidate_step_0250.pt' for step=250
H6 PASS cross-wired basename (step-250 report bound to candidate_step_0750.pt) refused
H7 PASS five symlinks to ONE checkpoint refused
K4/K5 PASS CLI refuses forged sha, writes no selection
K6 PASS CLI refuses an unreachable checkpoint.path
K7 PASS formal mode still refuses flat/manual candidate inputs
```

`H3` and `H4` are the checks that prove the selector is an **independent** layer rather than a
restatement of the schema: `H3` bypasses the schema entirely and still gets refused, and `H4` is a
case the schema explicitly lets through (it only hashes when the path is reachable) that the
selector catches.

Observation, not a defect: the `resolved_paths` duplicate check at `scripts/76:94-98` is
unreachable in practice — any two reports resolving to the same file necessarily carry the same live
sha, so the distinct-sha rule fires first (visible in `H7`, which is refused for the sha reason).
Defensive depth, no action needed.

## 4. The fix is not a blanket refusal — **PASS**

A fail-closed rule that refuses everything would also "close" the gap. It does not:

```text
L1 PASS five real main() runs stamped steps=[250, 375, 500, 625, 750]
L2 PASS five real candidate files have 5 distinct sha256
L3 PASS selector ACCEPTS untouched legitimate production reports steps=[250,375,500,625,750]
L4 PASS selector selected step=250
K1 PASS CLI accepts the legit set: rc=0 wrote=True selected_step=250
```

The trainer writes `candidate_step_{step:04d}.pt` (`discrete_lm_trainer.py:690`), which is exactly
the basename the selector now demands, so the real Wave-3 protocol is not broken by the new rule.

## 5. `--checkpoint-step` confirm-only — **PASS** (with one documented message change)

```text
C1 PASS CLI confirm of derived step accepted -> step=500
C2 PASS foreign CLI step refused: '--checkpoint-step=625 conflicts with inferred step=500'
C3 PASS CLI invent on best.pt refused: 'cannot invent a step for …'
C4 PASS payload-derived step (best.pt with payload step=625) confirmed via CLI -> 625
C5 PASS documented: require=False (non-formal) still allows invent -> 750
C6 PASS out-of-set step refused: 'checkpoint step 999 not in contract.train.candidate_steps'
C7 PASS payload/filename mismatch refused
```

Re-running the **round-4** production probe verbatim gives 39/41, with two expected deltas:

| Round-4 check | Now | Reading |
|---|---|---|
| `E5` `--checkpoint-step` override on `best.pt` → `step=625` | **refuses**, `step=None` | This *is* the round-5 fix. The round-4 probe asserts the behaviour round 5 was asked to remove. |
| `E4` `--checkpoint-step 999` on `best.pt` → `"not in contract"` | refuses with `"cannot invent"` | Still fail-closed, refused one layer earlier with a different message. Membership enforcement is intact and verified separately by `C6`. |

Neither is a fail-open. No other round-4 production check changed.

## 6. NEW finding — the CLI can no longer invent a step, but `mv` can — **MEDIUM**

### What I found

`resolve_checkpoint_step` treats the filename as authoritative when the payload has no step:

```text
M3 INFO formal path with NO payload step: resolve -> 625 err='' (filename alone is authoritative)
```

So I took **one** model state, saved it five times under the five candidate names with **no payload
step** (a one-byte junk key makes the files byte-distinct, satisfying the new distinct-sha rule), and
drove production `main()` five times:

```text
M1 INFO rename-only attack: five reports from ONE model state with NO payload step ->
        steps=[250, 375, 500, 625, 750] distinct_sha=5
M2 FAIL SELECTOR ACCEPTED five renamed copies of ONE model state (selected step=250)
```

Round-5's rules prove the five *files* are distinct. They do not prove the five *training steps*
are. `--checkpoint-step` invent is closed; `mv` / re-save is not.

### Severity and why it is not a blocker under the stated rules

It is not one of the four blocker conditions: the round-4 duplicate-file attack is closed, hash
verification is present and live, the CLI no longer invents steps, and nothing regressed. Exploiting
this requires deliberately re-writing checkpoint bytes — a materially higher bar than round 4's
"pass a documented flag" — and the real Wave-3 protocol evaluates trainer-written candidates whose
payloads always carry `step`. But it is the same plan §8.4 property ("one checkpoint hash/step match
per report") reached by a different route, and it undermines the tamper-evidence the v2 schema
exists to provide.

### Recommended fix (before Wave 3/4)

1. **Require `payload['step']` on the D1c formal path.** The trainer always stamps it
   (`discrete_lm_trainer.py:646-648`: `step`, `global_step`, `optimizer_step`), so making it
   mandatory when `require=True` costs nothing and removes the filename-only route. Keep the
   existing payload-vs-filename cross-check as the second source.
2. **Bind candidates to the sealed run.** The trainer already writes
   `payload["hashes"] = dict(self.repro_hashes)` (run manifest / recipe / recipe source manifest /
   eval contract) into every checkpoint, and the evaluator **never reads it** (`RY1 INFO … no`).
   Stamp those hashes into `checkpoint` in the v2 report, verify `eval_contract_hash` against the
   contract being enforced, and have the selector require all five candidates to share one run
   identity. That turns "five distinct files" into "five checkpoints from one sealed run".

## 7. NEW finding — `WAVE1_REPORT.md` publishes superseded digests — **MEDIUM–LOW**

```text
H-REPAIR_ROUND5.md          PASS live_digest_present=True  round4_digest_present=False
H-d1c_reliability_gate.json PASS live_digest_present=True  round4_digest_present=False
H-WAVE1_REPORT.md           FAIL live_digest_present=False round4_digest_present=True
```

`WAVE1_REPORT.md:41-43` still publishes, under the heading "Contract 2.0.0", for the live path
`configs/eval_contracts/d1c_semantic_eos_v2_0_0.json`:

```text
sha256                        9922ce4d75932dcaff255dd20601f731f48c369647677c68249b9ff652d938af
recipe_sha256                 9922ce4d…
recipe_source_manifest_sha256 62eeeae2c6360e88923dd6416e8c5a22403d869cbbdf806aabce31e6a512b81d
```

The live values are `cff9e9f5…` / `e505acf9…`. The same stale table exists in the Windows mirror at
`C:\projects\tts-self\.cursor\results\d1c_semantic_eos_repair_b81ecc3e\WAVE1_REPORT.md`.

This is a documentation regression introduced by round 5, not a code defect — every executable
binding (`W1`/`W2` YAML, `v2.0.1` parent pointer, `d1c_reliability_gate.json`, `scripts/72`) is
correctly rebound to the live digest, and `resolve_repro_hashes` now refuses a config pinned to the
round-4 value (`R7b2`). But rounds 3 and 4 both kept this artifact in sync (round 3's `H6`, round
4's `H-WAVE1_REPORT.md`), and a human auditor verifying the contract against the Wave-1 report would
verify against the wrong digest. **Update `WAVE1_REPORT.md` (both copies) to `cff9e9f5…` /
`e505acf9…` before human sign-off.**

## 8. Do the round-5 tests actually protect the round-5 fix? — **4 of 5**

Each guard was spliced out **in memory** and the round-5 negative tests re-run:

```text
Y0a-e PASS unmutated: all five round-5 tests green
Y1a  PASS MUTATION selector skips _require_distinct_checkpoint_identities
          -> test_selector_refuses_duplicate_checkpoint_hashes DID NOT RAISE
Y1b  PASS  … -> test_selector_refuses_checkpoint_hash_mismatch DID NOT RAISE
Y1c  PASS  … -> test_main_stamps_checkpoint_step_and_selector_e2e DID NOT RAISE
Y2a  PASS MUTATION drop on-disk verify -> duplicate test still green (different guard)
Y2b  PASS MUTATION drop on-disk verify -> hash-mismatch test DID NOT RAISE
Y3   FAIL MUTATION drop candidate_step_<step>.pt basename rule -> caught by NOTHING
Y4   PASS MUTATION drop distinct-sha rule -> caught by 2 tests
Y5   PASS MUTATION schema skips on-disk hash -> schema test DID NOT RAISE
Y6   PASS MUTATION restore round-4 --checkpoint-step invent -> CLI test DID NOT RAISE
Y7   PASS at least one attack test runs the REAL schema before the selector
Y8   INFO the two new selector tests stub validate_formal_report to a pass-through (3 sites)
```

**`Y3` is a coverage hole.** `REPAIR_ROUND5.md` lists the basename rule as fix rule #2, but deleting
it leaves the entire 407-test suite green. The rule *works* in the shipped code (`H5`, `H6`), so this
is missing coverage, not a live defect. **[LOW] Add a selector test with a report whose
`checkpoint.path` basename disagrees with its own step** — the plan's "checkpoint hash/step mismatch
rejected" case is only half covered without it.

`Y8` is a disclosure, not a defect: the two new selector unit tests monkeypatch
`validate_formal_report` to a pass-through (restored in `finally`), which is a legitimate way to
unit-test the selector in isolation. It is acceptable only because
`test_main_stamps_checkpoint_step_and_selector_e2e` runs the **real** schema on the attack shape
before handing it to the selector (`Y7`), and my `D1`/`D2` prove the un-bypassed path independently.

## 9. Prior FAILs and spot checks — all still hold

### FAIL #2 — contract / recipe / manifest binding after the round-5 reseal — **PASS**

```text
R2a  PASS declared sha256 == independent recompute cff9e9f5dd11ae66
R2b  PASS REPAIR_ROUND5 claimed contract_sha256 matches live
R2c  PASS recipe_sha256 == contract sha
R2c2 PASS round-4 contract digest superseded (old=9922ce4d75932dca)
R2d  PASS tampering infer.max_frames changes the content sha (seal is live)
R2e  PASS recipe_source_manifest n_locked=19 mismatches=[]
R2f  PASS recipe_source_manifest_sha256 live=e505acf99640a3c8 == claim
R2f2 PASS round-4 manifest digest superseded (old=62eeeae2c6360e88)
R2g  PASS round-5 edited sources are inside the locked manifest (missing=[])
R2h-w1/w2 PASS both recipes bind contract/recipe/manifest to the live digests
R2i  PASS v2.0.1 parent pointer repointed at the live 2.0.0 digest
R2j  PASS v2.0.1 no longer references the superseded round-4 contract digest
R2k  PASS scripts/72 still asserts declared eval_contract_hash before stamping
```

`R2g` is the check that matters after an edit-then-reseal round: the three files round 5 changed
(`scripts/74`, `scripts/76`, `discrete_report_schema.py`) are all inside the 19 locked sources, so
the reseal covers the edits rather than routing around them.

### FAIL #7 — resume hash checks — **PASS**

```text
R7a-w1/w2 PASS resolve_repro_hashes inert keys=[] (n=8)
R7b  PASS stale eval_contract_hash raises
R7b2 PASS a config pinned to the superseded ROUND-4 digest is now refused
R7c-{run_manifest,recipe,recipe_source_manifest,eval_contract}_hash PASS backfill still works
R7d  PASS D1c resume fail-closed message still present in the trainer
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

### main()-level regression strength — **PASS (unchanged)**

The round-4 mutation probe re-run verbatim against round-5 sources: `X0`–`X6`, **0 fails**. The
round-2 bare-float rewire and the D1b-gate swap still turn the suite red.

## 10. Pytest attestation — **VALID**

| Check | Result |
|---|---|
| Claimed | `404 passed, 3 skipped, 2 warnings in 55.52s`, 407 collected |
| Independent reproduction | `404 passed, 3 skipped, 2 warnings in 52.67s` — **exact match**, identical dot/skip pattern and warning block |
| Collect-only | **407** ids, all unique |
| Three claimed NEW tests | `test_selector_refuses_duplicate_checkpoint_hashes`, `test_selector_refuses_checkpoint_hash_mismatch`, `test_validate_formal_report_rejects_checkpoint_hash_mismatch` — all collected |
| Two claimed CHANGED tests | `test_resolve_checkpoint_step_payload_filename_cli`, `test_main_stamps_checkpoint_step_and_selector_e2e` — collected |
| Arithmetic | round-4 404 collected → round-5 407; +3 matches the three new tests; 404 passed + 3 skipped = 407 |
| Skip identities | opt-in gloo DDP, opt-in DDP smoke, `CUDA required` — GPU masking disclosed correctly |
| Focused claim | `28 passed in 3.29s` — matches the log |
| Claimed log mtime | `2026-08-04 16:42:19 UTC` |
| Newest attested source | `tests/test_d1c_infra_repair.py` @ `2026-08-04 16:41:02 UTC` — **no attested source post-dates the log** |

Standing caveat, correctly disclosed by the repair agent: the green result requires
`GIT_DIR`/`GIT_WORK_TREE` pointing at `/mnt/c/projects/tts-self/.git/worktrees/tts-self-d1c`,
otherwise `tests/test_mel_baseline.py::test_git_sha_and_run_manifest` fails. That is an artifact of
running a Windows worktree through `/mnt/c` and would not occur in the canonical WSL clone.

**Attestation caveat.** A green suite is still not evidence that the D1c formal path works, and this
suite would **not** catch the M2 rename attack (no test asserts that a candidate's step is bound to
its payload) nor a removal of the basename rule (`Y3`).

## 11. LOW-item claims in `REPAIR_ROUND5.md`

| Claim | This audit |
|---|---|
| Cover real `_load_model` payload step (round-4 X4) — deferred | **Agreed, still open**; round-4's `B2` and my `A4` cover it live, the suite does not |
| Weight the EOS accuracy aggregate — deferred | **Agreed, still open** (selector uses strict `== 1.0`, so only the published corpus number is skewed) |
| Pin `verify_recipe_source_manifest` to repo root — deferred | **Agreed, still open** (fail-closed, therefore safe) |
| Assert rather than overwrite `recipe_hash` in `scripts/72` — deferred | **Agreed, still open** |
| Non-D1c resume fail-open — deferred | **Agreed, still open** (D1c path is fail-closed) |
| CER margin tolerance at `scripts/76` — deferred | **Agreed, still open** |
| "Round-4 digests are superseded" | **Partly false** — superseded in code and in the gate JSON, **not** in `WAVE1_REPORT.md` (§7) |

## 12. Remaining defects, in fix order

1. **[MEDIUM — NEW] Bind a candidate to its training step, not just to distinct bytes.** Require
   `payload['step']` when `require=True`, and verify/propagate `payload['hashes']` so all five
   candidates must come from one sealed run. (`M1`–`M3`)
2. **[MEDIUM–LOW — NEW] Update `WAVE1_REPORT.md`** (worktree **and** the `C:\projects\tts-self`
   mirror) to the live `cff9e9f5…` / `e505acf9…` digests. (`H-WAVE1_REPORT.md`)
3. **[LOW — NEW] Test the basename rule.** Deleting it leaves the suite green. (`Y3`)
4. **[LOW] `--promote-best` on the D1c formal branch has no symlink guard.** `scripts/76:584-596`
   lacks the `src.is_symlink()` refusal that the D1b branch grew at `scripts/76:707-710`
   (commit `dc74937`). With `--best-path` pointing at an existing `best.pt` symlink, `shutil.copy2`
   would write **through** the link into whatever it targets. Pre-existing, not a round-5
   regression, but this is now the only promote path for D1c. (`RY2`)
5. **[LOW] Redundant full-file hashing.** `validate_formal_report` hashes the checkpoint whenever the
   path is reachable, `_require_distinct_checkpoint_identities` hashes it again, and
   `build_formal_report` re-hashes at emit time — ~10 full reads of multi-GB checkpoints per
   selection, over `/mnt/c` in this environment. Cache per resolved path.
6. **[LOW] Cover the real `_load_model`** with one assertion that it returns a payload carrying
   `step` (round-4's open X4 note).
7. **[LOW] Weight the EOS accuracy aggregate** by utterance count (or publish min alongside mean).
8. **[LOW] Pin `verify_recipe_source_manifest` to the repo root** in `enforce_eval_contract`.
9. **[LOW] Assert rather than overwrite** `recipe_hash` / `recipe_source_manifest_hash` at
   `scripts/72:483-488`.
10. **[LOW] Close the non-D1c resume fail-open** at `discrete_lm_trainer.py:784-792`.
11. **[LOW] Use a tolerance** for `macro_cer <= copy_macro + 0.03` at `scripts/76:180-182`
    (the round-4 report cited `133-135`; round 5 shifted the line numbers).

## Standing non-code blockers (unchanged, correctly disclosed by the repair agent)

1. No clean committed D1c source SHA. Branch `d1c/semantic-eos` is still at baseline `0661d10` with
   all D1c work uncommitted, so `require_clean_sha` refuses — correctly. Requires explicit user
   commit/push authorization.
2. External confirmation-seed commitment still absent and correctly **not** invented.
3. Wave 3 GPU smoke and Wave 4 DEV remain NO-GO pending human sign-off.

## Scope confirmations and post-audit integrity

- No commit created, no push, no branch mutation. `git rev-parse HEAD` = `0661d10eb616…` before and
  after; `git status --porcelain` = **31 entries** before and after.
- No GPU work; `CUDA_VISIBLE_DEVICES=` on every run. No training, decode, or model synthesis, and no
  new `best.pt` / `last.pt`, so **no listen-demo obligation arises**. (The disclosed
  `free_running_asr_eval` stub writes placeholder PCM ramps into `TemporaryDirectory` scratch space
  that is deleted on exit; nothing is generated by a model and nothing survives the probe.)
- No seed invented; no methodology file touched; no Wave 0–2 PASS claim.
- Probe scratch files were written only under `/tmp` and
  `.cursor/results/d1c_semantic_eos_repair_b81ecc3e/logs/`. No round-4 log or artifact was
  overwritten (my re-runs of the round-4 probes tee to `reaudit5_rerun_r4_*.log`).
- Post-audit source integrity is byte-identical to the pre-audit snapshot
  (`logs/reaudit5_post_probe_hashes.json`):

```text
7361cd65886f2317  scripts/74_eval_discrete_lm.py
fe529dda61c0f450  scripts/76_select_discrete_checkpoint.py
49dffe0e8a9e7fbd  src/minispeech/eval/discrete_report_schema.py
7d85475e6ab3db19  src/minispeech/eval/discrete_n2.py
8b665b8c9faf8304  src/minispeech/train/discrete_lm_trainer.py
77e0d30f12ef2a2d  tests/test_d1c_infra_repair.py
4a55bba4eaf5ebbc  configs/eval_contracts/d1c_semantic_eos_v2_0_0.json
7c77590723a1d002  configs/eval_contracts/d1c_semantic_eos_v2_0_1.json
0b8954784e67fe75  configs/discrete_speech_lm_d1c_semantic_eos_w1.yaml
73ac1e27ecd5ddbe  configs/discrete_speech_lm_d1c_semantic_eos_w2.yaml
```

- The frozen manifest still verifies after every probe: `n_locked = 19`, `mismatches = []`,
  `contract sha256 = cff9e9f5dd11ae66d66e44eddc8768f4989daa8c0339822d77252289a155baad`.
- Waves 0–2 remain **not PASS**. Wave 3 remains **NO-GO**.
